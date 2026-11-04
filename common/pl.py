import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Reuse DropNN, bce_sum_loss, conditional_entropy from 01_baseline without copying
_baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../01_baseline"))
if _baseline_dir not in sys.path:
    sys.path.insert(0, _baseline_dir)

from model import DropNN, bce_sum_loss, conditional_entropy
from lock import LockState, confident_mask, update_lock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_INIT = 1.5
LR_DECAY = 0.998
P_INIT = 0.5
P_FINAL = 0.99
T_MOM = 500
T1 = 100
T2 = 600
ALPHA_F = 3.0
BATCH_L = 32
BATCH_U = 256
EPOCHS = 1000

torch._dynamo.config.trace_autograd_ops = True
_compile = torch.compile if device.type == "cuda" else (lambda fn: fn)


def load_mnist(root: str = "./data"):
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True)
    X_train = (mnist_train.data.float() / 255.0).view(-1, 784).to(device)
    y_train = mnist_train.targets.to(device)
    X_test = (mnist_test.data.float() / 255.0).view(-1, 784).to(device)
    y_test = mnist_test.targets.to(device)
    return X_train, y_train, X_test, y_test


def make_splits(X: torch.Tensor, y: torch.Tensor, n_labeled: int, n_val: int = 1000, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    n_classes = 10
    val_per_class = n_val // n_classes
    lab_per_class = n_labeled // n_classes

    val_idx, lab_idx, unlab_idx = [], [], []
    for c in range(n_classes):
        c_idx = (y == c).nonzero(as_tuple=True)[0]
        c_perm = torch.randperm(len(c_idx), generator=g)
        c_shuffled = c_idx[c_perm.to(c_idx.device)]
        val_idx.append(c_shuffled[:val_per_class])
        lab_idx.append(c_shuffled[val_per_class : val_per_class + lab_per_class])
        unlab_idx.append(c_shuffled[val_per_class + lab_per_class :])

    idx_v = torch.cat(val_idx)
    idx_l = torch.cat(lab_idx)
    idx_u = torch.cat(unlab_idx)

    assert len(idx_v) == n_val, f"Validation size mismatch: {len(idx_v)} != {n_val}"
    assert len(idx_l) == n_labeled, f"Labeled size mismatch: {len(idx_l)} != {n_labeled}"
    assert len(idx_u) == len(X) - n_val - n_labeled, f"Unlabeled size mismatch: {len(idx_u)}"
    assert (torch.bincount(y[idx_v], minlength=10) == val_per_class).all(), "Val class imbalance"
    assert (torch.bincount(y[idx_l], minlength=10) == lab_per_class).all(), "Labeled class imbalance"

    return X[idx_l], y[idx_l], X[idx_u], y[idx_u], X[idx_v], y[idx_v]


def alpha_schedule(t: int, use_pl: bool) -> float:
    if not use_pl or t < T1:
        return 0.0
    if t < T2:
        return (t - T1) / (T2 - T1) * ALPHA_F
    return ALPHA_F


@_compile
def _step_sup(w1, b1, w2, b2, vw1, vb1, vw2, vb2, x_l, y_l_oh, neg_eps, p_t, one_minus_p):
    h = F.relu(F.linear(x_l, w1, b1))
    h = F.dropout(h, p=0.5, training=True) * 0.5
    logits_l = F.linear(h, w2, b2)
    loss = bce_sum_loss(logits_l, y_l_oh)
    gw1, gb1, gw2, gb2 = torch.autograd.grad(loss, (w1, b1, w2, b2))
    with torch.no_grad():
        vw1.mul_(p_t).add_(gw1 * one_minus_p)
        vb1.mul_(p_t).add_(gb1 * one_minus_p)
        vw2.mul_(p_t).add_(gw2 * one_minus_p)
        vb2.mul_(p_t).add_(gb2 * one_minus_p)
        w1.add_(vw1 * neg_eps)
        b1.add_(vb1 * neg_eps)
        w2.add_(vw2 * neg_eps)
        b2.add_(vb2 * neg_eps)


@_compile
def _step_pl(w1, b1, w2, b2, vw1, vb1, vw2, vb2, x_cat, y_l_oh, fixed_mask, fixed_y, w_u, alpha_t, neg_eps, p_t, one_minus_p):
    h = F.relu(F.linear(x_cat, w1, b1))
    h = F.dropout(h, p=0.5, training=True) * 0.5
    logits_cat = F.linear(h, w2, b2)
    logits_l = logits_cat[:BATCH_L]
    logits_u = logits_cat[BATCH_L:]
    argmax_y = F.one_hot(logits_u.detach().argmax(dim=1), 10).to(logits_u.dtype)
    tgt = torch.where(fixed_mask[:, None], fixed_y, argmax_y)
    per = F.binary_cross_entropy_with_logits(logits_u, tgt, reduction="none").sum(dim=1)
    loss = bce_sum_loss(logits_l, y_l_oh) + alpha_t * (per * w_u).mean()
    gw1, gb1, gw2, gb2 = torch.autograd.grad(loss, (w1, b1, w2, b2))
    with torch.no_grad():
        vw1.mul_(p_t).add_(gw1 * one_minus_p)
        vb1.mul_(p_t).add_(gb1 * one_minus_p)
        vw2.mul_(p_t).add_(gw2 * one_minus_p)
        vb2.mul_(p_t).add_(gb2 * one_minus_p)
        w1.add_(vw1 * neg_eps)
        b1.add_(vb1 * neg_eps)
        w2.add_(vw2 * neg_eps)
        b2.add_(vb2 * neg_eps)


def test_error(m: DropNN, X: torch.Tensor, y: torch.Tensor) -> float:
    m.eval()
    with torch.no_grad():
        preds = m(X).argmax(dim=1)
        return (preds != y).float().mean().item() * 100.0


def train(
    X_l: torch.Tensor,
    y_l: torch.Tensor,
    X_u: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    mode: str,
    y_u: torch.Tensor = None,
    epochs: int = EPOCHS,
    hi: float = 0.9,
    lo: float = 0.3,
    patience: int = 2,
    check_every: int = 5,
    verbose: bool = True,
):
    model = DropNN().to(device)
    w1, b1, w2, b2 = model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias
    vw1 = torch.zeros_like(w1)
    vb1 = torch.zeros_like(b1)
    vw2 = torch.zeros_like(w2)
    vb2 = torch.zeros_like(b2)

    n_u = len(X_u)
    st = LockState(n_u, device=device)
    conf = torch.zeros(n_u, dtype=torch.bool, device=device)
    active = torch.arange(n_u, device=device)
    all_u_idx = torch.arange(n_u, device=device)

    best_val_err = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    Y_l_oh = F.one_hot(y_l, 10).float()

    coverage = float("nan")
    precision = float("nan")
    last_probs = None
    steps_last_epoch = n_u // BATCH_U

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for t in range(epochs):
        eps_t = LR_INIT * (LR_DECAY ** t)
        p_t = (t / T_MOM) * P_FINAL + (1.0 - t / T_MOM) * P_INIT if t < T_MOM else P_FINAL
        use_pl = (mode != "dropNN")
        alpha = alpha_schedule(t, use_pl)
        sched = torch.tensor([-eps_t, p_t, 1.0 - p_t, alpha], device=device)
        neg_eps, p_ten, one_minus_p, alpha_ten = sched[0], sched[1], sched[2], sched[3]

        # Periodic lock / threshold evaluation on full unlabeled set
        if alpha > 0.0 and mode in {"thr", "lock", "retire"} and (t % check_every == 0):
            model.eval()
            probs_chunks = []
            with torch.no_grad():
                for i in range(0, n_u, 8192):
                    probs_chunks.append(torch.sigmoid(model(X_u[i : i + 8192])))
                last_probs = torch.cat(probs_chunks, dim=0)
            if mode == "thr":
                conf = confident_mask(last_probs, hi, lo)
            else:
                update_lock(last_probs, st, hi, lo, patience)
                if mode == "retire":
                    active = (~st.locked).nonzero().squeeze(1)
            model.train()

        # Batch construction and optimization step
        if alpha == 0.0 or mode == "dropNN":
            steps = n_u // BATCH_U
            steps_last_epoch = steps
            l_indices = torch.randint(0, len(X_l), (steps, BATCH_L), device=device)
            x_l_all = X_l[l_indices]
            y_l_all = Y_l_oh[l_indices]
            model.train()
            for step in range(steps):
                _step_sup(w1, b1, w2, b2, vw1, vb1, vw2, vb2, x_l_all[step], y_l_all[step], neg_eps, p_ten, one_minus_p)
        else:
            pool = active if mode == "retire" else all_u_idx
            steps = len(pool) // BATCH_U
            steps_last_epoch = steps
            if steps == 0:
                # Retire pool exhausted: fall back to supervised-only step path
                steps = n_u // BATCH_U
                steps_last_epoch = steps
                l_indices = torch.randint(0, len(X_l), (steps, BATCH_L), device=device)
                x_l_all = X_l[l_indices]
                y_l_all = Y_l_oh[l_indices]
                model.train()
                for step in range(steps):
                    _step_sup(w1, b1, w2, b2, vw1, vb1, vw2, vb2, x_l_all[step], y_l_all[step], neg_eps, p_ten, one_minus_p)
            else:
                # In 'retire', steps per epoch shrink with the active pool to save compute
                perm = torch.randperm(len(pool), device=device)
                u_idx_all = pool[perm[: steps * BATCH_U]].view(steps, BATCH_U)
                l_indices = torch.randint(0, len(X_l), (steps, BATCH_L), device=device)
                x_cat_all = torch.cat([X_l[l_indices], X_u[u_idx_all]], dim=1)
                y_l_all = Y_l_oh[l_indices]
                fixed_mask_all = st.locked[u_idx_all] if mode == "lock" else torch.zeros((steps, BATCH_U), dtype=torch.bool, device=device)
                fixed_y_all = F.one_hot(st.label[u_idx_all], 10).float()
                w_all = conf[u_idx_all].float() if mode == "thr" else torch.ones((steps, BATCH_U), dtype=torch.float32, device=device)

                model.train()
                for step in range(steps):
                    _step_pl(
                        w1, b1, w2, b2, vw1, vb1, vw2, vb2,
                        x_cat_all[step], y_l_all[step],
                        fixed_mask_all[step], fixed_y_all[step], w_all[step],
                        alpha_ten, neg_eps, p_ten, one_minus_p,
                    )

        # Validation and diagnostics
        if (t + 1) % 25 == 0 or t == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_err = (model(X_val).argmax(dim=1) != y_val).float().mean().item() * 100.0
            if val_err < best_val_err:
                best_val_err = val_err
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if y_u is not None:
                if mode in {"lock", "retire"}:
                    mask = st.locked
                    coverage = mask.float().mean().item() * 100.0
                    precision = (st.label == y_u)[mask].float().mean().item() * 100.0 if mask.any() else float("nan")
                elif mode == "thr":
                    mask = conf
                    coverage = mask.float().mean().item() * 100.0
                    if mask.any() and last_probs is not None:
                        precision = (last_probs.argmax(dim=1) == y_u)[mask].float().mean().item() * 100.0
                    else:
                        precision = float("nan")
                else:
                    coverage = float("nan")
                    precision = float("nan")

            if verbose and ((t + 1) % 250 == 0 or t == epochs - 1):
                cur_pool = len(active) if mode == "retire" else n_u
                print(
                    f"[{mode}] epoch {t+1:4d}/{epochs} | val_err={val_err:.2f}% | "
                    f"cov={coverage:.2f}% | prec={precision:.2f}% | pool={cur_pool}",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - t0

    # Compute last test error before loading best state
    test_last = test_error(model, X_test, y_test)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    peak_mb = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0

    info = {
        "test_last": test_last,
        "best_val_err": best_val_err,
        "seconds": elapsed_seconds,
        "peak_mb": peak_mb,
        "coverage": coverage,
        "precision": precision,
        "steps_last_epoch": steps_last_epoch,
    }
    return model, info
