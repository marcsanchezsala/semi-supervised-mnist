import torch


class LockState:
    def __init__(self, n: int, device=None):
        self.locked = torch.zeros(n, dtype=torch.bool, device=device)
        self.label = torch.zeros(n, dtype=torch.long, device=device)
        self.cand = torch.full((n,), -1, dtype=torch.long, device=device)
        self.streak = torch.zeros(n, dtype=torch.long, device=device)


def confident_mask(probs: torch.Tensor, hi: float = 0.9, lo: float = 0.3) -> torch.Tensor:
    top, cls = probs.max(dim=1)
    others = probs.scatter(1, cls[:, None], 0.0).max(dim=1).values
    return (top >= hi) & (others < lo)


@torch.no_grad()
def update_lock(
    probs: torch.Tensor,
    st: LockState,
    hi: float = 0.9,
    lo: float = 0.3,
    patience: int = 2,
) -> LockState:
    conf = confident_mask(probs, hi, lo)
    cls = probs.argmax(dim=1)
    agree = conf & (cls == st.cand)
    st.streak = torch.where(agree, st.streak + 1, conf.long())
    st.cand = cls
    new = ~st.locked & (st.streak >= patience)
    st.locked |= new
    st.label = torch.where(new, cls, st.label)
    return st


if __name__ == "__main__":
    # 1. Confident mask checks
    p1 = torch.zeros(3, 10)
    p1[0, 2] = 0.95
    p1[0, 5] = 0.50  # top 0.95, second 0.50 -> NOT confident (others >= 0.3)
    p1[1, 3] = 0.95
    p1[1, 4] = 0.20  # top 0.95, second 0.20 -> confident
    p1[2, 7] = 0.85
    p1[2, 1] = 0.10  # top 0.85 (< 0.90) -> NOT confident

    m = confident_mask(p1, hi=0.9, lo=0.3)
    assert not m[0].item(), "Sample with top 0.95 but another class at 0.5 must not be confident"
    assert m[1].item(), "Sample with top 0.95 and others < 0.3 must be confident"
    assert not m[2].item(), "Sample with top 0.85 (< 0.90) must not be confident"

    # 2. Patience streak & agreeing check
    st = LockState(2)
    p_step1 = torch.zeros(2, 10)
    p_step1[0, 4] = 0.95  # sample 0 confident class 4
    p_step1[1, 6] = 0.95  # sample 1 confident class 6

    update_lock(p_step1, st, patience=2)
    assert not st.locked[0].item() and not st.locked[1].item(), "Sample must not lock after 1 check (patience=2)"
    assert st.streak[0].item() == 1 and st.streak[1].item() == 1

    # Check non-agreeing confident step for sample 1 (flips from 6 to 8)
    p_step2 = torch.zeros(2, 10)
    p_step2[0, 4] = 0.95  # sample 0 agrees on class 4 (2nd consecutive)
    p_step2[1, 8] = 0.95  # sample 1 confident on class 8 (disagrees with previous class 6)

    update_lock(p_step2, st, patience=2)
    assert st.locked[0].item(), "Sample 0 must lock after 2 consecutive agreeing checks"
    assert st.label[0].item() == 4, "Sample 0 locked label must be 4"
    assert not st.locked[1].item(), "Sample 1 must not lock due to class change"
    assert st.streak[1].item() == 1, "Sample 1 streak must reset to 1 on class change"

    # 3. Contradicting probs do NOT change label or locked flag once locked
    p_step3 = torch.zeros(2, 10)
    p_step3[0, 9] = 0.98  # sample 0 fed strong contradicting class 9
    p_step3[1, 8] = 0.95  # sample 1 agrees on class 8 (2nd consecutive)

    update_lock(p_step3, st, patience=2)
    assert st.locked[0].item(), "Sample 0 must remain locked"
    assert st.label[0].item() == 4, "Sample 0 label must never change once locked"
    assert st.locked[1].item(), "Sample 1 must now lock on class 8"
    assert st.label[1].item() == 8

    print("Lock self-check passed successfully.")
