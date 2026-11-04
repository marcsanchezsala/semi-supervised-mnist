import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 5000, input_dropout: float = 0.5, hidden_dropout: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, in_features))
        self.enc_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dec_bias = nn.Parameter(torch.zeros(in_features))
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        # Unscaled dropout matches Hinton (2012) / Eq. (9) half-step gradient scale
        x_tilde = F.dropout(x_flat, p=self.input_dropout, training=True) * (1.0 - self.input_dropout) if self.training else x_flat * (1.0 - self.input_dropout)
        h = F.relu(F.linear(x_tilde, self.weight, self.enc_bias))
        h = F.dropout(h, p=self.hidden_dropout, training=True) * (1.0 - self.hidden_dropout) if self.training else h * (1.0 - self.hidden_dropout)
        return F.linear(h, self.weight.t(), self.dec_bias)


class DropNN(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 5000, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

        for m in (self.fc1, self.fc2):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x.view(x.size(0), -1)))
        # Unscaled dropout matches Hinton (2012) / Eq. (9) half-step gradient scale with lr=1.5
        h = F.dropout(h, p=self.dropout, training=True) * (1.0 - self.dropout) if self.training else h * (1.0 - self.dropout)
        return self.fc2(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

    def load_dae(self, dae: DenoisingAutoencoder) -> None:
        with torch.no_grad():
            # Scale W by (1 - p_in) because DropNN receives clean x instead of 50%-masked x_tilde
            self.fc1.weight.copy_(dae.weight * (1.0 - dae.input_dropout))
            self.fc1.bias.copy_(dae.enc_bias)


def bce_sum_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(dim=1).mean()


def conditional_entropy(probs: torch.Tensor) -> float:
    return -(probs * probs.clamp(min=torch.finfo(probs.dtype).tiny).log()).sum(dim=1).mean().item()


if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. DAE checks: shapes, tied-weights, backward flow
    dae = DenoisingAutoencoder()
    x = torch.randn(4, 784)
    recon = dae(x)
    assert recon.shape == (4, 784), f"Expected DAE shape (4, 784), got {recon.shape}"

    loss_dae = bce_sum_loss(recon, torch.rand(4, 784))
    loss_dae.backward()
    assert dae.weight.grad is not None and torch.isfinite(dae.weight.grad).all()

    # Verify 5 SGD steps strictly decrease reconstruction loss on fixed binary batch
    dae_opt = DenoisingAutoencoder()
    opt = torch.optim.SGD(dae_opt.parameters(), lr=0.1)
    x_fixed = (torch.rand(8, 784) > 0.5).float()
    dae_opt.train()
    init_loss = bce_sum_loss(dae_opt(x_fixed), x_fixed).item()
    for _ in range(5):
        opt.zero_grad()
        loss = bce_sum_loss(dae_opt(x_fixed), x_fixed)
        loss.backward()
        opt.step()
    final_loss = bce_sum_loss(dae_opt(x_fixed), x_fixed).item()
    assert final_loss < init_loss, f"DAE loss did not decrease: {init_loss} -> {final_loss}"

    # 2. DropNN.load_dae match in eval mode
    m = DropNN()
    m.load_dae(dae)
    m.eval()
    with torch.no_grad():
        x_eval = torch.randn(4, 784)
        expected = F.linear(x_eval * (1.0 - dae.input_dropout), dae.weight, dae.enc_bias)
        actual = m.fc1(x_eval)
        assert torch.allclose(expected, actual, atol=1e-6), "Pre-activation mismatch after load_dae"

    # 3. DropNN forward/backward and conditional entropy bounds
    m.train()
    logits = m(x)
    assert logits.shape == (4, 10)
    loss_clf = bce_sum_loss(logits, F.one_hot(torch.tensor([0, 1, 2, 3]), 10).float())
    loss_clf.backward()
    assert m.fc1.weight.grad is not None and torch.isfinite(m.fc1.weight.grad).all()

    probs = m.predict_proba(x)
    h = conditional_entropy(probs)
    assert 0.0 <= h <= (10.0 / 2.7182818) + 1e-4, f"Entropy out of bounds: {h}"
    print("02_DAE_pretraining self-check passed successfully.")
