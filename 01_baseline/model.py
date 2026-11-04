import torch
import torch.nn as nn
import torch.nn.functional as F


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


def bce_sum_loss(logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none").sum(dim=1).mean()


def conditional_entropy(probs: torch.Tensor) -> float:
    return -(probs * probs.clamp(min=torch.finfo(probs.dtype).tiny).log()).sum(dim=1).mean().item()


if __name__ == "__main__":
    torch.manual_seed(42)
    m = DropNN()
    x = torch.randn(4, 784)

    # Output shapes
    logits = m(x)
    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"
    probs = m.predict_proba(x)
    assert probs.shape == (4, 10) and (probs >= 0.0).all() and (probs <= 1.0).all()

    # Eval determinism vs train stochasticity
    m.eval()
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
        assert torch.allclose(out1, out2), "Eval mode must be deterministic"

    m.train()
    train_out1 = m(x)
    train_out2 = m(x)
    assert not torch.allclose(train_out1, train_out2), "Train mode with dropout must be stochastic"

    # Loss and backward gradient check
    targets = F.one_hot(torch.tensor([0, 1, 2, 3]), 10).float()
    loss = bce_sum_loss(train_out1, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.fc1.weight.grad is not None and torch.isfinite(m.fc1.weight.grad).all()

    # Conditional entropy bounds for 10 independent sigmoid units: [0, 10 / e]
    h = conditional_entropy(probs)
    assert 0.0 <= h <= (10.0 / 2.7182818) + 1e-4, f"Entropy out of bounds: {h}"
    print("Self-check passed successfully.")
