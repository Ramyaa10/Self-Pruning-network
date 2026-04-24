"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Internship — Case Study

Architecture:
  - Custom PrunableLinear layers with learnable gate_scores
  - Gates = sigmoid(gate_scores), multiplied element-wise with weights
  - Total Loss = CrossEntropyLoss + λ * L1(gates)
  - Three λ values compared: 1e-5 (low), 1e-4 (medium), 1e-3 (high)

Usage:
  python self_pruning_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable gate ∈ (0, 1).  When a gate collapses to ~0 the corresponding
    weight is effectively removed from the network.

    Parameters
    ----------
    in_features  : int  — size of each input sample
    out_features : int  — size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias — same shapes as nn.Linear
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight.
        # sigmoid(gate_scores) gives the actual gate values ∈ (0, 1).
        # Initialise to 0  →  initial gates ≈ 0.5 (all connections open).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming-uniform init for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Compute gates  = sigmoid(gate_scores)       ∈ (0, 1)
          2. Gated weights  = weight  ⊙  gates           (element-wise)
          3. Output         = x @ gated_weights.T + bias

        Gradients flow through both `weight` and `gate_scores` via the
        chain rule:
          ∂L/∂weight      = ∂L/∂out · (x ⊙ gates)
          ∂L/∂gate_scores = ∂L/∂out · (x ⊙ weight) · sigmoid'(gate_scores)
        """
        gates         = torch.sigmoid(self.gate_scores)          # (out, in)
        pruned_weights = self.weight * gates                      # element-wise
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached for inspection)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-Pruning Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Four-layer feed-forward network for CIFAR-10 (10 classes).
    All linear layers are replaced with PrunableLinear.

    Input: 3 × 32 × 32  →  flattened to 3072
    Architecture: 3072 → 512 → 256 → 128 → 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,  num_classes)

        # Keep a list for easy iteration over all prunable layers
        self.prunable_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

        self.dropout = nn.Dropout(p=0.3)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten: (B, 3072)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # ------------------------------------------------------------------
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Why L1?  The L1 penalty (sum of absolute values) is well-known to
        promote *exact* sparsity — i.e., driving values to precisely 0 — as
        opposed to L2 which merely makes values small.  Because our gates are
        always in (0, 1) after the sigmoid, the L1 norm equals their sum.

        Returns a scalar tensor (differentiable w.r.t. gate_scores).
        """
        total = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)   # keep graph alive
            total = total + gates.sum()
        return total

    # ------------------------------------------------------------------
    def count_total_gates(self) -> int:
        return sum(
            layer.gate_scores.numel() for layer in self.prunable_layers
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256, data_root: str = "./data"):
    """
    Download (if needed) and return CIFAR-10 train / test DataLoaders.
    Uses standard per-channel mean/std normalisation.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:          SelfPruningNet,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    lambda_sparse:  float,
    device:         torch.device,
) -> tuple[float, float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_total_loss  : float
    avg_cls_loss    : float
    avg_sparse_loss : float
    """
    model.train()
    total_loss_sum = cls_loss_sum = sp_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits   = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()

        # ── Total Loss = Classification Loss + λ · Sparsity Loss ──────────
        loss = cls_loss + lambda_sparse * sp_loss
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        cls_loss_sum   += cls_loss.item()
        sp_loss_sum    += sp_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, sp_loss_sum / n


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return test accuracy (%) on the given DataLoader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


@torch.no_grad()
def compute_sparsity(
    model:     SelfPruningNet,
    threshold: float = 1e-2,
) -> tuple[float, np.ndarray]:
    """
    Percentage of gates whose value < threshold (treated as pruned),
    plus a flat numpy array of all gate values for plotting.
    """
    all_gates = [
        layer.get_gates().cpu().flatten()
        for layer in model.prunable_layers
    ]
    all_gates_tensor = torch.cat(all_gates)
    pruned_count     = (all_gates_tensor < threshold).sum().item()
    sparsity_pct     = pruned_count / all_gates_tensor.numel() * 100.0
    return sparsity_pct, all_gates_tensor.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Full Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lambda_sparse: float,
    train_loader:  DataLoader,
    test_loader:   DataLoader,
    device:        torch.device,
    epochs:        int   = 20,
    lr:            float = 1e-3,
) -> tuple[float, float, np.ndarray]:
    """
    Train a fresh SelfPruningNet for `epochs` epochs and return:
      (test_accuracy %, sparsity %, flat gate values array)
    """
    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'─'*60}")
    print(f"  λ = {lambda_sparse:.0e}  |  epochs = {epochs}  |  lr = {lr}")
    print(f"{'─'*60}")
    print(f"  {'Epoch':>5}  {'TotalLoss':>10}  {'ClsLoss':>9}  "
          f"{'SpLoss':>9}  {'TestAcc':>8}  {'Sparsity':>9}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*9}")

    for epoch in range(1, epochs + 1):
        total_l, cls_l, sp_l = train_one_epoch(
            model, train_loader, optimizer, lambda_sparse, device
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            acc,      _  = evaluate(model, test_loader, device), None
            acc           = evaluate(model, test_loader, device)
            sparsity, _   = compute_sparsity(model)
            print(f"  {epoch:>5}  {total_l:>10.4f}  {cls_l:>9.4f}  "
                  f"{sp_l:>9.2f}  {acc:>7.2f}%  {sparsity:>8.2f}%")

    final_acc             = evaluate(model, test_loader, device)
    final_sparsity, gates = compute_sparsity(model)
    print(f"\n  ✓  Final  →  Acc: {final_acc:.2f}%  |  "
          f"Sparsity: {final_sparsity:.2f}%")
    return final_acc, final_sparsity, gates


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distributions(
    gates_dict: dict[float, np.ndarray],
    save_path:  str = "gate_distribution.png",
) -> None:
    """
    Plot a histogram of final gate values for each λ.

    A successful self-pruning network shows:
      • A large spike at 0 (pruned connections)
      • A secondary cluster away from 0 (retained connections)
    """
    n_plots = len(gates_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    colours = ["#2196F3", "#FF9800", "#E91E63"]
    THRESHOLD = 0.01

    for ax, colour, (lam, gates) in zip(axes, colours, gates_dict.items()):
        pruned_pct = (gates < THRESHOLD).mean() * 100
        ax.hist(gates, bins=120, color=colour, edgecolor="none", alpha=0.85)
        ax.axvline(THRESHOLD, color="black", linestyle="--",
                   linewidth=1.2, label=f"Prune threshold ({THRESHOLD})")
        ax.set_title(
            f"λ = {lam:.0e}\nSparsity = {pruned_pct:.1f}%",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Gate Value σ(gate_score)", fontsize=10)
        ax.set_ylabel("# Weights",               fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle(
        "Distribution of Final Gate Values — Self-Pruning Network (CIFAR-10)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Self-Pruning Neural Network — CIFAR-10")
    print(f"  Device : {device}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=256)

    # ── Hyperparameter grid: low / medium / high λ ─────────────────────
    lambdas = [1e-5, 1e-4, 1e-3]
    results    : dict[float, tuple[float, float]] = {}
    gates_dict : dict[float, np.ndarray]          = {}

    for lam in lambdas:
        acc, sparsity, gates = run_experiment(
            lambda_sparse = lam,
            train_loader  = train_loader,
            test_loader   = test_loader,
            device        = device,
            epochs        = 20,
            lr            = 1e-3,
        )
        results[lam]    = (acc, sparsity)
        gates_dict[lam] = gates

    # ── Summary Table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*58}")
    print(f"  {'Lambda':<12}  {'Test Accuracy':>14}  {'Sparsity Level':>16}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*16}")
    for lam, (acc, sp) in results.items():
        print(f"  {lam:<12.0e}  {acc:>13.2f}%  {sp:>15.2f}%")
    print(f"{'='*58}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_gate_distributions(gates_dict, save_path="gate_distribution.png")

    print("\nDone.\n")
