"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters
that drive themselves to zero via L1 sparsity regularization,
effectively pruning unnecessary weights during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# ─────────────────────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A linear layer augmented with learnable gate_scores.

    Each weight w_ij has a corresponding gate_score s_ij.
    The gate is computed as: gate = sigmoid(s_ij) ∈ (0, 1)
    The effective weight used in the forward pass is:
        pruned_weight = weight * gate

    When gate → 0, the weight is effectively removed (pruned).
    The L1 penalty on gates during training drives many gates to ~0,
    producing a sparse network.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight.
        # Initialized to a small positive value so sigmoid(score) ≈ 0.5
        # meaning all gates start ~open and the network learns to close them.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialize weights with kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert raw scores to gates in (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: Element-wise multiply weight by gate to get pruned weights
        # Gradients flow through both self.weight and self.gate_scores
        pruned_weights = self.weight * gates              # shape: (out, in)

        # Step 3: Standard linear operation: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all gate values for this layer."""
        return torch.sigmoid(self.gate_scores).abs().sum()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 using PrunableLinear layers.
    CIFAR-10: 32×32×3 images → 10 classes
    """

    def __init__(self, hidden_sizes=(1024, 512, 256)):
        super().__init__()
        input_size = 32 * 32 * 3  # Flattened CIFAR-10 image

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(PrunableLinear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h

        self.hidden = nn.Sequential(*layers)
        self.classifier = PrunableLinear(prev, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten
        x = self.hidden(x)
        return self.classifier(x)

    def get_all_prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """Sum L1 penalty across all prunable layers."""
        total = sum(layer.sparsity_loss() for layer in self.get_all_prunable_layers())
        return total

    def sparsity_level(self, threshold: float = 0.20)->float:
        """
        Fraction of weights whose gate value is below `threshold`.
        A gate < threshold means the weight is effectively pruned.
        """
        all_gates = torch.cat([
            layer.get_gates().flatten()
            for layer in self.get_all_prunable_layers()
        ])
        pruned = (all_gates < threshold).sum().item()
        total = all_gates.numel()
        return pruned / total * 100.0

    def all_gate_values(self) -> np.ndarray:
        """Return all gate values as a numpy array for plotting."""
        gates = torch.cat([
            layer.get_gates().flatten()
            for layer in self.get_all_prunable_layers()
        ])
        return gates.cpu().numpy()


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256, data_dir: str = "./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Part 3: Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam):
    """
    Train for one epoch.
    Total Loss = CrossEntropy(logits, labels) + λ * SparsityLoss
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        # Classification loss
        clf_loss = F.cross_entropy(logits, labels)

        # Sparsity regularization: L1 norm of all gate values
        sparse_loss = model.sparsity_loss()

        loss = clf_loss + lam * sparse_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Return test accuracy (%)."""
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


def run_experiment(lam: float, train_loader, test_loader,
                   device, epochs: int = 30, seed: int = 42):
    """
    Train a fresh SelfPruningNet with the given λ value.
    Returns: (test_accuracy, sparsity_level, gate_values_array)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SelfPruningNet(hidden_sizes=(1024, 512, 256)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lam)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(model, test_loader, device)
            sparsity = model.sparsity_level()
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.1f}% | "
                  f"Test Acc: {test_acc:.1f}% | "
                  f"Sparsity: {sparsity:.1f}% | "
                  f"Time: {elapsed:.1f}s")

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()
    gate_values = model.all_gate_values()

    print(f"\n  ✓ Final → Test Acc: {final_test_acc:.2f}% | "
          f"Sparsity: {final_sparsity:.2f}%")
    return final_test_acc, final_sparsity, gate_values


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_gate_distribution(gate_values_dict: dict, save_path: str = "gate_distribution.png"):
    """
    Plot histogram of gate values for each λ.
    A good result shows a large spike near 0 (pruned) and a cluster near 1 (kept).
    """
    lambdas = list(gate_values_dict.keys())
    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#F44336"]
    for ax, lam, color in zip(axes, lambdas, colors):
        gates = gate_values_dict[lam]
        ax.hist(gates, bins=80, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(f"Gate Distribution\nλ = {lam}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.axvline(0.20 , color="black", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.20)")
        pruned_pct = (gates < 0.20).mean() * 100
        ax.text(0.55, 0.85, f"Pruned: {pruned_pct:.1f}%",
                transform=ax.transAxes, fontsize=11,
                
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Self-Pruning Network: Gate Value Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Gate distribution plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Three λ values: low, medium, high
    lambdas = [1e-3, 2e-3,5e-3]
    epochs =   10      # Increase to 50+ for better accuracy
    batch_size = 128

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    results = {}
    gate_values_dict = {}

    for lam in lambdas:
        acc, sparsity, gates = run_experiment(
            lam, train_loader, test_loader, device, epochs=epochs)
        results[lam] = {"test_accuracy": acc, "sparsity": sparsity}
        gate_values_dict[lam] = gates

    # ── Print Results Table ──────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>17}")
    print("=" * 55)
    for lam in lambdas:
        r = results[lam]
        print(f"  {lam:<12} {r['test_accuracy']:>14.2f}% {r['sparsity']:>16.2f}%")
    print("=" * 55)

    # ── Plot Gate Distributions ─────────────────────────────
    plot_gate_distribution(gate_values_dict, save_path="gate_distribution.png")

    # ── Identify Best Model ─────────────────────────────────
    best_lam = max(results, key=lambda l: results[l]["test_accuracy"])
    print(f"\n  Best λ by accuracy: {best_lam} "
          f"→ {results[best_lam]['test_accuracy']:.2f}% acc, "
          f"{results[best_lam]['sparsity']:.2f}% sparsity")


if __name__ == "__main__":
    main()
