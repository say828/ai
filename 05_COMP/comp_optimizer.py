"""
COMP: Compositional Optimizer with Multi-Primitives

A novel optimization approach that composes simple update strategies (primitives)
based on optimization context.

Philosophy:
  Complex optimization = Intelligent composition of simple primitives
  Context determines which primitives to emphasize
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from context import OptimizationContext
from primitives import get_default_primitives, get_primitive_names
from weight_functions import rule_based_weights


# ============================================================================
# Simple Neural Network (Same as previous experiments)
# ============================================================================

class SimpleNetwork:
    """Simple 2-layer neural network for regression"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def loss(self, X, y):
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

    def gradient(self, X, y):
        m = len(X)
        pred = self.forward(X)

        dz2 = 2 * (pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

    def get_weights(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_weights(self, w):
        idx = 0
        W1_size = self.W1.size
        self.W1 = w[idx:idx+W1_size].reshape(self.W1.shape)
        idx += W1_size

        b1_size = self.b1.size
        self.b1 = w[idx:idx+b1_size]
        idx += b1_size

        W2_size = self.W2.size
        self.W2 = w[idx:idx+W2_size].reshape(self.W2.shape)
        idx += W2_size

        b2_size = self.b2.size
        self.b2 = w[idx:idx+b2_size]


# ============================================================================
# COMP Optimizer
# ============================================================================

class COMP_Optimizer:
    """
    Compositional Optimizer with Multi-Primitives

    Composes multiple update strategies based on optimization context.
    """

    def __init__(self, network, primitives=None, weight_fn=None):
        """
        Args:
            network: Neural network to optimize
            primitives: List of Primitive objects (default: get_default_primitives())
            weight_fn: Function(context, n_primitives) → weights (default: rule_based_weights)
        """
        self.network = network
        self.primitives = primitives or get_default_primitives()
        self.weight_fn = weight_fn or rule_based_weights
        self.context = OptimizationContext()

        # Statistics tracking
        self.weight_history = []
        self.loss_history = []

    def step(self, X, y):
        """
        Single optimization step

        Args:
            X: Input data
            y: Target data

        Returns:
            tuple: (loss, weights) for this step
        """
        # Get current state
        theta = self.network.get_weights()
        grad = self.network.gradient(X, y)
        loss = self.network.loss(X, y)
        grad_norm = np.linalg.norm(grad)

        # Update context
        self.context.update(theta, loss, grad_norm)

        # Compute primitive weights
        weights = self.weight_fn(self.context, len(self.primitives))
        self.weight_history.append(weights.copy())

        # Execute primitives and compose
        deltas = []
        for primitive in self.primitives:
            delta = primitive(theta, grad, self.context)
            deltas.append(delta)

        # Weighted combination
        final_delta = sum(w * d for w, d in zip(weights, deltas))

        # Apply update
        new_theta = theta + final_delta
        self.network.set_weights(new_theta)

        return loss, weights

    def train(self, X, y, max_iters=100, tolerance=1e-3, verbose=True):
        """
        Full training loop

        Args:
            X: Input data
            y: Target data
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            dict: Training results
        """
        start_time = time.time()
        losses = []
        primitive_names = get_primitive_names(self.primitives)

        for i in range(max_iters):
            loss, weights = self.step(X, y)
            losses.append(loss)

            if verbose and i % 10 == 0:
                phase = self.context.phase
                success_rate = self.context.recent_success_rate
                dominant_idx = np.argmax(weights)
                dominant = primitive_names[dominant_idx]

                print(f"[{i:3d}] Loss: {loss:.5f} | "
                      f"Phase: {phase:12s} | "
                      f"Success: {success_rate:.0%} | "
                      f"Dominant: {dominant}")

            # Convergence check
            if i > 10 and abs(losses[-1] - losses[-2]) < tolerance:
                if verbose:
                    print(f"\n✓ Converged at iteration {i}")
                break

        elapsed = time.time() - start_time

        return {
            'final_loss': losses[-1],
            'losses': losses,
            'iterations': len(losses),
            'time': elapsed,
            'weight_history': np.array(self.weight_history),
            'primitive_names': primitive_names,
        }


# ============================================================================
# Baseline: Standard SGD
# ============================================================================

def train_sgd(network, X, y, lr=0.05, max_iters=100, verbose=True):
    """Standard SGD for comparison"""
    losses = []
    start_time = time.time()

    for i in range(max_iters):
        grad = network.gradient(X, y)
        loss = network.loss(X, y)
        losses.append(loss)

        if verbose and i % 10 == 0:
            print(f"[{i:3d}] Loss: {loss:.5f}")

        theta = network.get_weights()
        network.set_weights(theta - lr * grad)

    elapsed = time.time() - start_time

    return {
        'final_loss': losses[-1],
        'losses': losses,
        'iterations': len(losses),
        'time': elapsed,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(result_comp, result_sgd, dataset_name, save_path=None):
    """Plot comparison between COMP and SGD"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(result_comp['losses'], label='COMP', linewidth=2, color='#2E86AB')
    ax.plot(result_sgd['losses'], label='SGD', linewidth=2, color='#A23B72', linestyle='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Curves: {dataset_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Weight evolution (stacked area)
    ax = axes[0, 1]
    weights = result_comp['weight_history']
    names = result_comp['primitive_names']

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    ax.stackplot(range(len(weights)), weights.T, labels=names, colors=colors, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title('Primitive Weights Evolution')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Average primitive contribution
    ax = axes[1, 0]
    avg_weights = weights.mean(axis=0)
    bars = ax.bar(names, avg_weights, color=colors, alpha=0.8)
    ax.set_ylabel('Average Weight')
    ax.set_title('Average Primitive Contribution')
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 4: Performance comparison
    ax = axes[1, 1]
    metrics = ['Final Loss', 'Iterations', 'Time (s)']
    comp_values = [result_comp['final_loss'], result_comp['iterations'], result_comp['time']]
    sgd_values = [result_sgd['final_loss'], result_sgd['iterations'], result_sgd['time']]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for visualization
    comp_norm = np.array(comp_values) / np.array(sgd_values)
    sgd_norm = np.ones(len(metrics))

    bars1 = ax.bar(x - width/2, comp_norm, width, label='COMP', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, sgd_norm, width, label='SGD', color='#A23B72', alpha=0.8)

    ax.set_ylabel('Relative to SGD')
    ax.set_title('Performance Comparison (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement labels
    improvement = (1 - result_comp['final_loss'] / result_sgd['final_loss']) * 100
    ax.text(0, comp_norm[0], f'{improvement:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")

    plt.close()


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(dataset_name, X, y):
    """Run COMP vs SGD on given dataset"""
    print("=" * 80)
    print(f"🔬 COMP vs SGD: {dataset_name.upper()}")
    print("=" * 80)

    # COMP
    print("\n1️⃣  COMP (Compositional Optimizer)")
    print("-" * 80)
    net_comp = SimpleNetwork(X.shape[1], 8, y.shape[1])
    opt_comp = COMP_Optimizer(net_comp)
    result_comp = opt_comp.train(X, y, max_iters=100, verbose=True)

    # SGD
    print("\n2️⃣  Standard SGD")
    print("-" * 80)
    net_sgd = SimpleNetwork(X.shape[1], 8, y.shape[1])
    result_sgd = train_sgd(net_sgd, X, y, lr=0.05, max_iters=100, verbose=True)

    # Results
    print("\n" + "=" * 80)
    print("📊 Results")
    print("=" * 80)
    print(f"{'Metric':<30} {'COMP':>15} {'SGD':>15}")
    print("-" * 80)
    print(f"{'Final Loss':<30} {result_comp['final_loss']:>15.6f} {result_sgd['final_loss']:>15.6f}")
    print(f"{'Iterations':<30} {result_comp['iterations']:>15} {result_sgd['iterations']:>15}")
    print(f"{'Time (seconds)':<30} {result_comp['time']:>15.4f} {result_sgd['time']:>15.4f}")

    improvement = (result_sgd['final_loss'] - result_comp['final_loss']) / result_sgd['final_loss'] * 100
    print(f"\n🎯 COMP Improvement: {improvement:+.2f}%")
    print("=" * 80)

    if improvement > 0:
        print("✅ COMP wins!")
    else:
        print("❌ SGD wins")

    # Plot
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results", f"comp_{dataset_name.lower()}_results.png")
    plot_comparison(result_comp, result_sgd, dataset_name, save_path)

    return result_comp, result_sgd


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Dataset 1: Linear
    np.random.seed(42)
    X_linear = np.random.randn(100, 3)
    y_linear = (X_linear @ np.array([1.5, -2.0, 0.5])[:, None] + 1.0)

    result_comp_linear, result_sgd_linear = run_experiment("Linear", X_linear, y_linear)

    # Dataset 2: Nonlinear
    X_nonlinear = np.random.randn(100, 3)
    y_nonlinear = np.sin(X_nonlinear[:, 0:1]) + np.cos(X_nonlinear[:, 1:2]) * X_nonlinear[:, 2:3]

    result_comp_nonlinear, result_sgd_nonlinear = run_experiment("Nonlinear", X_nonlinear, y_nonlinear)

    # Dataset 3: XOR
    X_xor = np.random.randn(100, 2)
    y_xor = ((X_xor[:, 0:1] > 0) ^ (X_xor[:, 1:2] > 0)).astype(float)

    result_comp_xor, result_sgd_xor = run_experiment("XOR", X_xor, y_xor)

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 Final Summary: COMP vs SGD")
    print("=" * 80)

    results = [
        ("Linear", result_comp_linear, result_sgd_linear),
        ("Nonlinear", result_comp_nonlinear, result_sgd_nonlinear),
        ("XOR", result_comp_xor, result_sgd_xor),
    ]

    wins = 0
    for name, comp, sgd in results:
        improvement = (sgd['final_loss'] - comp['final_loss']) / sgd['final_loss'] * 100
        winner = "✅ COMP" if improvement > 0 else "❌ SGD"
        print(f"{name:<12} | {winner:<10} | {improvement:+6.2f}%")
        if improvement > 0:
            wins += 1

    print("=" * 80)
    print(f"COMP win rate: {wins}/3 ({wins/3*100:.1f}%)")
    print("=" * 80)

    print("\n🎉 COMP: Compositional optimization with interpretable primitives!")
