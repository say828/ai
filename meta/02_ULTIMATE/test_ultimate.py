"""
ULTIMATE: Comprehensive Testing
Test ULTIMATE on all datasets and compare with other algorithms
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultimate_optimizer import ULTIMATE_Optimizer


class SimpleNetwork:
    """Simple 2-layer neural network for testing"""
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        """Forward pass"""
        self.X = X
        self.h = np.tanh(np.dot(X, self.W1) + self.b1)
        self.y_pred = np.dot(self.h, self.W2) + self.b2
        return self.y_pred

    def loss(self, X, y):
        """MSE loss"""
        y_pred = self.forward(X)
        return np.mean((y_pred - y) ** 2)

    def compute_gradient(self, X, y):
        """Compute gradient via backpropagation"""
        batch_size = X.shape[0]
        y_pred = self.forward(X)

        # Backward pass
        d_y_pred = 2 * (y_pred - y) / batch_size

        d_W2 = np.dot(self.h.T, d_y_pred)
        d_b2 = np.sum(d_y_pred, axis=0)
        d_h = np.dot(d_y_pred, self.W2.T)

        d_h_pre = d_h * (1 - self.h ** 2)  # tanh derivative

        d_W1 = np.dot(X.T, d_h_pre)
        d_b1 = np.sum(d_h_pre, axis=0)

        # Flatten to 1D
        grad = np.concatenate([d_W1.flatten(), d_b1, d_W2.flatten(), d_b2])
        return grad

    def get_weights(self):
        """Get all weights as flat vector"""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_weights(self, weights):
        """Set all weights from flat vector"""
        idx = 0

        # W1
        size = self.W1.size
        self.W1 = weights[idx:idx+size].reshape(self.W1.shape)
        idx += size

        # b1
        size = self.b1.size
        self.b1 = weights[idx:idx+size]
        idx += size

        # W2
        size = self.W2.size
        self.W2 = weights[idx:idx+size].reshape(self.W2.shape)
        idx += size

        # b2
        size = self.b2.size
        self.b2 = weights[idx:idx+size]


def generate_datasets():
    """Generate test datasets"""
    np.random.seed(42)

    # 1. Linear
    X_linear = np.random.randn(100, 2)
    y_linear = 2 * X_linear[:, 0] + 3 * X_linear[:, 1] + np.random.randn(100) * 0.1
    y_linear = y_linear.reshape(-1, 1)

    # 2. Nonlinear (polynomial)
    X_nonlinear = np.random.randn(100, 2)
    y_nonlinear = X_nonlinear[:, 0]**2 + X_nonlinear[:, 1]**2 + np.random.randn(100) * 0.1
    y_nonlinear = y_nonlinear.reshape(-1, 1)

    # 3. XOR
    X_xor = np.random.randn(100, 2)
    y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(float)
    y_xor = y_xor.reshape(-1, 1)

    return {
        'Linear': (X_linear, y_linear),
        'Nonlinear': (X_nonlinear, y_nonlinear),
        'XOR': (X_xor, y_xor)
    }


def run_sgd_baseline(X, y, max_iterations=200):
    """Run SGD baseline for comparison"""
    network = SimpleNetwork()
    losses = []

    lr = 0.01

    for iteration in range(max_iterations):
        loss = network.loss(X, y)
        losses.append(loss)

        grad = network.compute_gradient(X, y)
        theta = network.get_weights()
        theta = theta - lr * grad
        network.set_weights(theta)

    return losses


def load_baseline_results():
    """
    Load results from previous algorithms

    Returns:
        dict of {dataset: {algorithm: final_loss}}
    """
    # From previous experiments
    baselines = {
        'Linear': {
            'SGD': 0.01339,
            'QED': 0.00989,
            'LAML-Q': 0.00949,
            'COMP': 0.07930,
            'PIO': 0.34022
        },
        'Nonlinear': {
            'SGD': 0.15580,
            'QED': 0.10541,
            'LAML-Q': 0.11423,
            'COMP': 0.10849,
            'PIO': 0.23638
        },
        'XOR': {
            'SGD': 0.80423,
            'QED': 0.27960,
            'LAML-Q': 0.45117,
            'COMP': 0.23783,
            'PIO': 0.18683
        }
    }
    return baselines


def test_ultimate_on_dataset(dataset_name, X, y, meta_learning_enabled=True):
    """
    Test ULTIMATE on a single dataset

    Args:
        dataset_name: Name of dataset
        X, y: Data
        meta_learning_enabled: Enable meta-learning

    Returns:
        final_loss, loss_history
    """
    print(f"\n{'='*60}")
    print(f"Testing ULTIMATE on {dataset_name} dataset")
    print(f"Meta-learning: {'Enabled' if meta_learning_enabled else 'Disabled'}")
    print(f"{'='*60}\n")

    # Create network
    network = SimpleNetwork()

    # Create optimizer
    optimizer = ULTIMATE_Optimizer(
        network,
        max_iterations=200,
        meta_learning_interval=50,
        meta_learning_enabled=meta_learning_enabled,
        verbose=True
    )

    # Optimize
    final_loss, loss_history = optimizer.optimize(X, y)

    # Generate SGD baseline for plotting
    sgd_losses = run_sgd_baseline(X, y)

    # Plot results
    optimizer.plot_results(
        dataset_name,
        baseline_losses={'SGD': sgd_losses}
    )

    # Get strategy summary
    strategy = optimizer.get_strategy_summary()
    print(f"\n--- Learned Strategy for {dataset_name} ---")
    sorted_strategy = sorted(strategy.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_strategy[:5]:
        print(f"  {name}: {weight:.4f}")

    return final_loss, loss_history


def compare_all_algorithms():
    """
    Compare ULTIMATE with all baseline algorithms
    """
    print("\n" + "="*80)
    print("ULTIMATE vs All Algorithms - Comprehensive Comparison")
    print("="*80)

    datasets = generate_datasets()
    baselines = load_baseline_results()

    results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n\n{'#'*80}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*80}")

        # Test ULTIMATE (with meta-learning)
        final_loss, history = test_ultimate_on_dataset(dataset_name, X, y, meta_learning_enabled=True)

        results[dataset_name] = {
            'ULTIMATE': final_loss
        }

        # Add baselines
        results[dataset_name].update(baselines[dataset_name])

    # Final comparison table
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL ALGORITHMS")
    print("="*80)
    print()

    # Header
    print(f"{'Dataset':<15} | {'ULTIMATE':<12} | {'QED':<12} | {'LAML-Q':<12} | {'COMP':<12} | {'PIO':<12} | {'SGD':<12}")
    print("-" * 105)

    # Results
    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        res = results[dataset_name]
        print(f"{dataset_name:<15} | {res['ULTIMATE']:<12.5f} | {res['QED']:<12.5f} | "
              f"{res['LAML-Q']:<12.5f} | {res['COMP']:<12.5f} | {res['PIO']:<12.5f} | {res['SGD']:<12.5f}")

    # Win count
    print("\n" + "-"*80)
    print("WIN COUNT (vs SGD)")
    print("-"*80)

    win_counts = {'ULTIMATE': 0, 'QED': 0, 'LAML-Q': 0, 'COMP': 0, 'PIO': 0}

    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        res = results[dataset_name]
        sgd_loss = res['SGD']

        print(f"\n{dataset_name}:")
        for algo in ['ULTIMATE', 'QED', 'LAML-Q', 'COMP', 'PIO']:
            if res[algo] < sgd_loss:
                improvement = (sgd_loss - res[algo]) / sgd_loss * 100
                print(f"  {algo}: WIN (+{improvement:.2f}%)")
                win_counts[algo] += 1
            else:
                degradation = (res[algo] - sgd_loss) / sgd_loss * 100
                print(f"  {algo}: LOSS (-{degradation:.2f}%)")

    print("\n" + "-"*80)
    print("TOTAL WINS (out of 3 datasets)")
    print("-"*80)
    for algo, wins in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {algo}: {wins}/3")

    # Best algorithm per dataset
    print("\n" + "-"*80)
    print("BEST ALGORITHM PER DATASET")
    print("-"*80)

    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        res = results[dataset_name]
        best_algo = min(res.items(), key=lambda x: x[1])
        print(f"  {dataset_name}: {best_algo[0]} (loss: {best_algo[1]:.5f})")

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    compare_all_algorithms()
