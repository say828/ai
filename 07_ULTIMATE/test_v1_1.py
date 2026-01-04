"""
ULTIMATE v1.1 Testing
Compare v1.0 (baseline) vs v1.1 (improvements)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultimate_optimizer import ULTIMATE_Optimizer
from test_ultimate import SimpleNetwork, generate_datasets


def test_version(dataset_name, X, y, version="v1.1", winner_take_all=True, tuned=True):
    """
    Test a specific version of ULTIMATE

    Args:
        dataset_name: Name of dataset
        X, y: Data
        version: Version string for display
        winner_take_all: Enable winner-take-all mode
        tuned: Use tuned primitive LRs

    Returns:
        final_loss, loss_history
    """
    print(f"\n{'='*60}")
    print(f"Testing ULTIMATE {version} on {dataset_name}")
    print(f"Winner-take-all: {'Enabled' if winner_take_all else 'Disabled'}")
    print(f"Tuned primitives: {'Yes' if tuned else 'No'}")
    print(f"{'='*60}\n")

    # Create network
    network = SimpleNetwork()

    # Create optimizer
    optimizer = ULTIMATE_Optimizer(
        network,
        max_iterations=200,
        meta_learning_interval=50,
        meta_learning_enabled=True,
        verbose=False,  # Less verbose for cleaner output
        winner_take_all=winner_take_all,
        confidence_threshold=0.85,
        tuned_primitives=tuned
    )

    # Optimize
    final_loss, loss_history = optimizer.optimize(X, y)

    # Get strategy summary
    strategy = optimizer.get_strategy_summary()
    print(f"\n--- Strategy for {dataset_name} ({version}) ---")
    sorted_strategy = sorted(strategy.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_strategy[:3]:
        print(f"  {name}: {weight:.4f}")

    return final_loss, loss_history, strategy


def compare_versions():
    """
    Compare v1.0 vs v1.1 on all datasets
    """
    print("\n" + "="*80)
    print("ULTIMATE v1.0 vs v1.1 Comparison")
    print("="*80)

    datasets = generate_datasets()

    results = {
        'v1.0': {},
        'v1.1': {}
    }

    for dataset_name, (X, y) in datasets.items():
        print(f"\n\n{'#'*80}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*80}")

        # Test v1.0 (baseline)
        final_loss_v1_0, history_v1_0, strategy_v1_0 = test_version(
            dataset_name, X, y,
            version="v1.0",
            winner_take_all=False,  # Disabled
            tuned=False             # Not tuned
        )
        results['v1.0'][dataset_name] = final_loss_v1_0

        # Test v1.1 (improved)
        final_loss_v1_1, history_v1_1, strategy_v1_1 = test_version(
            dataset_name, X, y,
            version="v1.1",
            winner_take_all=True,   # Enabled
            tuned=True              # Tuned
        )
        results['v1.1'][dataset_name] = final_loss_v1_1

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: v1.0 vs v1.1")
    print("="*80)
    print()

    print(f"{'Dataset':<15} | {'v1.0':<12} | {'v1.1':<12} | {'Improvement':<15} | {'Status':<10}")
    print("-" * 80)

    improvements = []
    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        loss_v1_0 = results['v1.0'][dataset_name]
        loss_v1_1 = results['v1.1'][dataset_name]

        if loss_v1_0 > 0:
            improvement = (loss_v1_0 - loss_v1_1) / loss_v1_0 * 100
        else:
            improvement = 0

        improvements.append(improvement)

        status = "✅ Better" if improvement > 0 else "❌ Worse"

        print(f"{dataset_name:<15} | {loss_v1_0:<12.5f} | {loss_v1_1:<12.5f} | "
              f"{improvement:>+6.2f}%        | {status}")

    print("\n" + "-"*80)
    print(f"Average Improvement: {np.mean(improvements):+.2f}%")

    # Winner count
    winners_v1_1 = sum(1 for imp in improvements if imp > 0)
    print(f"v1.1 wins: {winners_v1_1}/3 datasets")

    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    compare_versions()
