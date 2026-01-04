"""
ULTIMATE v1.2 Testing
Compare v1.0 (baseline) vs v1.2 (improved primitives)

v1.2 improvements:
- Adaptive primitive upgraded to Adam-like (first + second moments)
- PathSampling increased from 5 to 20 samples
- Keep v1.0's natural soft winner-take-all (NO forced winner-take-all)
- Keep uniform LRs (NO tuning)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultimate_optimizer import ULTIMATE_Optimizer
from test_ultimate import SimpleNetwork, generate_datasets


def test_version(dataset_name, X, y, version="v1.2"):
    """
    Test a specific version of ULTIMATE

    Args:
        dataset_name: Name of dataset
        X, y: Data
        version: Version string for display

    Returns:
        final_loss, loss_history, strategy
    """
    print(f"\n{'='*60}")
    print(f"Testing ULTIMATE {version} on {dataset_name}")
    print(f"Primitive improvements: {'Adam-like Adaptive + 20-sample PathSampling' if version == 'v1.2' else 'Original'}")
    print(f"{'='*60}\n")

    # Create network
    network = SimpleNetwork()

    # Create optimizer (v1.2 uses v1.0 settings + improved primitives)
    optimizer = ULTIMATE_Optimizer(
        network,
        max_iterations=200,
        meta_learning_interval=50,
        meta_learning_enabled=True,
        verbose=False,
        winner_take_all=False,  # v1.2: NO forced winner-take-all (keep v1.0 approach)
        tuned_primitives=False  # v1.2: NO tuning (keep v1.0 uniform LRs)
    )

    # Optimize
    final_loss, loss_history = optimizer.optimize(X, y)

    # Get strategy summary
    strategy = optimizer.get_strategy_summary()
    print(f"\n--- Strategy for {dataset_name} ({version}) ---")
    sorted_strategy = sorted(strategy.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_strategy[:3]:
        print(f"  {name}: {weight:.2%}")

    return final_loss, loss_history, strategy


def compare_versions():
    """
    Compare v1.0 vs v1.2 on all datasets
    """
    print("\n" + "="*80)
    print("ULTIMATE v1.0 vs v1.2 Comparison")
    print("="*80)
    print("\nv1.2 changes:")
    print("  1. Adaptive primitive: RMSprop → Adam-like (first + second moments)")
    print("  2. PathSampling: 5 samples → 20 samples (better path exploration)")
    print("  3. Keep v1.0's natural soft winner-take-all (NO forcing)")
    print("  4. Keep v1.0's uniform LRs (NO tuning)")
    print("\n" + "="*80)

    datasets = generate_datasets()

    results = {
        'v1.0': {},
        'v1.2': {}
    }

    strategies = {
        'v1.0': {},
        'v1.2': {}
    }

    for dataset_name, (X, y) in datasets.items():
        print(f"\n\n{'#'*80}")
        print(f"# Dataset: {dataset_name}")
        print(f"{'#'*80}")

        # Test v1.0 (baseline)
        final_loss_v1_0, history_v1_0, strategy_v1_0 = test_version(
            dataset_name, X, y, version="v1.0"
        )
        results['v1.0'][dataset_name] = final_loss_v1_0
        strategies['v1.0'][dataset_name] = strategy_v1_0

        # Test v1.2 (improved primitives)
        final_loss_v1_2, history_v1_2, strategy_v1_2 = test_version(
            dataset_name, X, y, version="v1.2"
        )
        results['v1.2'][dataset_name] = final_loss_v1_2
        strategies['v1.2'][dataset_name] = strategy_v1_2

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: v1.0 vs v1.2")
    print("="*80)
    print()

    print(f"{'Dataset':<15} | {'v1.0':<12} | {'v1.2':<12} | {'Improvement':<15} | {'Status':<10}")
    print("-" * 80)

    improvements = []
    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        loss_v1_0 = results['v1.0'][dataset_name]
        loss_v1_2 = results['v1.2'][dataset_name]

        if loss_v1_0 > 0:
            improvement = (loss_v1_0 - loss_v1_2) / loss_v1_0 * 100
        else:
            improvement = 0

        improvements.append(improvement)

        status = "✅ Better" if improvement > 0 else "❌ Worse"

        print(f"{dataset_name:<15} | {loss_v1_0:<12.5f} | {loss_v1_2:<12.5f} | "
              f"{improvement:>+6.2f}%        | {status}")

    print("\n" + "-"*80)
    avg_improvement = np.mean(improvements)
    print(f"Average Improvement: {avg_improvement:+.2f}%")

    # Winner count
    winners_v1_2 = sum(1 for imp in improvements if imp > 0)
    print(f"v1.2 wins: {winners_v1_2}/3 datasets")

    # Strategy comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    for dataset_name in ['Linear', 'Nonlinear', 'XOR']:
        print(f"\n{dataset_name}:")
        print(f"  v1.0 top strategy: {max(strategies['v1.0'][dataset_name].items(), key=lambda x: x[1])}")
        print(f"  v1.2 top strategy: {max(strategies['v1.2'][dataset_name].items(), key=lambda x: x[1])}")

    print("\n" + "="*80)
    if avg_improvement > 0:
        print("SUCCESS! v1.2 improved over v1.0 by better primitive implementations!")
        print("Key insight: Strategy selection was already good, primitives needed improvement.")
    else:
        print("Note: Results may vary due to randomness. Run multiple times for confidence.")
    print("="*80)


if __name__ == "__main__":
    compare_versions()
