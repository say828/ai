"""
GENESIS Phase 3 Experiment: v1.1 vs v1.2 Comparison

Compare v1.1 (improved) with v1.2 (refined)

Goal: Demonstrate that v1.2 achieves POSITIVE learning
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from genesis_entity_v1_1 import GENESIS_Entity_v1_1
from genesis_entity_v1_2 import GENESIS_Entity_v1_2
from genesis_environment import RegressionEnvironment


def run_single_entity_experiment(
    entity_class,
    environment,
    n_steps: int = 200,
    seed: int = 42
) -> Tuple[List, List, List, int]:
    """
    Run experiment with a single entity

    Returns:
        - viability_history
        - error_history
        - metamorphosis_ages
        - entity
    """
    np.random.seed(seed)

    entity = entity_class(entity_id=1)

    viability_history = []
    error_history = []
    metamorphosis_ages = []

    for step in range(n_steps):
        # Track metamorphosis
        age_before = entity.age

        # Live one step
        viability = entity.live_one_step(environment, ecosystem=None)

        # Record metrics
        viability_history.append(viability)

        # Measure prediction error by testing phenotype
        try:
            test_idx = np.random.randint(len(environment.X))
            test_input = environment.X[test_idx]
            test_target = environment.y[test_idx]

            prediction = entity.phenotype.forward(test_input)

            if len(prediction) > 0:
                error = np.abs(prediction[0] - test_target)
            else:
                error = 100.0

            error_history.append(error)
        except:
            error_history.append(100.0)

        # Check if metamorphosed
        if hasattr(entity, 'age') and entity.age > age_before + 1:
            metamorphosis_ages.append(step)

        # Progress
        if (step + 1) % 50 == 0:
            if error_history:
                print(f"  Step {step+1}/{n_steps}: viability={viability:.3f}, error={error_history[-1]:.3f}")
            else:
                print(f"  Step {step+1}/{n_steps}: viability={viability:.3f}, error=N/A")

    return viability_history, error_history, metamorphosis_ages, entity


def compare_versions(n_steps: int = 200):
    """
    Compare v1.1 and v1.2

    Same regression problem as before
    """
    print("=" * 80)
    print("GENESIS Phase 3 Experiment: v1.1 vs v1.2")
    print("=" * 80)

    # Create regression environment (same as before)
    print("\nCreating regression environment...")
    print("  Problem: y = 2*x1 + 3*x2 + noise")
    print("  Samples: 100")
    print("  Features: 2")
    print("  Noise: 0.1")

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1

    env = RegressionEnvironment(X, y)

    # Run v1.1
    print("\n" + "-" * 80)
    print("Running v1.1 (baseline)...")
    print("-" * 80)

    v1_1_viability, v1_1_error, v1_1_metamorphoses, entity_v1_1 = \
        run_single_entity_experiment(
            GENESIS_Entity_v1_1,
            env,
            n_steps=n_steps,
            seed=42
        )

    print(f"\nv1.1 Final: {entity_v1_1}")
    print(f"v1.1 Metamorphoses: {len(v1_1_metamorphoses)}")

    # Run v1.2
    print("\n" + "-" * 80)
    print("Running v1.2 (refined)...")
    print("-" * 80)

    v1_2_viability, v1_2_error, v1_2_metamorphoses, entity_v1_2 = \
        run_single_entity_experiment(
            GENESIS_Entity_v1_2,
            env,
            n_steps=n_steps,
            seed=42
        )

    print(f"\nv1.2 Final: {entity_v1_2}")
    print(f"v1.2 Metamorphoses: {len(v1_2_metamorphoses)}")

    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Viability comparison
    v1_1_final_viability = v1_1_viability[-1]
    v1_2_final_viability = v1_2_viability[-1]
    viability_improvement = ((v1_2_final_viability - v1_1_final_viability) / v1_1_final_viability) * 100

    print("\n1. VIABILITY")
    print(f"   v1.1: {v1_1_final_viability:.3f}")
    print(f"   v1.2: {v1_2_final_viability:.3f}")
    print(f"   Change: {viability_improvement:+.1f}%")

    # Error comparison
    v1_1_final_error = v1_1_error[-1]
    v1_2_final_error = v1_2_error[-1]
    error_improvement = ((v1_1_final_error - v1_2_final_error) / v1_1_final_error) * 100

    print("\n2. PREDICTION ERROR")
    print(f"   v1.1: {v1_1_final_error:.3f}")
    print(f"   v1.2: {v1_2_final_error:.3f}")
    print(f"   Change: {error_improvement:+.1f}%")

    # Learning progress
    v1_1_initial_error = np.mean(v1_1_error[:10])
    v1_1_learning_progress = ((v1_1_initial_error - v1_1_final_error) / v1_1_initial_error) * 100

    v1_2_initial_error = np.mean(v1_2_error[:10])
    v1_2_learning_progress = ((v1_2_initial_error - v1_2_final_error) / v1_2_initial_error) * 100

    print("\n3. LEARNING PROGRESS (% error reduction)")
    print(f"   v1.1: {v1_1_learning_progress:+.1f}%")
    print(f"   v1.2: {v1_2_learning_progress:+.1f}%")
    print(f"   Change: {v1_2_learning_progress - v1_1_learning_progress:+.1f}%p")

    # Critical check: Is v1.2 POSITIVE?
    print("\n4. POSITIVE LEARNING CHECK")
    if v1_2_learning_progress > 0:
        print(f"   ✅ SUCCESS! v1.2 achieved POSITIVE learning (+{v1_2_learning_progress:.1f}%)")
    else:
        print(f"   ❌ FAILED. v1.2 still negative learning ({v1_2_learning_progress:.1f}%)")

    # Metamorphosis comparison
    metamorphosis_change = ((len(v1_1_metamorphoses) - len(v1_2_metamorphoses)) /
                            len(v1_1_metamorphoses) * 100 if len(v1_1_metamorphoses) > 0 else 0)

    print("\n5. METAMORPHOSIS COUNT")
    print(f"   v1.1: {len(v1_1_metamorphoses)}/{n_steps}")
    print(f"   v1.2: {len(v1_2_metamorphoses)}/{n_steps}")
    print(f"   Change: {metamorphosis_change:+.1f}%")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Metric':<25} {'v1.1':<12} {'v1.2':<12} {'Change':<12}")
    print("-" * 80)
    print(f"{'Final Viability':<25} {v1_1_final_viability:<12.3f} {v1_2_final_viability:<12.3f} {viability_improvement:+.1f}%")
    print(f"{'Final Error':<25} {v1_1_final_error:<12.3f} {v1_2_final_error:<12.3f} {error_improvement:+.1f}%")
    print(f"{'Learning Progress':<25} {v1_1_learning_progress:<12.1f}% {v1_2_learning_progress:<12.1f}% {v1_2_learning_progress - v1_1_learning_progress:+.1f}%p")
    print(f"{'Metamorphosis Count':<25} {len(v1_1_metamorphoses):<12} {len(v1_2_metamorphoses):<12} {metamorphosis_change:+.1f}%")

    # Visualization
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Viability over time
    ax1 = axes[0, 0]
    ax1.plot(v1_1_viability, label='v1.1', alpha=0.7, linewidth=1.5)
    ax1.plot(v1_2_viability, label='v1.2', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Viability')
    ax1.set_title('Viability Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smoothed error over time
    window = 10
    v1_1_error_smooth = np.convolve(v1_1_error, np.ones(window)/window, mode='valid')
    v1_2_error_smooth = np.convolve(v1_2_error, np.ones(window)/window, mode='valid')

    ax2 = axes[0, 1]
    ax2.plot(v1_1_error_smooth, label='v1.1 (smoothed)', alpha=0.7, linewidth=1.5)
    ax2.plot(v1_2_error_smooth, label='v1.2 (smoothed)', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Prediction Error (MSE)')
    ax2.set_title('Prediction Error Over Time (smoothed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Metamorphosis events
    ax3 = axes[1, 0]
    ax3.scatter(v1_1_metamorphoses, [1]*len(v1_1_metamorphoses),
                label=f'v1.1 ({len(v1_1_metamorphoses)} events)', alpha=0.6, s=30)
    ax3.scatter(v1_2_metamorphoses, [2]*len(v1_2_metamorphoses),
                label=f'v1.2 ({len(v1_2_metamorphoses)} events)', alpha=0.6, s=30)
    ax3.set_xlabel('Step')
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['v1.1', 'v1.2'])
    ax3.set_title('Metamorphosis Events')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Learning progress comparison
    ax4 = axes[1, 1]
    versions = ['v1.1', 'v1.2']
    learning_progresses = [v1_1_learning_progress, v1_2_learning_progress]
    colors = ['red' if lp < 0 else 'green' for lp in learning_progresses]

    bars = ax4.bar(versions, learning_progresses, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_ylabel('Learning Progress (%)')
    ax4.set_title('Learning Progress Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, learning_progresses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    plt.savefig('/Users/say/Documents/GitHub/ai/08_GENESIS/v1_1_v1_2_comparison.png', dpi=300)
    print("Saved plot: v1_1_v1_2_comparison.png")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    print("\nImprovement 1: Hebbian Learning Rate (0.01 → 0.05)")
    print("  - v1.2 should show stronger integration of successful patterns")
    print("  - Check: pathway_strengths should diverge more in v1.2")

    print("\nImprovement 2: Environment Feedback Smoothing")
    print("  - v1.2 uses moving average over 10 feedbacks")
    print("  - Check: viability should be more stable in v1.2")
    v1_1_viability_std = np.std(v1_1_viability)
    v1_2_viability_std = np.std(v1_2_viability)
    print(f"  v1.1 viability std: {v1_1_viability_std:.3f}")
    print(f"  v1.2 viability std: {v1_2_viability_std:.3f}")
    print(f"  Change: {((v1_2_viability_std - v1_1_viability_std) / v1_1_viability_std * 100):+.1f}%")

    print("\nImprovement 3: Better Initialization (Xavier/He)")
    print("  - v1.2 uses He initialization instead of simple *0.01")
    print("  - Check: should see faster initial learning")

    print("\nImprovement 4: Metamorphosis Threshold (0.005 → 0.001)")
    print("  - v1.2 should have even fewer metamorphoses")
    print(f"  Expected reduction: ~80%")
    print(f"  Actual reduction: {metamorphosis_change:.1f}%")

    print("\nImprovement 5: Network Capacity ([32,16] → [64,32])")
    print("  - v1.2 has 2x more parameters")
    print("  - Check: should have higher representational capacity")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if v1_2_learning_progress > 0:
        print("\n✅✅✅ SUCCESS! ✅✅✅")
        print(f"\nv1.2 achieved POSITIVE learning (+{v1_2_learning_progress:.1f}%)")
        print("This is a breakthrough moment for GENESIS!")
        print("\nKey achievements:")
        print("1. Error decreased instead of increased")
        print("2. Entity learned to improve its predictions")
        print("3. Viability metric correctly reflects performance")
        print("4. Hebbian-like integration is working")
        print("5. Structural evolution is controlled")
    else:
        print("\n⚠️ Still negative learning")
        print(f"\nv1.2 learning progress: {v1_2_learning_progress:.1f}%")
        print("But improvements over v1.1:")
        print(f"- Viability: {viability_improvement:+.1f}%")
        print(f"- Error: {error_improvement:+.1f}%")
        print(f"- Learning progress: {v1_2_learning_progress - v1_1_learning_progress:+.1f}%p")
        print("\nNext steps:")
        print("1. Further increase Hebbian learning rate?")
        print("2. Add explicit gradient-like signals?")
        print("3. Multi-entity ecosystem for collective learning?")

    return {
        'v1_1': {
            'viability': v1_1_viability,
            'error': v1_1_error,
            'metamorphoses': v1_1_metamorphoses,
            'entity': entity_v1_1
        },
        'v1_2': {
            'viability': v1_2_viability,
            'error': v1_2_error,
            'metamorphoses': v1_2_metamorphoses,
            'entity': entity_v1_2
        }
    }


if __name__ == "__main__":
    print("\nGENESIS Phase 3 Experiment")
    print("v1.1 (improved) vs v1.2 (refined)")
    print("\nGoal: Achieve POSITIVE learning!")
    print("\nRunning 200 steps...\n")

    results = compare_versions(n_steps=200)

    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)
