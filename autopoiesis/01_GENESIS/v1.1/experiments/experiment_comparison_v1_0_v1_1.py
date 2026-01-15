"""
GENESIS: v1.0 vs v1.1 Comparison

Compare baseline vs improved versions
"""

import numpy as np
import matplotlib.pyplot as plt
from genesis_entity import GENESIS_Entity
from genesis_entity_v1_1 import GENESIS_Entity_v1_1
from genesis_environment import RegressionEnvironment


def run_entity(EntityClass, version_name, X, y, steps=200):
    """Run single entity and collect metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {version_name}")
    print(f"{'='*60}")

    entity = EntityClass(entity_id=1)
    env = RegressionEnvironment(X, y, noise_level=0.1)

    viability_history = []
    error_history = []
    metamorphosis_count = 0

    for step in range(steps):
        viability = entity.live_one_step(env, ecosystem=None)
        viability_history.append(viability)

        # Measure prediction error
        try:
            test_idx = np.random.randint(len(X))
            test_input = X[test_idx]
            test_target = y[test_idx]

            prediction = entity.phenotype.forward(test_input)

            if len(prediction) > 0:
                error = np.abs(prediction[0] - test_target[0])
            else:
                error = 100.0

            error_history.append(error)
        except:
            error_history.append(100.0)

        # Count metamorphosis
        if step > 0 and len(entity.viability_history) >= 2:
            if entity.age > 0 and hasattr(entity, 'metamorphose'):
                # Track if metamorphosis happened (indirect)
                pass

        if step % 40 == 0 and step > 0:
            recent_viability = np.mean(viability_history[-20:])
            recent_error = np.mean(error_history[-20:])
            print(f"  Step {step:3d}: viability={recent_viability:.3f}, error={recent_error:.3f}")

    # Final metrics
    final_viability = np.mean(viability_history[-20:])
    final_error = np.mean(error_history[-20:])
    initial_error = np.mean(error_history[:20])

    print(f"\n{version_name} Results:")
    print(f"  Final viability: {final_viability:.3f}")
    print(f"  Initial error: {initial_error:.3f}")
    print(f"  Final error: {final_error:.3f}")
    print(f"  Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")
    print(f"  Entity age: {entity.age}")

    return {
        'viability_history': viability_history,
        'error_history': error_history,
        'final_viability': final_viability,
        'final_error': final_error,
        'initial_error': initial_error,
        'improvement': (initial_error - final_error) / initial_error,
        'entity': entity
    }


def main():
    """Compare v1.0 and v1.1"""
    print("="*70)
    print("GENESIS: v1.0 vs v1.1 Comparison")
    print("="*70)

    # Generate problem
    print("\n1. Generating regression problem...")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1).reshape(-1, 1)
    print(f"   Dataset: {X.shape[0]} samples")

    # Run v1.0
    print("\n2. Running v1.0 (baseline)...")
    results_v1_0 = run_entity(GENESIS_Entity, "GENESIS v1.0", X, y, steps=200)

    # Run v1.1
    print("\n2. Running v1.1 (improved)...")
    results_v1_1 = run_entity(GENESIS_Entity_v1_1, "GENESIS v1.1", X, y, steps=200)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print(f"\nViability:")
    print(f"  v1.0: {results_v1_0['final_viability']:.3f}")
    print(f"  v1.1: {results_v1_1['final_viability']:.3f}")
    viability_improvement = (results_v1_1['final_viability'] - results_v1_0['final_viability']) / results_v1_0['final_viability'] * 100
    print(f"  Change: {viability_improvement:+.1f}%")

    print(f"\nPrediction Error:")
    print(f"  v1.0: {results_v1_0['final_error']:.3f}")
    print(f"  v1.1: {results_v1_1['final_error']:.3f}")
    error_improvement = (results_v1_0['final_error'] - results_v1_1['final_error']) / results_v1_0['final_error'] * 100
    print(f"  Improvement: {error_improvement:+.1f}%")

    print(f"\nLearning Progress:")
    print(f"  v1.0: {results_v1_0['improvement']*100:.1f}%")
    print(f"  v1.1: {results_v1_1['improvement']*100:.1f}%")

    # Visualization
    print("\n3. Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Viability comparison
    ax = axes[0, 0]
    ax.plot(results_v1_0['viability_history'], label='v1.0', color='blue', alpha=0.7)
    ax.plot(results_v1_1['viability_history'], label='v1.1', color='green', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Viability')
    ax.set_title('Viability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    ax = axes[0, 1]
    # Smooth errors
    window = 10
    v1_0_smooth = np.convolve(results_v1_0['error_history'], np.ones(window)/window, mode='valid')
    v1_1_smooth = np.convolve(results_v1_1['error_history'], np.ones(window)/window, mode='valid')

    ax.plot(v1_0_smooth, label='v1.0', color='blue', alpha=0.7)
    ax.plot(v1_1_smooth, label='v1.1', color='green', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error Over Time (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Improvement bars
    ax = axes[1, 0]
    versions = ['v1.0', 'v1.1']
    improvements = [results_v1_0['improvement']*100, results_v1_1['improvement']*100]
    colors = ['blue', 'green']
    bars = ax.bar(versions, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Learning Improvement')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top')

    # Plot 4: Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    GENESIS v1.0 vs v1.1 Summary
    {'='*40}

    Phase 2 Improvements:
    ✓ Viability ↔ Performance connection
    ✓ Hebbian-like integration
    ✓ Controlled metamorphosis

    Results:
    {'─'*40}
    Viability:
      v1.0: {results_v1_0['final_viability']:.3f}
      v1.1: {results_v1_1['final_viability']:.3f}
      Change: {viability_improvement:+.1f}%

    Prediction Error:
      v1.0: {results_v1_0['final_error']:.3f}
      v1.1: {results_v1_1['final_error']:.3f}
      Improvement: {error_improvement:+.1f}%

    Learning Progress:
      v1.0: {results_v1_0['improvement']*100:.1f}%
      v1.1: {results_v1_1['improvement']*100:.1f}%

    Status:
    """

    if error_improvement > 0 and results_v1_1['improvement'] > results_v1_0['improvement']:
        summary_text += "    ✅ v1.1 IMPROVED over v1.0!"
    elif error_improvement > 0:
        summary_text += "    ✓ v1.1 shows promise"
    else:
        summary_text += "    ⚠️  Further refinement needed"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    save_path = '/Users/say/Documents/GitHub/ai/08_GENESIS/experiment_v1_0_v1_1_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if error_improvement > 10 and results_v1_1['improvement'] > 0:
        print("\n✅ SUCCESS! v1.1 shows significant improvement!")
        print(f"   Prediction error reduced by {error_improvement:.1f}%")
        print(f"   Learning progress: {results_v1_1['improvement']*100:.1f}%")
    elif error_improvement > 0:
        print("\n✓ Progress made. v1.1 is better than v1.0.")
        print(f"   Prediction error reduced by {error_improvement:.1f}%")
    else:
        print("\n⚠️  v1.1 did not improve over v1.0")
        print("   Possible reasons:")
        print("   - Random initialization variance")
        print("   - More steps needed")
        print("   - Further refinements needed")

    print("\nPhase 2 improvements implemented:")
    print("  ✓ Viability metric now reflects environment feedback (40% weight)")
    print("  ✓ Hebbian-like integration strengthens successful pathways")
    print("  ✓ Metamorphosis threshold tightened (0.01 → 0.005)")

    print("\n" + "="*70)

    return {
        'v1_0': results_v1_0,
        'v1_1': results_v1_1
    }


if __name__ == "__main__":
    results = main()
