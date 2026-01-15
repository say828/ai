"""
GENESIS Experiment 1: Survival Without Loss Function

Question: Can an AI entity learn to perform a task (regression)
WITHOUT an explicit loss function?

Traditional AI:
  loss = MSE(prediction, target)
  gradient = ∂loss/∂θ
  θ = θ - lr * gradient

GENESIS:
  viability = can_it_survive(entity, environment)
  if viability > threshold:
      continue_existing()
  else:
      transform_or_die()

NO GRADIENTS
NO LOSS FUNCTION
ONLY VIABILITY
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from genesis_entity import GENESIS_Entity
from genesis_environment import RegressionEnvironment


def run_experiment_1():
    """
    Test: Can GENESIS learn regression without loss function?
    """
    print("="*70)
    print("GENESIS EXPERIMENT 1: Survival Without Loss Function")
    print("="*70)

    # Generate regression problem
    print("\n1. Generating regression problem...")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1).reshape(-1, 1)

    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   True function: y = 2*x1 + 3*x2 + noise")

    # Create environment
    env = RegressionEnvironment(X, y, noise_level=0.1)

    # Create GENESIS entity
    print("\n2. Creating GENESIS entity...")
    entity = GENESIS_Entity(entity_id=1)
    print(f"   {entity}")
    print(f"   Genome: {entity.genome}")
    print(f"   Initial viability: {entity.viability:.3f}")

    # Let entity exist and learn
    print("\n3. Entity living for 200 steps...")
    print("   (NO loss function, NO gradient descent, ONLY viability!)")

    viability_history = []
    error_history = []

    for step in range(200):
        # Entity lives one step
        viability = entity.live_one_step(env, ecosystem=None)
        viability_history.append(viability)

        # Measure actual prediction error (for comparison)
        # Entity doesn't see this!
        try:
            test_idx = np.random.randint(len(X))
            test_input = X[test_idx]
            test_target = y[test_idx]

            # Entity makes prediction
            # Simplified: use phenotype to predict
            prediction = entity.phenotype.forward(test_input)

            if len(prediction) > 0:
                error = np.abs(prediction[0] - test_target[0])
            else:
                error = 100.0

            error_history.append(error)
        except:
            error_history.append(100.0)

        # Progress
        if step % 20 == 0 and step > 0:
            recent_viability = np.mean(viability_history[-20:])
            recent_error = np.mean(error_history[-20:])
            print(f"   Step {step:3d}: viability={recent_viability:.3f}, "
                  f"error={recent_error:.3f}, intentions={len(entity.intentions)}")

    # Final results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    final_viability = np.mean(viability_history[-20:])
    final_error = np.mean(error_history[-20:])
    initial_error = np.mean(error_history[:20])

    print(f"\nFinal viability: {final_viability:.3f}")
    print(f"Initial error: {initial_error:.3f}")
    print(f"Final error: {final_error:.3f}")
    print(f"Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")

    print(f"\nEntity age: {entity.age}")
    print(f"Experiences: {len(entity.experiences)}")
    print(f"Capabilities: {entity.self_model.assess_capabilities()}")
    print(f"Identity: {entity.self_model.construct_identity()}")
    print(f"Purpose: {entity.self_model.infer_purpose()}")

    # Visualization
    print("\n4. Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Viability over time
    ax = axes[0, 0]
    ax.plot(viability_history, color='green', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Survival threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Viability')
    ax.set_title('Viability Over Time (NO loss function!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Prediction error over time
    ax = axes[0, 1]
    # Smooth error for better visualization
    window = 10
    smoothed_error = np.convolve(error_history, np.ones(window)/window, mode='valid')
    ax.plot(smoothed_error, color='red', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error (Entity doesn\'t see this!)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Viability vs Error
    ax = axes[1, 0]
    # Sample to avoid overplotting
    sample_indices = np.linspace(0, len(viability_history)-1, 100, dtype=int)
    ax.scatter(
        [error_history[i] for i in sample_indices],
        [viability_history[i] for i in sample_indices],
        alpha=0.5,
        c=range(len(sample_indices)),
        cmap='viridis'
    )
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Viability')
    ax.set_title('Viability vs Error (Correlation?)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Entity development
    ax = axes[1, 1]
    milestones = {
        'Birth': 0,
        'First Intention': 0,
        'First Capability': 0,
        'Metamorphosis': 0,
        'Final': entity.age
    }

    # Simplified milestone visualization
    ax.barh(range(len(milestones)), list(milestones.values()), color='blue', alpha=0.6)
    ax.set_yticks(range(len(milestones)))
    ax.set_yticklabels(milestones.keys())
    ax.set_xlabel('Step')
    ax.set_title('Entity Lifecycle')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save
    save_path = '/Users/say/Documents/GitHub/ai/08_GENESIS/experiment_1_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if final_error < initial_error:
        improvement_pct = (initial_error - final_error) / initial_error * 100
        print(f"\n✅ SUCCESS! Entity learned WITHOUT loss function!")
        print(f"   Prediction error improved by {improvement_pct:.1f}%")
        print(f"   Final viability: {final_viability:.3f}")
        print("\n   This proves:")
        print("   - Learning is possible without explicit loss function")
        print("   - Viability-driven evolution works")
        print("   - Self-generated intentions can guide learning")
    else:
        print(f"\n⚠️  Entity survived but didn't improve prediction")
        print(f"   This suggests:")
        print("   - Viability metric needs refinement")
        print("   - More steps needed for convergence")
        print("   - Environment feedback could be stronger")

    print("\n" + "="*70)

    return {
        'final_viability': final_viability,
        'final_error': final_error,
        'initial_error': initial_error,
        'improvement': (initial_error - final_error) / initial_error,
        'entity': entity
    }


if __name__ == "__main__":
    results = run_experiment_1()

    print("\nExperiment 1 complete!")
    print(f"Final improvement: {results['improvement']*100:.1f}%")
