"""
GENESIS v2.0 Single-Task Validation Experiment
Author: GENESIS Project
Date: 2026-01-03
Version: 2.0

Purpose:
    Validate GENESIS v2.0 on single-task regression learning.
    Compare against v1.1 baseline to measure improvements.

Expected Improvements:
    - Faster learning (gradient-based vs Hebbian-only)
    - Lower final error (global optimization)
    - More stable viability (direct feedback loop)
    - Maintained catastrophic forgetting resistance

Metrics:
    - Final prediction error
    - Learning progress (% improvement)
    - Viability trajectory
    - Convergence speed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from genesis_entity_v2_0 import GENESIS_Entity_v2_0, Genome_v2_0


class RegressionEnvironment:
    """
    Simple regression task: y = W_true · x + noise

    This is the same task used in v1.1 experiments for fair comparison.
    """

    def __init__(self, input_dim: int = 10, noise_level: float = 0.1, seed: int = 42):
        """
        Args:
            input_dim: Input dimensionality
            noise_level: Gaussian noise standard deviation
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        self.input_dim = input_dim
        self.noise_level = noise_level

        # True function: y = W_true · x
        self.W_true = np.random.randn(input_dim) * 2.0
        self.current_input = None

        print(f"Environment initialized: input_dim={input_dim}, noise={noise_level}")
        print(f"True weights norm: {np.linalg.norm(self.W_true):.3f}")

    def get_input(self) -> np.ndarray:
        """Generate random input sample."""
        self.current_input = np.random.randn(self.input_dim)
        return self.current_input

    def get_target(self, x: np.ndarray) -> np.ndarray:
        """
        Get ground truth target for input.

        Args:
            x: Input sample

        Returns:
            target: True output (with noise)
        """
        true_output = np.dot(x, self.W_true)
        noise = np.random.randn() * self.noise_level
        return np.array([[true_output + noise]])

    def get_task_context(self) -> dict:
        """Provide task context for router."""
        return {'task_name': 'regression'}


def run_experiment(n_steps: int = 500, learning_rate: float = 0.01, seed: int = 42):
    """
    Run single-task learning experiment.

    Args:
        n_steps: Number of training steps
        learning_rate: Gradient learning rate
        seed: Random seed

    Returns:
        results: Dictionary with metrics and history
    """
    print("\n" + "=" * 70)
    print("GENESIS v2.0 Single-Task Validation")
    print("=" * 70)

    # Set random seed
    np.random.seed(seed)

    # Initialize environment
    env = RegressionEnvironment(input_dim=10, noise_level=0.1, seed=seed)

    # Initialize entity
    genome = Genome_v2_0(
        input_size=10,
        shared_size=32,
        module_size=16,
        learning_rate_gradient=learning_rate,
        learning_rate_hebbian=0.01,
        success_threshold=1.0
    )

    entity = GENESIS_Entity_v2_0(genome)

    print(f"\nEntity initialized: id={entity.id}")
    print(f"  Shared encoder: {genome.shared_size} hidden units")
    print(f"  Module size: {genome.module_size} hidden units")
    print(f"  Learning rates: gradient={learning_rate}, hebbian={genome.learning_rate_hebbian}")

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training for {n_steps} steps...")
    print(f"{'='*70}\n")

    error_history = []
    viability_history = []
    success_history = []

    for step in range(n_steps):
        viability = entity.live_one_step(env)

        # Record metrics
        if len(entity.error_history) > 0:
            error_history.append(entity.error_history[-1])
            viability_history.append(viability)
            success_history.append(entity.success_history[-1])

        # Progress reporting
        if step % 50 == 0 or step == n_steps - 1:
            recent_error = np.mean(list(entity.error_history)[-20:]) if len(entity.error_history) >= 20 else error_history[-1]
            recent_viability = np.mean(viability_history[-20:]) if len(viability_history) >= 20 else viability
            recent_success = np.mean(success_history[-20:]) if len(success_history) >= 20 else success_history[-1]

            print(f"Step {step:3d} | Error: {recent_error:6.3f} | Viability: {recent_viability:.3f} | Success: {recent_success:.1%}")

    # Compute final metrics
    print(f"\n{'='*70}")
    print("Final Results")
    print(f"{'='*70}")

    initial_error = np.mean(error_history[:20])
    final_error = np.mean(error_history[-20:])
    learning_progress = ((initial_error - final_error) / initial_error) * 100

    initial_viability = np.mean(viability_history[:20])
    final_viability = np.mean(viability_history[-20:])
    viability_improvement = ((final_viability - initial_viability) / initial_viability) * 100

    final_success_rate = np.mean(success_history[-100:])

    print(f"\n**Error Metrics**:")
    print(f"  Initial Error (steps 0-20):  {initial_error:.3f}")
    print(f"  Final Error (steps 480-500): {final_error:.3f}")
    print(f"  Error Reduction:              {((initial_error - final_error) / initial_error * 100):.1f}%")
    print(f"  Learning Progress:            {learning_progress:.1f}%")

    print(f"\n**Viability Metrics**:")
    print(f"  Initial Viability:  {initial_viability:.3f}")
    print(f"  Final Viability:    {final_viability:.3f}")
    print(f"  Improvement:        {viability_improvement:.1f}%")

    print(f"\n**Success Rate**:")
    print(f"  Final 100 steps: {final_success_rate:.1%}")

    print(f"\n**Architecture**:")
    entity_summary = entity.get_summary()
    print(f"  Tasks learned: {entity_summary['n_tasks']}")
    print(f"  Modules used:  {entity_summary['n_modules']}")

    # Return results
    results = {
        'error_history': error_history,
        'viability_history': viability_history,
        'success_history': success_history,
        'metrics': {
            'initial_error': initial_error,
            'final_error': final_error,
            'learning_progress': learning_progress,
            'initial_viability': initial_viability,
            'final_viability': final_viability,
            'viability_improvement': viability_improvement,
            'final_success_rate': final_success_rate,
        },
        'entity_summary': entity_summary
    }

    return results


def plot_results(results: dict, save_path: str = None):
    """
    Plot learning curves.

    Args:
        results: Experiment results
        save_path: Path to save figure (if None, display only)
    """
    error_history = results['error_history']
    viability_history = results['viability_history']
    success_history = results['success_history']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Error over time
    ax = axes[0]
    ax.plot(error_history, alpha=0.3, color='red', label='Raw')

    # Smooth with moving average
    window = 20
    if len(error_history) >= window:
        smoothed = np.convolve(error_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(error_history)), smoothed, color='red', linewidth=2, label='Smoothed')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Viability over time
    ax = axes[1]
    ax.plot(viability_history, alpha=0.3, color='blue', label='Raw')

    if len(viability_history) >= window:
        smoothed = np.convolve(viability_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(viability_history)), smoothed, color='blue', linewidth=2, label='Smoothed')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Viability')
    ax.set_title('Viability Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Success rate over time
    ax = axes[2]
    if len(success_history) >= window:
        success_rate = np.convolve(success_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(success_history)), success_rate, color='green', linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (20-step window)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()  # Close instead of show to prevent hanging


def compare_with_v1_1():
    """
    Compare v2.0 results with v1.1 baseline.

    v1.1 Baseline (from Phase 2 analysis):
        - Initial Error: 4.977
        - Final Error: 3.072
        - Learning Progress: +10.2%
        - Viability Improvement: +43.0%
        - Error Reduction: -22.4%
    """
    print("\n" + "=" * 70)
    print("v1.1 vs v2.0 Comparison")
    print("=" * 70)

    v1_1_baseline = {
        'initial_error': 4.977,
        'final_error': 3.072,
        'learning_progress': 10.2,
        'viability_improvement': 43.0,
        'error_reduction': -22.4
    }

    print("\n**Expected v2.0 Advantages**:")
    print("  1. Faster convergence (gradient-based learning)")
    print("  2. Lower final error (global optimization)")
    print("  3. More stable training (direct feedback loop)")
    print("  4. Maintained catastrophic forgetting resistance")

    print("\n**v1.1 Baseline**:")
    print(f"  Initial Error:     {v1_1_baseline['initial_error']:.3f}")
    print(f"  Final Error:       {v1_1_baseline['final_error']:.3f}")
    print(f"  Learning Progress: +{v1_1_baseline['learning_progress']:.1f}%")
    print(f"  Viability Improv:  +{v1_1_baseline['viability_improvement']:.1f}%")

    print("\n*Run experiment to compare v2.0 performance...*")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║               GENESIS v2.0 Validation Experiment                  ║
    ║                                                                   ║
    ║  Purpose: Test v2.0 improvements over v1.1 on single-task        ║
    ║  Expected: Better learning, lower error, faster convergence      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Show v1.1 baseline
    compare_with_v1_1()

    # Run v2.0 experiment with balanced learning rate
    results = run_experiment(n_steps=500, learning_rate=0.001, seed=42)

    # Plot results
    result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, 'v2_0_single_task_validation.png')
    plot_results(results, save_path=save_path)

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
