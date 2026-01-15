"""
ULTIMATE: Meta-Conscious Optimizer
궁극의 최적화 시스템

3-Layer Architecture:
    Layer 1: Primitive Pool (10 universal primitives)
    Layer 2: Strategy Selector (Policy Network)
    Layer 3: Meta-Learner (Experience → Knowledge)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from primitives import get_all_primitives
from context import OptimizationContext
from policy_network import PolicyNetwork
from meta_learner import MetaLearner, ExperienceBuffer


class ULTIMATE_Optimizer:
    """
    Meta-conscious optimizer that:
    1. Observes problem characteristics (Context)
    2. Selects optimal strategy (Policy Network)
    3. Learns from experience (Meta-Learner)
    4. Continuously improves (Evolution)

    v1.1 improvements:
    - Winner-take-all mode (decisive when confident)
    - Dynamic LR scaling
    - Tuned primitive LRs
    """
    def __init__(self, network, max_iterations=200, meta_learning_interval=50,
                 meta_learning_enabled=True, verbose=False,
                 winner_take_all=True, confidence_threshold=0.85,
                 tuned_primitives=True):
        """
        Args:
            network: Neural network to optimize
            max_iterations: Maximum optimization iterations
            meta_learning_interval: Update policy every N iterations
            meta_learning_enabled: Enable meta-learning (False for cold start)
            verbose: Print detailed info
            winner_take_all: Enable winner-take-all mode (v1.1)
            confidence_threshold: Threshold for winner-take-all (v1.1)
            tuned_primitives: Use tuned LRs for primitives (v1.1)
        """
        self.network = network
        self.max_iterations = max_iterations
        self.meta_learning_interval = meta_learning_interval
        self.meta_learning_enabled = meta_learning_enabled
        self.verbose = verbose

        # v1.1 features
        self.winner_take_all = winner_take_all
        self.confidence_threshold = confidence_threshold

        # Layer 1: Primitive Pool
        self.primitives = get_all_primitives(tuned=tuned_primitives)
        self.n_primitives = len(self.primitives)

        # Layer 2: Policy Network
        self.policy_network = PolicyNetwork(context_dim=12, n_primitives=self.n_primitives)

        # Layer 3: Meta-Learner
        self.meta_learner = MetaLearner(self.policy_network, learning_rate=0.001)
        self.experience_buffer = ExperienceBuffer(max_size=1000)

        # Context Tracker
        self.context = OptimizationContext(max_iterations=max_iterations)

        # History
        self.loss_history = []
        self.weight_history = []  # Primitive weights over time

    def step(self, X, y):
        """
        Single optimization step

        Returns:
            new_loss: Loss after update
        """
        # 1. Compute context
        context_vector = self.context.get_context_vector(self.network, X, y)

        # 2. Get strategy from policy network
        primitive_weights = self.policy_network.forward(context_vector)

        if self.verbose and self.context.iteration % 20 == 0:
            print(f"\nIteration {self.context.iteration}")
            print(f"  Context: progress={context_vector[3]:.2f}, improvement_rate={context_vector[4]:.4f}")
            print(f"  Top 3 primitives: {np.argsort(primitive_weights)[-3:][::-1]}")

        # 3. Compute updates from all primitives
        current_theta = self.network.get_weights()
        updates = []

        for primitive in self.primitives:
            try:
                update = primitive.compute_update(self.network, X, y, self.context)
                # Check for NaN
                if np.any(np.isnan(update)):
                    if self.verbose:
                        print(f"  Warning: Primitive {type(primitive).__name__} returned NaN, using zeros")
                    update = np.zeros_like(current_theta)
                updates.append(update)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Primitive {type(primitive).__name__} failed: {e}")
                updates.append(np.zeros_like(current_theta))

        # 4. Weighted combination (v1.1: winner-take-all or ensemble)
        max_weight = np.max(primitive_weights)
        max_idx = np.argmax(primitive_weights)

        if self.winner_take_all and max_weight >= self.confidence_threshold:
            # Winner-take-all: Use only the dominant primitive
            final_update = updates[max_idx]

            if self.verbose and self.context.iteration % 20 == 0:
                primitive_names = [
                    'GradDescent', 'Momentum', 'Adaptive', 'ParticleSwarm', 'BestAttr',
                    'StochJump', 'PathSample', 'ActionGuide', 'MultiScale', 'Ensemble'
                ]
                print(f"  Winner-take-all: {primitive_names[max_idx]} ({max_weight:.1%})")
        else:
            # Ensemble: Weighted combination
            final_update = sum(w * u for w, u in zip(primitive_weights, updates))

        # Check final update for NaN
        if np.any(np.isnan(final_update)):
            if self.verbose:
                print(f"  Warning: Final update has NaN, using gradient descent")
            grad = self.network.compute_gradient(X, y)
            final_update = -0.01 * grad

        # 5. Apply update
        old_loss = self.network.loss(X, y)
        new_theta = current_theta + final_update
        self.network.set_weights(new_theta)
        new_loss = self.network.loss(X, y)

        # 6. Compute improvement and action
        improvement = old_loss - new_loss
        kinetic = 0.5 * np.sum(final_update ** 2)
        action = kinetic + new_loss

        # 7. Update context
        grad = self.network.compute_gradient(X, y)
        self.context.update(new_loss, grad, improvement, action)

        # 8. Record experience
        self.experience_buffer.add(context_vector, primitive_weights, improvement)

        # 9. Meta-learning (periodic)
        if (self.meta_learning_enabled and
            self.context.iteration % self.meta_learning_interval == 0 and
            len(self.experience_buffer) >= 50):

            if self.verbose:
                print(f"\n=== Meta-Learning Update at iteration {self.context.iteration} ===")

            self.meta_learner.update(self.experience_buffer, n_epochs=5, batch_size=32)

            if self.verbose:
                error = self.meta_learner.evaluate(self.experience_buffer)
                print(f"  Policy prediction error: {error:.6f}")

        # 10. Record history
        self.loss_history.append(new_loss)
        self.weight_history.append(primitive_weights.copy())

        return new_loss

    def optimize(self, X, y):
        """
        Full optimization run

        Returns:
            final_loss: Final loss value
            history: Loss history
        """
        print(f"\n{'='*60}")
        print(f"ULTIMATE Optimizer")
        print(f"Meta-learning: {'Enabled' if self.meta_learning_enabled else 'Disabled (Cold Start)'}")
        print(f"{'='*60}\n")

        for iteration in range(self.max_iterations):
            loss = self.step(X, y)

            if iteration % 20 == 0:
                print(f"Iteration {iteration:3d} | Loss: {loss:.6f}")

        final_loss = self.loss_history[-1]
        print(f"\nFinal Loss: {final_loss:.6f}")

        return final_loss, self.loss_history

    def plot_results(self, dataset_name, baseline_losses=None):
        """
        Plot optimization results

        Args:
            dataset_name: Name of dataset
            baseline_losses: Dict of {name: loss_history} for comparison
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "results", f"ultimate_{dataset_name.lower()}_results.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Loss curve with comparisons
        ax = axes[0, 0]
        ax.plot(self.loss_history, label='ULTIMATE', linewidth=2, color='red')

        if baseline_losses:
            for name, losses in baseline_losses.items():
                ax.plot(losses, label=name, alpha=0.7, linestyle='--')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Curve - {dataset_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 2. Primitive weights over time
        ax = axes[0, 1]
        weight_array = np.array(self.weight_history)

        primitive_names = [
            'GradDescent', 'Momentum', 'Adaptive', 'ParticleSwarm', 'BestAttr',
            'StochJump', 'PathSample', 'ActionGuide', 'MultiScale', 'Ensemble'
        ]

        for i in range(self.n_primitives):
            ax.plot(weight_array[:, i], label=primitive_names[i], alpha=0.7)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title('Primitive Weights Evolution')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. Average primitive usage
        ax = axes[1, 0]
        avg_weights = np.mean(weight_array, axis=0)
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_primitives))

        bars = ax.bar(range(self.n_primitives), avg_weights, color=colors)
        ax.set_xticks(range(self.n_primitives))
        ax.set_xticklabels(primitive_names, rotation=45, ha='right')
        ax.set_ylabel('Average Weight')
        ax.set_title('Average Primitive Usage')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Improvement rate over time
        ax = axes[1, 1]
        improvements = [self.loss_history[i] - self.loss_history[i+1]
                       for i in range(len(self.loss_history)-1)]
        improvements.append(0)

        ax.plot(improvements, color='green', alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement')
        ax.set_title('Step-wise Improvement')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")
        plt.close()

    def get_strategy_summary(self):
        """
        Get summary of learned strategy

        Returns:
            dict with average primitive weights
        """
        if len(self.weight_history) == 0:
            return {}

        weight_array = np.array(self.weight_history)
        avg_weights = np.mean(weight_array, axis=0)

        primitive_names = [
            'GradientDescent', 'Momentum', 'Adaptive', 'ParticleSwarm', 'BestAttractor',
            'StochasticJump', 'PathSampling', 'ActionGuided', 'MultiScale', 'EnsembleAverage'
        ]

        summary = {name: weight for name, weight in zip(primitive_names, avg_weights)}
        return summary
