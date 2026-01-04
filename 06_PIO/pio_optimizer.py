"""
PIO: Path Integral Optimizer

Directly applies Feynman's path integral formulation to neural network optimization.

Philosophy:
  - Nature doesn't "plan" - it superposes all possibilities
  - All updates are considered simultaneously
  - Actions (efficiency) naturally select the best path
  - Temperature controls exploration/exploitation

Based on:
  - Feynman path integrals (quantum mechanics)
  - Euclidean action (statistical mechanics)
  - Langevin dynamics (sampling)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# Simple Neural Network
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
# Path Integral Optimizer
# ============================================================================

class PathIntegralOptimizer:
    """
    Path Integral Optimizer (PIO)

    Uses Feynman path integral formulation:
      θ_{t+1} = θ_t + ⟨Δθ⟩
      ⟨Δθ⟩ = (1/Z) ∫ Δθ · e^(-S[Δθ]/T) D(Δθ)

    Where:
      S[Δθ] = (1/2)||Δθ||² + λ·L(θ + Δθ)  (Action)
      T = temperature (controls exploration)
      Z = partition function (normalization)
    """

    def __init__(self, network, n_samples=10, temperature=0.3,
                 lambda_loss=1.0, temp_decay=0.95):
        """
        Args:
            network: Neural network to optimize
            n_samples: Number of Monte Carlo samples
            temperature: Initial temperature
            lambda_loss: Weight of loss in action
            temp_decay: Temperature decay per iteration
        """
        self.network = network
        self.n_samples = n_samples
        self.temperature = temperature
        self.temp_init = temperature
        self.lambda_loss = lambda_loss
        self.temp_decay = temp_decay

        # Tracking
        self.loss_history = []
        self.temperature_history = []
        self.action_history = []

    def compute_action(self, delta, theta_current, X, y):
        """
        Compute action S[Δθ]

        S[Δθ] = (1/2)||Δθ||² + λ·L(θ + Δθ)

        Args:
            delta: Update vector Δθ
            theta_current: Current parameters
            X, y: Data

        Returns:
            float: Action value
        """
        # Kinetic term (penalize large updates)
        kinetic = 0.5 * np.sum(delta ** 2)

        # Potential term (loss at new position)
        theta_new = theta_current + delta
        self.network.set_weights(theta_new)
        loss = self.network.loss(X, y)
        potential = self.lambda_loss * loss

        action = kinetic + potential
        return action, loss

    def sample_path_integral(self, theta_current, X, y):
        """
        Sample updates via Langevin dynamics

        Langevin equation:
          dΔθ/dt = -∇S[Δθ] + √(2T) · η(t)

        Where η(t) is white noise.

        This generates samples from distribution:
          P(Δθ) ∝ e^(-S[Δθ]/T)

        Args:
            theta_current: Current parameters
            X, y: Data

        Returns:
            samples: List of Δθ samples
            weights: Boltzmann weights e^(-S/T)
            actions: Action values
        """
        # Initialize
        delta = np.zeros_like(theta_current)
        samples = []
        weights = []
        actions = []

        # Compute gradient once (for efficiency)
        grad = self.network.gradient(X, y)

        # Langevin dynamics parameters
        dt = 0.01  # Time step
        n_steps = self.n_samples

        for step in range(n_steps):
            # Gradient of action
            # ∇S = Δθ + λ·∇L
            grad_action = delta + self.lambda_loss * grad

            # Langevin update
            noise = np.random.randn(len(delta)) * np.sqrt(2 * dt * self.temperature)
            delta = delta - dt * grad_action + noise

            # Compute action and weight
            action, loss = self.compute_action(delta, theta_current, X, y)
            weight = np.exp(-action / self.temperature)

            samples.append(delta.copy())
            weights.append(weight)
            actions.append(action)

        return samples, np.array(weights), np.array(actions)

    def step(self, X, y):
        """
        Single optimization step

        Args:
            X, y: Training data

        Returns:
            loss: Current loss
            avg_action: Average action of samples
        """
        theta = self.network.get_weights()

        # Sample from path integral
        samples, weights, actions = self.sample_path_integral(theta, X, y)

        # Normalize weights
        weights = weights / (weights.sum() + 1e-10)

        # Weighted average (expectation value)
        final_update = sum(w * s for w, s in zip(weights, samples))

        # Apply update
        theta_new = theta + final_update
        self.network.set_weights(theta_new)

        # Compute loss
        loss = self.network.loss(X, y)
        avg_action = np.mean(actions)

        # Track
        self.loss_history.append(loss)
        self.temperature_history.append(self.temperature)
        self.action_history.append(avg_action)

        # Temperature decay (simulated annealing)
        self.temperature *= self.temp_decay

        return loss, avg_action

    def train(self, X, y, max_iters=100, tolerance=1e-3, verbose=True):
        """
        Full training loop

        Args:
            X, y: Training data
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            dict: Training results
        """
        start_time = time.time()

        if verbose:
            print("=" * 80)
            print("🌌 PIO: Path Integral Optimizer")
            print("=" * 80)
            print(f"Samples: {self.n_samples} | Initial T: {self.temp_init:.3f} | "
                  f"λ: {self.lambda_loss}")
            print("-" * 80)

        for i in range(max_iters):
            loss, avg_action = self.step(X, y)

            if verbose and i % 10 == 0:
                print(f"[{i:3d}] Loss: {loss:.5f} | "
                      f"Action: {avg_action:.3f} | "
                      f"T: {self.temperature:.4f}")

            # Convergence check
            if i > 10 and abs(self.loss_history[-1] - self.loss_history[-2]) < tolerance:
                if verbose:
                    print(f"\n✓ Converged at iteration {i}")
                break

        elapsed = time.time() - start_time

        return {
            'final_loss': self.loss_history[-1],
            'losses': self.loss_history,
            'iterations': len(self.loss_history),
            'time': elapsed,
            'temperatures': self.temperature_history,
            'actions': self.action_history,
        }


# ============================================================================
# Baseline: Standard SGD
# ============================================================================

def train_sgd(network, X, y, lr=0.05, max_iters=100, verbose=True):
    """Standard SGD for comparison"""
    losses = []
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 80)
        print("📊 Standard SGD (Baseline)")
        print("=" * 80)
        print(f"Learning rate: {lr}")
        print("-" * 80)

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

def plot_results(result_pio, result_sgd, dataset_name, save_path=None):
    """Plot PIO vs SGD comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(result_pio['losses'], label='PIO', linewidth=2, color='#8E44AD')
    ax.plot(result_sgd['losses'], label='SGD', linewidth=2, color='#E74C3C', linestyle='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Curves: {dataset_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Temperature evolution
    ax = axes[0, 1]
    temps = result_pio['temperatures']
    ax.plot(temps, color='#E67E22', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Evolution (Simulated Annealing)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='T=0.1')
    ax.legend()

    # Plot 3: Action evolution
    ax = axes[1, 0]
    actions = result_pio['actions']
    ax.plot(actions, color='#3498DB', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Action')
    ax.set_title('Action Evolution (Efficiency Metric)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance comparison
    ax = axes[1, 1]
    metrics = ['Final Loss', 'Iterations', 'Time (s)']
    pio_values = [result_pio['final_loss'], result_pio['iterations'], result_pio['time']]
    sgd_values = [result_sgd['final_loss'], result_sgd['iterations'], result_sgd['time']]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for visualization
    pio_norm = np.array(pio_values) / np.array(sgd_values)
    sgd_norm = np.ones(len(metrics))

    bars1 = ax.bar(x - width/2, pio_norm, width, label='PIO', color='#8E44AD', alpha=0.8)
    bars2 = ax.bar(x + width/2, sgd_norm, width, label='SGD', color='#E74C3C', alpha=0.8)

    ax.set_ylabel('Relative to SGD')
    ax.set_title('Performance Comparison (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement label
    improvement = (1 - result_pio['final_loss'] / result_sgd['final_loss']) * 100
    ax.text(0, pio_norm[0], f'{improvement:+.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {save_path}")

    plt.close()


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(dataset_name, X, y):
    """Run PIO vs SGD on given dataset"""
    print("\n" + "=" * 80)
    print(f"🔬 PIO vs SGD: {dataset_name.upper()}")
    print("=" * 80)

    # PIO
    net_pio = SimpleNetwork(X.shape[1], 8, y.shape[1])
    opt_pio = PathIntegralOptimizer(net_pio, n_samples=10, temperature=0.3,
                                    lambda_loss=1.0, temp_decay=0.95)
    result_pio = opt_pio.train(X, y, max_iters=100, verbose=True)

    # SGD
    net_sgd = SimpleNetwork(X.shape[1], 8, y.shape[1])
    result_sgd = train_sgd(net_sgd, X, y, lr=0.05, max_iters=100, verbose=True)

    # Results
    print("\n" + "=" * 80)
    print("📊 Results")
    print("=" * 80)
    print(f"{'Metric':<30} {'PIO':>15} {'SGD':>15}")
    print("-" * 80)
    print(f"{'Final Loss':<30} {result_pio['final_loss']:>15.6f} {result_sgd['final_loss']:>15.6f}")
    print(f"{'Iterations':<30} {result_pio['iterations']:>15} {result_sgd['iterations']:>15}")
    print(f"{'Time (seconds)':<30} {result_pio['time']:>15.4f} {result_sgd['time']:>15.4f}")

    improvement = (result_sgd['final_loss'] - result_pio['final_loss']) / result_sgd['final_loss'] * 100
    print(f"\n🎯 PIO Improvement: {improvement:+.2f}%")
    print("=" * 80)

    if improvement > 0:
        print("✅ PIO wins!")
    else:
        print("❌ SGD wins")

    # Plot
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "results", f"pio_{dataset_name.lower()}_results.png")
    plot_results(result_pio, result_sgd, dataset_name, save_path)

    return result_pio, result_sgd


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🌌 PIO: PATH INTEGRAL OPTIMIZER")
    print("   Feynman's quantum path integral → AI optimization")
    print("=" * 80)

    # Dataset 1: Linear
    np.random.seed(42)
    X_linear = np.random.randn(100, 3)
    y_linear = (X_linear @ np.array([1.5, -2.0, 0.5])[:, None] + 1.0)

    result_pio_linear, result_sgd_linear = run_experiment("Linear", X_linear, y_linear)

    # Dataset 2: Nonlinear
    X_nonlinear = np.random.randn(100, 3)
    y_nonlinear = np.sin(X_nonlinear[:, 0:1]) + np.cos(X_nonlinear[:, 1:2]) * X_nonlinear[:, 2:3]

    result_pio_nonlinear, result_sgd_nonlinear = run_experiment("Nonlinear", X_nonlinear, y_nonlinear)

    # Dataset 3: XOR
    X_xor = np.random.randn(100, 2)
    y_xor = ((X_xor[:, 0:1] > 0) ^ (X_xor[:, 1:2] > 0)).astype(float)

    result_pio_xor, result_sgd_xor = run_experiment("XOR", X_xor, y_xor)

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 Final Summary: PIO vs SGD")
    print("=" * 80)

    results = [
        ("Linear", result_pio_linear, result_sgd_linear),
        ("Nonlinear", result_pio_nonlinear, result_sgd_nonlinear),
        ("XOR", result_pio_xor, result_sgd_xor),
    ]

    wins = 0
    for name, pio, sgd in results:
        improvement = (sgd['final_loss'] - pio['final_loss']) / sgd['final_loss'] * 100
        winner = "✅ PIO" if improvement > 0 else "❌ SGD"
        print(f"{name:<12} | {winner:<10} | {improvement:+6.2f}%")
        if improvement > 0:
            wins += 1

    print("=" * 80)
    print(f"PIO win rate: {wins}/3 ({wins/3*100:.1f}%)")
    print("=" * 80)

    print("\n🌌 Path Integral Optimization: Nature's way of optimizing!")
    print("   All paths superposed, action selects the best.")
