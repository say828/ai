"""
ULTIMATE: Universal Primitives
10개의 범용 최적화 기본 요소
"""

import numpy as np


class Primitive:
    """Base class for all primitives"""
    def compute_update(self, network, X, y, context):
        """
        Compute update for network parameters

        Args:
            network: Neural network object
            X, y: Training data
            context: OptimizationContext object

        Returns:
            update: numpy array of parameter updates
        """
        raise NotImplementedError


class GradientDescent(Primitive):
    """P1: Basic gradient descent direction"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def compute_update(self, network, X, y, context):
        grad = network.compute_gradient(X, y)
        return -self.lr * grad


class MomentumUpdate(Primitive):
    """P2: Momentum-based updates (inertia)"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def compute_update(self, network, X, y, context):
        grad = network.compute_gradient(X, y)

        if self.velocity is None:
            self.velocity = np.zeros_like(grad)

        self.velocity = self.momentum * self.velocity - self.lr * grad
        return self.velocity


class AdaptiveStep(Primitive):
    """P3: Per-parameter adaptive learning rates (Adam-like)"""
    def __init__(self, lr=0.01, epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1  # First moment decay
        self.beta2 = beta2  # Second moment decay
        self.m = None  # First moment (mean)
        self.v = None  # Second moment (variance)
        self.t = 0     # Time step

    def compute_update(self, network, X, y, context):
        grad = network.compute_gradient(X, y)

        # Initialize moments
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        # Update time step
        self.t += 1

        # Update biased first moment estimate (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # Update biased second moment estimate (RMSprop)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Adaptive update (Adam-like)
        adapted_lr = self.lr / (np.sqrt(v_hat) + self.epsilon)

        return -adapted_lr * m_hat


class ParticleSwarm(Primitive):
    """P4: Collective exploration (simplified swarm)"""
    def __init__(self, lr=0.01, n_particles=5):
        self.lr = lr
        self.n_particles = n_particles
        self.particles = []
        self.best_particle = None
        self.best_loss = float('inf')

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()

        # Initialize particles
        if len(self.particles) == 0:
            self.particles = [current_theta.copy() for _ in range(self.n_particles)]
            self.best_particle = current_theta.copy()

        try:
            # Evaluate particles
            for i, particle in enumerate(self.particles):
                network.set_weights(particle)
                loss = network.loss(X, y)

                if loss < self.best_loss and not np.isnan(loss):
                    self.best_loss = loss
                    self.best_particle = particle.copy()

            # Update particles toward best
            for i in range(len(self.particles)):
                toward_best = self.best_particle - self.particles[i]
                random_explore = np.random.randn(len(current_theta)) * 0.01
                self.particles[i] += 0.1 * toward_best + random_explore

            # Return direction toward best particle
            update = self.lr * (self.best_particle - current_theta)
        finally:
            # Always restore current weights
            network.set_weights(current_theta)

        return update if not np.any(np.isnan(update)) else np.zeros_like(current_theta)


class BestAttractor(Primitive):
    """P5: Move toward historically best point"""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.best_theta = None
        self.best_loss = float('inf')

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()
        current_loss = network.loss(X, y)

        # Update best
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_theta = current_theta.copy()

        if self.best_theta is None:
            return np.zeros_like(current_theta)

        # Move toward best
        direction = self.best_theta - current_theta
        return self.lr * direction


class StochasticJump(Primitive):
    """P6: Random exploration jumps (escape)"""
    def __init__(self, jump_size=0.01):
        self.jump_size = jump_size

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()
        return np.random.randn(len(current_theta)) * self.jump_size


class PathSampling(Primitive):
    """P7: Sample multiple paths (simplified path integral)"""
    def __init__(self, lr=0.01, n_samples=20, temperature=0.1):
        self.lr = lr
        self.n_samples = n_samples
        self.temperature = temperature

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()
        grad = network.compute_gradient(X, y)

        samples = []
        weights = []

        try:
            for _ in range(self.n_samples):
                # Sample a direction
                noise = np.random.randn(len(current_theta)) * 0.01
                delta = -self.lr * grad + noise

                # Compute action (loss)
                network.set_weights(current_theta + delta)
                loss = network.loss(X, y)

                if not np.isnan(loss):
                    # Boltzmann weight (clip for stability)
                    weight = np.exp(np.clip(-loss / self.temperature, -10, 10))
                    samples.append(delta)
                    weights.append(weight)

            # Weighted average
            if len(weights) > 0:
                weights = np.array(weights)
                weights = weights / (weights.sum() + 1e-10)
                weighted_update = sum(w * s for w, s in zip(weights, samples))
            else:
                weighted_update = -self.lr * grad

        finally:
            # Always restore weights
            network.set_weights(current_theta)

        return weighted_update if not np.any(np.isnan(weighted_update)) else np.zeros_like(current_theta)


class ActionGuided(Primitive):
    """P8: LAML-Q style action-guided updates"""
    def __init__(self, lr=0.01, n_candidates=3):
        self.lr = lr
        self.n_candidates = n_candidates

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()
        grad = network.compute_gradient(X, y)

        candidates = []
        actions = []

        try:
            for _ in range(self.n_candidates):
                # Generate candidate update
                noise = np.random.randn(len(grad)) * 0.1
                delta = -self.lr * (grad + noise)

                # Compute action (kinetic + potential)
                kinetic = 0.5 * np.sum(delta ** 2)

                network.set_weights(current_theta + delta)
                loss = network.loss(X, y)

                if not np.isnan(loss):
                    potential = loss
                    action = kinetic + potential
                    candidates.append(delta)
                    actions.append(action)

            # Select candidate with minimum action
            if len(actions) > 0:
                best_idx = np.argmin(actions)
                update = candidates[best_idx]
            else:
                update = -self.lr * grad

        finally:
            # Always restore weights
            network.set_weights(current_theta)

        return update if not np.any(np.isnan(update)) else np.zeros_like(current_theta)


class MultiScale(Primitive):
    """P9: Multi-scale predictions"""
    def __init__(self, lr=0.01, scales=[1, 2, 5]):
        self.lr = lr
        self.scales = scales

    def compute_update(self, network, X, y, context):
        grad = network.compute_gradient(X, y)
        current_theta = network.get_weights()

        updates = []
        losses = []

        try:
            for scale in self.scales:
                delta = -self.lr * scale * grad

                network.set_weights(current_theta + delta)
                loss = network.loss(X, y)

                if not np.isnan(loss):
                    updates.append(delta)
                    losses.append(loss)

            # Weight by improvement
            if len(losses) > 0:
                losses = np.array(losses)
                # Clip losses for stability
                weights = np.exp(np.clip(-losses, -10, 10))
                weights = weights / (weights.sum() + 1e-10)
                weighted_update = sum(w * u for w, u in zip(weights, updates))
            else:
                weighted_update = -self.lr * grad

        finally:
            # Always restore weights
            network.set_weights(current_theta)

        return weighted_update if not np.any(np.isnan(weighted_update)) else np.zeros_like(current_theta)


class EnsembleAverage(Primitive):
    """P10: Maintain ensemble of hypotheses"""
    def __init__(self, lr=0.01, n_ensemble=3):
        self.lr = lr
        self.n_ensemble = n_ensemble
        self.ensemble = []

    def compute_update(self, network, X, y, context):
        current_theta = network.get_weights()

        # Initialize ensemble
        if len(self.ensemble) == 0:
            for _ in range(self.n_ensemble):
                noise = np.random.randn(len(current_theta)) * 0.01
                self.ensemble.append(current_theta + noise)

        try:
            # Update each ensemble member
            for i in range(len(self.ensemble)):
                network.set_weights(self.ensemble[i])
                grad_i = network.compute_gradient(X, y)

                if not np.any(np.isnan(grad_i)):
                    self.ensemble[i] = self.ensemble[i] - self.lr * grad_i

            # Average ensemble direction
            ensemble_mean = np.mean(self.ensemble, axis=0)
            update = ensemble_mean - current_theta

        finally:
            # Always restore weights
            network.set_weights(current_theta)

        return update if not np.any(np.isnan(update)) else np.zeros_like(current_theta)


def get_all_primitives(tuned=True):
    """
    Get all 10 universal primitives

    Args:
        tuned: If True, use tuned learning rates. If False, use default 0.01 for all.

    v1.2 changes:
    - AdaptiveStep now uses Adam-like algorithm (first + second moments)
    - PathSampling uses 20 samples instead of 5 (better exploration)
    """
    if tuned:
        # v1.1: Tuned learning rates for each primitive (FAILED - DO NOT USE)
        return [
            GradientDescent(lr=0.005),        # 작게: 안정적이어야 함
            MomentumUpdate(lr=0.008, momentum=0.9),   # 중간
            AdaptiveStep(lr=0.01),            # 그대로: 자체 적응
            ParticleSwarm(lr=0.015, n_particles=5),   # 크게: 탐색 강화
            BestAttractor(lr=0.01),           # 그대로
            StochasticJump(jump_size=0.02),   # 크게: 탈출 강화
            PathSampling(lr=0.008, n_samples=20, temperature=0.1),  # v1.2: 20 samples
            ActionGuided(lr=0.01, n_candidates=3),     # 그대로
            MultiScale(lr=0.006, scales=[1, 2, 5]),    # 작게: 이미 여러 스케일
            EnsembleAverage(lr=0.008, n_ensemble=3)   # 중간
        ]
    else:
        # v1.2: Improved primitives with uniform learning rates (RECOMMENDED)
        return [
            GradientDescent(lr=0.01),
            MomentumUpdate(lr=0.01, momentum=0.9),
            AdaptiveStep(lr=0.01),  # Now Adam-like!
            ParticleSwarm(lr=0.01, n_particles=5),
            BestAttractor(lr=0.01),
            StochasticJump(jump_size=0.01),
            PathSampling(lr=0.01, n_samples=20, temperature=0.1),  # v1.2: 20 samples
            ActionGuided(lr=0.01, n_candidates=3),
            MultiScale(lr=0.01, scales=[1, 2, 5]),
            EnsembleAverage(lr=0.01, n_ensemble=3)
        ]
