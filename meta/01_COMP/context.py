"""
Optimization Context Tracking

Tracks the state of optimization process and provides
contextual information for intelligent strategy selection.
"""

import numpy as np
from collections import deque


class OptimizationContext:
    """
    Tracks optimization state and computes contextual features

    Features tracked:
    - Current state (loss, gradient norm, theta)
    - Historical data (loss history, gradient history)
    - Statistics (success rate, variance, improvement rate)
    - Phase (exploration, exploitation, refinement)
    """

    def __init__(self, history_size=20):
        """
        Args:
            history_size: Number of recent steps to keep in memory
        """
        # Current state
        self.iteration = 0
        self.current_loss = float('inf')
        self.current_grad_norm = 0.0
        self.current_theta = None

        # History (recent N steps only)
        self.loss_history = deque(maxlen=history_size)
        self.grad_norm_history = deque(maxlen=history_size)
        self.update_history = deque(maxlen=history_size)

        # Success tracking
        self.success_count = 0
        self.total_count = 0
        self.recent_successes = deque(maxlen=10)  # Last 10 steps

    def update(self, theta, loss, grad_norm):
        """
        Update context with new information

        Call this at the end of each optimization step.

        Args:
            theta: Current parameters
            loss: Current loss value
            grad_norm: L2 norm of gradient
        """
        # Check if improved
        improved = loss < self.current_loss
        self.recent_successes.append(1 if improved else 0)

        if improved:
            self.success_count += 1
        self.total_count += 1

        # Update current state
        self.current_theta = theta.copy() if theta is not None else None
        self.current_loss = loss
        self.current_grad_norm = grad_norm

        # Record history
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

        self.iteration += 1

    @property
    def success_rate(self):
        """Overall success rate (fraction of improving steps)"""
        if self.total_count == 0:
            return 0.5  # Neutral assumption
        return self.success_count / self.total_count

    @property
    def recent_success_rate(self):
        """Recent success rate (last 10 steps)"""
        if len(self.recent_successes) == 0:
            return 0.5
        return sum(self.recent_successes) / len(self.recent_successes)

    @property
    def loss_variance(self):
        """Variance of recent losses (indicates stability)"""
        if len(self.loss_history) < 2:
            return 1.0
        return float(np.var(list(self.loss_history)))

    @property
    def grad_variance(self):
        """Variance of recent gradient norms"""
        if len(self.grad_norm_history) < 2:
            return 1.0
        return float(np.var(list(self.grad_norm_history)))

    @property
    def improvement_rate(self):
        """
        Rate of improvement in recent steps

        Returns:
            Positive value indicates improvement (loss decreasing)
            Negative value indicates deterioration
        """
        if len(self.loss_history) < 2:
            return 0.0

        recent = list(self.loss_history)[-5:]  # Last 5 steps
        if len(recent) < 2:
            return 0.0

        # Avoid division by zero
        if recent[0] < 1e-10:
            return 0.0

        return (recent[0] - recent[-1]) / max(recent[0], 1e-10)

    @property
    def phase(self):
        """
        Automatically determine optimization phase

        Returns:
            'exploration': Early phase, need exploration
            'exploitation': Mid phase, rapid convergence
            'refinement': Late phase, fine-tuning
        """
        # Early phase: explore
        if self.iteration < 15:
            return 'exploration'

        # If improving rapidly: exploit
        if self.improvement_rate > 0.05:
            return 'exploitation'

        # Otherwise: refine
        return 'refinement'

    def to_vector(self):
        """
        Convert context to feature vector

        Useful for learned weight functions.

        Returns:
            np.ndarray: Feature vector of shape (8,)
        """
        return np.array([
            self.current_loss,
            self.current_grad_norm,
            self.recent_success_rate,
            self.loss_variance,
            self.improvement_rate,
            1.0 if self.phase == 'exploration' else 0.0,
            1.0 if self.phase == 'exploitation' else 0.0,
            1.0 if self.phase == 'refinement' else 0.0,
        ])

    def get_summary(self):
        """Get human-readable summary of current context"""
        return {
            'iteration': self.iteration,
            'loss': self.current_loss,
            'grad_norm': self.current_grad_norm,
            'success_rate': self.success_rate,
            'recent_success_rate': self.recent_success_rate,
            'phase': self.phase,
            'improvement_rate': self.improvement_rate,
        }

    def __repr__(self):
        """String representation"""
        return (f"OptimizationContext(iter={self.iteration}, "
                f"loss={self.current_loss:.5f}, "
                f"phase={self.phase})")
