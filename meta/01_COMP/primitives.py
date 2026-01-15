"""
Optimization Primitives

Each primitive provides a different "perspective" on how to update parameters.
The COMP optimizer composes these primitives intelligently based on context.
"""

import numpy as np


class Primitive:
    """
    Base class for all optimization primitives

    Each primitive implements a simple, interpretable update strategy.
    """

    def __init__(self, name):
        """
        Args:
            name: Human-readable name for this primitive
        """
        self.name = name

    def __call__(self, theta, grad, context):
        """
        Compute update direction

        Args:
            theta: Current parameters (np.ndarray)
            grad: Current gradient (np.ndarray)
            context: OptimizationContext object

        Returns:
            np.ndarray: Proposed update (same shape as theta)
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


# ============================================================================
# Exploration Primitives: Focus on finding good regions
# ============================================================================

class GradientDescent(Primitive):
    """
    Standard gradient descent

    Philosophy: Follow the direction of steepest descent
    Use case: Reliable local optimization
    """

    def __init__(self, lr=0.05):
        """
        Args:
            lr: Learning rate (step size)
        """
        super().__init__("GradientDescent")
        self.lr = lr

    def __call__(self, theta, grad, context):
        """Standard gradient descent step"""
        return -self.lr * grad


class StochasticJump(Primitive):
    """
    Random exploration with temperature

    Philosophy: Inject noise to escape local minima
    Use case: Exploration, escaping plateaus
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Scale of random perturbations
        """
        super().__init__("StochasticJump")
        self.temperature = temperature

    def __call__(self, theta, grad, context):
        """
        Random jump with context-dependent temperature

        Temperature is higher during exploration phase.
        """
        # Adapt temperature to phase
        if context.phase == 'exploration':
            effective_temp = self.temperature
        elif context.phase == 'exploitation':
            effective_temp = self.temperature * 0.5
        else:  # refinement
            effective_temp = self.temperature * 0.2

        return np.random.randn(len(theta)) * effective_temp


class Momentum(Primitive):
    """
    Momentum-based updates

    Philosophy: Maintain velocity from past gradients
    Use case: Accelerate convergence, smooth optimization
    """

    def __init__(self, decay=0.9, lr=0.05):
        """
        Args:
            decay: Momentum decay factor (0 to 1)
            lr: Learning rate for gradient component
        """
        super().__init__("Momentum")
        self.decay = decay
        self.lr = lr
        self.velocity = None

    def __call__(self, theta, grad, context):
        """
        Update with momentum

        velocity = decay * velocity - lr * grad
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

        # Update velocity
        self.velocity = self.decay * self.velocity - self.lr * grad

        return self.velocity.copy()


class BestDirection(Primitive):
    """
    Direction toward best parameters seen so far

    Philosophy: Move toward proven good regions
    Use case: Exploitation of known good solutions
    """

    def __init__(self, step_size=0.1):
        """
        Args:
            step_size: Fraction of distance to best to move
        """
        super().__init__("BestDirection")
        self.step_size = step_size
        self.best_theta = None
        self.best_loss = float('inf')

    def __call__(self, theta, grad, context):
        """
        Move toward best parameters

        If current is better, update best.
        """
        # Update best if current is better
        if context.current_loss < self.best_loss:
            self.best_loss = context.current_loss
            self.best_theta = theta.copy()

        # If no best yet, return zero
        if self.best_theta is None:
            return np.zeros_like(theta)

        # Direction toward best
        direction = self.best_theta - theta
        norm = np.linalg.norm(direction)

        if norm > 1e-10:
            # Normalize and scale
            return (direction / norm) * self.step_size
        else:
            return np.zeros_like(theta)


class AdaptiveStep(Primitive):
    """
    Gradient descent with adaptive learning rate

    Philosophy: Increase LR when succeeding, decrease when failing
    Use case: Automatic step size tuning
    """

    def __init__(self, base_lr=0.05, lr_mult_up=1.5, lr_mult_down=0.5):
        """
        Args:
            base_lr: Initial learning rate
            lr_mult_up: Multiplier when success rate is high
            lr_mult_down: Multiplier when success rate is low
        """
        super().__init__("AdaptiveStep")
        self.base_lr = base_lr
        self.lr_mult_up = lr_mult_up
        self.lr_mult_down = lr_mult_down

    def __call__(self, theta, grad, context):
        """
        Gradient descent with success-based LR adaptation

        High success rate → increase LR
        Low success rate → decrease LR
        """
        success_rate = context.recent_success_rate

        if success_rate > 0.7:
            lr = self.base_lr * self.lr_mult_up
        elif success_rate < 0.3:
            lr = self.base_lr * self.lr_mult_down
        else:
            lr = self.base_lr

        return -lr * grad


# ============================================================================
# Primitive Registry
# ============================================================================

def get_default_primitives():
    """
    Get default set of 5 primitives

    Returns:
        List of Primitive objects
    """
    return [
        GradientDescent(lr=0.05),
        StochasticJump(temperature=0.1),
        Momentum(decay=0.9, lr=0.05),
        BestDirection(step_size=0.1),
        AdaptiveStep(base_lr=0.05),
    ]


def get_primitive_names(primitives):
    """
    Get names of primitives

    Args:
        primitives: List of Primitive objects

    Returns:
        List of strings (primitive names)
    """
    return [p.name for p in primitives]
