"""
Weight Functions: Context → Primitive Weights

These functions determine how much each primitive should contribute
based on the current optimization context.
"""

import numpy as np


def rule_based_weights(context, n_primitives=5):
    """
    Rule-based weight assignment

    Uses heuristic rules based on optimization phase, success rate,
    and gradient norm to assign weights to primitives.

    Primitive ordering (default):
        0: GradientDescent
        1: StochasticJump
        2: Momentum
        3: BestDirection
        4: AdaptiveStep

    Args:
        context: OptimizationContext object
        n_primitives: Number of primitives (default: 5)

    Returns:
        np.ndarray: Weights summing to 1.0
    """
    weights = np.zeros(n_primitives)

    # ========================================================================
    # Phase-based base weights
    # ========================================================================

    if context.phase == 'exploration':
        # Early phase: Need strong exploration
        weights[0] = 0.25  # Gradient (moderate)
        weights[1] = 0.35  # Stochastic (high - explore!)
        weights[2] = 0.20  # Momentum (moderate)
        weights[3] = 0.10  # Best (low - don't have good best yet)
        weights[4] = 0.10  # Adaptive (low)

    elif context.phase == 'exploitation':
        # Mid phase: Rapid convergence
        weights[0] = 0.30  # Gradient (increase)
        weights[1] = 0.15  # Stochastic (decrease - less exploration)
        weights[2] = 0.30  # Momentum (increase - accelerate)
        weights[3] = 0.15  # Best (increase - have good best now)
        weights[4] = 0.10  # Adaptive (moderate)

    else:  # refinement
        # Late phase: Fine-tuning
        weights[0] = 0.25  # Gradient (moderate)
        weights[1] = 0.05  # Stochastic (minimal - avoid disruption)
        weights[2] = 0.25  # Momentum (moderate)
        weights[3] = 0.20  # Best (higher - stick to good regions)
        weights[4] = 0.25  # Adaptive (higher - tune step size)

    # ========================================================================
    # Success rate adjustments
    # ========================================================================

    success_rate = context.recent_success_rate

    if success_rate < 0.3:
        # Failing frequently → need more exploration
        weights[1] *= 2.0  # Stochastic (increase exploration)
        weights[0] *= 0.7  # Gradient (decrease - not working well)
        weights[3] *= 0.8  # Best (decrease - current best may be bad)

    elif success_rate > 0.8:
        # Succeeding consistently → exploit more
        weights[0] *= 1.2  # Gradient (increase - working well)
        weights[2] *= 1.2  # Momentum (increase - keep going)
        weights[1] *= 0.5  # Stochastic (decrease - don't need noise)

    # ========================================================================
    # Gradient norm adjustments
    # ========================================================================

    grad_norm = context.current_grad_norm

    if grad_norm < 0.01:
        # Very small gradient → might be stuck
        weights[1] *= 1.5  # Stochastic (increase - escape!)
        weights[3] *= 1.3  # Best (increase - jump to known good)
        weights[0] *= 0.7  # Gradient (decrease - too small anyway)

    elif grad_norm > 1.0:
        # Very large gradient → might overshoot
        weights[0] *= 0.8  # Gradient (decrease - avoid overshoot)
        weights[4] *= 1.3  # Adaptive (increase - need careful step)

    # ========================================================================
    # Loss variance adjustments
    # ========================================================================

    loss_var = context.loss_variance

    if loss_var > 0.1:
        # High variance → unstable
        weights[1] *= 0.7  # Stochastic (decrease - don't add noise)
        weights[2] *= 1.2  # Momentum (increase - smooth out)
        weights[3] *= 1.2  # Best (increase - stick to safe regions)

    # ========================================================================
    # Normalize
    # ========================================================================

    weights = np.maximum(weights, 0.0)  # Ensure non-negative
    weight_sum = weights.sum()

    if weight_sum > 1e-10:
        weights = weights / weight_sum
    else:
        # Fallback to uniform if all weights are zero
        weights = np.ones(n_primitives) / n_primitives

    return weights


def uniform_weights(context, n_primitives=5):
    """
    Uniform weights (baseline)

    All primitives contribute equally.
    Useful for debugging and comparison.

    Args:
        context: OptimizationContext object (unused)
        n_primitives: Number of primitives

    Returns:
        np.ndarray: Uniform weights
    """
    return np.ones(n_primitives) / n_primitives


def phase_only_weights(context, n_primitives=5):
    """
    Weights based only on phase (simpler version)

    Ignores success rate, gradient norm, etc.
    Useful for ablation studies.

    Args:
        context: OptimizationContext object
        n_primitives: Number of primitives

    Returns:
        np.ndarray: Phase-based weights
    """
    weights = np.zeros(n_primitives)

    if context.phase == 'exploration':
        weights[0] = 0.25
        weights[1] = 0.35
        weights[2] = 0.20
        weights[3] = 0.10
        weights[4] = 0.10

    elif context.phase == 'exploitation':
        weights[0] = 0.30
        weights[1] = 0.15
        weights[2] = 0.30
        weights[3] = 0.15
        weights[4] = 0.10

    else:  # refinement
        weights[0] = 0.25
        weights[1] = 0.05
        weights[2] = 0.25
        weights[3] = 0.20
        weights[4] = 0.25

    return weights / weights.sum()


# ============================================================================
# Weight Function Registry
# ============================================================================

WEIGHT_FUNCTIONS = {
    'rule_based': rule_based_weights,
    'uniform': uniform_weights,
    'phase_only': phase_only_weights,
}


def get_weight_function(name='rule_based'):
    """
    Get weight function by name

    Args:
        name: Name of weight function

    Returns:
        Function: weight_fn(context, n_primitives) → weights

    Raises:
        ValueError: If name is not recognized
    """
    if name not in WEIGHT_FUNCTIONS:
        raise ValueError(f"Unknown weight function: {name}. "
                         f"Available: {list(WEIGHT_FUNCTIONS.keys())}")

    return WEIGHT_FUNCTIONS[name]
