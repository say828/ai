"""
ULTIMATE: Context Computation
상황 인식을 위한 context vector 계산
"""

import numpy as np


class OptimizationContext:
    """
    Tracks optimization state and computes context vector
    """
    def __init__(self, max_iterations=200, history_size=20):
        self.max_iterations = max_iterations
        self.history_size = history_size

        # Current state
        self.iteration = 0
        self.current_loss = float('inf')
        self.current_grad_norm = 0.0

        # History
        self.loss_history = []
        self.grad_history = []
        self.improvement_history = []

        # Best tracking
        self.best_loss = float('inf')
        self.iterations_since_improvement = 0

        # Action tracking (LAML-Q style)
        self.action_history = []

    def update(self, loss, grad, improvement=0.0, action=0.0):
        """Update context with new information"""
        self.iteration += 1
        self.current_loss = loss
        self.current_grad_norm = np.linalg.norm(grad)

        # Update histories
        self.loss_history.append(loss)
        self.grad_history.append(self.current_grad_norm)
        self.improvement_history.append(improvement)
        self.action_history.append(action)

        # Keep only recent history
        if len(self.loss_history) > self.history_size:
            self.loss_history = self.loss_history[-self.history_size:]
            self.grad_history = self.grad_history[-self.history_size:]
            self.improvement_history = self.improvement_history[-self.history_size:]
            self.action_history = self.action_history[-self.history_size:]

        # Update best
        if loss < self.best_loss:
            self.best_loss = loss
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

    def get_context_vector(self, network, X, y):
        """
        Compute 12-dimensional context vector

        Returns:
            context: numpy array of shape (12,)
        """
        context = np.zeros(12)

        # 1. Current loss (normalized, clipped)
        context[0] = np.clip(np.log(max(self.current_loss, 1e-10)), -10, 10)

        # 2. Gradient norm (normalized, clipped)
        context[1] = np.clip(np.log(max(self.current_grad_norm, 1e-10)), -10, 10)

        # 3. Loss variance (stability, clipped)
        if len(self.loss_history) >= 2:
            loss_var = np.var(self.loss_history)
            context[2] = np.clip(loss_var, 0, 100)
        else:
            context[2] = 0.0

        # 4. Progress (iteration / max_iterations)
        context[3] = self.iteration / max(self.max_iterations, 1)

        # 5. Improvement rate (recent, clipped)
        if len(self.improvement_history) >= 5:
            imp_rate = np.mean(self.improvement_history[-5:])
            context[4] = np.clip(imp_rate, -10, 10) if not np.isnan(imp_rate) else 0.0
        else:
            context[4] = 0.0

        # 6. Success rate (positive improvements)
        if len(self.improvement_history) >= 5:
            successes = sum(1 for imp in self.improvement_history[-5:] if imp > 0 and not np.isnan(imp))
            context[5] = successes / 5.0
        else:
            context[5] = 0.5

        # 7. Landscape smoothness (approximate via grad variance, clipped)
        if len(self.grad_history) >= 2:
            grad_variance = np.var(self.grad_history)
            context[6] = np.clip(np.log(max(grad_variance, 1e-10)), -10, 10)
        else:
            context[6] = 0.0

        # 8. Problem dimensionality (log scale)
        theta = network.get_weights()
        context[7] = np.clip(np.log(len(theta) + 1), 0, 10)

        # 9. Data complexity (approximate via loss scale, clipped)
        if len(self.loss_history) >= 5:
            mean_loss = np.mean(self.loss_history[-5:])
            context[8] = np.clip(np.log(max(mean_loss, 1e-10)), -10, 10)
        else:
            context[8] = 0.0

        # 10. Best loss so far (clipped)
        context[9] = np.clip(np.log(max(self.best_loss, 1e-10)), -10, 10)

        # 11. Iterations since improvement (normalized, clipped)
        context[10] = np.clip(self.iterations_since_improvement / 20.0, 0, 5)

        # 12. Average action (efficiency, clipped)
        if len(self.action_history) >= 5:
            avg_action = np.mean(self.action_history[-5:])
            context[11] = np.clip(avg_action, -10, 10) if not np.isnan(avg_action) else 0.0
        else:
            context[11] = 0.0

        # Final check: replace any NaN with 0
        context = np.nan_to_num(context, nan=0.0, posinf=10.0, neginf=-10.0)

        return context

    @property
    def progress(self):
        """Return current progress [0, 1]"""
        return self.iteration / self.max_iterations

    @property
    def improvement_rate(self):
        """Return recent improvement rate"""
        if len(self.improvement_history) >= 5:
            return np.mean(self.improvement_history[-5:])
        return 0.0

    @property
    def success_rate(self):
        """Return success rate (fraction of improving steps)"""
        if len(self.improvement_history) >= 5:
            successes = sum(1 for imp in self.improvement_history[-5:] if imp > 0)
            return successes / 5.0
        return 0.5
