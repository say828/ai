"""
MetaController for GENESIS v2.0
Author: GENESIS Project
Date: 2026-01-03
Version: 2.0

Purpose:
    High-level decision making for architecture evolution.

    Key innovations over v1.1:
    - Task similarity-based module sharing
    - Performance-based module addition
    - Automatic specialization vs sharing decisions
    - Metamorphosis triggers based on data

Components:
    1. PerformanceHistory: Tracks task-specific metrics
    2. SharingPolicy: Decides when to share vs specialize
    3. MetaController: High-level coordinator
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class PerformanceHistory:
    """
    Tracks performance metrics for a single task.

    Metrics:
        - Error history (for plateau detection)
        - Success rate (for viability)
        - Growth trend (for learning progress)
    """

    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Size of sliding window for statistics
        """
        self.window_size = window_size

        # Core metrics
        self.errors: deque = deque(maxlen=window_size)
        self.successes: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

        # Aggregated statistics
        self.total_steps = 0
        self.total_success = 0

    def record(self, error: float, success: bool) -> None:
        """
        Record a new observation.

        Args:
            error: Prediction error
            success: Whether the prediction was successful
        """
        self.errors.append(float(error))
        self.successes.append(int(success))
        self.timestamps.append(self.total_steps)

        self.total_steps += 1
        if success:
            self.total_success += 1

    def get_recent_errors(self, n: Optional[int] = None) -> np.ndarray:
        """Get recent n errors (default: all in window)."""
        if n is None:
            return np.array(list(self.errors))
        else:
            return np.array(list(self.errors)[-n:])

    def get_success_rate(self, n: Optional[int] = None) -> float:
        """
        Compute success rate over recent n observations.

        Args:
            n: Number of recent observations (default: entire window)

        Returns:
            success_rate: Fraction of successful predictions
        """
        if len(self.successes) == 0:
            return 0.0

        if n is None:
            recent_successes = list(self.successes)
        else:
            recent_successes = list(self.successes)[-n:]

        return float(np.mean(recent_successes))

    def is_plateaued(self, window: int = 20, threshold: float = 0.05) -> bool:
        """
        Detect if performance has plateaued.

        Method:
            Compare recent errors to older errors in window.
            If improvement < threshold, consider plateaued.

        Args:
            window: Size of comparison window
            threshold: Minimum improvement to not be plateaued

        Returns:
            is_plateaued: True if performance has stagnated
        """
        if len(self.errors) < window * 2:
            return False  # Not enough data

        # Split into two halves
        older_errors = list(self.errors)[:window]
        recent_errors = list(self.errors)[-window:]

        older_mean = np.mean(older_errors)
        recent_mean = np.mean(recent_errors)

        # Improvement = reduction in error
        improvement = (older_mean - recent_mean) / (older_mean + 1e-8)

        return improvement < threshold

    def compute_growth_trend(self, window: int = 20) -> float:
        """
        Compute learning trend (negative = improving, positive = degrading).

        Method:
            Linear regression slope on recent errors.

        Args:
            window: Size of trend window

        Returns:
            trend: Slope of error over time (normalized)
        """
        if len(self.errors) < window:
            return 0.0

        recent_errors = np.array(list(self.errors)[-window:])
        x = np.arange(len(recent_errors))

        # Linear regression: y = mx + b
        # Slope m = cov(x,y) / var(x)
        mean_x = np.mean(x)
        mean_y = np.mean(recent_errors)

        cov_xy = np.mean((x - mean_x) * (recent_errors - mean_y))
        var_x = np.mean((x - mean_x) ** 2)

        slope = cov_xy / (var_x + 1e-8)

        # Normalize by mean error
        normalized_slope = slope / (mean_y + 1e-8)

        return float(normalized_slope)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_steps': self.total_steps,
            'recent_error_mean': float(np.mean(self.errors)) if len(self.errors) > 0 else 0.0,
            'recent_error_std': float(np.std(self.errors)) if len(self.errors) > 0 else 0.0,
            'success_rate': self.get_success_rate(),
            'is_plateaued': self.is_plateaued(),
            'growth_trend': self.compute_growth_trend()
        }


class SharingPolicy:
    """
    Determines when tasks should share modules vs specialize.

    Strategy:
        - High similarity → share modules (positive transfer)
        - Low similarity → specialize (prevent interference)
    """

    def __init__(self, specialization_threshold: float = 0.7):
        """
        Args:
            specialization_threshold: Similarity threshold for sharing
                (> threshold: share, < threshold: specialize)
        """
        self.specialization_threshold = specialization_threshold

    def decide(self, task_similarity: float) -> str:
        """
        Decide sharing policy for two tasks.

        Args:
            task_similarity: Cosine similarity between task embeddings (0-1)

        Returns:
            policy: 'share' or 'specialize'
        """
        if task_similarity >= self.specialization_threshold:
            return 'share'
        else:
            return 'specialize'

    def recommend_modules(self,
                          task_similarity: float,
                          source_modules: List[str],
                          default_modules: List[str]) -> List[str]:
        """
        Recommend module set for a new task based on similarity.

        Args:
            task_similarity: Similarity to most similar existing task
            source_modules: Module set of the similar task
            default_modules: Default module set

        Returns:
            recommended_modules: Module set for the new task
        """
        policy = self.decide(task_similarity)

        if policy == 'share':
            # Share modules from similar task
            return source_modules.copy()
        else:
            # Use default modules (independent start)
            return default_modules.copy()


class MetaController:
    """
    High-level coordinator for architecture evolution.

    Responsibilities:
        1. Track task performance over time
        2. Decide when to add new modules
        3. Manage sharing vs specialization
        4. Trigger metamorphosis when needed

    Design Philosophy:
        - Data-driven decisions (not hand-crafted heuristics)
        - Conservative evolution (don't change unnecessarily)
        - Prioritize existing modules before adding new ones
    """

    def __init__(self,
                 specialization_threshold: float = 0.7,
                 plateau_threshold: float = 0.05,
                 default_modules: Optional[List[str]] = None):
        """
        Args:
            specialization_threshold: Similarity threshold for sharing
            plateau_threshold: Performance improvement threshold
            default_modules: Default module set for new tasks
        """
        # Task tracking
        self.task_memory: Dict[str, PerformanceHistory] = {}
        self.task_similarities: Dict[Tuple[str, str], float] = {}

        # Module tracking
        self.module_usage: Dict[str, int] = {}  # module_id → usage count
        self.module_performance: Dict[str, List[float]] = {}  # module_id → error history

        # Policies
        self.sharing_policy = SharingPolicy(specialization_threshold)
        self.plateau_threshold = plateau_threshold
        self.default_modules = default_modules or ['linear', 'nonlinear', 'interaction']

        # Evolution counters
        self.total_decisions = 0
        self.metamorphosis_count = 0

    def register_task(self, task_id: str) -> None:
        """
        Register a new task for tracking.

        Args:
            task_id: Unique task identifier
        """
        if task_id not in self.task_memory:
            self.task_memory[task_id] = PerformanceHistory()
            print(f"MetaController: Registered task '{task_id}'")

    def record_performance(self, task_id: str, error: float, success: bool) -> None:
        """
        Record performance for a task.

        Args:
            task_id: Task identifier
            error: Prediction error
            success: Whether prediction was successful
        """
        if task_id not in self.task_memory:
            self.register_task(task_id)

        self.task_memory[task_id].record(error, success)

    def update_task_similarity(self, task_i: str, task_j: str, similarity: float) -> None:
        """
        Update similarity between two tasks.

        Args:
            task_i: First task ID
            task_j: Second task ID
            similarity: Cosine similarity (0-1)
        """
        key = tuple(sorted([task_i, task_j]))
        self.task_similarities[key] = similarity

    def handle_new_task(self,
                       task_id: str,
                       task_similarities: Dict[str, float]) -> Dict:
        """
        Decide how to initialize a new task.

        Decision logic:
            1. Find most similar existing task
            2. If similarity > threshold, share modules
            3. Otherwise, create independent task head

        Args:
            task_id: New task identifier
            task_similarities: {existing_task_id: similarity}

        Returns:
            decision: {
                'action': 'share' | 'create_new',
                'from': Optional[str],  # source task for sharing
                'modules': List[str]
            }
        """
        self.total_decisions += 1
        self.register_task(task_id)

        # Find most similar task
        if len(task_similarities) == 0:
            # First task ever
            return {
                'action': 'create_new',
                'from': None,
                'modules': self.default_modules.copy()
            }

        most_similar_task = max(task_similarities, key=task_similarities.get)
        max_similarity = task_similarities[most_similar_task]

        # Update similarity matrix
        self.update_task_similarity(task_id, most_similar_task, max_similarity)

        # Decide: share or create new?
        policy = self.sharing_policy.decide(max_similarity)

        if policy == 'share':
            # Share modules from similar task
            print(f"MetaController: Task '{task_id}' similar to '{most_similar_task}' "
                  f"(sim={max_similarity:.3f}) → sharing modules")

            return {
                'action': 'share',
                'from': most_similar_task,
                'modules': None  # Will inherit from source task
            }
        else:
            # Create independent task
            print(f"MetaController: Task '{task_id}' different from all tasks "
                  f"(max_sim={max_similarity:.3f}) → creating new head")

            return {
                'action': 'create_new',
                'from': None,
                'modules': self.default_modules.copy()
            }

    def should_add_module(self, task_id: str) -> Dict:
        """
        Decide whether to add a new functional module.

        Conditions for adding module:
            1. Performance has plateaued
            2. Task has sufficient observations
            3. Not too many modules already

        Args:
            task_id: Task to evaluate

        Returns:
            decision: {
                'add': bool,
                'type': Optional[str],  # module type to add
                'reason': str
            }
        """
        if task_id not in self.task_memory:
            return {'add': False, 'type': None, 'reason': 'Task not registered'}

        history = self.task_memory[task_id]

        # Condition 1: Enough data?
        if history.total_steps < 100:
            return {'add': False, 'type': None, 'reason': 'Insufficient data'}

        # Condition 2: Plateaued?
        if not history.is_plateaued(window=20, threshold=self.plateau_threshold):
            return {'add': False, 'type': None, 'reason': 'Still improving'}

        # Condition 3: Diagnose missing capability
        missing_type = self._diagnose_missing_capability(history)

        if missing_type is None:
            return {'add': False, 'type': None, 'reason': 'No clear deficiency'}

        print(f"MetaController: Task '{task_id}' plateaued → adding '{missing_type}' module")

        return {
            'add': True,
            'type': missing_type,
            'reason': f'Performance plateaued, missing {missing_type} capability'
        }

    def _diagnose_missing_capability(self, history: PerformanceHistory) -> Optional[str]:
        """
        Diagnose what type of module might help.

        Heuristic:
            - High error variance → need nonlinear module
            - Steady high error → need interaction module
            - Otherwise → unknown

        Args:
            history: Performance history

        Returns:
            module_type: Suggested module type or None
        """
        recent_errors = history.get_recent_errors(n=20)

        if len(recent_errors) < 20:
            return None

        error_mean = np.mean(recent_errors)
        error_std = np.std(recent_errors)

        # High variance → nonlinear patterns
        if error_std / (error_mean + 1e-8) > 0.5:
            return 'nonlinear'

        # Steady high error → need interactions
        if error_mean > 1.0 and error_std < 0.3:
            return 'interaction'

        return None  # No clear deficiency

    def should_trigger_metamorphosis(self, task_id: str) -> Dict:
        """
        Decide whether to trigger metamorphosis (structural change).

        Metamorphosis triggers:
            1. Critical failure (viability < 0.2 for extended period)
            2. Prolonged stagnation (plateaued + poor performance)
            3. Strategic exploration (random, low probability)

        Args:
            task_id: Task to evaluate

        Returns:
            decision: {
                'trigger': bool,
                'reason': str,
                'action': Optional[str]  # 'add_module' | 'reset' | 'explore'
            }
        """
        if task_id not in self.task_memory:
            return {'trigger': False, 'reason': 'Task not registered', 'action': None}

        history = self.task_memory[task_id]

        # Trigger 1: Critical failure
        recent_success_rate = history.get_success_rate(n=50)
        if recent_success_rate < 0.2 and history.total_steps > 100:
            self.metamorphosis_count += 1
            print(f"MetaController: Task '{task_id}' critical failure → metamorphosis")
            return {
                'trigger': True,
                'reason': f'Critical failure (success rate={recent_success_rate:.2f})',
                'action': 'add_module'
            }

        # Trigger 2: Prolonged stagnation
        if history.is_plateaued(window=50, threshold=0.01):
            recent_error = np.mean(history.get_recent_errors(n=20))
            if recent_error > 1.0:
                self.metamorphosis_count += 1
                print(f"MetaController: Task '{task_id}' prolonged stagnation → metamorphosis")
                return {
                    'trigger': True,
                    'reason': f'Prolonged stagnation (error={recent_error:.3f})',
                    'action': 'add_module'
                }

        # Trigger 3: Strategic exploration (5% chance after 200 steps)
        if history.total_steps > 200 and np.random.rand() < 0.05:
            self.metamorphosis_count += 1
            print(f"MetaController: Task '{task_id}' strategic exploration → metamorphosis")
            return {
                'trigger': True,
                'reason': 'Strategic exploration',
                'action': 'explore'
            }

        return {'trigger': False, 'reason': 'No trigger conditions met', 'action': None}

    def get_task_summary(self, task_id: str) -> Dict:
        """
        Get comprehensive summary for a task.

        Args:
            task_id: Task identifier

        Returns:
            summary: Full performance statistics
        """
        if task_id not in self.task_memory:
            return {'error': 'Task not registered'}

        history = self.task_memory[task_id]
        return history.get_summary()

    def get_global_summary(self) -> Dict:
        """
        Get summary of all tasks and meta-controller state.

        Returns:
            summary: {
                'n_tasks': int,
                'total_decisions': int,
                'metamorphosis_count': int,
                'tasks': {task_id: summary}
            }
        """
        summary = {
            'n_tasks': len(self.task_memory),
            'total_decisions': self.total_decisions,
            'metamorphosis_count': self.metamorphosis_count,
            'tasks': {}
        }

        for task_id in self.task_memory.keys():
            summary['tasks'][task_id] = self.get_task_summary(task_id)

        return summary


# =======================
# Testing and Validation
# =======================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MetaController v2.0")
    print("=" * 60)

    # Initialize controller
    controller = MetaController(
        specialization_threshold=0.7,
        plateau_threshold=0.05
    )

    # Test 1: New task handling (first task)
    print("\n[Test 1] First task - should create new")
    decision_1 = controller.handle_new_task('task_1', {})
    print(f"Decision: {decision_1}")
    assert decision_1['action'] == 'create_new'

    # Test 2: Similar task (should share)
    print("\n[Test 2] Similar task - should share")
    decision_2 = controller.handle_new_task('task_2', {'task_1': 0.85})
    print(f"Decision: {decision_2}")
    assert decision_2['action'] == 'share'
    assert decision_2['from'] == 'task_1'

    # Test 3: Different task (should create new)
    print("\n[Test 3] Different task - should create new")
    decision_3 = controller.handle_new_task('task_3', {'task_1': 0.3, 'task_2': 0.25})
    print(f"Decision: {decision_3}")
    assert decision_3['action'] == 'create_new'

    # Test 4: Performance tracking
    print("\n[Test 4] Performance tracking")
    for i in range(50):
        error = 5.0 * np.exp(-i / 10.0)  # Exponential decay
        success = (error < 1.0)
        controller.record_performance('task_1', error, success)

    summary = controller.get_task_summary('task_1')
    print(f"Task 1 summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Recent error: {summary['recent_error_mean']:.3f}")
    print(f"  Success rate: {summary['success_rate']:.3f}")
    print(f"  Growth trend: {summary['growth_trend']:.3f}")
    print(f"  Is plateaued: {summary['is_plateaued']}")

    # Test 5: Plateau detection
    print("\n[Test 5] Plateau detection")
    # Add 50 more steps with constant error (plateau)
    for i in range(50):
        controller.record_performance('task_1', 1.0, True)

    summary = controller.get_task_summary('task_1')
    print(f"After plateau:")
    print(f"  Is plateaued: {summary['is_plateaued']}")
    assert summary['is_plateaued'] == True

    # Test 6: Module addition decision
    print("\n[Test 6] Module addition decision")
    module_decision = controller.should_add_module('task_1')
    print(f"Module decision: {module_decision}")

    # Test 7: Metamorphosis trigger (critical failure)
    print("\n[Test 7] Metamorphosis trigger - critical failure")
    controller.register_task('failing_task')
    for i in range(101):  # Need > 100 steps
        controller.record_performance('failing_task', 10.0, False)  # Constant failure

    metamorphosis = controller.should_trigger_metamorphosis('failing_task')
    print(f"Metamorphosis decision: {metamorphosis}")
    assert metamorphosis['trigger'] == True
    assert 'Critical failure' in metamorphosis['reason']

    # Test 8: Global summary
    print("\n[Test 8] Global summary")
    global_summary = controller.get_global_summary()
    print(f"Global summary:")
    print(f"  Total tasks: {global_summary['n_tasks']}")
    print(f"  Total decisions: {global_summary['total_decisions']}")
    print(f"  Metamorphosis count: {global_summary['metamorphosis_count']}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    print("\n[Summary]")
    print("- PerformanceHistory tracks task metrics over time")
    print("- SharingPolicy decides when to share vs specialize")
    print("- MetaController handles:")
    print("  • New task initialization (share or create)")
    print("  • Module addition (plateau detection)")
    print("  • Metamorphosis triggers (failure, stagnation)")
    print("- All decisions are data-driven (no hand-crafted rules)")
