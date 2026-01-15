"""
Task Router for GENESIS v2.0
Author: GENESIS Project
Date: 2026-01-03
Version: 2.0

Purpose:
    Identify tasks and route to appropriate functional modules.

    Key innovations over v1.1:
    - Task identification via embedding similarity
    - Dynamic module selection per task
    - Automatic new task registration

Components:
    1. TaskDetector: Identifies tasks from input features
    2. ModuleSelector: Selects appropriate modules for each task
    3. TaskRouter: Coordinates detection and routing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import uuid


class TaskDetector:
    """
    Identifies tasks based on feature embedding similarity.

    Method:
        - Each task has an embedding vector (averaged features)
        - New inputs are compared to all known task embeddings
        - If similarity > threshold, classify as known task
        - Otherwise, register as new task
    """

    def __init__(self, feature_size: int = 32, threshold: float = 0.8):
        """
        Args:
            feature_size: Dimensionality of feature vectors
            threshold: Cosine similarity threshold for task recognition
        """
        self.feature_size = feature_size
        self.threshold = threshold

        # Task embeddings: task_id → embedding vector
        self.task_embeddings: Dict[str, np.ndarray] = {}

        # Task statistics for online embedding updates
        self.task_counts: Dict[str, int] = {}  # task_id → observation count

        # History for debugging
        self.detection_history: List[Dict] = []

    def identify_task(self, features: np.ndarray, context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Identify task from input features.

        Args:
            features: Feature vector from shared encoder (shape: [feature_size] or [1, feature_size])
            context: Optional context information (e.g., environment metadata)

        Returns:
            (task_id, confidence): Task identifier and confidence score
                - If known task: (existing_task_id, similarity)
                - If new task: ('new_task', 0.0)
        """
        # Flatten to 1D if needed
        features = features.flatten()

        # Normalize features for cosine similarity
        features_norm = features / (np.linalg.norm(features) + 1e-8)

        # If no tasks registered yet, this is a new task
        if len(self.task_embeddings) == 0:
            self._log_detection('new_task', 0.0, features)
            return 'new_task', 0.0

        # Compute similarity to all known tasks
        similarities = {}
        for task_id, embedding in self.task_embeddings.items():
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            similarity = np.dot(features_norm, embedding_norm)
            similarities[task_id] = float(similarity)

        # Find most similar task
        best_task_id = max(similarities, key=similarities.get)
        best_similarity = similarities[best_task_id]

        # Decision: known task or new task?
        if best_similarity >= self.threshold:
            self._log_detection(best_task_id, best_similarity, features)
            return best_task_id, best_similarity
        else:
            self._log_detection('new_task', best_similarity, features)
            return 'new_task', 0.0

    def register_task(self, task_id: str, initial_features: np.ndarray) -> None:
        """
        Register a new task with initial feature embedding.

        Args:
            task_id: Unique identifier for the task
            initial_features: Initial feature vector
        """
        if task_id in self.task_embeddings:
            print(f"Warning: Task {task_id} already exists. Skipping registration.")
            return

        # Initialize embedding as the first observed features (flatten)
        self.task_embeddings[task_id] = initial_features.flatten().copy()
        self.task_counts[task_id] = 1

        print(f"TaskDetector: Registered new task '{task_id}'")

    def update_embedding(self, task_id: str, features: np.ndarray) -> None:
        """
        Update task embedding with new observation (online averaging).

        Args:
            task_id: Task to update
            features: New feature observation
        """
        if task_id not in self.task_embeddings:
            print(f"Warning: Task {task_id} not registered. Cannot update.")
            return

        # Flatten features
        features = features.flatten()

        # Online mean update: new_mean = old_mean + (x - old_mean) / (n + 1)
        count = self.task_counts[task_id]
        old_embedding = self.task_embeddings[task_id]
        new_embedding = old_embedding + (features - old_embedding) / (count + 1)

        self.task_embeddings[task_id] = new_embedding
        self.task_counts[task_id] = count + 1

    def get_task_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix between all known tasks.

        Returns:
            similarity_matrix: [n_tasks, n_tasks] cosine similarity matrix
        """
        task_ids = list(self.task_embeddings.keys())
        n_tasks = len(task_ids)

        if n_tasks == 0:
            return np.array([])

        # Normalize all embeddings
        embeddings = np.array([self.task_embeddings[tid] for tid in task_ids])
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

        return similarity_matrix

    def _log_detection(self, task_id: str, confidence: float, features: np.ndarray) -> None:
        """Log detection event for debugging."""
        self.detection_history.append({
            'task_id': task_id,
            'confidence': confidence,
            'features_norm': float(np.linalg.norm(features)),
            'timestamp': len(self.detection_history)
        })


class ModuleSelector:
    """
    Selects appropriate functional modules for each task.

    Strategy:
        - Each task has a preferred set of modules
        - Initially, all modules are active (exploration)
        - Over time, prune ineffective modules (specialization)
    """

    def __init__(self, default_modules: Optional[List[str]] = None):
        """
        Args:
            default_modules: Default module set for new tasks
        """
        self.default_modules = default_modules or ['linear', 'nonlinear', 'interaction']

        # Task-module mapping: task_id → [module_ids]
        self.task_module_map: Dict[str, List[str]] = {}

        # Module usage statistics
        self.module_usage: Dict[Tuple[str, str], int] = {}  # (task_id, module_id) → count

    def select_modules(self, task_id: str, all_modules: List[str]) -> List[str]:
        """
        Select active modules for a task.

        Args:
            task_id: Task identifier
            all_modules: All available module names

        Returns:
            active_modules: List of module names to activate
        """
        # Known task: use cached module set
        if task_id in self.task_module_map:
            cached_modules = self.task_module_map[task_id]
            # Filter to only include modules that still exist
            active_modules = [m for m in cached_modules if m in all_modules]
            return active_modules

        # Unknown task: use default modules
        active_modules = [m for m in self.default_modules if m in all_modules]
        return active_modules

    def register_task(self, task_id: str, initial_modules: Optional[List[str]] = None) -> None:
        """
        Register a new task with initial module set.

        Args:
            task_id: Task identifier
            initial_modules: Initial module set (if None, use default)
        """
        if task_id in self.task_module_map:
            print(f"Warning: Task {task_id} already has module mapping. Skipping.")
            return

        modules = initial_modules if initial_modules is not None else self.default_modules
        self.task_module_map[task_id] = modules.copy()

        # Initialize usage counters
        for module_id in modules:
            self.module_usage[(task_id, module_id)] = 0

        print(f"ModuleSelector: Registered task '{task_id}' with modules {modules}")

    def update_module_set(self, task_id: str, new_modules: List[str]) -> None:
        """
        Update the module set for a task (manual override).

        Args:
            task_id: Task to update
            new_modules: New module set
        """
        if task_id not in self.task_module_map:
            print(f"Warning: Task {task_id} not registered. Use register_task() first.")
            return

        old_modules = self.task_module_map[task_id]
        self.task_module_map[task_id] = new_modules.copy()

        print(f"ModuleSelector: Updated {task_id} modules: {old_modules} → {new_modules}")

    def record_usage(self, task_id: str, modules: List[str]) -> None:
        """
        Record module usage for statistics.

        Args:
            task_id: Task that used the modules
            modules: Modules that were activated
        """
        for module_id in modules:
            key = (task_id, module_id)
            self.module_usage[key] = self.module_usage.get(key, 0) + 1

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get module usage statistics per task.

        Returns:
            stats: {task_id: {module_id: usage_count}}
        """
        stats = {}
        for (task_id, module_id), count in self.module_usage.items():
            if task_id not in stats:
                stats[task_id] = {}
            stats[task_id][module_id] = count
        return stats


class TaskRouter:
    """
    High-level coordinator for task detection and module routing.

    Workflow:
        1. Receive input features from shared encoder
        2. Detect task ID (or register new task)
        3. Select appropriate modules for the task
        4. Return routing decision
    """

    def __init__(self,
                 feature_size: int = 32,
                 detection_threshold: float = 0.8,
                 default_modules: Optional[List[str]] = None):
        """
        Args:
            feature_size: Dimensionality of encoder features
            detection_threshold: Task similarity threshold
            default_modules: Default module set for new tasks
        """
        self.task_detector = TaskDetector(feature_size, detection_threshold)
        self.module_selector = ModuleSelector(default_modules)

        # Auto-generate task IDs for new tasks
        self.task_counter = 0

    def route(self, features: np.ndarray, all_modules: List[str],
              context: Optional[Dict] = None) -> Dict:
        """
        Main routing method: detect task and select modules.

        Args:
            features: Input features from shared encoder
            all_modules: All available module names
            context: Optional context (e.g., user-provided task_id)

        Returns:
            routing_decision: {
                'task_id': str,
                'is_new_task': bool,
                'active_modules': List[str],
                'confidence': float
            }
        """
        # Step 1: Check if explicit task_name in context (overrides detection)
        if context and 'task_name' in context:
            explicit_task_id = context['task_name']

            # Check if task exists
            is_new_task = explicit_task_id not in self.task_detector.task_embeddings

            if is_new_task:
                detected_task = 'new_task'
                confidence = 0.0
            else:
                detected_task = explicit_task_id
                confidence = 1.0  # Explicit task ID has perfect confidence
        else:
            # Auto-detect task from features
            detected_task, confidence = self.task_detector.identify_task(features, context)

        # Step 2: Handle new task registration
        is_new_task = (detected_task == 'new_task')
        if is_new_task:
            # Use explicit task_name from context, or auto-generate
            if context and 'task_name' in context:
                task_id = context['task_name']
            else:
                task_id = self._generate_task_id(context)

            # Register in both detector and selector
            self.task_detector.register_task(task_id, features)
            self.module_selector.register_task(task_id)
        else:
            task_id = detected_task

            # Update task embedding with new observation
            self.task_detector.update_embedding(task_id, features)

        # Step 3: Select modules
        active_modules = self.module_selector.select_modules(task_id, all_modules)

        # Step 4: Record usage
        self.module_selector.record_usage(task_id, active_modules)

        # Return routing decision
        return {
            'task_id': task_id,
            'is_new_task': is_new_task,
            'active_modules': active_modules,
            'confidence': confidence
        }

    def _generate_task_id(self, context: Optional[Dict] = None) -> str:
        """
        Generate unique task ID for new tasks.

        Args:
            context: Optional context with 'task_name' hint

        Returns:
            task_id: Unique identifier
        """
        # Use context hint if available
        if context and 'task_name' in context:
            base_name = context['task_name']
        else:
            base_name = f"task_{self.task_counter}"
            self.task_counter += 1

        # Ensure uniqueness
        task_id = base_name
        counter = 1
        while task_id in self.task_detector.task_embeddings:
            task_id = f"{base_name}_{counter}"
            counter += 1

        return task_id

    def get_all_tasks(self) -> List[str]:
        """Get list of all registered task IDs."""
        return list(self.task_detector.task_embeddings.keys())

    def get_task_summary(self) -> Dict:
        """
        Get comprehensive summary of all tasks.

        Returns:
            summary: {
                'n_tasks': int,
                'tasks': {task_id: {
                    'embedding_norm': float,
                    'observation_count': int,
                    'active_modules': List[str],
                    'module_usage': Dict[str, int]
                }}
            }
        """
        summary = {
            'n_tasks': len(self.task_detector.task_embeddings),
            'tasks': {}
        }

        for task_id in self.get_all_tasks():
            # Task detector info
            embedding = self.task_detector.task_embeddings[task_id]
            count = self.task_detector.task_counts[task_id]

            # Module selector info
            active_modules = self.module_selector.task_module_map.get(task_id, [])
            usage_stats = self.module_selector.get_usage_stats().get(task_id, {})

            summary['tasks'][task_id] = {
                'embedding_norm': float(np.linalg.norm(embedding)),
                'observation_count': count,
                'active_modules': active_modules,
                'module_usage': usage_stats
            }

        return summary


# =======================
# Testing and Validation
# =======================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TaskRouter v2.0")
    print("=" * 60)

    # Initialize router
    router = TaskRouter(
        feature_size=8,
        detection_threshold=0.8,
        default_modules=['linear', 'nonlinear', 'interaction']
    )

    all_modules = ['linear', 'nonlinear', 'interaction']

    # Test 1: First task (should create new task)
    print("\n[Test 1] First input - should create new task")
    features_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    decision_1 = router.route(features_1, all_modules, context={'task_name': 'linear_task'})
    print(f"Decision: {decision_1}")
    assert decision_1['is_new_task'] == True
    assert decision_1['task_id'] == 'linear_task'

    # Test 2: Similar input (should recognize same task)
    print("\n[Test 2] Similar input - should recognize linear_task")
    features_2 = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Very similar to features_1
    decision_2 = router.route(features_2, all_modules)
    print(f"Decision: {decision_2}")
    assert decision_2['is_new_task'] == False
    assert decision_2['task_id'] == 'linear_task'
    print(f"Confidence: {decision_2['confidence']:.3f}")

    # Test 3: Different input (should create new task)
    print("\n[Test 3] Different input - should create new task")
    features_3 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Orthogonal to features_1
    decision_3 = router.route(features_3, all_modules, context={'task_name': 'nonlinear_task'})
    print(f"Decision: {decision_3}")
    assert decision_3['is_new_task'] == True
    assert decision_3['task_id'] == 'nonlinear_task'

    # Test 4: Back to first task
    print("\n[Test 4] Back to first pattern - should recognize linear_task")
    features_4 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    decision_4 = router.route(features_4, all_modules)
    print(f"Decision: {decision_4}")
    assert decision_4['is_new_task'] == False
    assert decision_4['task_id'] == 'linear_task'

    # Test 5: Task summary
    print("\n[Test 5] Task summary")
    summary = router.get_task_summary()
    print(f"Total tasks: {summary['n_tasks']}")
    for task_id, info in summary['tasks'].items():
        print(f"\n  Task: {task_id}")
        print(f"    Observations: {info['observation_count']}")
        print(f"    Active modules: {info['active_modules']}")
        print(f"    Module usage: {info['module_usage']}")

    # Test 6: Task similarity matrix
    print("\n[Test 6] Task similarity matrix")
    similarity_matrix = router.task_detector.get_task_similarity_matrix()
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity matrix:\n{similarity_matrix}")
    print(f"Tasks are orthogonal (similarity ≈ 0): {np.abs(similarity_matrix[0, 1]) < 0.1}")

    # Test 7: Module selection override
    print("\n[Test 7] Module selection override")
    router.module_selector.update_module_set('linear_task', ['linear'])  # Only linear module
    decision_7 = router.route(features_1, all_modules)
    print(f"Decision: {decision_7}")
    assert decision_7['active_modules'] == ['linear']

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    print("\n[Summary]")
    print(f"- TaskDetector correctly identifies tasks via embedding similarity")
    print(f"- ModuleSelector routes tasks to appropriate modules")
    print(f"- TaskRouter coordinates detection + routing seamlessly")
    print(f"- Online embedding updates improve task recognition over time")
    print(f"- Module specialization supported via update_module_set()")
