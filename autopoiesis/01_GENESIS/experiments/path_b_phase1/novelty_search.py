"""
GENESIS Phase 4B: Novelty Search

Rewards behavioral novelty instead of objective performance.

Key insight: Sometimes the best way to solve a problem is to stop trying to solve it
and instead explore the space of possible behaviors. Novelty search avoids deceptive
local optima by rewarding agents that do something different from what's been seen before.

References:
- Lehman & Stanley (2011) "Abandoning Objectives: Evolution through the Search for Novelty Alone"
- Lehman & Stanley (2008) "Exploiting Open-Endedness to Solve Problems Through the Search for Novelty"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
import copy


class BehaviorCharacterizer:
    """
    Converts agent experiences into behavior descriptor

    Behavior descriptor = compact representation of "what the agent did"
    Different characterizations reveal different aspects of behavior
    """

    def __init__(self, characterization_type: str = 'trajectory'):
        """
        Args:
            characterization_type: Type of behavior characterization
                - 'trajectory': Final position (x, y)
                - 'resource': Resource usage pattern (energy, material, reproductions)
                - 'movement': Movement statistics (speed, turns, exploration)
                - 'composite': Combination of multiple aspects
        """
        self.characterization_type = characterization_type

    def characterize(self, agent_history: Dict) -> np.ndarray:
        """
        Convert agent history to behavior descriptor

        Args:
            agent_history: Dictionary with agent's history
                - 'positions': List of (x, y) positions
                - 'energies': List of energy values
                - 'materials': List of material values
                - 'actions': List of action vectors
                - 'reproductions': Count of reproductions

        Returns:
            Behavior descriptor (fixed-size numpy array)
        """
        if self.characterization_type == 'trajectory':
            return self._characterize_trajectory(agent_history)
        elif self.characterization_type == 'resource':
            return self._characterize_resource(agent_history)
        elif self.characterization_type == 'movement':
            return self._characterize_movement(agent_history)
        elif self.characterization_type == 'composite':
            return self._characterize_composite(agent_history)
        else:
            raise ValueError(f"Unknown characterization type: {self.characterization_type}")

    def _characterize_trajectory(self, history: Dict) -> np.ndarray:
        """
        Characterize by final position

        Simple but effective for maze/navigation tasks
        """
        positions = history.get('positions', [])
        if not positions:
            return np.array([0.0, 0.0])

        # Convert to list if it's a deque
        if not isinstance(positions, (list, np.ndarray)):
            positions = list(positions)

        # Final position
        final_pos = positions[-1]
        return np.array([final_pos[0], final_pos[1]], dtype=np.float32)

    def _characterize_resource(self, history: Dict) -> np.ndarray:
        """
        Characterize by resource usage pattern

        Good for understanding survival strategies
        """
        energies = history.get('energies', [])
        materials = history.get('materials', [])
        reproductions = history.get('reproductions', 0)

        # Convert to numpy arrays if needed
        if energies and not isinstance(energies, np.ndarray):
            energies = np.array(list(energies))
        if materials and not isinstance(materials, np.ndarray):
            materials = np.array(list(materials))

        # Statistics
        avg_energy = np.mean(energies) if len(energies) > 0 else 0.0
        std_energy = np.std(energies) if len(energies) > 0 else 0.0
        avg_material = np.mean(materials) if len(materials) > 0 else 0.0
        reproduction_rate = reproductions / max(len(energies) if len(energies) > 0 else 1, 1)

        return np.array([
            avg_energy,
            std_energy,
            avg_material,
            reproduction_rate
        ], dtype=np.float32)

    def _characterize_movement(self, history: Dict) -> np.ndarray:
        """
        Characterize by movement style

        Good for understanding exploration vs exploitation
        """
        positions = history.get('positions', [])
        actions = history.get('actions', [])

        # Convert to numpy array if it's a deque or list
        if not isinstance(positions, np.ndarray):
            positions = np.array(list(positions))

        if len(positions) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Movement statistics
        movements = np.diff(positions, axis=0)
        speeds = np.linalg.norm(movements, axis=1)

        avg_speed = np.mean(speeds)
        std_speed = np.std(speeds)

        # Turn rate (change in direction)
        if len(movements) > 1:
            angles = np.arctan2(movements[:, 1], movements[:, 0])
            turn_rates = np.abs(np.diff(angles))
            avg_turn_rate = np.mean(turn_rates)
        else:
            avg_turn_rate = 0.0

        # Exploration radius (distance from starting point)
        start_pos = positions[0]
        distances = np.linalg.norm(positions - start_pos, axis=1)
        max_distance = np.max(distances)

        return np.array([
            avg_speed,
            std_speed,
            avg_turn_rate,
            max_distance
        ], dtype=np.float32)

    def _characterize_composite(self, history: Dict) -> np.ndarray:
        """
        Composite characterization using multiple aspects

        Most informative but higher dimensional
        """
        trajectory = self._characterize_trajectory(history)
        resource = self._characterize_resource(history)
        movement = self._characterize_movement(history)

        return np.concatenate([trajectory, resource, movement])


class NoveltyArchive:
    """
    Archive of discovered behaviors

    Used to compute novelty scores (distance to archived behaviors)
    """

    def __init__(self,
                 max_size: int = 10000,
                 novelty_threshold: float = 0.5):
        """
        Args:
            max_size: Maximum archive size (oldest behaviors pruned when exceeded)
            novelty_threshold: Minimum novelty to add to archive
        """
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold

        # Archive storage
        self.behaviors = []  # List of behavior descriptors
        self.timestamps = []  # When each behavior was added
        self.metadata = []  # Optional metadata (agent ID, fitness, etc.)

        # Statistics
        self.total_added = 0
        self.total_rejected = 0

    def add(self,
            behavior: np.ndarray,
            novelty: float,
            metadata: Optional[Dict] = None):
        """
        Add behavior to archive if novel enough

        Args:
            behavior: Behavior descriptor
            novelty: Computed novelty score
            metadata: Optional metadata to store

        Returns:
            True if added, False if rejected
        """
        if novelty >= self.novelty_threshold:
            self.behaviors.append(behavior.copy())
            self.timestamps.append(self.total_added + self.total_rejected)
            self.metadata.append(metadata or {})

            # Prune if too large (FIFO)
            if len(self.behaviors) > self.max_size:
                self.behaviors.pop(0)
                self.timestamps.pop(0)
                self.metadata.pop(0)

            self.total_added += 1
            return True
        else:
            self.total_rejected += 1
            return False

    def get_behaviors(self) -> List[np.ndarray]:
        """Get all archived behaviors"""
        return self.behaviors

    def get_statistics(self) -> Dict:
        """Get archive statistics"""
        return {
            'size': len(self.behaviors),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'total_rejected': self.total_rejected,
            'acceptance_rate': self.total_added / max(self.total_added + self.total_rejected, 1)
        }


class NoveltySearch:
    """
    Novelty Search algorithm

    Evolution guided by behavioral novelty instead of objective fitness
    """

    def __init__(self,
                 behavior_characterizer: BehaviorCharacterizer,
                 k_nearest: int = 15,
                 archive_max_size: int = 10000,
                 novelty_threshold: float = 0.5,
                 combine_with_fitness: bool = False,
                 fitness_weight: float = 0.0):
        """
        Args:
            behavior_characterizer: How to convert agent history to behavior
            k_nearest: Number of nearest neighbors for novelty computation
            archive_max_size: Max behaviors to store in archive
            novelty_threshold: Min novelty to add to archive
            combine_with_fitness: Whether to combine novelty with fitness
            fitness_weight: Weight for fitness (1-weight for novelty)
        """
        self.characterizer = behavior_characterizer
        self.k_nearest = k_nearest
        self.archive = NoveltyArchive(archive_max_size, novelty_threshold)

        self.combine_with_fitness = combine_with_fitness
        self.fitness_weight = fitness_weight
        self.novelty_weight = 1.0 - fitness_weight

        # Statistics
        self.evaluations = 0
        self.unique_behaviors = 0
        self.novelty_history = deque(maxlen=1000)

    def compute_novelty(self,
                       agent_history: Dict,
                       population_behaviors: Optional[List[np.ndarray]] = None) -> float:
        """
        Compute novelty of agent's behavior

        Args:
            agent_history: Agent's behavioral history
            population_behaviors: Current population's behaviors (for local novelty)

        Returns:
            Novelty score (higher = more novel)
        """
        # Characterize behavior
        behavior = self.characterizer.characterize(agent_history)

        # Get comparison set (archive + optional current population)
        archived_behaviors = self.archive.get_behaviors()

        if population_behaviors:
            comparison_behaviors = archived_behaviors + population_behaviors
        else:
            comparison_behaviors = archived_behaviors

        # Compute novelty
        if len(comparison_behaviors) < self.k_nearest:
            # Archive too small - everything is novel
            novelty = 1.0
        else:
            # Compute distances to all behaviors
            distances = [
                self._behavior_distance(behavior, other)
                for other in comparison_behaviors
            ]

            # Average distance to k-nearest neighbors
            nearest_k = sorted(distances)[:self.k_nearest]
            novelty = np.mean(nearest_k)

        # Track statistics
        self.evaluations += 1
        self.novelty_history.append(novelty)

        return novelty

    def evaluate_agent(self,
                      agent_history: Dict,
                      fitness: float = 0.0,
                      population_behaviors: Optional[List[np.ndarray]] = None) -> Tuple[float, Dict]:
        """
        Evaluate agent using novelty (and optionally fitness)

        Args:
            agent_history: Agent's behavioral history
            fitness: Objective fitness (if combining with novelty)
            population_behaviors: Current population behaviors

        Returns:
            (score, info_dict) where score is novelty or novelty+fitness combination
        """
        # Compute novelty
        novelty = self.compute_novelty(agent_history, population_behaviors)

        # Compute final score
        if self.combine_with_fitness:
            score = self.novelty_weight * novelty + self.fitness_weight * fitness
        else:
            score = novelty

        # Add to archive
        behavior = self.characterizer.characterize(agent_history)
        added = self.archive.add(behavior, novelty, {
            'fitness': fitness,
            'score': score,
            'evaluation': self.evaluations
        })

        if added:
            self.unique_behaviors += 1

        info = {
            'novelty': novelty,
            'fitness': fitness,
            'score': score,
            'added_to_archive': added
        }

        return score, info

    def _behavior_distance(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """
        Compute distance between two behaviors

        Uses Euclidean distance (can be swapped for other metrics)
        """
        return np.linalg.norm(b1 - b2)

    def get_statistics(self) -> Dict:
        """Get novelty search statistics"""
        archive_stats = self.archive.get_statistics()

        return {
            'evaluations': self.evaluations,
            'unique_behaviors': self.unique_behaviors,
            'avg_novelty': np.mean(self.novelty_history) if self.novelty_history else 0.0,
            'std_novelty': np.std(self.novelty_history) if self.novelty_history else 0.0,
            'archive': archive_stats
        }

    def get_diverse_exemplars(self, n: int = 10) -> List[Dict]:
        """
        Get n diverse behaviors from archive

        Uses farthest-first traversal to maximize diversity
        """
        behaviors = self.archive.get_behaviors()
        metadata = self.archive.metadata

        if len(behaviors) == 0:
            return []

        if len(behaviors) <= n:
            return [
                {'behavior': b, 'metadata': m}
                for b, m in zip(behaviors, metadata)
            ]

        # Farthest-first traversal
        selected_indices = []

        # Start with random behavior
        selected_indices.append(np.random.randint(len(behaviors)))

        while len(selected_indices) < n:
            # Find behavior farthest from selected set
            max_min_distance = -1
            best_idx = None

            for i in range(len(behaviors)):
                if i in selected_indices:
                    continue

                # Min distance to selected set
                min_dist = min(
                    self._behavior_distance(behaviors[i], behaviors[j])
                    for j in selected_indices
                )

                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_idx = i

            if best_idx is not None:
                selected_indices.append(best_idx)
            else:
                break

        return [
            {'behavior': behaviors[i], 'metadata': metadata[i]}
            for i in selected_indices
        ]


def create_novelty_search(characterization_type: str = 'trajectory',
                         k_nearest: int = 15,
                         pure_novelty: bool = True) -> NoveltySearch:
    """
    Convenience function to create NoveltySearch instance

    Args:
        characterization_type: Type of behavior characterization
        k_nearest: Number of nearest neighbors
        pure_novelty: If True, only novelty. If False, combine with fitness 50/50

    Returns:
        Configured NoveltySearch instance
    """
    characterizer = BehaviorCharacterizer(characterization_type)

    if pure_novelty:
        ns = NoveltySearch(
            behavior_characterizer=characterizer,
            k_nearest=k_nearest,
            combine_with_fitness=False
        )
    else:
        ns = NoveltySearch(
            behavior_characterizer=characterizer,
            k_nearest=k_nearest,
            combine_with_fitness=True,
            fitness_weight=0.5  # 50% fitness, 50% novelty
        )

    return ns
