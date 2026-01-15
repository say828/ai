"""
GENESIS Phase 4B: MAP-Elites (Quality-Diversity)

Illuminates the search space by maintaining high-quality solutions across behavioral dimensions.

Key insight: Instead of finding ONE best solution, find MANY good solutions that differ in
interesting ways. This gives us:
1. Robustness (multiple strategies for same task)
2. Creativity (unexpected solutions in unexplored regions)
3. Stepping stones (solutions useful for transfer learning)

References:
- Mouret & Clune (2015) "Illuminating the Search Space by Mapping Elites"
- Pugh et al. (2016) "Quality Diversity: A New Frontier for Evolutionary Computation"
- Cully & Demiris (2017) "Quality and Diversity Optimization"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy


class BehaviorSpace:
    """
    Defines discretized behavior space for MAP-Elites

    Behavior space = multi-dimensional grid where each cell represents a behavioral niche
    """

    def __init__(self,
                 behavior_dims: List[str],
                 bins_per_dim: int = 10,
                 behavior_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            behavior_dims: Names of behavioral dimensions (e.g., ['speed', 'exploration'])
            bins_per_dim: Number of bins per dimension
            behavior_ranges: Min/max for each dimension (auto-detected if None)
        """
        self.behavior_dims = behavior_dims
        self.bins_per_dim = bins_per_dim
        self.n_dims = len(behavior_dims)

        # Behavior ranges (will be learned if not provided)
        if behavior_ranges:
            self.behavior_ranges = behavior_ranges
            self.ranges_fixed = True
        else:
            # Initialize with placeholder ranges (will be updated)
            self.behavior_ranges = {dim: (0.0, 1.0) for dim in behavior_dims}
            self.ranges_fixed = False

        # Observations for automatic range detection
        self.observations = {dim: [] for dim in behavior_dims}

    def behavior_to_bin(self, behavior: Dict[str, float]) -> Tuple[int, ...]:
        """
        Convert continuous behavior to discrete bin coordinates

        Args:
            behavior: Dict mapping dimension name → value

        Returns:
            Tuple of bin indices (one per dimension)
        """
        bin_coords = []

        for dim in self.behavior_dims:
            value = behavior.get(dim, 0.0)

            # Update range if learning
            if not self.ranges_fixed:
                self.observations[dim].append(value)
                if len(self.observations[dim]) > 100:
                    # Update range based on observed values
                    obs = self.observations[dim]
                    self.behavior_ranges[dim] = (min(obs), max(obs))

            # Normalize to [0, 1]
            min_val, max_val = self.behavior_ranges[dim]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.5

            # Clip and convert to bin
            normalized = np.clip(normalized, 0.0, 0.999)  # 0.999 to avoid edge case
            bin_idx = int(normalized * self.bins_per_dim)

            bin_coords.append(bin_idx)

        return tuple(bin_coords)

    def bin_to_behavior(self, bin_coords: Tuple[int, ...]) -> Dict[str, float]:
        """
        Convert bin coordinates to behavior (center of bin)

        Args:
            bin_coords: Tuple of bin indices

        Returns:
            Dict mapping dimension name → value (center of bin)
        """
        behavior = {}

        for i, dim in enumerate(self.behavior_dims):
            bin_idx = bin_coords[i]

            # Center of bin
            normalized = (bin_idx + 0.5) / self.bins_per_dim

            # Denormalize
            min_val, max_val = self.behavior_ranges[dim]
            value = min_val + normalized * (max_val - min_val)

            behavior[dim] = value

        return behavior

    def total_bins(self) -> int:
        """Total number of bins in behavior space"""
        return self.bins_per_dim ** self.n_dims

    def get_statistics(self) -> Dict:
        """Get behavior space statistics"""
        return {
            'dimensions': self.n_dims,
            'bins_per_dim': self.bins_per_dim,
            'total_bins': self.total_bins(),
            'behavior_ranges': self.behavior_ranges,
            'ranges_fixed': self.ranges_fixed
        }


class EliteArchive:
    """
    Archive storing the best solution in each behavioral niche

    Each bin contains: (agent, fitness, behavior, metadata)
    """

    def __init__(self, behavior_space: BehaviorSpace):
        """
        Args:
            behavior_space: Defines discretization of behavior space
        """
        self.behavior_space = behavior_space

        # Archive: bin_coords → (agent, fitness, behavior, metadata)
        self.archive = {}

        # Statistics
        self.total_adds = 0
        self.total_improvements = 0
        self.total_rejections = 0

    def add(self,
            agent,
            fitness: float,
            behavior: Dict[str, float],
            metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Add solution to archive (if it improves the bin)

        Args:
            agent: Agent (will be deep copied)
            fitness: Quality/fitness score
            behavior: Behavioral characterization
            metadata: Optional metadata

        Returns:
            (added, reason) where:
                added = True if solution was added
                reason = "new_bin", "improvement", or "rejected"
        """
        # Get bin
        bin_coords = self.behavior_space.behavior_to_bin(behavior)

        # Check if bin empty or new solution better
        if bin_coords not in self.archive:
            # New bin!
            self.archive[bin_coords] = (
                copy.deepcopy(agent),
                fitness,
                behavior.copy(),
                metadata or {}
            )
            self.total_adds += 1
            return True, "new_bin"

        else:
            # Bin occupied - check if new solution better
            current_fitness = self.archive[bin_coords][1]

            if fitness > current_fitness:
                # Improvement!
                self.archive[bin_coords] = (
                    copy.deepcopy(agent),
                    fitness,
                    behavior.copy(),
                    metadata or {}
                )
                self.total_improvements += 1
                return True, "improvement"
            else:
                # Rejected
                self.total_rejections += 1
                return False, "rejected"

    def get(self, bin_coords: Tuple[int, ...]) -> Optional[Tuple]:
        """
        Get elite from specific bin

        Returns:
            (agent, fitness, behavior, metadata) or None if bin empty
        """
        return self.archive.get(bin_coords)

    def get_all_elites(self) -> List[Tuple]:
        """
        Get all elites in archive

        Returns:
            List of (agent, fitness, behavior, metadata, bin_coords)
        """
        return [
            (agent, fitness, behavior, metadata, bin_coords)
            for bin_coords, (agent, fitness, behavior, metadata) in self.archive.items()
        ]

    def sample_random_elite(self) -> Optional[Tuple]:
        """
        Sample random elite from archive

        Returns:
            (agent, fitness, behavior, metadata) or None if empty
        """
        if not self.archive:
            return None

        bin_coords = np.random.choice(list(self.archive.keys()))
        return self.archive[bin_coords]

    def get_statistics(self) -> Dict:
        """Get archive statistics"""
        if not self.archive:
            return {
                'size': 0,
                'coverage': 0.0,
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'total_adds': self.total_adds,
                'total_improvements': self.total_improvements,
                'total_rejections': self.total_rejections
            }

        fitnesses = [f for _, f, _, _ in self.archive.values()]

        return {
            'size': len(self.archive),
            'coverage': len(self.archive) / self.behavior_space.total_bins(),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'total_adds': self.total_adds,
            'total_improvements': self.total_improvements,
            'total_rejections': self.total_rejections
        }


class MAPElites:
    """
    MAP-Elites algorithm

    Illuminates search space by evolving diverse high-quality solutions
    """

    def __init__(self,
                 behavior_dims: List[str],
                 bins_per_dim: int = 10,
                 behavior_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            behavior_dims: Names of behavioral dimensions
            bins_per_dim: Number of bins per dimension
            behavior_ranges: Min/max for each dimension (auto-detected if None)
        """
        # Behavior space
        self.behavior_space = BehaviorSpace(
            behavior_dims=behavior_dims,
            bins_per_dim=bins_per_dim,
            behavior_ranges=behavior_ranges
        )

        # Elite archive
        self.archive = EliteArchive(self.behavior_space)

        # Statistics
        self.generation = 0
        self.evaluations = 0

    def add_solution(self,
                    agent,
                    fitness: float,
                    behavior: Dict[str, float],
                    metadata: Optional[Dict] = None) -> Dict:
        """
        Add solution to MAP-Elites archive

        Args:
            agent: Agent to add
            fitness: Fitness/quality score
            behavior: Behavioral characterization (dict of dimension → value)
            metadata: Optional metadata

        Returns:
            Info dict with 'added' (bool) and 'reason' (str)
        """
        added, reason = self.archive.add(agent, fitness, behavior, metadata)

        self.evaluations += 1

        return {
            'added': added,
            'reason': reason,
            'bin': self.behavior_space.behavior_to_bin(behavior)
        }

    def sample_elite(self) -> Optional[Tuple]:
        """
        Sample random elite for mutation/crossover

        Returns:
            (agent, fitness, behavior, metadata) or None if archive empty
        """
        return self.archive.sample_random_elite()

    def get_coverage(self) -> float:
        """Get fraction of behavior space covered"""
        return self.archive.get_statistics()['coverage']

    def get_statistics(self) -> Dict:
        """Get MAP-Elites statistics"""
        space_stats = self.behavior_space.get_statistics()
        archive_stats = self.archive.get_statistics()

        return {
            'generation': self.generation,
            'evaluations': self.evaluations,
            'behavior_space': space_stats,
            'archive': archive_stats
        }

    def get_behavior_heatmap(self, dim1: str, dim2: str) -> np.ndarray:
        """
        Get 2D heatmap of fitness across two behavioral dimensions

        Args:
            dim1: First dimension name
            dim2: Second dimension name

        Returns:
            2D array of fitnesses (NaN for empty bins)
        """
        if dim1 not in self.behavior_space.behavior_dims:
            raise ValueError(f"Dimension {dim1} not in behavior space")
        if dim2 not in self.behavior_space.behavior_dims:
            raise ValueError(f"Dimension {dim2} not in behavior space")

        # Get dimension indices
        idx1 = self.behavior_space.behavior_dims.index(dim1)
        idx2 = self.behavior_space.behavior_dims.index(dim2)

        # Create heatmap
        bins = self.behavior_space.bins_per_dim
        heatmap = np.full((bins, bins), np.nan)

        # Fill with fitnesses
        for bin_coords, (_, fitness, _, _) in self.archive.archive.items():
            i = bin_coords[idx1]
            j = bin_coords[idx2]
            heatmap[i, j] = fitness

        return heatmap

    def get_diverse_elites(self, n: int = 10) -> List[Tuple]:
        """
        Get n diverse elites (maximally spread across behavior space)

        Args:
            n: Number of elites to return

        Returns:
            List of (agent, fitness, behavior, metadata)
        """
        all_elites = self.archive.get_all_elites()

        if len(all_elites) <= n:
            return [(a, f, b, m) for a, f, b, m, _ in all_elites]

        # Use farthest-first traversal in behavior space
        selected_indices = []

        # Start with random elite
        selected_indices.append(np.random.randint(len(all_elites)))

        while len(selected_indices) < n:
            # Find elite farthest from selected set (in behavior space)
            max_min_distance = -1
            best_idx = None

            for i in range(len(all_elites)):
                if i in selected_indices:
                    continue

                # Min distance to selected set
                bin_i = all_elites[i][4]  # bin_coords

                min_dist = min(
                    self._bin_distance(bin_i, all_elites[j][4])
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
            (all_elites[i][0], all_elites[i][1], all_elites[i][2], all_elites[i][3])
            for i in selected_indices
        ]

    def _bin_distance(self, bin1: Tuple[int, ...], bin2: Tuple[int, ...]) -> float:
        """Distance between two bins in discretized space"""
        return np.sqrt(sum((b1 - b2)**2 for b1, b2 in zip(bin1, bin2)))


def create_map_elites(behavior_dims: List[str],
                     bins_per_dim: int = 10,
                     auto_range: bool = True) -> MAPElites:
    """
    Convenience function to create MAP-Elites instance

    Args:
        behavior_dims: Names of behavioral dimensions
        bins_per_dim: Number of bins per dimension
        auto_range: If True, automatically detect behavior ranges

    Returns:
        Configured MAP-Elites instance
    """
    if auto_range:
        behavior_ranges = None
    else:
        # Use default range [0, 1] for all dimensions
        behavior_ranges = {dim: (0.0, 1.0) for dim in behavior_dims}

    return MAPElites(
        behavior_dims=behavior_dims,
        bins_per_dim=bins_per_dim,
        behavior_ranges=behavior_ranges
    )
