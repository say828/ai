"""
GENESIS Phase 2: Stigmergy System

Environmental memory - indirect communication through environmental modifications.

Inspired by:
- Ant pheromone trails (coordination without direct communication)
- Termite mound construction (collective architecture)
- Slime mold path optimization (distributed problem solving)

Agents leave "marks" in the environment that guide other agents,
enabling emergent collective intelligence.
"""

import numpy as np
from typing import Dict, Tuple


class StigmergyField:
    """
    Multi-channel environmental memory field

    Different "channels" for different information types:
    - Pheromone trails: Movement paths
    - Danger markers: Hazard warnings
    - Resource indicators: Food/energy locations
    - Success markers: High coherence zones

    Each channel decays over time (evaporation) to prevent obsolete information.
    """

    def __init__(self, grid_size: int, decay_rate: float = 0.98):
        """
        Args:
            grid_size: Size of square grid
            decay_rate: Per-step decay multiplier (0.98 = 2% evaporation per step)
        """
        self.grid_size = grid_size
        self.decay_rate = decay_rate

        # Multi-channel fields
        self.pheromone_trail = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.danger_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.resource_field = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.success_field = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Statistics
        self.total_deposits = 0
        self.total_readings = 0

    def deposit_pheromone(self, x: int, y: int, strength: float = 1.0):
        """
        Deposit movement pheromone (agent passed through here)

        Creates trails that other agents can follow, like ant pheromone trails.

        Args:
            x, y: Grid coordinates
            strength: Deposit amount
        """
        if self._in_bounds(x, y):
            self.pheromone_trail[x, y] += strength
            self.total_deposits += 1

    def mark_danger(self, x: int, y: int, intensity: float = 1.0):
        """
        Mark dangerous location (death, energy loss, low coherence)

        Args:
            x, y: Grid coordinates
            intensity: Danger level (0.0 = safe, 1.0 = deadly)
        """
        if self._in_bounds(x, y):
            self.danger_field[x, y] = max(self.danger_field[x, y], intensity)
            self.total_deposits += 1

    def mark_resource(self, x: int, y: int, amount: float = 1.0):
        """
        Mark resource location (food, energy sources)

        Args:
            x, y: Grid coordinates
            amount: Resource abundance
        """
        if self._in_bounds(x, y):
            self.resource_field[x, y] += amount
            self.total_deposits += 1

    def mark_success(self, x: int, y: int, coherence: float):
        """
        Mark location where high coherence was achieved

        Creates "success zones" that attract other agents.

        Args:
            x, y: Grid coordinates
            coherence: Coherence value achieved here
        """
        if self._in_bounds(x, y):
            self.success_field[x, y] = max(self.success_field[x, y], coherence)
            self.total_deposits += 1

    def get_field_at(self, x: int, y: int, radius: int = 3) -> Dict[str, float]:
        """
        Read stigmergy information in local neighborhood

        Agents "smell" the environment in their vicinity.

        Args:
            x, y: Center coordinates
            radius: Sensing radius (default 3 = 7×7 neighborhood)

        Returns:
            Dictionary with aggregated field values
        """
        if not self._in_bounds(x, y):
            return self._empty_reading()

        x_min = max(0, x - radius)
        x_max = min(self.grid_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.grid_size, y + radius + 1)

        self.total_readings += 1

        return {
            'pheromone': float(self.pheromone_trail[x_min:x_max, y_min:y_max].mean()),
            'danger': float(self.danger_field[x_min:x_max, y_min:y_max].max()),
            'resource': float(self.resource_field[x_min:x_max, y_min:y_max].sum()),
            'success': float(self.success_field[x_min:x_max, y_min:y_max].max())
        }

    def get_gradient(self, x: int, y: int, field_type: str = 'pheromone') -> Tuple[float, float]:
        """
        Compute gradient of specified field at location

        Gradient points toward increasing field strength, enabling
        chemotaxis-like behavior (follow the gradient).

        Args:
            x, y: Location
            field_type: One of 'pheromone', 'danger', 'resource', 'success'

        Returns:
            (dx, dy): Gradient direction
        """
        if not self._in_bounds(x, y):
            return 0.0, 0.0

        field = self._get_field(field_type)

        # Compute 8-directional gradient
        dx = field[min(x + 1, self.grid_size - 1), y] - field[max(x - 1, 0), y]
        dy = field[x, min(y + 1, self.grid_size - 1)] - field[x, max(y - 1, 0)]

        return float(dx), float(dy)

    def decay_all(self):
        """
        Apply decay to all fields (evaporation)

        Older marks fade over time, preventing stale information from
        persisting indefinitely.
        """
        self.pheromone_trail *= self.decay_rate
        self.danger_field *= self.decay_rate
        self.resource_field *= self.decay_rate
        self.success_field *= self.decay_rate

    def diffuse_all(self, diffusion_rate: float = 0.1):
        """
        Apply diffusion to all fields (spreading)

        Marks spread to neighboring cells, creating smooth gradients
        rather than sharp peaks.

        Args:
            diffusion_rate: How much to spread (0.1 = 10% spreads to neighbors)
        """
        from scipy.ndimage import gaussian_filter

        # Simple Gaussian blur approximation of diffusion
        self.pheromone_trail = gaussian_filter(self.pheromone_trail, sigma=1.0) * (1 + diffusion_rate)
        self.danger_field = gaussian_filter(self.danger_field, sigma=1.0) * (1 + diffusion_rate)
        self.resource_field = gaussian_filter(self.resource_field, sigma=1.0) * (1 + diffusion_rate)
        self.success_field = gaussian_filter(self.success_field, sigma=1.0) * (1 + diffusion_rate)

    def get_field_visualization(self, field_type: str = 'pheromone') -> np.ndarray:
        """
        Get field for visualization

        Args:
            field_type: Which field to visualize

        Returns:
            2D array of field values
        """
        return self._get_field(field_type).copy()

    def get_statistics(self) -> Dict:
        """Get stigmergy statistics"""
        return {
            'total_deposits': self.total_deposits,
            'total_readings': self.total_readings,
            'field_statistics': {
                'pheromone': {
                    'mean': float(self.pheromone_trail.mean()),
                    'max': float(self.pheromone_trail.max()),
                    'std': float(self.pheromone_trail.std()),
                    'coverage': float((self.pheromone_trail > 0.01).sum() / self.pheromone_trail.size)
                },
                'danger': {
                    'mean': float(self.danger_field.mean()),
                    'max': float(self.danger_field.max()),
                    'coverage': float((self.danger_field > 0.1).sum() / self.danger_field.size)
                },
                'resource': {
                    'mean': float(self.resource_field.mean()),
                    'max': float(self.resource_field.max()),
                    'sum': float(self.resource_field.sum())
                },
                'success': {
                    'mean': float(self.success_field.mean()),
                    'max': float(self.success_field.max()),
                    'coverage': float((self.success_field > 0.8).sum() / self.success_field.size)
                }
            }
        }

    def reset(self):
        """Clear all stigmergy fields"""
        self.pheromone_trail.fill(0)
        self.danger_field.fill(0)
        self.resource_field.fill(0)
        self.success_field.fill(0)

    # Helper methods

    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid"""
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _get_field(self, field_type: str) -> np.ndarray:
        """Get field array by name"""
        field_map = {
            'pheromone': self.pheromone_trail,
            'danger': self.danger_field,
            'resource': self.resource_field,
            'success': self.success_field
        }
        return field_map.get(field_type, self.pheromone_trail)

    def _empty_reading(self) -> Dict[str, float]:
        """Return empty reading for out-of-bounds queries"""
        return {
            'pheromone': 0.0,
            'danger': 0.0,
            'resource': 0.0,
            'success': 0.0
        }
