"""
GENESIS Path B Phase 0: Minimal Environment

16x16 Torus Grid with Logistic Resource Growth
"""

import numpy as np
from typing import Dict, Tuple


class MinimalGrid:
    """
    16x16 토러스 그리드 환경
    
    Features:
    - Torus topology (경계 연결)
    - Logistic resource growth
    - Local sensing (8 directions)
    """
    
    def __init__(self, size: int = 16, growth_rate: float = 0.15, capacity: float = 1.0):
        """
        Args:
            size: Grid size (size x size)
            growth_rate: Resource growth rate
            capacity: Maximum resource per cell
        """
        self.size = size
        self.growth_rate = growth_rate
        self.capacity = capacity
        
        # Initialize resources randomly in [0.2, 0.8]
        self.resources = np.random.uniform(0.2, 0.8, (size, size))
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
    def _wrap(self, x: int, y: int) -> Tuple[int, int]:
        """Wrap coordinates for torus topology"""
        return x % self.size, y % self.size
    
    def step(self) -> None:
        """
        Resource growth step (logistic growth)
        r_t+1 = r_t + growth_rate * r_t * (1 - r_t/capacity)
        """
        growth = self.growth_rate * self.resources * (1 - self.resources / self.capacity)
        self.resources = np.clip(self.resources + growth, 0, self.capacity)
    
    def get_local_view(self, x: int, y: int) -> np.ndarray:
        """
        Get local sensor view (8 directions + current position)
        
        Args:
            x, y: Agent position
            
        Returns:
            sensor_input: shape (18,) - 9 positions x 2 features (resource, occupied)
                          [current_resource, current_occupied,
                           N_resource, N_occupied, NE_resource, NE_occupied, ...]
        """
        sensor = np.zeros(18)
        
        # Current position
        wx, wy = self._wrap(x, y)
        sensor[0] = self.resources[wx, wy]
        sensor[1] = 0.0  # Occupied will be set by population manager
        
        # 8 directions
        for i, (dx, dy) in enumerate(self.directions):
            nx, ny = self._wrap(x + dx, y + dy)
            sensor[2 + i * 2] = self.resources[nx, ny]
            sensor[2 + i * 2 + 1] = 0.0  # Occupied placeholder
            
        return sensor
    
    def consume(self, x: int, y: int, amount: float) -> float:
        """
        Consume resources at position
        
        Args:
            x, y: Position
            amount: Requested amount
            
        Returns:
            actual_consumed: Actual amount consumed (may be less than requested)
        """
        wx, wy = self._wrap(x, y)
        actual = min(amount, self.resources[wx, wy])
        self.resources[wx, wy] -= actual
        return actual
    
    def get_statistics(self) -> Dict:
        """Get environment statistics"""
        return {
            'mean_resource': float(np.mean(self.resources)),
            'std_resource': float(np.std(self.resources)),
            'min_resource': float(np.min(self.resources)),
            'max_resource': float(np.max(self.resources)),
            'total_resource': float(np.sum(self.resources))
        }
    
    def get_resource_map(self) -> np.ndarray:
        """Get resource map for visualization"""
        return self.resources.copy()


if __name__ == "__main__":
    # Test
    env = MinimalGrid(size=16, growth_rate=0.1)
    
    print("Initial state:")
    print(f"  Mean resource: {np.mean(env.resources):.3f}")
    
    # Test growth
    for i in range(10):
        env.step()
    
    print(f"\nAfter 10 steps:")
    print(f"  Mean resource: {np.mean(env.resources):.3f}")
    
    # Test sensing
    view = env.get_local_view(8, 8)
    print(f"\nLocal view at (8,8): shape={view.shape}")
    print(f"  Current resource: {view[0]:.3f}")
    
    # Test consumption
    before = env.resources[8, 8]
    consumed = env.consume(8, 8, 0.5)
    after = env.resources[8, 8]
    print(f"\nConsumption test at (8,8):")
    print(f"  Before: {before:.3f}, Consumed: {consumed:.3f}, After: {after:.3f}")
    
    print("\nEnvironment test passed!")
