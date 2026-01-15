"""
GENESIS Path B Phase 1: Full Environment

64x64 Torus Grid with:
- Two resource types: Energy (fast regen) and Material (slow regen)
- Spatial heterogeneity (resource patches)
- Logistic growth dynamics
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResourceConfig:
    """Resource configuration parameters"""
    energy_growth_rate: float = 0.15    # Fast regeneration
    material_growth_rate: float = 0.05  # Slow regeneration
    energy_capacity: float = 1.0
    material_capacity: float = 1.0
    n_patches: int = 8                  # Number of resource-rich patches
    patch_radius_min: int = 5
    patch_radius_max: int = 15


class FullALifeEnvironment:
    """
    64x64 Torus Grid Environment
    
    Features:
    - Toroidal topology (wraparound boundaries)
    - Two resource types with different dynamics
    - Spatial heterogeneity through patches
    - Visual field sensing for agents
    """
    
    def __init__(self, 
                 size: int = 64, 
                 config: Optional[ResourceConfig] = None,
                 seed: Optional[int] = None):
        """
        Args:
            size: Grid size (size x size)
            config: Resource configuration
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.size = size
        self.config = config or ResourceConfig()
        
        # Initialize resource grids
        # Energy: more abundant, faster regeneration
        self.energy_grid = np.random.uniform(0.3, 0.7, (size, size))
        # Material: less abundant, slower regeneration
        self.material_grid = np.random.uniform(0.1, 0.4, (size, size))
        
        # Create spatial heterogeneity
        self._create_resource_patches()
        
        # Occupation grid (for collision detection)
        self.occupation_grid = np.zeros((size, size), dtype=np.int32)
        
        # Statistics
        self.step_count = 0
        self.total_energy_consumed = 0.0
        self.total_material_consumed = 0.0
        
    def _create_resource_patches(self):
        """
        Create resource-rich and resource-poor regions
        Creates spatial structure for niche formation
        """
        cfg = self.config
        
        # Energy-rich patches
        for _ in range(cfg.n_patches):
            cx = np.random.randint(0, self.size)
            cy = np.random.randint(0, self.size)
            radius = np.random.randint(cfg.patch_radius_min, cfg.patch_radius_max)
            richness = np.random.uniform(0.5, 1.0)
            
            for x in range(self.size):
                for y in range(self.size):
                    # Toroidal distance
                    dx = min(abs(x - cx), self.size - abs(x - cx))
                    dy = min(abs(y - cy), self.size - abs(y - cy))
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < radius:
                        weight = 1 - dist / radius
                        self.energy_grid[x, y] += richness * weight * 0.3
        
        # Material-rich patches (different locations)
        for _ in range(cfg.n_patches // 2):
            cx = np.random.randint(0, self.size)
            cy = np.random.randint(0, self.size)
            radius = np.random.randint(cfg.patch_radius_min, cfg.patch_radius_max)
            richness = np.random.uniform(0.3, 0.7)
            
            for x in range(self.size):
                for y in range(self.size):
                    dx = min(abs(x - cx), self.size - abs(x - cx))
                    dy = min(abs(y - cy), self.size - abs(y - cy))
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < radius:
                        weight = 1 - dist / radius
                        self.material_grid[x, y] += richness * weight * 0.3
        
        # Clip to capacity
        self.energy_grid = np.clip(self.energy_grid, 0, cfg.energy_capacity)
        self.material_grid = np.clip(self.material_grid, 0, cfg.material_capacity)
    
    def step(self):
        """
        Environment step: resource regeneration using logistic growth
        r_t+1 = r_t + growth_rate * r_t * (1 - r_t / capacity)
        """
        cfg = self.config
        
        # Energy growth (faster)
        energy_growth = (cfg.energy_growth_rate * self.energy_grid * 
                        (1 - self.energy_grid / cfg.energy_capacity))
        self.energy_grid = np.clip(
            self.energy_grid + energy_growth, 
            0, 
            cfg.energy_capacity
        )
        
        # Material growth (slower)
        material_growth = (cfg.material_growth_rate * self.material_grid * 
                          (1 - self.material_grid / cfg.material_capacity))
        self.material_grid = np.clip(
            self.material_grid + material_growth, 
            0, 
            cfg.material_capacity
        )
        
        self.step_count += 1
    
    def get_visual_field(self, x: int, y: int, radius: int = 3) -> np.ndarray:
        """
        Get visual field centered at (x, y)
        
        Args:
            x, y: Center position
            radius: Visual field radius (results in (2*radius+1)^2 cells)
            
        Returns:
            field: shape (2*radius+1, 2*radius+1, 3)
                   channels: [energy, material, occupation]
        """
        field_size = 2 * radius + 1
        field = np.zeros((field_size, field_size, 3))
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Toroidal wrapping
                nx = (x + dx) % self.size
                ny = (y + dy) % self.size
                
                fx = dx + radius
                fy = dy + radius
                
                field[fx, fy, 0] = self.energy_grid[nx, ny]
                field[fx, fy, 1] = self.material_grid[nx, ny]
                field[fx, fy, 2] = min(1.0, self.occupation_grid[nx, ny])
        
        return field
    
    def get_local_gradient(self, x: int, y: int) -> np.ndarray:
        """
        Compute resource gradient at position
        
        Returns:
            gradient: shape (4,) - [energy_dx, energy_dy, material_dx, material_dy]
        """
        # Sobel-like gradient computation
        energy_grad = np.zeros(2)
        material_grad = np.zeros(2)
        
        for dx, dy, weight in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]:
            nx = (x + dx) % self.size
            ny = (y + dy) % self.size
            
            if dx != 0:
                energy_grad[0] += dx * weight * self.energy_grid[nx, ny]
                material_grad[0] += dx * weight * self.material_grid[nx, ny]
            if dy != 0:
                energy_grad[1] += dy * weight * self.energy_grid[nx, ny]
                material_grad[1] += dy * weight * self.material_grid[nx, ny]
        
        return np.concatenate([energy_grad, material_grad])
    
    def consume(self, x: int, y: int, 
                energy_amt: float, 
                material_amt: float) -> Tuple[float, float]:
        """
        Consume resources at position
        
        Args:
            x, y: Position
            energy_amt: Requested energy amount
            material_amt: Requested material amount
            
        Returns:
            (energy_gained, material_gained): Actual amounts consumed
        """
        x, y = x % self.size, y % self.size
        
        # Consume energy
        energy_gained = min(energy_amt, self.energy_grid[x, y])
        self.energy_grid[x, y] -= energy_gained
        self.total_energy_consumed += energy_gained
        
        # Consume material
        material_gained = min(material_amt, self.material_grid[x, y])
        self.material_grid[x, y] -= material_gained
        self.total_material_consumed += material_gained
        
        return energy_gained, material_gained
    
    def set_occupation(self, positions: list):
        """
        Update occupation grid with agent positions
        
        Args:
            positions: List of (x, y) tuples
        """
        self.occupation_grid.fill(0)
        for x, y in positions:
            self.occupation_grid[x % self.size, y % self.size] += 1
    
    def toroidal_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Compute distance on torus"""
        dx = min(abs(x2 - x1), self.size - abs(x2 - x1))
        dy = min(abs(y2 - y1), self.size - abs(y2 - y1))
        return np.sqrt(dx**2 + dy**2)
    
    def get_statistics(self) -> Dict:
        """Get environment statistics"""
        return {
            'step': self.step_count,
            'energy_mean': float(np.mean(self.energy_grid)),
            'energy_std': float(np.std(self.energy_grid)),
            'energy_total': float(np.sum(self.energy_grid)),
            'material_mean': float(np.mean(self.material_grid)),
            'material_std': float(np.std(self.material_grid)),
            'material_total': float(np.sum(self.material_grid)),
            'total_energy_consumed': self.total_energy_consumed,
            'total_material_consumed': self.total_material_consumed
        }
    
    def get_resource_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get copies of resource grids for visualization"""
        return self.energy_grid.copy(), self.material_grid.copy()


if __name__ == "__main__":
    print("Testing FullALifeEnvironment...")
    
    env = FullALifeEnvironment(size=64, seed=42)
    
    print(f"Grid size: {env.size}x{env.size}")
    print(f"\nInitial state:")
    stats = env.get_statistics()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Test visual field
    visual = env.get_visual_field(32, 32, radius=3)
    print(f"\nVisual field shape: {visual.shape}")
    print(f"  Energy range: [{visual[:,:,0].min():.3f}, {visual[:,:,0].max():.3f}]")
    print(f"  Material range: [{visual[:,:,1].min():.3f}, {visual[:,:,1].max():.3f}]")
    
    # Test gradient
    gradient = env.get_local_gradient(32, 32)
    print(f"\nLocal gradient at (32,32): {gradient}")
    
    # Test consumption
    before_e, before_m = env.energy_grid[32, 32], env.material_grid[32, 32]
    gained_e, gained_m = env.consume(32, 32, 0.3, 0.2)
    after_e, after_m = env.energy_grid[32, 32], env.material_grid[32, 32]
    print(f"\nConsumption test:")
    print(f"  Energy: {before_e:.3f} -> {after_e:.3f} (gained {gained_e:.3f})")
    print(f"  Material: {before_m:.3f} -> {after_m:.3f} (gained {gained_m:.3f})")
    
    # Run for some steps
    for i in range(100):
        env.step()
    
    print(f"\nAfter 100 steps:")
    stats = env.get_statistics()
    print(f"  Energy mean: {stats['energy_mean']:.4f}")
    print(f"  Material mean: {stats['material_mean']:.4f}")
    
    print("\nEnvironment test PASSED!")
