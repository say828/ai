"""
GENESIS Path B: 2D Grid World Environment
==========================================

Open-ended artificial life environment for validating autopoietic agents.

Features:
- 100x100 grid world
- Resources: random spawn, deplete on consumption
- Predators: simple rule-based chase behavior
- Physics: movement, collision, vision cone

Author: GENESIS Project
Date: 2026-01-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import random


class CellType(IntEnum):
    """Grid cell types"""
    EMPTY = 0
    RESOURCE = 1
    OBSTACLE = 2
    PREDATOR = 3
    AGENT = 4


@dataclass
class Predator:
    """Simple rule-based predator"""
    x: int
    y: int
    speed: float = 0.7  # Move probability per step
    vision_range: int = 15
    
    def move_towards(self, target_x: int, target_y: int, grid_size: int) -> Tuple[int, int]:
        """Move one step towards target"""
        dx = np.sign(target_x - self.x)
        dy = np.sign(target_y - self.y)
        
        new_x = np.clip(self.x + dx, 0, grid_size - 1)
        new_y = np.clip(self.y + dy, 0, grid_size - 1)
        
        return int(new_x), int(new_y)


@dataclass
class Resource:
    """Resource that can be consumed"""
    x: int
    y: int
    energy: float = 1.0
    respawn_time: int = 50  # Steps until respawn after depletion


class GridWorld:
    """
    2D Grid World Environment
    
    Provides an open-ended environment for artificial life experiments.
    No explicit rewards - agents must maintain their own viability.
    """
    
    def __init__(
        self,
        size: int = 100,
        n_resources: int = 200,
        n_predators: int = 5,
        n_obstacles: int = 50,
        resource_respawn_rate: float = 0.02,
        resource_cluster_prob: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            size: Grid dimension (size x size)
            n_resources: Initial number of resources
            n_predators: Number of predators
            n_obstacles: Number of static obstacles
            resource_respawn_rate: Probability of new resource spawn per step
            resource_cluster_prob: Probability resources spawn near existing ones
            seed: Random seed for reproducibility
        """
        self.size = size
        self.n_resources = n_resources
        self.n_predators = n_predators
        self.n_obstacles = n_obstacles
        self.resource_respawn_rate = resource_respawn_rate
        self.resource_cluster_prob = resource_cluster_prob
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize grid
        self.grid = np.zeros((size, size), dtype=np.int8)
        
        # Entity lists
        self.resources: Dict[Tuple[int, int], Resource] = {}
        self.predators: List[Predator] = []
        self.obstacles: set = set()
        self.agent_positions: Dict[str, Tuple[int, int]] = {}  # agent_id -> (x, y)
        
        # Statistics
        self.step_count = 0
        self.total_resources_consumed = 0
        self.total_agent_deaths = 0
        
        # Initialize world
        self._initialize_world()
        
        print(f"GridWorld initialized: {size}x{size}")
        print(f"  Resources: {n_resources}")
        print(f"  Predators: {n_predators}")
        print(f"  Obstacles: {n_obstacles}")
    
    def _initialize_world(self):
        """Initialize all world elements"""
        # Place obstacles first (permanent)
        self._place_obstacles()
        
        # Place resources
        self._place_initial_resources()
        
        # Place predators
        self._place_predators()
    
    def _place_obstacles(self):
        """Place static obstacles"""
        placed = 0
        while placed < self.n_obstacles:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            if self.grid[x, y] == CellType.EMPTY:
                self.grid[x, y] = CellType.OBSTACLE
                self.obstacles.add((x, y))
                placed += 1
    
    def _place_initial_resources(self):
        """Place initial resources with clustering"""
        placed = 0
        while placed < self.n_resources:
            # Decide: cluster near existing or random
            if placed > 0 and random.random() < self.resource_cluster_prob:
                # Cluster near existing resource
                existing = random.choice(list(self.resources.keys()))
                x = np.clip(existing[0] + np.random.randint(-5, 6), 0, self.size - 1)
                y = np.clip(existing[1] + np.random.randint(-5, 6), 0, self.size - 1)
            else:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
            
            if self.grid[x, y] == CellType.EMPTY:
                self.grid[x, y] = CellType.RESOURCE
                self.resources[(x, y)] = Resource(x=x, y=y)
                placed += 1
    
    def _place_predators(self):
        """Place predators"""
        for _ in range(self.n_predators):
            while True:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                
                if self.grid[x, y] == CellType.EMPTY:
                    predator = Predator(x=x, y=y)
                    self.predators.append(predator)
                    self.grid[x, y] = CellType.PREDATOR
                    break
    
    def add_agent(self, agent_id: str, position: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Add an agent to the world
        
        Args:
            agent_id: Unique agent identifier
            position: Optional specific position, random if None
            
        Returns:
            (x, y) position where agent was placed
        """
        if position is not None:
            x, y = position
            if self.grid[x, y] == CellType.EMPTY:
                self.grid[x, y] = CellType.AGENT
                self.agent_positions[agent_id] = (x, y)
                return (x, y)
        
        # Find random empty position
        attempts = 0
        while attempts < 1000:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            if self.grid[x, y] == CellType.EMPTY:
                self.grid[x, y] = CellType.AGENT
                self.agent_positions[agent_id] = (x, y)
                return (x, y)
            
            attempts += 1
        
        raise RuntimeError("Could not find empty position for agent")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the world"""
        if agent_id in self.agent_positions:
            x, y = self.agent_positions[agent_id]
            self.grid[x, y] = CellType.EMPTY
            del self.agent_positions[agent_id]
            self.total_agent_deaths += 1
    
    def get_local_observation(self, agent_id: str, vision_size: int = 9) -> np.ndarray:
        """
        Get local grid observation for an agent
        
        Args:
            agent_id: Agent identifier
            vision_size: Size of vision grid (must be odd)
            
        Returns:
            (vision_size, vision_size, n_channels) observation
            Channels: [empty, resource, obstacle, predator, other_agent]
        """
        if agent_id not in self.agent_positions:
            return np.zeros((vision_size, vision_size, 5))
        
        x, y = self.agent_positions[agent_id]
        half = vision_size // 2
        
        # Create observation grid
        obs = np.zeros((vision_size, vision_size, 5), dtype=np.float32)
        
        for i in range(vision_size):
            for j in range(vision_size):
                gx = x - half + i
                gy = y - half + j
                
                # Check bounds
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    cell = self.grid[gx, gy]
                    
                    if cell == CellType.EMPTY:
                        obs[i, j, 0] = 1.0
                    elif cell == CellType.RESOURCE:
                        obs[i, j, 1] = 1.0
                    elif cell == CellType.OBSTACLE:
                        obs[i, j, 2] = 1.0
                    elif cell == CellType.PREDATOR:
                        obs[i, j, 3] = 1.0
                    elif cell == CellType.AGENT:
                        if (gx, gy) != (x, y):  # Other agent
                            obs[i, j, 4] = 1.0
                else:
                    # Out of bounds = obstacle
                    obs[i, j, 2] = 1.0
        
        return obs
    
    def step_agent(self, agent_id: str, action: int) -> Dict:
        """
        Execute agent action
        
        Actions:
            0: Stay
            1: Up (y+1)
            2: Down (y-1)
            3: Left (x-1)
            4: Right (x+1)
            5: Consume (if on resource)
            
        Returns:
            result dict with: moved, consumed, energy_gained, hit_predator, new_position
        """
        if agent_id not in self.agent_positions:
            return {'valid': False, 'reason': 'agent_not_found'}
        
        x, y = self.agent_positions[agent_id]
        result = {
            'valid': True,
            'moved': False,
            'consumed': False,
            'energy_gained': 0.0,
            'hit_predator': False,
            'hit_obstacle': False,
            'new_position': (x, y)
        }
        
        # Movement actions
        dx, dy = 0, 0
        if action == 1:  # Up
            dy = 1
        elif action == 2:  # Down
            dy = -1
        elif action == 3:  # Left
            dx = -1
        elif action == 4:  # Right
            dx = 1
        elif action == 5:  # Consume
            # Check for resource at current position
            if (x, y) in self.resources:
                resource = self.resources[(x, y)]
                result['consumed'] = True
                result['energy_gained'] = resource.energy
                del self.resources[(x, y)]
                self.grid[x, y] = CellType.AGENT  # Remains agent position
                self.total_resources_consumed += 1
            return result
        
        if action == 0:  # Stay
            return result
        
        # Calculate new position
        new_x = np.clip(x + dx, 0, self.size - 1)
        new_y = np.clip(y + dy, 0, self.size - 1)
        
        # Check collision
        target_cell = self.grid[new_x, new_y]
        
        if target_cell == CellType.OBSTACLE:
            result['hit_obstacle'] = True
            return result
        
        if target_cell == CellType.PREDATOR:
            result['hit_predator'] = True
            # Agent dies - handled by caller
            return result
        
        # Move agent
        self.grid[x, y] = CellType.EMPTY
        
        # Check if moving onto resource
        if target_cell == CellType.RESOURCE:
            # Move onto resource cell (can consume next turn)
            pass
        
        self.grid[new_x, new_y] = CellType.AGENT
        self.agent_positions[agent_id] = (new_x, new_y)
        result['moved'] = True
        result['new_position'] = (new_x, new_y)
        
        return result
    
    def step_world(self):
        """
        Advance world state by one step
        
        Updates:
        - Predator movements
        - Resource respawning
        - Statistics
        """
        self.step_count += 1
        
        # Move predators
        self._update_predators()
        
        # Respawn resources
        self._respawn_resources()
    
    def _update_predators(self):
        """Update predator positions"""
        for predator in self.predators:
            if random.random() > predator.speed:
                continue  # Predator doesn't move this step
            
            # Find nearest agent within vision
            nearest_agent = None
            min_dist = float('inf')
            
            for agent_id, (ax, ay) in self.agent_positions.items():
                dist = abs(ax - predator.x) + abs(ay - predator.y)  # Manhattan
                if dist < min_dist and dist <= predator.vision_range:
                    min_dist = dist
                    nearest_agent = (ax, ay)
            
            # Clear old position
            self.grid[predator.x, predator.y] = CellType.EMPTY
            
            if nearest_agent is not None:
                # Chase nearest agent
                new_x, new_y = predator.move_towards(
                    nearest_agent[0], nearest_agent[1], self.size
                )
            else:
                # Random walk
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                new_x = np.clip(predator.x + dx, 0, self.size - 1)
                new_y = np.clip(predator.y + dy, 0, self.size - 1)
            
            # Check if new position is valid
            if self.grid[new_x, new_y] in [CellType.EMPTY, CellType.RESOURCE]:
                predator.x = new_x
                predator.y = new_y
            
            # Mark new position
            self.grid[predator.x, predator.y] = CellType.PREDATOR
    
    def _respawn_resources(self):
        """Respawn resources"""
        # Random spawn
        if random.random() < self.resource_respawn_rate:
            # Find empty position
            attempts = 0
            while attempts < 50:
                if random.random() < self.resource_cluster_prob and len(self.resources) > 0:
                    # Near existing resource
                    existing = random.choice(list(self.resources.keys()))
                    x = np.clip(existing[0] + np.random.randint(-5, 6), 0, self.size - 1)
                    y = np.clip(existing[1] + np.random.randint(-5, 6), 0, self.size - 1)
                else:
                    x = np.random.randint(0, self.size)
                    y = np.random.randint(0, self.size)
                
                if self.grid[x, y] == CellType.EMPTY:
                    self.grid[x, y] = CellType.RESOURCE
                    self.resources[(x, y)] = Resource(x=x, y=y)
                    break
                
                attempts += 1
    
    def get_agent_distances_to_resources(self, agent_id: str) -> List[float]:
        """Get sorted list of distances to all resources"""
        if agent_id not in self.agent_positions:
            return []
        
        x, y = self.agent_positions[agent_id]
        distances = []
        
        for (rx, ry) in self.resources.keys():
            dist = np.sqrt((rx - x)**2 + (ry - y)**2)
            distances.append(dist)
        
        return sorted(distances)
    
    def get_agent_distances_to_predators(self, agent_id: str) -> List[float]:
        """Get sorted list of distances to all predators"""
        if agent_id not in self.agent_positions:
            return []
        
        x, y = self.agent_positions[agent_id]
        distances = []
        
        for predator in self.predators:
            dist = np.sqrt((predator.x - x)**2 + (predator.y - y)**2)
            distances.append(dist)
        
        return sorted(distances)
    
    def get_world_state(self) -> Dict:
        """Get current world state summary"""
        return {
            'step': self.step_count,
            'n_agents': len(self.agent_positions),
            'n_resources': len(self.resources),
            'n_predators': len(self.predators),
            'total_consumed': self.total_resources_consumed,
            'total_deaths': self.total_agent_deaths
        }
    
    def render_grid(self) -> np.ndarray:
        """
        Get grid for visualization
        
        Returns:
            RGB array (size, size, 3) with values 0-255
        """
        # Color map
        colors = {
            CellType.EMPTY: [240, 240, 240],      # Light gray
            CellType.RESOURCE: [0, 200, 0],        # Green
            CellType.OBSTACLE: [100, 100, 100],    # Dark gray
            CellType.PREDATOR: [200, 0, 0],        # Red
            CellType.AGENT: [0, 0, 200],           # Blue
        }
        
        rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                rgb[i, j] = colors.get(cell, [0, 0, 0])
        
        return rgb


# =====================
# Testing
# =====================

if __name__ == "__main__":
    print("=" * 70)
    print("GridWorld Test")
    print("=" * 70)
    
    # Create world
    world = GridWorld(
        size=100,
        n_resources=200,
        n_predators=5,
        n_obstacles=50,
        seed=42
    )
    
    # Add test agents
    for i in range(5):
        pos = world.add_agent(f"agent_{i}")
        print(f"Agent {i} placed at {pos}")
    
    # Run simulation
    print("\nRunning 100 steps...")
    for step in range(100):
        # Each agent takes random action
        for agent_id in list(world.agent_positions.keys()):
            action = np.random.randint(0, 6)
            result = world.step_agent(agent_id, action)
            
            if result.get('hit_predator'):
                print(f"  Step {step}: {agent_id} hit predator!")
                world.remove_agent(agent_id)
        
        # Update world
        world.step_world()
        
        if step % 20 == 0:
            state = world.get_world_state()
            print(f"Step {step}: agents={state['n_agents']}, "
                  f"resources={state['n_resources']}, consumed={state['total_consumed']}")
    
    print("\nFinal state:")
    print(world.get_world_state())
    
    print("\n" + "=" * 70)
    print("GridWorld test complete!")
    print("=" * 70)
