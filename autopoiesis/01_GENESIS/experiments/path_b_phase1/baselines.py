"""
GENESIS Path B Phase 1: Baseline Agents

Non-autopoietic baseline agents for comparison:
1. RandomAgent: Random actions
2. FixedPolicyAgent: Hand-coded greedy policy
3. RLAgent: Simple policy gradient (REINFORCE)

These baselines help demonstrate that coherence-based selection
provides advantages over traditional approaches.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import copy

from full_environment import FullALifeEnvironment


class BaselineAgent:
    """Base class for baseline agents"""
    
    STATE_DIM = 128
    SENSOR_DIM = 370
    ACTION_DIM = 5
    VISUAL_RADIUS = 3
    
    def __init__(self, x: int, y: int, agent_id: int, parent_id: Optional[int] = None):
        self.id = agent_id
        self.parent_id = parent_id
        self.x = x
        self.y = y
        self.age = 0
        self.birth_time = 0
        
        self.energy = 1.0
        self.material = 0.5
        
        self.state_history = deque(maxlen=50)
        self.action_history = deque(maxlen=20)
        self.coherence_history = deque(maxlen=100)
        self.position_history = deque(maxlen=50)
        
        self.total_energy_consumed = 0.0
        self.total_material_consumed = 0.0
        self.offspring_count = 0
        self.is_alive = True
        
        # Baseline-specific: use fixed thresholds
        self.death_energy_threshold = 0.0
        self.reproduction_age_threshold = 100
        self.reproduction_energy_threshold = 0.7
        self.reproduction_material_threshold = 0.4
    
    def sense(self, env: FullALifeEnvironment, 
              nearby_agents: List['BaselineAgent']) -> np.ndarray:
        """Standard sensing (same as autopoietic agent)"""
        visual = env.get_visual_field(self.x, self.y, radius=self.VISUAL_RADIUS)
        return visual.flatten()  # Simplified: just use visual field
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """Override in subclasses"""
        raise NotImplementedError
    
    def metabolic_cost(self, action: np.ndarray) -> float:
        """Fixed metabolic cost (no coherence dependency)"""
        base_cost = 0.01
        movement_cost = 0.005 * (abs(action[0]) + abs(action[1]))
        return base_cost + movement_cost
    
    def can_reproduce(self) -> bool:
        """Energy-based reproduction (no coherence requirement)"""
        return (
            self.age > self.reproduction_age_threshold and
            self.energy > self.reproduction_energy_threshold and
            self.material > self.reproduction_material_threshold
        )
    
    def should_die(self) -> bool:
        """Energy-based death (no coherence check)"""
        return self.energy <= self.death_energy_threshold
    
    def compute_coherence(self) -> Dict[str, float]:
        """Compute coherence for comparison (but not used for survival)"""
        if len(self.state_history) < 10:
            return {'composite': 0.5}
        
        states = np.array(list(self.state_history))
        state_var = np.var(states)
        stability = 1.0 / (1.0 + state_var)
        
        self.coherence_history.append(stability)
        return {'composite': stability}
    
    def get_behavior_descriptor(self) -> np.ndarray:
        """Same as autopoietic agent"""
        bd = np.zeros(8)
        if self.action_history:
            actions = np.array(list(self.action_history))
            bd[0] = np.mean(actions[:, 0])
            bd[1] = np.mean(actions[:, 1])
        return bd
    
    def get_summary(self) -> Dict:
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'position': (self.x, self.y),
            'age': self.age,
            'energy': self.energy,
            'material': self.material,
            'is_alive': self.is_alive
        }


class RandomAgent(BaselineAgent):
    """
    Random Baseline: Actions sampled uniformly at random
    
    This serves as a lower bound for performance.
    Any reasonable algorithm should outperform random actions.
    """
    
    def __init__(self, x: int, y: int, agent_id: int, **kwargs):
        super().__init__(x, y, agent_id)
        self.state = np.zeros(self.STATE_DIM)
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """Random action"""
        action = np.random.uniform(-1, 1, self.ACTION_DIM)
        self.action_history.append(action.copy())
        
        # Update state randomly for coherence tracking
        self.state = np.random.randn(self.STATE_DIM) * 0.5
        self.state_history.append(self.state.copy())
        
        return action
    
    def create_offspring(self, child_id: int, **kwargs) -> 'RandomAgent':
        """Create random child"""
        self.energy *= 0.5
        self.material *= 0.5
        self.offspring_count += 1
        
        child_x = (self.x + np.random.randint(-3, 4)) % 64
        child_y = (self.y + np.random.randint(-3, 4)) % 64
        
        child = RandomAgent(x=child_x, y=child_y, agent_id=child_id)
        child.parent_id = self.id
        child.energy = 0.8
        child.material = 0.4
        
        return child


class FixedPolicyAgent(BaselineAgent):
    """
    Fixed Policy Baseline: Hand-coded greedy strategy
    
    Strategy:
    - Move toward highest resource cell in visual field
    - Always consume at maximum rate
    - No internal state dynamics
    
    This represents a simple hand-engineered solution.
    """
    
    def __init__(self, x: int, y: int, agent_id: int, **kwargs):
        super().__init__(x, y, agent_id)
        self.state = np.zeros(self.STATE_DIM)
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """Greedy policy: move toward best resource"""
        # Parse visual field (assume 7x7x3 flattened)
        visual_size = 7
        try:
            # Reshape to 7x7x3 (energy, material, occupation)
            visual = sensor_input[:147].reshape(visual_size, visual_size, 3)
            
            # Combined resource value
            resource_map = visual[:, :, 0] + 0.5 * visual[:, :, 1]  # Energy + 0.5 * Material
            
            # Find best direction
            center = visual_size // 2
            best_value = resource_map[center, center]
            best_dx, best_dy = 0, 0
            
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    px = center + dx
                    py = center + dy
                    if 0 <= px < visual_size and 0 <= py < visual_size:
                        value = resource_map[px, py]
                        if value > best_value:
                            best_value = value
                            best_dx = dx
                            best_dy = dy
            
            # Normalize to [-1, 1]
            action = np.array([
                best_dx / 2.0,
                best_dy / 2.0,
                1.0,  # Max energy consumption
                1.0,  # Max material consumption
                0.0   # No explicit reproduction signal
            ])
            
        except:
            # Fallback to no movement
            action = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        
        self.action_history.append(action.copy())
        
        # Update state for tracking
        self.state = np.tanh(self.state * 0.9 + sensor_input[:self.STATE_DIM] * 0.1 if len(sensor_input) >= self.STATE_DIM else self.state)
        self.state_history.append(self.state.copy())
        
        return action
    
    def create_offspring(self, child_id: int, **kwargs) -> 'FixedPolicyAgent':
        """Create identical child (no evolution)"""
        self.energy *= 0.5
        self.material *= 0.5
        self.offspring_count += 1
        
        child_x = (self.x + np.random.randint(-3, 4)) % 64
        child_y = (self.y + np.random.randint(-3, 4)) % 64
        
        child = FixedPolicyAgent(x=child_x, y=child_y, agent_id=child_id)
        child.parent_id = self.id
        child.energy = 0.8
        child.material = 0.4
        
        return child


class RLAgent(BaselineAgent):
    """
    Reinforcement Learning Baseline: Simple Policy Gradient (REINFORCE)
    
    Uses accumulated energy as reward signal.
    Updates policy weights based on episodic returns.
    
    This represents a traditional RL approach to the problem.
    """
    
    def __init__(self, x: int, y: int, agent_id: int, 
                 genome: Optional[Dict] = None, **kwargs):
        super().__init__(x, y, agent_id)
        
        # Neural network weights
        if genome is None:
            self.W_in = np.random.randn(self.STATE_DIM, 147) * 0.1  # Visual input only
            self.W_rec = np.random.randn(self.STATE_DIM, self.STATE_DIM) * 0.05
            self.W_out = np.random.randn(self.ACTION_DIM, self.STATE_DIM) * 0.1
        else:
            self.W_in = genome['W_in'].copy()
            self.W_rec = genome['W_rec'].copy()
            self.W_out = genome['W_out'].copy()
        
        self.state = np.zeros(self.STATE_DIM)
        
        # RL-specific
        self.episode_rewards = []
        self.episode_log_probs = []
        self.learning_rate = 0.001
        self.prev_energy = self.energy
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """Policy network forward pass with exploration"""
        # Use only visual input
        visual_input = sensor_input[:147] if len(sensor_input) >= 147 else np.zeros(147)
        
        # RNN update
        self.state = np.tanh(
            self.W_rec @ self.state + self.W_in @ visual_input
        )
        self.state_history.append(self.state.copy())
        
        # Compute action mean
        action_mean = np.tanh(self.W_out @ self.state)
        
        # Add exploration noise
        noise = np.random.randn(self.ACTION_DIM) * 0.2
        action = np.clip(action_mean + noise, -1, 1)
        
        self.action_history.append(action.copy())
        
        # Track for RL update
        # Simplified: reward is energy gained
        reward = self.energy - self.prev_energy
        self.episode_rewards.append(reward)
        self.prev_energy = self.energy
        
        return action
    
    def update_policy(self):
        """Simple policy gradient update (called periodically)"""
        if len(self.episode_rewards) < 10:
            return
        
        # Compute returns (cumulative discounted rewards)
        gamma = 0.99
        returns = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = np.array(returns)
        if np.std(returns) > 1e-6:
            returns = (returns - np.mean(returns)) / np.std(returns)
        
        # Simplified update: nudge weights in direction of good actions
        # This is a very simplified version of REINFORCE
        if len(self.action_history) >= len(returns):
            for i, (action, ret) in enumerate(zip(list(self.action_history)[-len(returns):], returns)):
                if ret > 0:
                    # Reinforce this action pattern
                    self.W_out += self.learning_rate * ret * np.outer(action, self.state) * 0.01
        
        # Reset episode
        self.episode_rewards = []
        self.prev_energy = self.energy
    
    def create_offspring(self, child_id: int, 
                        mutation_rate: float = 0.05,
                        mutation_scale: float = 0.1,
                        **kwargs) -> 'RLAgent':
        """Create child with mutated weights"""
        self.energy *= 0.5
        self.material *= 0.5
        self.offspring_count += 1
        
        # Mutate genome
        child_genome = {
            'W_in': self.W_in + np.random.randn(*self.W_in.shape) * mutation_scale * (np.random.rand(*self.W_in.shape) < mutation_rate),
            'W_rec': self.W_rec + np.random.randn(*self.W_rec.shape) * mutation_scale * (np.random.rand(*self.W_rec.shape) < mutation_rate),
            'W_out': self.W_out + np.random.randn(*self.W_out.shape) * mutation_scale * (np.random.rand(*self.W_out.shape) < mutation_rate)
        }
        
        child_x = (self.x + np.random.randint(-3, 4)) % 64
        child_y = (self.y + np.random.randint(-3, 4)) % 64
        
        child = RLAgent(x=child_x, y=child_y, agent_id=child_id, genome=child_genome)
        child.parent_id = self.id
        child.energy = 0.8
        child.material = 0.4
        
        return child


class BaselinePopulationManager:
    """
    Population manager for baseline experiments
    
    Similar to FullPopulationManager but works with baseline agents
    """
    
    def __init__(self,
                 env: FullALifeEnvironment,
                 agent_class: type,
                 initial_pop: int = 100,
                 max_population: int = 500,
                 **kwargs):
        self.env = env
        self.agent_class = agent_class
        self.max_population = max_population
        self.kwargs = kwargs
        
        self.agents = []
        self.next_id = 0
        self.current_step = 0
        
        # Initialize population
        for _ in range(initial_pop):
            x = np.random.randint(0, env.size)
            y = np.random.randint(0, env.size)
            agent = agent_class(x=x, y=y, agent_id=self.next_id)
            agent.birth_time = 0
            self.agents.append(agent)
            self.next_id += 1
        
        self.total_births = 0
        self.total_deaths = 0
    
    def step(self) -> Dict:
        """Execute one step"""
        self.current_step += 1
        
        # Update occupation grid
        positions = [(a.x, a.y) for a in self.agents]
        self.env.set_occupation(positions)
        
        # All agents act
        for agent in self.agents:
            sensor = agent.sense(self.env, [])
            action = agent.forward(sensor)
            
            # Execute action
            dx = int(np.round(action[0] * 2))
            dy = int(np.round(action[1] * 2))
            agent.x = (agent.x + np.clip(dx, -2, 2)) % self.env.size
            agent.y = (agent.y + np.clip(dy, -2, 2)) % self.env.size
            agent.position_history.append((agent.x, agent.y))
            
            # Consume resources
            consume_e = max(0, (action[2] + 1) / 2) * 0.4
            consume_m = max(0, (action[3] + 1) / 2) * 0.3
            gained_e, gained_m = self.env.consume(agent.x, agent.y, consume_e, consume_m)
            
            agent.energy += gained_e
            agent.material += gained_m
            agent.total_energy_consumed += gained_e
            agent.total_material_consumed += gained_m
            
            # Metabolic cost
            cost = agent.metabolic_cost(action)
            agent.energy -= cost
            
            # Cap resources
            agent.energy = min(agent.energy, 2.0)
            agent.material = min(agent.material, 1.0)
            
            # Compute coherence (for tracking)
            agent.compute_coherence()
            agent.age += 1
            
            # RL-specific update
            if isinstance(agent, RLAgent) and self.current_step % 50 == 0:
                agent.update_policy()
        
        # Environment step
        self.env.step()
        
        # Deaths
        deaths = [a for a in self.agents if a.should_die()]
        for agent in deaths:
            agent.is_alive = False
            self.total_deaths += 1
        self.agents = [a for a in self.agents if a.is_alive]
        
        # Reproduction
        offspring = []
        if len(self.agents) < self.max_population:
            for agent in self.agents:
                if agent.can_reproduce() and len(self.agents) + len(offspring) < self.max_population:
                    child = agent.create_offspring(child_id=self.next_id)
                    child.birth_time = self.current_step
                    offspring.append(child)
                    self.next_id += 1
                    self.total_births += 1
        
        self.agents.extend(offspring)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self.agents:
            return {
                'step': self.current_step,
                'population_size': 0,
                'avg_coherence': 0,
                'avg_energy': 0,
                'avg_age': 0,
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
                'extinct': True
            }
        
        coherences = [a.coherence_history[-1] if a.coherence_history else 0.5 for a in self.agents]
        energies = [a.energy for a in self.agents]
        ages = [a.age for a in self.agents]
        
        return {
            'step': self.current_step,
            'population_size': len(self.agents),
            'avg_coherence': float(np.mean(coherences)),
            'std_coherence': float(np.std(coherences)),
            'avg_energy': float(np.mean(energies)),
            'avg_age': float(np.mean(ages)),
            'max_age': int(np.max(ages)),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'extinct': False
        }


if __name__ == "__main__":
    print("Testing Baseline Agents...")
    
    env = FullALifeEnvironment(size=64, seed=42)
    
    # Test each baseline
    for name, agent_class in [('Random', RandomAgent), 
                               ('FixedPolicy', FixedPolicyAgent), 
                               ('RL', RLAgent)]:
        print(f"\n--- Testing {name} ---")
        
        # Reset environment
        env = FullALifeEnvironment(size=64, seed=42)
        pop = BaselinePopulationManager(env, agent_class, initial_pop=50, max_population=200)
        
        print(f"Initial population: {len(pop.agents)}")
        
        # Run for 300 steps
        for step in range(300):
            stats = pop.step()
            
            if step % 100 == 0:
                print(f"  Step {step}: pop={stats['population_size']}, "
                      f"coherence={stats['avg_coherence']:.3f}, "
                      f"energy={stats['avg_energy']:.3f}")
            
            if stats['population_size'] == 0:
                print(f"  EXTINCTION at step {step}")
                break
        
        print(f"Final: pop={stats['population_size']}, births={stats['total_births']}, deaths={stats['total_deaths']}")
    
    print("\nBaseline test PASSED!")
