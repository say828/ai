"""
GENESIS Path B Phase 0: Minimal Autopoietic Agent

Simple RNN-based agent with coherence-based viability
Based on: v2.0/core/autopoietic_entity.py
"""

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import deque
import uuid

if TYPE_CHECKING:
    from minimal_environment import MinimalGrid


class MinimalAutopoieticAgent:
    """
    Simplified Autopoietic Agent
    
    Core Principles:
    - Internal dynamics drive behavior (not external rewards)
    - Coherence = organizational maintenance
    - Survival = maintaining coherence above threshold
    """
    
    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: Optional[str] = None,
                 state_dim: int = 5,
                 sensor_dim: int = 18,
                 action_dim: int = 3):
        """
        Args:
            x, y: Initial position
            agent_id: Unique identifier
            state_dim: Internal state dimension
            sensor_dim: Sensor input dimension
            action_dim: Action output dimension (dx, dy, consume)
        """
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.x = x
        self.y = y
        
        # Dimensions
        self.state_dim = state_dim
        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        
        # Internal state (RNN hidden state)
        self.state = np.random.randn(state_dim) * 0.1
        
        # Network weights
        # W_rec: recurrent weights (state -> state)
        self.W_rec = np.random.randn(state_dim, state_dim) * 0.3
        # W_in: input weights (sensor -> state)
        self.W_in = np.random.randn(state_dim, sensor_dim) * 0.2
        # W_out: output weights (state -> action)
        self.W_out = np.random.randn(action_dim, state_dim) * 0.3
        
        # State history for coherence computation
        self.state_history = deque(maxlen=50)
        
        # Vital statistics
        self.energy = 1.0  # Start with full energy
        self.age = 0
        self.is_alive = True
        
        # Coherence thresholds
        self.death_threshold = 0.3
        self.reproduction_threshold = 0.6
        
        # Statistics
        self.total_consumed = 0.0
        self.coherence_history = deque(maxlen=100)
        
    def sense(self, env: 'MinimalGrid') -> np.ndarray:
        """
        Get sensor input from environment
        
        Args:
            env: Environment
            
        Returns:
            sensor_input: shape (18,)
        """
        return env.get_local_view(self.x, self.y)
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """
        Forward pass through RNN
        
        state_t+1 = tanh(W_rec @ state_t + W_in @ sensor)
        action = tanh(W_out @ state_t+1)
        
        Args:
            sensor_input: shape (sensor_dim,)
            
        Returns:
            action: shape (3,) - [dx, dy, consume_intensity]
        """
        # RNN step
        new_state = np.tanh(
            self.W_rec @ self.state + 
            self.W_in @ sensor_input
        )
        
        # Update state
        self.state = new_state
        self.state_history.append(self.state.copy())
        
        # Compute action
        action = np.tanh(self.W_out @ self.state)
        
        return action
    
    def compute_coherence(self) -> Dict[str, float]:
        """
        Compute organizational coherence
        
        Based on CoherenceAssessor from autopoietic_entity.py
        
        Components:
            1. Predictability: Low variance in state changes
            2. Stability: Low variance in recent states
            3. Complexity: Moderate state variance (not too ordered, not too chaotic)
            4. Circularity: Autocorrelation (recurrent patterns)
        
        Returns:
            coherence: Dict with component scores and composite
        """
        if len(self.state_history) < 10:
            default = {
                'predictability': 0.5,
                'stability': 0.5,
                'complexity': 0.5,
                'circularity': 0.5,
                'composite': 0.5
            }
            return default
        
        states = np.array(list(self.state_history))
        
        # 1. Predictability: low variance in state changes
        state_changes = np.diff(states, axis=0)
        var_changes = np.mean(np.var(state_changes, axis=0))
        predictability = 1.0 / (1.0 + var_changes)
        
        # 2. Stability: low variance in recent states
        recent_states = states[-20:] if len(states) >= 20 else states
        stability = 1.0 / (1.0 + np.std(recent_states))
        
        # 3. Complexity: optimal variance around 0.3-0.5
        state_variance = np.var(states)
        # Bell curve around 0.4 variance
        complexity = np.exp(-((state_variance - 0.4) ** 2) / 0.2)
        
        # 4. Circularity: autocorrelation
        if len(states) >= 15:
            try:
                flat_early = states[:-5].flatten()
                flat_late = states[5:].flatten()
                if len(flat_early) == len(flat_late) and np.std(flat_early) > 0 and np.std(flat_late) > 0:
                    corr = np.corrcoef(flat_early, flat_late)[0, 1]
                    circularity = abs(corr) if not np.isnan(corr) else 0.5
                else:
                    circularity = 0.5
            except:
                circularity = 0.5
        else:
            circularity = 0.5
        
        # Composite score
        composite = (
            0.3 * predictability +
            0.3 * stability +
            0.2 * complexity +
            0.2 * circularity
        )
        
        coherence = {
            'predictability': float(np.clip(predictability, 0, 1)),
            'stability': float(np.clip(stability, 0, 1)),
            'complexity': float(np.clip(complexity, 0, 1)),
            'circularity': float(np.clip(circularity, 0, 1)),
            'composite': float(np.clip(composite, 0, 1))
        }
        
        self.coherence_history.append(coherence['composite'])
        
        return coherence
    
    def update_energy(self, consumed: float, action: np.ndarray, coherence: float = 0.5) -> None:
        """
        Update energy based on consumption, action, and coherence

        Key insight: Lower coherence = higher metabolic cost
        This creates selection pressure for coherent agents

        Args:
            consumed: Resources consumed
            action: Action taken
            coherence: Current coherence score
        """
        base_cost = 0.005  # Reduced base existence cost
        action_cost = 0.01 * np.sum(np.abs(action[:2]))  # Reduced movement cost

        # COHERENCE-DEPENDENT COST: Low coherence = high cost
        # coherence in [0,1], cost multiplier in [0.5, 2.0]
        coherence_multiplier = 2.0 - 1.5 * coherence  # 0.5 at coherence=1, 2.0 at coherence=0
        total_cost = (base_cost + action_cost) * coherence_multiplier

        self.energy += consumed - total_cost
        self.total_consumed += consumed

        # Clamp energy
        self.energy = min(self.energy, 2.0)  # Max energy cap
    
    def can_reproduce(self) -> bool:
        """
        Check if agent can reproduce
        
        Conditions:
        - Coherence > 0.6
        - Age > 50
        - Energy > 0.5
        """
        if len(self.coherence_history) == 0:
            return False
            
        recent_coherence = np.mean(list(self.coherence_history)[-10:])
        return (
            recent_coherence > self.reproduction_threshold and
            self.age > 50 and
            self.energy > 0.5
        )
    
    def should_die(self) -> bool:
        """
        Check if agent should die
        
        Conditions:
        - Coherence < 0.3
        - OR Energy < 0
        """
        if self.energy < 0:
            return True
            
        if len(self.coherence_history) >= 5:
            recent_coherence = np.mean(list(self.coherence_history)[-5:])
            if recent_coherence < self.death_threshold:
                return True
                
        return False
    
    def mutate_child(self, mutation_rate: float = 0.1) -> 'MinimalAutopoieticAgent':
        """
        Create mutated offspring
        
        Args:
            mutation_rate: Standard deviation of Gaussian noise
            
        Returns:
            child: New agent with mutated weights
        """
        # Create child at same position (will be moved by population manager)
        child = MinimalAutopoieticAgent(
            x=self.x,
            y=self.y,
            state_dim=self.state_dim,
            sensor_dim=self.sensor_dim,
            action_dim=self.action_dim
        )
        
        # Copy weights with mutation
        child.W_rec = self.W_rec + np.random.randn(*self.W_rec.shape) * mutation_rate
        child.W_in = self.W_in + np.random.randn(*self.W_in.shape) * mutation_rate
        child.W_out = self.W_out + np.random.randn(*self.W_out.shape) * mutation_rate
        
        # Child starts with half parent's energy
        child.energy = self.energy / 2
        self.energy = self.energy / 2  # Parent also loses energy
        
        return child
    
    def act(self, env: 'MinimalGrid') -> Dict:
        """
        Complete action cycle: sense -> process -> act

        Args:
            env: Environment

        Returns:
            result: Action results
        """
        # Sense
        sensor = self.sense(env)

        # Process (forward pass)
        action = self.forward(sensor)

        # Execute action
        # Movement: discretize dx, dy to {-1, 0, 1}
        dx = int(np.round(action[0]))
        dy = int(np.round(action[1]))
        dx = np.clip(dx, -1, 1)
        dy = np.clip(dy, -1, 1)

        # Update position (with wrapping)
        self.x = (self.x + dx) % env.size
        self.y = (self.y + dy) % env.size

        # Consumption: consume_intensity in [0, 1]
        consume_intensity = (action[2] + 1) / 2  # Map from [-1,1] to [0,1]
        consumed = env.consume(self.x, self.y, consume_intensity * 0.3)  # Max 0.3 per step

        # Get current coherence for energy update
        current_coherence = self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5

        # Update energy (coherence affects metabolic cost)
        self.update_energy(consumed, action, current_coherence)

        # Increment age
        self.age += 1

        return {
            'action': action,
            'dx': dx,
            'dy': dy,
            'consumed': consumed,
            'position': (self.x, self.y)
        }
    
    def get_summary(self) -> Dict:
        """Get agent summary"""
        return {
            'id': self.id,
            'position': (self.x, self.y),
            'age': self.age,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5,
            'avg_coherence': np.mean(list(self.coherence_history)) if len(self.coherence_history) > 0 else 0.5,
            'total_consumed': self.total_consumed
        }


if __name__ == "__main__":
    from minimal_environment import MinimalGrid
    
    print("Testing MinimalAutopoieticAgent...")
    
    # Create environment and agent
    env = MinimalGrid(size=16)
    agent = MinimalAutopoieticAgent(x=8, y=8)
    
    print(f"Agent ID: {agent.id}")
    print(f"Initial position: ({agent.x}, {agent.y})")
    print(f"Initial energy: {agent.energy}")
    
    # Run for 100 steps
    for step in range(100):
        result = agent.act(env)
        coherence = agent.compute_coherence()
        env.step()
        
        if step % 20 == 0:
            print(f"Step {step}: pos=({agent.x},{agent.y}), "
                  f"energy={agent.energy:.3f}, "
                  f"coherence={coherence['composite']:.3f}")
    
    print(f"\nFinal summary:")
    summary = agent.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Test reproduction
    print(f"\nCan reproduce: {agent.can_reproduce()}")
    
    if agent.energy > 0.3:
        child = agent.mutate_child()
        print(f"Child created: {child.id}")
        print(f"Parent energy after: {agent.energy:.3f}")
        print(f"Child energy: {child.energy:.3f}")
    
    print("\nAgent test passed!")
