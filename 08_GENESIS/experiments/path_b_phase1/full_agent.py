"""
GENESIS Path B Phase 1: Full Autopoietic Agent

Complete agent with:
- 370-dimensional sensory input (visual field + proprioception + memory)
- 128-dimensional internal RNN state
- Continuous action space
- 4D coherence metric
- Coherence-based metabolic costs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
import copy

if TYPE_CHECKING:
    from full_environment import FullALifeEnvironment


class FullAutopoieticAgent:
    """
    Complete Autopoietic Agent
    
    Architecture:
    - Sensor: 370 dimensions
      - Visual field: 7x7x3 = 147 (energy, material, occupation)
      - Proprioception: 32 (position, energy, material, age, etc.)
      - Memory: 128 (previous internal state)
      - Gradient: 4 (resource gradients)
      - Temporal: 59 (recent action/coherence history)
    - Internal State: 128-dim RNN
    - Action: 5-dim continuous (dx, dy, consume_e, consume_m, reproduce_signal)
    
    Coherence Components:
    1. Predictability: Low entropy in state transitions
    2. Stability: Low variance in recent states
    3. Complexity: Moderate variance (edge of chaos)
    4. Circularity: Autocorrelation patterns
    """
    
    # Class-level configuration
    STATE_DIM = 128
    VISUAL_RADIUS = 3
    VISUAL_DIM = (2 * VISUAL_RADIUS + 1) ** 2 * 3  # 7x7x3 = 147
    PROPRIO_DIM = 32
    MEMORY_DIM = 128  # Previous state
    GRADIENT_DIM = 4
    TEMPORAL_DIM = 59  # Recent history features
    SENSOR_DIM = VISUAL_DIM + PROPRIO_DIM + MEMORY_DIM + GRADIENT_DIM + TEMPORAL_DIM  # 370
    ACTION_DIM = 5
    
    def __init__(self, 
                 x: int, 
                 y: int, 
                 agent_id: int,
                 genome: Optional[Dict] = None,
                 parent_id: Optional[int] = None):
        """
        Args:
            x, y: Initial position
            agent_id: Unique identifier
            genome: Evolvable weights (if None, initialize randomly)
            parent_id: Parent agent ID (for phylogeny tracking)
        """
        self.id = agent_id
        self.parent_id = parent_id
        self.x = x
        self.y = y
        self.age = 0
        self.birth_time = 0  # Will be set by population manager
        
        # Resources
        self.energy = 1.0
        self.material = 0.5
        
        # Internal state
        self.state = np.zeros(self.STATE_DIM)
        self.prev_state = np.zeros(self.STATE_DIM)
        
        # History for coherence and analysis
        self.state_history = deque(maxlen=50)
        self.action_history = deque(maxlen=20)
        self.coherence_history = deque(maxlen=100)
        self.position_history = deque(maxlen=50)
        
        # Genome (evolvable parameters)
        if genome is None:
            self.genome = self._initialize_genome()
        else:
            self.genome = copy.deepcopy(genome)
        
        # Extract weights from genome
        self._load_weights()
        
        # Statistics
        self.total_energy_consumed = 0.0
        self.total_material_consumed = 0.0
        self.total_distance_traveled = 0.0
        self.offspring_count = 0
        self.is_alive = True
        
        # Thresholds
        self.death_coherence_threshold = 0.25
        self.reproduction_coherence_threshold = 0.55
        self.reproduction_age_threshold = 100
        self.reproduction_energy_threshold = 0.7
        self.reproduction_material_threshold = 0.4
        
    def _initialize_genome(self) -> Dict:
        """Initialize evolvable parameters with Xavier-like initialization"""
        genome = {}
        
        # Input weights (sensor -> state)
        fan_in = self.SENSOR_DIM
        fan_out = self.STATE_DIM
        scale_in = np.sqrt(2.0 / (fan_in + fan_out))
        genome['W_in'] = np.random.randn(self.STATE_DIM, self.SENSOR_DIM) * scale_in
        
        # Recurrent weights (state -> state)
        scale_rec = np.sqrt(2.0 / (self.STATE_DIM * 2))
        genome['W_rec'] = np.random.randn(self.STATE_DIM, self.STATE_DIM) * scale_rec
        # Initialize closer to identity for stability
        genome['W_rec'] += np.eye(self.STATE_DIM) * 0.5
        
        # Output weights (state -> action)
        scale_out = np.sqrt(2.0 / (self.STATE_DIM + self.ACTION_DIM))
        genome['W_out'] = np.random.randn(self.ACTION_DIM, self.STATE_DIM) * scale_out
        
        # Biases
        genome['b_state'] = np.zeros(self.STATE_DIM)
        genome['b_action'] = np.zeros(self.ACTION_DIM)
        
        # Coherence sensitivity (how much coherence affects behavior)
        genome['coherence_sensitivity'] = np.random.uniform(0.5, 1.5)
        
        # Metabolic efficiency (evolvable)
        genome['metabolic_efficiency'] = np.random.uniform(0.8, 1.2)
        
        return genome
    
    def _load_weights(self):
        """Extract weights from genome for fast access"""
        self.W_in = self.genome['W_in']
        self.W_rec = self.genome['W_rec']
        self.W_out = self.genome['W_out']
        self.b_state = self.genome['b_state']
        self.b_action = self.genome['b_action']
        
    def sense(self, env: 'FullALifeEnvironment', 
              nearby_agents: List['FullAutopoieticAgent']) -> np.ndarray:
        """
        Construct 370-dimensional sensory input
        
        Components:
        - Visual field (147): 7x7x3 grid centered on agent
        - Proprioception (32): Internal state summary
        - Memory (128): Previous internal state
        - Gradient (4): Resource gradients
        - Temporal (59): Recent history features
        """
        sensor_parts = []
        
        # 1. Visual field (147 dim)
        visual = env.get_visual_field(self.x, self.y, radius=self.VISUAL_RADIUS)
        visual_flat = visual.flatten()
        sensor_parts.append(visual_flat)
        
        # 2. Proprioception (32 dim)
        proprio = np.zeros(self.PROPRIO_DIM)
        proprio[0] = self.x / env.size  # Normalized x
        proprio[1] = self.y / env.size  # Normalized y
        proprio[2] = self.energy / 2.0  # Normalized energy (max 2.0)
        proprio[3] = self.material      # Material (max 1.0)
        proprio[4] = min(1.0, self.age / 1000.0)  # Normalized age
        proprio[5] = len(nearby_agents) / 10.0  # Nearby density
        
        # Mean resources of nearby agents
        if nearby_agents:
            proprio[6] = np.mean([a.energy for a in nearby_agents]) / 2.0
            proprio[7] = np.mean([a.material for a in nearby_agents])
        
        # Recent coherence
        if self.coherence_history:
            proprio[8] = self.coherence_history[-1]
            proprio[9] = np.mean(list(self.coherence_history)[-5:]) if len(self.coherence_history) >= 5 else 0.5
        else:
            proprio[8] = 0.5
            proprio[9] = 0.5
        
        # Energy/material rates
        if len(self.position_history) > 5:
            recent_positions = list(self.position_history)[-5:]
            movement = sum(1 for i in range(1, len(recent_positions)) 
                         if recent_positions[i] != recent_positions[i-1])
            proprio[10] = movement / 4.0  # Movement rate
        
        # State statistics
        if self.state_history:
            recent_states = np.array(list(self.state_history)[-10:])
            proprio[11] = np.mean(recent_states)
            proprio[12] = np.std(recent_states)
            proprio[13] = np.max(np.abs(recent_states))
        
        # Action statistics
        if self.action_history:
            recent_actions = np.array(list(self.action_history)[-5:])
            proprio[14:19] = np.mean(recent_actions, axis=0) if len(recent_actions) > 0 else 0
        
        # Fill remaining with zeros (padding)
        sensor_parts.append(proprio)
        
        # 3. Memory (128 dim) - Previous internal state
        sensor_parts.append(self.prev_state.copy())
        
        # 4. Gradient (4 dim)
        gradient = env.get_local_gradient(self.x, self.y)
        sensor_parts.append(gradient)
        
        # 5. Temporal features (59 dim)
        temporal = np.zeros(self.TEMPORAL_DIM)
        
        # Coherence history features
        if len(self.coherence_history) >= 10:
            coh_arr = np.array(list(self.coherence_history)[-10:])
            temporal[0:10] = coh_arr
            temporal[10] = np.mean(coh_arr)
            temporal[11] = np.std(coh_arr)
            temporal[12] = coh_arr[-1] - coh_arr[0]  # Trend
        
        # State change features
        if len(self.state_history) >= 5:
            states = np.array(list(self.state_history)[-5:])
            state_changes = np.diff(states, axis=0)
            temporal[13:17] = np.mean(np.abs(state_changes), axis=0)[:4]  # First 4 dims
        
        sensor_parts.append(temporal)
        
        # Concatenate all parts
        sensor = np.concatenate(sensor_parts)
        
        # Ensure correct dimension
        assert len(sensor) == self.SENSOR_DIM, f"Sensor dim mismatch: {len(sensor)} vs {self.SENSOR_DIM}"
        
        return sensor
    
    def forward(self, sensor_input: np.ndarray) -> np.ndarray:
        """
        RNN forward pass
        
        state_t+1 = tanh(W_rec @ state_t + W_in @ sensor_t + b_state)
        action = tanh(W_out @ state_t+1 + b_action)
        
        Args:
            sensor_input: shape (SENSOR_DIM,)
            
        Returns:
            action: shape (ACTION_DIM,) - [dx, dy, consume_e, consume_m, reproduce]
        """
        # Store previous state
        self.prev_state = self.state.copy()
        
        # RNN update
        pre_activation = (
            self.W_rec @ self.state + 
            self.W_in @ sensor_input + 
            self.b_state
        )
        self.state = np.tanh(pre_activation)
        
        # Store in history
        self.state_history.append(self.state.copy())
        
        # Compute action
        action = np.tanh(self.W_out @ self.state + self.b_action)
        self.action_history.append(action.copy())
        
        return action
    
    def compute_coherence(self) -> Dict[str, float]:
        """
        Compute 4-dimensional organizational coherence
        
        Components:
        1. Predictability: Low entropy in state transitions (consistent behavior)
        2. Stability: Low variance in recent states (not chaotic)
        3. Complexity: Moderate variance (edge of chaos, not dead)
        4. Circularity: Temporal autocorrelation (self-maintaining patterns)
        
        Returns:
            Dictionary with component scores and composite
        """
        if len(self.state_history) < 10:
            default = {
                'predictability': 0.5,
                'stability': 0.5,
                'complexity': 0.5,
                'circularity': 0.5,
                'composite': 0.5
            }
            self.coherence_history.append(0.5)
            return default
        
        states = np.array(list(self.state_history))
        
        # 1. PREDICTABILITY: Low variance in state changes
        state_changes = np.diff(states, axis=0)
        var_changes = np.mean(np.var(state_changes, axis=0))
        predictability = 1.0 / (1.0 + var_changes * 5)  # Scale factor
        
        # 2. STABILITY: Low variance in recent states
        recent_states = states[-20:] if len(states) >= 20 else states
        state_variance = np.mean(np.var(recent_states, axis=0))
        stability = 1.0 / (1.0 + state_variance * 3)
        
        # 3. COMPLEXITY: Peak at moderate variance (~0.3-0.5)
        # Too low = dead/frozen, too high = chaotic
        total_variance = np.var(states)
        # Gaussian peak at 0.4, with sigma=0.2
        complexity = np.exp(-((total_variance - 0.4) ** 2) / (2 * 0.2**2))
        
        # 4. CIRCULARITY: Autocorrelation at lag 5
        if len(states) >= 15:
            try:
                early = states[:-5].flatten()
                late = states[5:].flatten()
                
                if np.std(early) > 1e-6 and np.std(late) > 1e-6:
                    corr = np.corrcoef(early, late)[0, 1]
                    circularity = max(0, corr) if not np.isnan(corr) else 0.5
                else:
                    circularity = 0.3  # Low but not zero for constant states
            except Exception:
                circularity = 0.5
        else:
            circularity = 0.5
        
        # Composite score (weighted average)
        composite = (
            0.30 * predictability +
            0.30 * stability +
            0.20 * complexity +
            0.20 * circularity
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
    
    def metabolic_cost(self, action: np.ndarray) -> float:
        """
        Compute metabolic cost based on coherence
        
        Key mechanism: Low coherence = high cost (wasteful)
                      High coherence = low cost (efficient)
        
        This creates selection pressure for coherent agents.
        """
        # Get recent coherence
        if self.coherence_history:
            coherence = np.mean(list(self.coherence_history)[-5:])
        else:
            coherence = 0.5
        
        # Base cost
        base_cost = 0.008
        
        # Movement cost (based on action magnitude)
        movement_cost = 0.003 * (abs(action[0]) + abs(action[1]))
        
        # Coherence multiplier: 2.0 at coherence=0, 0.5 at coherence=1
        coherence_multiplier = 2.0 - 1.5 * coherence
        
        # Apply metabolic efficiency from genome
        efficiency = self.genome.get('metabolic_efficiency', 1.0)
        
        total_cost = (base_cost + movement_cost) * coherence_multiplier / efficiency
        
        return total_cost
    
    def can_reproduce(self) -> bool:
        """
        Check reproduction conditions
        
        Requirements:
        - High coherence (> threshold)
        - Sufficient age
        - Sufficient energy and material
        """
        if len(self.coherence_history) < 10:
            return False
        
        recent_coherence = np.mean(list(self.coherence_history)[-10:])
        
        return (
            recent_coherence > self.reproduction_coherence_threshold and
            self.age > self.reproduction_age_threshold and
            self.energy > self.reproduction_energy_threshold and
            self.material > self.reproduction_material_threshold
        )
    
    def should_die(self) -> bool:
        """
        Check death conditions
        
        Death occurs when:
        - Energy depleted
        - Coherence too low for too long
        """
        if self.energy <= 0:
            return True
        
        if len(self.coherence_history) >= 10:
            recent_coherence = np.mean(list(self.coherence_history)[-10:])
            if recent_coherence < self.death_coherence_threshold:
                return True
        
        return False
    
    def create_offspring(self, child_id: int, 
                        mutation_rate: float = 0.05,
                        mutation_scale: float = 0.1) -> 'FullAutopoieticAgent':
        """
        Create mutated offspring
        
        Args:
            child_id: ID for the child
            mutation_rate: Probability of mutating each weight
            mutation_scale: Standard deviation of mutation
            
        Returns:
            New agent with mutated genome
        """
        # Reproduction cost
        self.energy *= 0.5
        self.material *= 0.5
        self.offspring_count += 1
        
        # Mutate genome
        child_genome = {}
        
        for key, value in self.genome.items():
            if isinstance(value, np.ndarray):
                # Apply sparse mutations
                mutations = np.random.randn(*value.shape) * mutation_scale
                mask = np.random.rand(*value.shape) < mutation_rate
                child_genome[key] = value + mutations * mask
            elif isinstance(value, (int, float)):
                # Scalar parameters
                if np.random.rand() < mutation_rate:
                    child_genome[key] = value * (1 + np.random.randn() * mutation_scale)
                else:
                    child_genome[key] = value
            else:
                child_genome[key] = copy.deepcopy(value)
        
        # Clip metabolic efficiency to reasonable range
        if 'metabolic_efficiency' in child_genome:
            child_genome['metabolic_efficiency'] = np.clip(
                child_genome['metabolic_efficiency'], 0.5, 2.0
            )
        
        # Create child at nearby position
        child_x = (self.x + np.random.randint(-3, 4)) % 64
        child_y = (self.y + np.random.randint(-3, 4)) % 64
        
        child = FullAutopoieticAgent(
            x=child_x,
            y=child_y,
            agent_id=child_id,
            genome=child_genome,
            parent_id=self.id
        )
        
        # Child starts with some resources
        child.energy = 0.8
        child.material = 0.4
        
        return child
    
    def get_behavior_descriptor(self) -> np.ndarray:
        """
        Compute 8-dimensional behavior descriptor for QD metrics
        
        Dimensions:
        0-1: Movement pattern (mean dx, dy)
        2-3: Resource consumption pattern (energy ratio, material ratio)
        4-5: Spatial distribution (position variance x, y)
        6-7: Temporal dynamics (state variance, coherence trend)
        """
        bd = np.zeros(8)
        
        # Movement pattern
        if self.action_history:
            actions = np.array(list(self.action_history))
            bd[0] = np.mean(actions[:, 0])  # Mean dx
            bd[1] = np.mean(actions[:, 1])  # Mean dy
            bd[2] = np.mean(np.clip(actions[:, 2], 0, 1))  # Energy consumption
            bd[3] = np.mean(np.clip(actions[:, 3], 0, 1))  # Material consumption
        
        # Spatial distribution
        if self.position_history:
            positions = np.array(list(self.position_history))
            bd[4] = np.std(positions[:, 0]) / 64  # Normalized x variance
            bd[5] = np.std(positions[:, 1]) / 64  # Normalized y variance
        
        # Temporal dynamics
        if self.state_history:
            states = np.array(list(self.state_history))
            bd[6] = np.std(states)  # State variance
        
        if len(self.coherence_history) >= 5:
            coh = list(self.coherence_history)
            bd[7] = coh[-1] - coh[-5]  # Coherence trend
        
        return bd
    
    def get_summary(self) -> Dict:
        """Get agent summary for logging/analysis"""
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'position': (self.x, self.y),
            'age': self.age,
            'energy': self.energy,
            'material': self.material,
            'is_alive': self.is_alive,
            'coherence': self.coherence_history[-1] if self.coherence_history else 0.5,
            'avg_coherence': np.mean(list(self.coherence_history)) if self.coherence_history else 0.5,
            'total_energy_consumed': self.total_energy_consumed,
            'total_material_consumed': self.total_material_consumed,
            'offspring_count': self.offspring_count
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_phase1')
    from full_environment import FullALifeEnvironment
    
    print("Testing FullAutopoieticAgent...")
    
    # Create environment and agent
    env = FullALifeEnvironment(size=64, seed=42)
    agent = FullAutopoieticAgent(x=32, y=32, agent_id=0)
    
    print(f"Agent ID: {agent.id}")
    print(f"Sensor dimension: {agent.SENSOR_DIM}")
    print(f"State dimension: {agent.STATE_DIM}")
    print(f"Action dimension: {agent.ACTION_DIM}")
    print(f"Initial position: ({agent.x}, {agent.y})")
    print(f"Initial energy: {agent.energy}")
    print(f"Initial material: {agent.material}")
    
    # Run for 200 steps
    print("\nRunning 200 steps...")
    for step in range(200):
        # Sense
        sensor = agent.sense(env, [])
        
        # Forward pass
        action = agent.forward(sensor)
        
        # Execute action
        dx = int(np.round(action[0] * 2))
        dy = int(np.round(action[1] * 2))
        agent.x = (agent.x + np.clip(dx, -2, 2)) % env.size
        agent.y = (agent.y + np.clip(dy, -2, 2)) % env.size
        agent.position_history.append((agent.x, agent.y))
        
        # Consume resources
        consume_e = max(0, action[2]) * 0.3
        consume_m = max(0, action[3]) * 0.2
        gained_e, gained_m = env.consume(agent.x, agent.y, consume_e, consume_m)
        agent.energy += gained_e
        agent.material += gained_m
        agent.total_energy_consumed += gained_e
        agent.total_material_consumed += gained_m
        
        # Metabolic cost
        cost = agent.metabolic_cost(action)
        agent.energy -= cost
        agent.energy = min(agent.energy, 2.0)
        agent.material = min(agent.material, 1.0)
        
        # Compute coherence
        coherence = agent.compute_coherence()
        
        # Age
        agent.age += 1
        
        # Environment step
        env.step()
        
        if step % 50 == 0:
            print(f"  Step {step}: pos=({agent.x},{agent.y}), "
                  f"energy={agent.energy:.3f}, material={agent.material:.3f}, "
                  f"coherence={coherence['composite']:.3f}")
    
    # Final summary
    print("\nFinal Summary:")
    summary = agent.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Test behavior descriptor
    bd = agent.get_behavior_descriptor()
    print(f"\nBehavior descriptor: {bd}")
    
    # Test reproduction
    print(f"\nCan reproduce: {agent.can_reproduce()}")
    
    if agent.energy > 0.5:
        child = agent.create_offspring(child_id=1)
        print(f"Child created: ID={child.id}, parent={child.parent_id}")
        print(f"Parent energy after: {agent.energy:.3f}")
        print(f"Child energy: {child.energy:.3f}")
    
    print("\nAgent test PASSED!")
