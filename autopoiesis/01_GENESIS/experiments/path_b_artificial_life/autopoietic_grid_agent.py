"""
GENESIS Path B: Autopoietic Grid Agent
======================================

Extension of AutopoieticEntity for 2D grid world survival.

Key Features:
- Sensory: 9x9 local grid perception (vision)
- Internal: 50 units recurrent dynamics
- Motor: 4-direction movement + consume action
- Coherence -> survival mapping
- Reproduction: coherence > 0.7

NO external rewards, NO objective functions.
ONLY organizational coherence maintenance.

Author: GENESIS Project
Date: 2026-01-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import uuid


class RecurrentDynamics:
    """
    Extended recurrent dynamics for grid world perception
    
    Key difference from v2.0:
    - Sensory integration layer
    - Larger internal state (50 units)
    - Motor output layer
    """
    
    def __init__(
        self,
        sensory_dim: int = 405,  # 9x9x5 flattened
        n_internal: int = 50,
        motor_dim: int = 6,  # 6 actions
        connectivity: float = 0.3
    ):
        self.sensory_dim = sensory_dim
        self.n_internal = n_internal
        self.motor_dim = motor_dim
        
        # Sensory -> Internal weights (random projection)
        self.W_sensory = np.random.randn(n_internal, sensory_dim) * 0.1
        
        # Internal recurrent weights (sparse)
        mask = np.random.rand(n_internal, n_internal) < connectivity
        self.W_internal = np.random.randn(n_internal, n_internal) * 0.2 * mask
        np.fill_diagonal(self.W_internal, 0)  # No self-loops
        
        # Internal -> Motor weights
        self.W_motor = np.random.randn(motor_dim, n_internal) * 0.3
        
        # Internal state
        self.state = np.random.randn(n_internal) * 0.1
        self.state_history = deque(maxlen=100)
        
        # Sensory buffer
        self.last_sensory = np.zeros(sensory_dim)
    
    def step(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        One step of dynamics
        
        Args:
            sensory_input: Flattened sensory observation
            
        Returns:
            New internal state
        """
        # Ensure correct dimension
        if len(sensory_input) != self.sensory_dim:
            sensory_input = np.zeros(self.sensory_dim)
        
        self.last_sensory = sensory_input
        
        # Sensory integration
        sensory_effect = np.tanh(np.dot(self.W_sensory, sensory_input))
        
        # Internal dynamics (recurrent)
        internal_effect = np.tanh(np.dot(self.W_internal, self.state))
        
        # Leaky integration
        tau = 0.6
        new_state = tau * self.state + (1 - tau) * (internal_effect + 0.3 * sensory_effect)
        
        # Update state
        self.state = np.clip(new_state, -2, 2)
        self.state_history.append(self.state.copy())
        
        return self.state
    
    def get_motor_output(self) -> np.ndarray:
        """
        Generate motor output from internal state
        
        Returns:
            Motor activations (6 actions)
        """
        motor = np.dot(self.W_motor, self.state)
        return motor  # Raw values, will be converted to action
    
    def get_action_probabilities(self) -> np.ndarray:
        """
        Get action probabilities via softmax
        
        Returns:
            Probability distribution over actions
        """
        motor = self.get_motor_output()
        
        # Softmax with temperature
        temperature = 1.0
        exp_motor = np.exp((motor - np.max(motor)) / temperature)
        probs = exp_motor / np.sum(exp_motor)
        
        return probs


class GridCoherenceAssessor:
    """
    Coherence assessment adapted for grid world survival
    
    Additions:
    - Energy integration into coherence
    - Survival pressure
    """
    
    def __init__(self):
        self.coherence_history = deque(maxlen=100)
    
    def assess(
        self,
        dynamics: RecurrentDynamics,
        energy: float,
        recent_consumed: int,
        recent_danger: int
    ) -> Dict[str, float]:
        """
        Multi-dimensional coherence assessment
        
        Args:
            dynamics: Agent's internal dynamics
            energy: Current energy level (0-1 normalized)
            recent_consumed: Resources consumed in last 50 steps
            recent_danger: Danger encounters in last 50 steps
            
        Returns:
            Coherence scores dictionary
        """
        if len(dynamics.state_history) < 10:
            return {
                'predictability': 0.5,
                'stability': 0.5,
                'responsiveness': 0.5,
                'metabolic': 0.5,
                'composite': 0.5
            }
        
        states = np.array(list(dynamics.state_history))
        
        # 1. Predictability: 내부 상태의 예측가능성
        state_changes = np.diff(states, axis=0)
        predictability = 1.0 / (1.0 + np.mean(np.var(state_changes, axis=0)))
        
        # 2. Stability: 역학의 안정성
        recent = states[-20:]
        stability = 1.0 / (1.0 + np.std(recent))
        
        # 3. Responsiveness: 감각 입력에 대한 반응
        # 상태 변화와 감각 상관
        sensory_norm = np.linalg.norm(dynamics.last_sensory)
        state_change_norm = np.linalg.norm(states[-1] - states[-2]) if len(states) > 1 else 0
        responsiveness = min(1.0, 0.5 + 0.5 * (state_change_norm * sensory_norm))
        
        # 4. Metabolic: 에너지 관련 일관성
        # 에너지가 적절히 유지되면 높음
        metabolic = min(1.0, energy * 1.5)  # Energy 0.67 이상이면 1.0
        
        # Bonus for recent consumption
        if recent_consumed > 0:
            metabolic = min(1.0, metabolic + 0.1 * recent_consumed)
        
        # Penalty for danger
        if recent_danger > 0:
            metabolic = max(0.1, metabolic - 0.1 * recent_danger)
        
        # Composite coherence
        composite = (
            0.25 * predictability +
            0.25 * stability +
            0.20 * responsiveness +
            0.30 * metabolic  # Metabolic is important for survival
        )
        
        coherence = {
            'predictability': float(np.clip(predictability, 0, 1)),
            'stability': float(np.clip(stability, 0, 1)),
            'responsiveness': float(np.clip(responsiveness, 0, 1)),
            'metabolic': float(np.clip(metabolic, 0, 1)),
            'composite': float(np.clip(composite, 0, 1))
        }
        
        self.coherence_history.append(coherence['composite'])
        
        return coherence


class StructuralDrift:
    """
    Structural plasticity through coherence-guided drift
    
    NOT gradient descent!
    Random perturbations accepted if coherence maintained.
    """
    
    def __init__(self, drift_rate: float = 0.02):
        self.drift_rate = drift_rate
        self.changes_accepted = 0
        self.changes_rejected = 0
    
    def maybe_drift(
        self,
        dynamics: RecurrentDynamics,
        current_coherence: float,
        min_coherence: float = 0.4
    ) -> bool:
        """
        Attempt structural drift
        
        Args:
            dynamics: Agent's dynamics to modify
            current_coherence: Current coherence score
            min_coherence: Minimum coherence to maintain
            
        Returns:
            Whether drift was accepted
        """
        # Only drift if coherence is low
        if current_coherence > 0.7:
            return False
        
        # Save original weights
        W_internal_orig = dynamics.W_internal.copy()
        W_sensory_orig = dynamics.W_sensory.copy()
        W_motor_orig = dynamics.W_motor.copy()
        
        # Random perturbation
        drift_internal = np.random.randn(*dynamics.W_internal.shape) * self.drift_rate
        drift_sensory = np.random.randn(*dynamics.W_sensory.shape) * self.drift_rate * 0.5
        drift_motor = np.random.randn(*dynamics.W_motor.shape) * self.drift_rate
        
        # Apply (maintain sparsity)
        internal_mask = (dynamics.W_internal != 0)
        dynamics.W_internal += drift_internal * internal_mask
        dynamics.W_sensory += drift_sensory
        dynamics.W_motor += drift_motor
        
        # This is a simple accept - could be extended with actual coherence evaluation
        # For now, accept all drifts (evolutionary pressure will select)
        self.changes_accepted += 1
        
        return True


class AutopoieticGridAgent:
    """
    Autopoietic agent for grid world
    
    Core principle: Survive by maintaining organizational coherence
    
    NO external rewards.
    NO optimization targets.
    ONLY internal coherence maintenance.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        n_internal: int = 50,
        connectivity: float = 0.3,
        drift_rate: float = 0.02,
        initial_energy: float = 1.0,
        energy_decay: float = 0.002,
        coherence_death_threshold: float = 0.2,
        reproduction_threshold: float = 0.7
    ):
        """
        Args:
            agent_id: Unique identifier
            n_internal: Internal units count
            connectivity: Recurrent connectivity density
            drift_rate: Structural drift rate
            initial_energy: Starting energy
            energy_decay: Energy loss per step
            coherence_death_threshold: Death if coherence below this
            reproduction_threshold: Can reproduce if coherence above this
        """
        self.id = agent_id or str(uuid.uuid4())[:8]
        
        # Internal dynamics
        self.dynamics = RecurrentDynamics(
            sensory_dim=405,  # 9x9x5
            n_internal=n_internal,
            motor_dim=6,
            connectivity=connectivity
        )
        
        # Coherence assessment
        self.assessor = GridCoherenceAssessor()
        
        # Structural plasticity
        self.plasticity = StructuralDrift(drift_rate)
        
        # Energy and survival
        self.energy = initial_energy
        self.energy_decay = energy_decay
        self.coherence_death_threshold = coherence_death_threshold
        self.reproduction_threshold = reproduction_threshold
        
        # State
        self.is_alive = True
        self.age = 0
        self.can_reproduce = False
        
        # Statistics
        self.total_consumed = 0
        self.total_moved = 0
        self.total_danger = 0
        self.recent_consumed = deque(maxlen=50)
        self.recent_danger = deque(maxlen=50)
        self.coherence_history = deque(maxlen=200)
        self.action_history = deque(maxlen=100)
        self.position_history = deque(maxlen=500)
    
    def perceive_and_act(self, observation: np.ndarray) -> int:
        """
        Perceive environment and generate action
        
        Args:
            observation: 9x9x5 grid observation
            
        Returns:
            action: 0-5 (stay, up, down, left, right, consume)
        """
        if not self.is_alive:
            return 0
        
        # Flatten observation
        sensory = observation.flatten()
        
        # Run internal dynamics
        self.dynamics.step(sensory)
        
        # Get action from internal dynamics
        probs = self.dynamics.get_action_probabilities()
        
        # Stochastic action selection
        action = np.random.choice(6, p=probs)
        
        self.action_history.append(action)
        
        return action
    
    def update_state(self, action_result: Dict, position: Tuple[int, int]):
        """
        Update agent state after action
        
        Args:
            action_result: Result from world.step_agent()
            position: Current position
        """
        if not self.is_alive:
            return
        
        self.age += 1
        self.position_history.append(position)
        
        # Energy dynamics
        self.energy -= self.energy_decay  # Constant decay
        
        if action_result.get('moved'):
            self.total_moved += 1
            self.energy -= 0.001  # Small movement cost
        
        if action_result.get('consumed'):
            energy_gained = action_result.get('energy_gained', 0.1)
            self.energy = min(1.0, self.energy + energy_gained * 0.3)
            self.total_consumed += 1
            self.recent_consumed.append(1)
        else:
            self.recent_consumed.append(0)
        
        if action_result.get('hit_predator'):
            self.total_danger += 1
            self.recent_danger.append(1)
            # Large energy loss from predator encounter
            self.energy -= 0.5
        else:
            self.recent_danger.append(0)
        
        # Assess coherence
        coherence = self.assessor.assess(
            self.dynamics,
            self.energy,
            sum(self.recent_consumed),
            sum(self.recent_danger)
        )
        self.coherence_history.append(coherence['composite'])
        
        # Structural drift (if needed)
        if coherence['composite'] < 0.5 and self.age % 10 == 0:
            self.plasticity.maybe_drift(
                self.dynamics,
                coherence['composite']
            )
        
        # Check survival
        self._check_survival(coherence['composite'])
        
        # Check reproduction eligibility
        self._check_reproduction(coherence['composite'])
    
    def _check_survival(self, coherence: float):
        """Check if agent survives"""
        # Death by energy depletion
        if self.energy <= 0:
            self.is_alive = False
            return
        
        # Death by coherence collapse
        if coherence < self.coherence_death_threshold and self.age > 50:
            self.is_alive = False
            return
    
    def _check_reproduction(self, coherence: float):
        """Check if agent can reproduce"""
        # Need high coherence AND sufficient energy
        self.can_reproduce = (
            coherence > self.reproduction_threshold and
            self.energy > 0.6 and
            self.age > 100
        )
    
    def reproduce(self, mutation_rate: float = 0.1) -> 'AutopoieticGridAgent':
        """
        Create offspring with mutated structure
        
        Args:
            mutation_rate: Mutation intensity
            
        Returns:
            Offspring agent
        """
        offspring = AutopoieticGridAgent(
            n_internal=self.dynamics.n_internal,
            connectivity=0.3,
            drift_rate=self.plasticity.drift_rate,
            initial_energy=0.5,  # Offspring starts with less energy
            energy_decay=self.energy_decay,
            coherence_death_threshold=self.coherence_death_threshold,
            reproduction_threshold=self.reproduction_threshold
        )
        
        # Copy structure with mutation
        offspring.dynamics.W_internal = self.dynamics.W_internal.copy()
        offspring.dynamics.W_sensory = self.dynamics.W_sensory.copy()
        offspring.dynamics.W_motor = self.dynamics.W_motor.copy()
        
        # Mutate
        mask = (offspring.dynamics.W_internal != 0)
        offspring.dynamics.W_internal += np.random.randn(*offspring.dynamics.W_internal.shape) * mutation_rate * mask
        offspring.dynamics.W_sensory += np.random.randn(*offspring.dynamics.W_sensory.shape) * mutation_rate * 0.5
        offspring.dynamics.W_motor += np.random.randn(*offspring.dynamics.W_motor.shape) * mutation_rate
        
        # Parent loses energy from reproduction
        self.energy -= 0.3
        self.can_reproduce = False
        
        return offspring
    
    def get_trajectory_features(self) -> np.ndarray:
        """
        Extract features from trajectory for behavioral analysis
        
        Returns:
            Feature vector for clustering
        """
        if len(self.position_history) < 10:
            return np.zeros(20)
        
        positions = np.array(list(self.position_history))
        actions = np.array(list(self.action_history)) if len(self.action_history) > 0 else np.array([0])
        
        features = []
        
        # Movement statistics
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            features.append(np.mean(np.abs(displacements)))  # Average movement
            features.append(np.std(np.abs(displacements)))   # Movement variance
            
            # Directional bias
            features.append(np.mean(displacements[:, 0]))  # X bias
            features.append(np.mean(displacements[:, 1]))  # Y bias
        else:
            features.extend([0, 0, 0, 0])
        
        # Action statistics
        action_counts = np.bincount(actions.astype(int), minlength=6) / len(actions)
        features.extend(action_counts.tolist())  # 6 values
        
        # Energy/coherence statistics
        if len(self.coherence_history) > 0:
            coherence = np.array(list(self.coherence_history))
            features.append(np.mean(coherence))
            features.append(np.std(coherence))
            features.append(np.min(coherence))
            features.append(np.max(coherence))
        else:
            features.extend([0.5, 0, 0.5, 0.5])
        
        # Consumption pattern
        features.append(self.total_consumed / max(1, self.age))
        features.append(self.total_danger / max(1, self.age))
        
        # Exploration (unique positions)
        unique_positions = len(set(map(tuple, positions)))
        features.append(unique_positions / max(1, len(positions)))
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def get_summary(self) -> Dict:
        """Get agent state summary"""
        return {
            'id': self.id,
            'age': self.age,
            'is_alive': self.is_alive,
            'energy': self.energy,
            'coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5,
            'avg_coherence': np.mean(list(self.coherence_history)) if len(self.coherence_history) > 0 else 0.5,
            'total_consumed': self.total_consumed,
            'total_moved': self.total_moved,
            'total_danger': self.total_danger,
            'can_reproduce': self.can_reproduce
        }


# =====================
# Testing
# =====================

if __name__ == "__main__":
    print("=" * 70)
    print("AutopoieticGridAgent Test")
    print("=" * 70)
    
    # Create agent
    agent = AutopoieticGridAgent(
        n_internal=50,
        connectivity=0.3,
        drift_rate=0.02,
        initial_energy=1.0
    )
    
    print(f"\nAgent created: {agent.id}")
    print(f"  Internal units: {agent.dynamics.n_internal}")
    print(f"  Energy: {agent.energy}")
    
    # Simulate with random observations
    print("\nSimulating 200 steps with random observations...")
    
    for step in range(200):
        # Random observation (9x9x5)
        obs = np.random.rand(9, 9, 5) * 0.5
        
        # Perceive and act
        action = agent.perceive_and_act(obs)
        
        # Simulate action result
        result = {
            'moved': action in [1, 2, 3, 4],
            'consumed': action == 5 and np.random.random() < 0.3,
            'energy_gained': 0.2 if action == 5 else 0,
            'hit_predator': np.random.random() < 0.02
        }
        
        # Update state
        agent.update_state(result, (50 + step % 10, 50 + step % 10))
        
        if not agent.is_alive:
            print(f"\nAgent died at step {step}")
            break
        
        if step % 50 == 0:
            summary = agent.get_summary()
            print(f"Step {step}: energy={summary['energy']:.3f}, "
                  f"coherence={summary['coherence']:.3f}, "
                  f"consumed={summary['total_consumed']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    
    summary = agent.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Trajectory features
    features = agent.get_trajectory_features()
    print(f"\nTrajectory features shape: {features.shape}")
    print(f"  Features: {features[:5]}...")
    
    print("\n" + "=" * 70)
    print("Autopoietic paradigm: NO rewards, ONLY coherence!")
    print("=" * 70)
