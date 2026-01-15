"""
GENESIS Phase 2: Agent Extensions

Extends Phase 1 agents with Phase 2 capabilities without modifying base code.
Uses composition pattern to add new features.
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from full_agent import FullAutopoieticAgent
    from full_environment import FullALifeEnvironment
    from stigmergy import StigmergyField
    from semantic_memory import SemanticMemory


class Phase2AgentExtension:
    """
    Extension wrapper for Phase 1 agents

    Adds:
    - Stigmergy sensing (4 additional sensor dimensions)
    - Semantic knowledge guidance
    - Experience priority calculation
    - Environmental marking behavior
    """

    def __init__(self, agent: 'FullAutopoieticAgent'):
        """
        Args:
            agent: Phase 1 agent to extend
        """
        self.agent = agent
        self.last_observation = None
        self.last_action = None
        self.last_coherence = 0.5

    def sense_with_stigmergy(self,
                            env: 'FullALifeEnvironment',
                            nearby_agents,
                            stigmergy: Optional['StigmergyField'] = None) -> np.ndarray:
        """
        Enhanced sensing with stigmergy information

        Args:
            env: Environment
            nearby_agents: List of nearby agents
            stigmergy: Stigmergy field (optional)

        Returns:
            374-dimensional sensor vector (370 base + 4 stigmergy)
        """
        # Base Phase 1 sensing (370-dim)
        base_sensors = self.agent.sense(env, nearby_agents)

        # Add stigmergy information (4-dim)
        if stigmergy is not None:
            stigmergy_info = stigmergy.get_field_at(self.agent.x, self.agent.y, radius=5)
            stigmergy_sensors = np.array([
                stigmergy_info['pheromone'],
                stigmergy_info['danger'],
                stigmergy_info['resource'],
                stigmergy_info['success']
            ], dtype=np.float32)
        else:
            stigmergy_sensors = np.zeros(4, dtype=np.float32)

        # Combine (374-dim total)
        full_sensors = np.concatenate([base_sensors, stigmergy_sensors])

        # Store for experience replay
        self.last_observation = base_sensors.copy()

        return full_sensors

    def act_with_semantic_guidance(self,
                                   observation: np.ndarray,
                                   semantic_memory: Optional['SemanticMemory'] = None,
                                   guidance_strength: float = 0.3) -> np.ndarray:
        """
        Enhanced action with semantic knowledge guidance

        Combines:
        1. Base agent action (implicit teacher knowledge)
        2. Semantic memory guidance (explicit rules)

        Args:
            observation: Full sensor input (374-dim or 370-dim)
            semantic_memory: Semantic memory system (optional)
            guidance_strength: How much to blend semantic guidance (0-1)

        Returns:
            5-dimensional action vector
        """
        # Use base observation (first 370 dims) for semantic query
        base_obs = observation[:370]

        # Get base action from Phase 1 agent
        # Note: Agent's act method expects 370-dim, so we pass base observation
        base_action = self._compute_base_action(base_obs)

        # Query semantic memory for guidance
        if semantic_memory is not None and semantic_memory.rules:
            suggested_action, confidence = semantic_memory.query_knowledge(base_obs)

            if suggested_action is not None and confidence > 0.7:
                # Blend actions based on confidence
                blend_factor = guidance_strength * confidence
                final_action = (1 - blend_factor) * base_action + blend_factor * suggested_action
            else:
                final_action = base_action
        else:
            final_action = base_action

        # Store for experience replay
        self.last_action = final_action.copy()

        return final_action

    def mark_environment(self,
                        stigmergy: Optional['StigmergyField'] = None,
                        coherence: float = 0.5):
        """
        Leave marks in environment based on agent state

        Marking rules:
        - Always: Deposit pheromone (movement trail)
        - High coherence (>0.9): Mark success zone
        - Low energy (<0.2): Mark danger zone
        - Recently consumed resource: Mark resource location

        Args:
            stigmergy: Stigmergy field to mark
            coherence: Current coherence value
        """
        if stigmergy is None:
            return

        x, y = self.agent.x, self.agent.y

        # Always leave pheromone trail
        stigmergy.deposit_pheromone(x, y, strength=0.1)

        # Mark success if high coherence
        if coherence > 0.9:
            stigmergy.mark_success(x, y, coherence)

        # Mark danger if low energy (starvation zone)
        if self.agent.energy < 0.2:
            stigmergy.mark_danger(x, y, intensity=0.5)

        # Mark resource if recently consumed significant amount
        recent_consumption = self.agent.total_energy_consumed
        if recent_consumption > 0:  # Simplified check
            stigmergy.mark_resource(x, y, amount=0.3)

        self.last_coherence = coherence

    def compute_experience_priority(self, coherence: float) -> float:
        """
        Compute priority for storing this experience

        Priority factors:
        1. Coherence gain (learned something new)
        2. Survival value (avoided death, found food)
        3. Novelty (unusual situation)

        Args:
            coherence: Current coherence

        Returns:
            Priority score (0.0 - 3.0+)
        """
        priority = coherence  # Base priority

        # Coherence gain bonus
        if self.agent.coherence_history:
            prev_coherence = self.agent.coherence_history[-1]
            coherence_gain = coherence - prev_coherence
            if coherence_gain > 0.1:  # Significant improvement
                priority += 1.0

        # Survival bonus (low energy but still alive)
        if self.agent.energy < 0.3:
            priority += 2.0  # High priority - learned to survive

        # High performance bonus
        if coherence > 0.9:
            priority += 1.0

        return priority

    def get_experience_dict(self, coherence: float) -> Dict:
        """
        Package agent experience for episodic memory

        Returns:
            Dictionary with observation, action, outcome, context
        """
        return {
            'observation': self.last_observation.copy() if self.last_observation is not None else np.zeros(370),
            'action': self.last_action.copy() if self.last_action is not None else np.zeros(5),
            'coherence': coherence,
            'energy': self.agent.energy,
            'age': self.agent.age,
            'position': (self.agent.x, self.agent.y),
            'state': self.agent.state.copy()
        }

    # Helper methods

    def _compute_base_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute base action using Phase 1 agent's RNN

        This replicates the agent's act() method without executing it.
        """
        # Use agent's act method with 370-dim observation
        # But we need to be careful not to update agent state twice

        # Simple approach: just use the agent's forward pass
        # state_input = np.concatenate([observation, self.agent.state])
        # new_state = np.tanh(self.agent.W_in @ observation +
        #                     self.agent.W_rec @ self.agent.state +
        #                     self.agent.b_state)
        # action = np.tanh(self.agent.W_out @ new_state + self.agent.b_action)

        # Actually, let's just pass through to agent.act but store state first
        # and restore it after (to avoid double-updating)

        # For simplicity in Phase 2, we'll compute action directly
        state_input = observation
        new_state = np.tanh(
            self.agent.W_in @ state_input +
            self.agent.W_rec @ self.agent.state +
            self.agent.b_state
        )
        action = np.tanh(self.agent.W_out @ new_state + self.agent.b_action)

        return action
