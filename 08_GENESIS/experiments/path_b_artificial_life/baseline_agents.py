"""
GENESIS Path B: Baseline Agents
================================

Comparison agents for artificial life experiments:
1. RL Agent (PPO with intrinsic motivation only - no explicit reward)
2. NEAT Agent (neuroevolution)
3. Random Agent (control)

For fair comparison:
- RL uses ONLY intrinsic motivation (curiosity), no explicit rewards
- All agents have same sensory/motor interface

Author: GENESIS Project
Date: 2026-01-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ===================================
# Random Agent (Control)
# ===================================

class RandomAgent:
    """
    Random agent for baseline comparison
    
    Takes completely random actions.
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        self.id = agent_id or f"random_{str(uuid.uuid4())[:8]}"
        self.agent_type = "random"
        
        # State tracking
        self.is_alive = True
        self.age = 0
        self.energy = 1.0
        self.energy_decay = 0.002
        
        # Statistics
        self.total_consumed = 0
        self.total_moved = 0
        self.total_danger = 0
        self.coherence_history = deque(maxlen=200)
        self.action_history = deque(maxlen=100)
        self.position_history = deque(maxlen=500)
    
    def perceive_and_act(self, observation: np.ndarray) -> int:
        """Random action selection"""
        if not self.is_alive:
            return 0
        
        action = np.random.randint(0, 6)
        self.action_history.append(action)
        return action
    
    def update_state(self, action_result: Dict, position: Tuple[int, int]):
        """Update state after action"""
        if not self.is_alive:
            return
        
        self.age += 1
        self.position_history.append(position)
        
        # Energy dynamics
        self.energy -= self.energy_decay
        
        if action_result.get('moved'):
            self.total_moved += 1
            self.energy -= 0.001
        
        if action_result.get('consumed'):
            energy_gained = action_result.get('energy_gained', 0.1)
            self.energy = min(1.0, self.energy + energy_gained * 0.3)
            self.total_consumed += 1
        
        if action_result.get('hit_predator'):
            self.total_danger += 1
            self.energy -= 0.5
        
        # Pseudo-coherence (just energy-based for comparison)
        self.coherence_history.append(self.energy)
        
        # Check survival
        if self.energy <= 0:
            self.is_alive = False
    
    @property
    def can_reproduce(self) -> bool:
        return self.energy > 0.6 and self.age > 100
    
    def reproduce(self, mutation_rate: float = 0.1) -> 'RandomAgent':
        """Create offspring (no mutation for random agent)"""
        offspring = RandomAgent()
        offspring.energy = 0.5
        self.energy -= 0.3
        return offspring
    
    def get_trajectory_features(self) -> np.ndarray:
        """Extract trajectory features"""
        if len(self.position_history) < 10:
            return np.zeros(20)
        
        positions = np.array(list(self.position_history))
        actions = np.array(list(self.action_history)) if len(self.action_history) > 0 else np.array([0])
        
        features = []
        
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            features.extend([
                np.mean(np.abs(displacements)),
                np.std(np.abs(displacements)),
                np.mean(displacements[:, 0]),
                np.mean(displacements[:, 1])
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        action_counts = np.bincount(actions.astype(int), minlength=6) / max(1, len(actions))
        features.extend(action_counts.tolist())
        
        if len(self.coherence_history) > 0:
            coh = np.array(list(self.coherence_history))
            features.extend([np.mean(coh), np.std(coh), np.min(coh), np.max(coh)])
        else:
            features.extend([0.5, 0, 0.5, 0.5])
        
        features.append(self.total_consumed / max(1, self.age))
        features.append(self.total_danger / max(1, self.age))
        
        unique_positions = len(set(map(tuple, positions)))
        features.append(unique_positions / max(1, len(positions)))
        
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def get_summary(self) -> Dict:
        return {
            'id': self.id,
            'type': self.agent_type,
            'age': self.age,
            'is_alive': self.is_alive,
            'energy': self.energy,
            'coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5,
            'total_consumed': self.total_consumed,
            'total_moved': self.total_moved,
            'total_danger': self.total_danger
        }


# ===================================
# RL Agent (PPO with Intrinsic Motivation)
# ===================================

class IntrinsicCuriosity(nn.Module):
    """
    Intrinsic Curiosity Module (ICM)
    
    Provides intrinsic motivation without explicit external rewards.
    """
    
    def __init__(self, state_dim: int = 405, action_dim: int = 6, feature_dim: int = 64):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Forward model: predict next state features from current + action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Inverse model: predict action from states
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor):
        """
        Compute intrinsic reward
        
        Args:
            state: Current state
            next_state: Next state
            action: Action taken (one-hot)
            
        Returns:
            intrinsic_reward, forward_loss, inverse_loss
        """
        # Encode states
        state_feat = self.encoder(state)
        next_state_feat = self.encoder(next_state)
        
        # Forward model prediction
        pred_next_feat = self.forward_model(torch.cat([state_feat, action], dim=-1))
        
        # Forward loss (prediction error = curiosity)
        forward_loss = torch.mean((pred_next_feat - next_state_feat.detach()) ** 2, dim=-1)
        
        # Inverse model prediction
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=-1))
        
        # Inverse loss
        inverse_loss = nn.functional.cross_entropy(
            pred_action, 
            action.argmax(dim=-1), 
            reduction='none'
        )
        
        # Intrinsic reward = forward prediction error
        intrinsic_reward = forward_loss.detach()
        
        return intrinsic_reward, forward_loss.mean(), inverse_loss.mean()


class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO
    """
    
    def __init__(self, state_dim: int = 405, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor):
        """Get action probabilities and value"""
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class RLAgent:
    """
    RL Agent using PPO with Intrinsic Curiosity (NO external rewards)
    
    IMPORTANT: For fair comparison with autopoietic agents,
    this agent ONLY uses intrinsic motivation (curiosity).
    No explicit rewards for consumption, survival, etc.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        state_dim: int = 405,
        action_dim: int = 6,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        curiosity_scale: float = 0.1,
        device: str = "cpu"
    ):
        self.id = agent_id or f"rl_{str(uuid.uuid4())[:8]}"
        self.agent_type = "rl_ppo"
        self.device = device
        
        # Networks
        self.policy = PPONetwork(state_dim, action_dim).to(device)
        self.curiosity = IntrinsicCuriosity(state_dim, action_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.curiosity_optimizer = optim.Adam(self.curiosity.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.curiosity_scale = curiosity_scale
        self.action_dim = action_dim
        
        # Experience buffer (limited size to prevent memory issues)
        self.buffer_size = 128
        self.states = deque(maxlen=self.buffer_size)
        self.actions = deque(maxlen=self.buffer_size)
        self.rewards = deque(maxlen=self.buffer_size)  # Intrinsic rewards only!
        self.log_probs = deque(maxlen=self.buffer_size)
        self.values = deque(maxlen=self.buffer_size)
        self.dones = deque(maxlen=self.buffer_size)
        
        # State tracking
        self.is_alive = True
        self.age = 0
        self.energy = 1.0
        self.energy_decay = 0.002
        
        self.last_state = None
        self.last_action = None
        
        # Statistics
        self.total_consumed = 0
        self.total_moved = 0
        self.total_danger = 0
        self.coherence_history = deque(maxlen=200)
        self.action_history = deque(maxlen=100)
        self.position_history = deque(maxlen=500)
    
    def perceive_and_act(self, observation: np.ndarray) -> int:
        """Select action using policy network"""
        if not self.is_alive:
            return 0
        
        state = torch.FloatTensor(observation.flatten()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store for learning
        if self.last_state is not None:
            self.states.append(self.last_state)
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
        
        self.last_state = state
        self.last_action = action.item()
        
        self.action_history.append(action.item())
        
        return action.item()
    
    def update_state(self, action_result: Dict, position: Tuple[int, int]):
        """Update state and compute intrinsic reward"""
        if not self.is_alive:
            return
        
        self.age += 1
        self.position_history.append(position)
        
        # Energy dynamics (same as autopoietic)
        self.energy -= self.energy_decay
        
        if action_result.get('moved'):
            self.total_moved += 1
            self.energy -= 0.001
        
        if action_result.get('consumed'):
            energy_gained = action_result.get('energy_gained', 0.1)
            self.energy = min(1.0, self.energy + energy_gained * 0.3)
            self.total_consumed += 1
        
        if action_result.get('hit_predator'):
            self.total_danger += 1
            self.energy -= 0.5
        
        # Pseudo-coherence (energy-based for comparison)
        self.coherence_history.append(self.energy)
        
        # Compute intrinsic reward if we have previous state
        if self.last_state is not None and len(self.states) > 0:
            current_state = self.last_state
            action_onehot = torch.zeros(1, self.action_dim).to(self.device)
            action_onehot[0, self.last_action] = 1.0
            
            # For intrinsic reward, we need next state observation
            # This will be computed in next perceive_and_act
            # For now, use 0 as placeholder
            self.rewards.append(0.0)  # Will be updated
            self.dones.append(not self.is_alive)
            
            states_list = list(self.states)
            if len(states_list) > 1:
                # Update intrinsic reward for previous transition
                prev_state = states_list[-2]
                curr_state = states_list[-1] if len(states_list) > 0 else current_state
                
                with torch.no_grad():
                    intrinsic_r, _, _ = self.curiosity(
                        prev_state,
                        curr_state,
                        action_onehot
                    )
                    # Update second to last reward in deque
                    if len(self.rewards) >= 2:
                        rewards_list = list(self.rewards)
                        rewards_list[-2] = intrinsic_r.item() * self.curiosity_scale
                        self.rewards.clear()
                        self.rewards.extend(rewards_list)
        
        # Check survival
        if self.energy <= 0:
            self.is_alive = False
        
        # Periodic learning update
        if len(self.states) >= 64 and self.age % 32 == 0:
            self._learn()
    
    def _learn(self):
        """PPO learning update"""
        if len(self.states) < 10:
            return

        # Prepare data - convert deque to list for slicing
        states_list = list(self.states)[-64:]
        if len(states_list) == 0:
            return
        states = torch.cat(states_list).to(self.device)
        actions = torch.LongTensor(list(self.action_history)[-64:]).to(self.device)
        rewards = torch.FloatTensor(list(self.rewards)[-64:]).to(self.device)
        old_log_probs = torch.FloatTensor(list(self.log_probs)[-64:]).to(self.device)
        values = torch.FloatTensor(list(self.values)[-64:]).to(self.device)
        
        # Compute advantages
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # K epochs
            action_probs, new_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = nn.functional.mse_loss(new_values.squeeze(), rewards)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
        
        # Update curiosity module
        if len(self.states) >= 2:
            states_list = list(self.states)
            if len(states_list) < 2:
                return
            for _ in range(2):
                prev_states = torch.cat(states_list[-64:-1]).to(self.device)
                next_states = torch.cat(states_list[-63:]).to(self.device)

                action_onehot = torch.zeros(len(prev_states), self.action_dim).to(self.device)
                for i, a in enumerate(list(self.action_history)[-64:-1]):
                    action_onehot[i, a] = 1.0

                _, forward_loss, inverse_loss = self.curiosity(
                    prev_states, next_states, action_onehot
                )

                curiosity_loss = forward_loss + 0.2 * inverse_loss
                
                self.curiosity_optimizer.zero_grad()
                curiosity_loss.backward()
                self.curiosity_optimizer.step()
    
    @property
    def can_reproduce(self) -> bool:
        return self.energy > 0.6 and self.age > 100
    
    def reproduce(self, mutation_rate: float = 0.1) -> 'RLAgent':
        """Create offspring with mutated network"""
        offspring = RLAgent(
            state_dim=405,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Copy and mutate policy network
        offspring.policy.load_state_dict(self.policy.state_dict())
        with torch.no_grad():
            for param in offspring.policy.parameters():
                param.add_(torch.randn_like(param) * mutation_rate)
        
        offspring.energy = 0.5
        self.energy -= 0.3
        
        return offspring
    
    def get_trajectory_features(self) -> np.ndarray:
        """Extract trajectory features"""
        if len(self.position_history) < 10:
            return np.zeros(20)
        
        positions = np.array(list(self.position_history))
        actions = np.array(list(self.action_history)) if len(self.action_history) > 0 else np.array([0])
        
        features = []
        
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            features.extend([
                np.mean(np.abs(displacements)),
                np.std(np.abs(displacements)),
                np.mean(displacements[:, 0]),
                np.mean(displacements[:, 1])
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        action_counts = np.bincount(actions.astype(int), minlength=6) / max(1, len(actions))
        features.extend(action_counts.tolist())
        
        if len(self.coherence_history) > 0:
            coh = np.array(list(self.coherence_history))
            features.extend([np.mean(coh), np.std(coh), np.min(coh), np.max(coh)])
        else:
            features.extend([0.5, 0, 0.5, 0.5])
        
        features.append(self.total_consumed / max(1, self.age))
        features.append(self.total_danger / max(1, self.age))
        
        unique_positions = len(set(map(tuple, positions)))
        features.append(unique_positions / max(1, len(positions)))
        
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def get_summary(self) -> Dict:
        return {
            'id': self.id,
            'type': self.agent_type,
            'age': self.age,
            'is_alive': self.is_alive,
            'energy': self.energy,
            'coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5,
            'total_consumed': self.total_consumed,
            'total_moved': self.total_moved,
            'total_danger': self.total_danger
        }


# ===================================
# NEAT Agent (Neuroevolution)
# ===================================

class NEATNetwork:
    """
    Simple feedforward network evolved via NEAT-like mutation
    
    Simplified version without full NEAT complexity.
    """
    
    def __init__(
        self,
        input_dim: int = 405,
        hidden_dim: int = 32,
        output_dim: int = 6
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simple 2-layer network
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        h = np.tanh(np.dot(self.W1, x) + self.b1)
        out = np.dot(self.W2, h) + self.b2
        
        # Softmax
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / np.sum(exp_out)
        
        return probs
    
    def mutate(self, rate: float = 0.1):
        """Mutate weights"""
        self.W1 += np.random.randn(*self.W1.shape) * rate
        self.b1 += np.random.randn(*self.b1.shape) * rate
        self.W2 += np.random.randn(*self.W2.shape) * rate
        self.b2 += np.random.randn(*self.b2.shape) * rate
    
    def copy(self) -> 'NEATNetwork':
        """Create copy of network"""
        new_net = NEATNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        new_net.W1 = self.W1.copy()
        new_net.b1 = self.b1.copy()
        new_net.W2 = self.W2.copy()
        new_net.b2 = self.b2.copy()
        return new_net


class NEATAgent:
    """
    NEAT-style neuroevolution agent
    
    Evolves network structure through mutation.
    Fitness = survival time (implicit, not optimized directly).
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        network: Optional[NEATNetwork] = None
    ):
        self.id = agent_id or f"neat_{str(uuid.uuid4())[:8]}"
        self.agent_type = "neat"
        
        # Network
        self.network = network or NEATNetwork()
        
        # State tracking
        self.is_alive = True
        self.age = 0
        self.energy = 1.0
        self.energy_decay = 0.002
        
        # Statistics
        self.total_consumed = 0
        self.total_moved = 0
        self.total_danger = 0
        self.coherence_history = deque(maxlen=200)
        self.action_history = deque(maxlen=100)
        self.position_history = deque(maxlen=500)
    
    def perceive_and_act(self, observation: np.ndarray) -> int:
        """Select action using evolved network"""
        if not self.is_alive:
            return 0
        
        state = observation.flatten()
        probs = self.network.forward(state)
        
        # Stochastic action selection
        action = np.random.choice(6, p=probs)
        self.action_history.append(action)
        
        return action
    
    def update_state(self, action_result: Dict, position: Tuple[int, int]):
        """Update state after action"""
        if not self.is_alive:
            return
        
        self.age += 1
        self.position_history.append(position)
        
        # Energy dynamics
        self.energy -= self.energy_decay
        
        if action_result.get('moved'):
            self.total_moved += 1
            self.energy -= 0.001
        
        if action_result.get('consumed'):
            energy_gained = action_result.get('energy_gained', 0.1)
            self.energy = min(1.0, self.energy + energy_gained * 0.3)
            self.total_consumed += 1
        
        if action_result.get('hit_predator'):
            self.total_danger += 1
            self.energy -= 0.5
        
        # Pseudo-coherence
        self.coherence_history.append(self.energy)
        
        # Check survival
        if self.energy <= 0:
            self.is_alive = False
    
    @property
    def can_reproduce(self) -> bool:
        return self.energy > 0.6 and self.age > 100
    
    def reproduce(self, mutation_rate: float = 0.1) -> 'NEATAgent':
        """Create offspring with mutated network"""
        new_network = self.network.copy()
        new_network.mutate(mutation_rate)
        
        offspring = NEATAgent(network=new_network)
        offspring.energy = 0.5
        self.energy -= 0.3
        
        return offspring
    
    def get_trajectory_features(self) -> np.ndarray:
        """Extract trajectory features"""
        if len(self.position_history) < 10:
            return np.zeros(20)
        
        positions = np.array(list(self.position_history))
        actions = np.array(list(self.action_history)) if len(self.action_history) > 0 else np.array([0])
        
        features = []
        
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            features.extend([
                np.mean(np.abs(displacements)),
                np.std(np.abs(displacements)),
                np.mean(displacements[:, 0]),
                np.mean(displacements[:, 1])
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        action_counts = np.bincount(actions.astype(int), minlength=6) / max(1, len(actions))
        features.extend(action_counts.tolist())
        
        if len(self.coherence_history) > 0:
            coh = np.array(list(self.coherence_history))
            features.extend([np.mean(coh), np.std(coh), np.min(coh), np.max(coh)])
        else:
            features.extend([0.5, 0, 0.5, 0.5])
        
        features.append(self.total_consumed / max(1, self.age))
        features.append(self.total_danger / max(1, self.age))
        
        unique_positions = len(set(map(tuple, positions)))
        features.append(unique_positions / max(1, len(positions)))
        
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def get_summary(self) -> Dict:
        return {
            'id': self.id,
            'type': self.agent_type,
            'age': self.age,
            'is_alive': self.is_alive,
            'energy': self.energy,
            'coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.5,
            'total_consumed': self.total_consumed,
            'total_moved': self.total_moved,
            'total_danger': self.total_danger
        }


# ===================================
# Agent Factory
# ===================================

def create_agent(agent_type: str, **kwargs):
    """
    Factory function to create agents
    
    Args:
        agent_type: "autopoietic", "rl", "neat", or "random"
        **kwargs: Agent-specific arguments
        
    Returns:
        Agent instance
    """
    if agent_type == "random":
        return RandomAgent(**kwargs)
    elif agent_type == "rl":
        return RLAgent(**kwargs)
    elif agent_type == "neat":
        return NEATAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# =====================
# Testing
# =====================

if __name__ == "__main__":
    print("=" * 70)
    print("Baseline Agents Test")
    print("=" * 70)
    
    # Test each agent type
    agents = [
        RandomAgent(agent_id="test_random"),
        RLAgent(agent_id="test_rl"),
        NEATAgent(agent_id="test_neat")
    ]
    
    for agent in agents:
        print(f"\nTesting {agent.agent_type} agent: {agent.id}")
        
        # Simulate 100 steps
        for step in range(100):
            obs = np.random.rand(9, 9, 5) * 0.5
            action = agent.perceive_and_act(obs)
            
            result = {
                'moved': action in [1, 2, 3, 4],
                'consumed': action == 5 and np.random.random() < 0.3,
                'energy_gained': 0.2 if action == 5 else 0,
                'hit_predator': np.random.random() < 0.02
            }
            
            agent.update_state(result, (50 + step % 10, 50 + step % 10))
            
            if not agent.is_alive:
                print(f"  Died at step {step}")
                break
        
        summary = agent.get_summary()
        print(f"  Final: age={summary['age']}, energy={summary['energy']:.3f}, "
              f"consumed={summary['total_consumed']}")
        
        # Test reproduction
        if agent.can_reproduce:
            offspring = agent.reproduce()
            print(f"  Reproduced! Offspring: {offspring.id}")
    
    print("\n" + "=" * 70)
    print("Baseline agents test complete!")
    print("=" * 70)
