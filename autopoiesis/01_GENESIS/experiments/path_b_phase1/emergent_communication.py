"""
GENESIS Phase 4C: Emergent Communication

Agents develop communication protocols through evolution.

Key capabilities:
- Message encoding/decoding
- Attention mechanisms for selective listening
- Multiple communication channels
- Emergent signal conventions
- Social learning through communication

References:
- Lazaridou & Baroni (2020) "Emergent Multi-Agent Communication in the Deep Learning Era"
- Foerster et al. (2016) "Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import copy


class MessageEncoder(nn.Module):
    """
    Encodes agent's internal state into communication signal

    Learns what information to communicate
    """

    def __init__(self, state_dim: int = 128, message_dim: int = 8):
        """
        Args:
            state_dim: Dimension of agent's internal state
            message_dim: Dimension of message signal
        """
        super().__init__()

        self.state_dim = state_dim
        self.message_dim = message_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, message_dim),
            nn.Tanh()  # Bounded signals [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state into message

        Args:
            state: Internal state tensor

        Returns:
            Message signal
        """
        return self.encoder(state)

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Encode numpy state

        Args:
            state: Internal state (numpy)

        Returns:
            Message signal (numpy)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            message = self.forward(state_tensor)
            return message.squeeze(0).numpy()


class MessageDecoder(nn.Module):
    """
    Decodes received message into internal influence

    Learns how to interpret messages
    """

    def __init__(self, message_dim: int = 8, influence_dim: int = 32):
        """
        Args:
            message_dim: Dimension of message signal
            influence_dim: Dimension of influence on agent
        """
        super().__init__()

        self.message_dim = message_dim
        self.influence_dim = influence_dim

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, 32),
            nn.ReLU(),
            nn.Linear(32, influence_dim),
            nn.Tanh()
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Decode message into influence

        Args:
            message: Message signal

        Returns:
            Influence vector
        """
        return self.decoder(message)

    def decode(self, message: np.ndarray) -> np.ndarray:
        """
        Decode numpy message

        Args:
            message: Message signal (numpy)

        Returns:
            Influence vector (numpy)
        """
        with torch.no_grad():
            message_tensor = torch.FloatTensor(message).unsqueeze(0)
            influence = self.forward(message_tensor)
            return influence.squeeze(0).numpy()


class MessageAttention(nn.Module):
    """
    Attention mechanism for selective message processing

    Learns which messages to pay attention to
    """

    def __init__(self, state_dim: int = 128, message_dim: int = 8):
        """
        Args:
            state_dim: Dimension of agent's state
            message_dim: Dimension of message
        """
        super().__init__()

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(state_dim + message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Attention weight [0, 1]
        )

    def forward(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weight

        Args:
            state: Agent's state
            message: Message signal

        Returns:
            Attention weight
        """
        combined = torch.cat([state, message], dim=-1)
        return self.attention(combined)

    def compute_attention(self, state: np.ndarray, message: np.ndarray) -> float:
        """
        Compute attention weight (numpy)

        Args:
            state: Agent's state
            message: Message signal

        Returns:
            Attention weight
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            message_tensor = torch.FloatTensor(message).unsqueeze(0)
            attention = self.forward(state_tensor, message_tensor)
            return attention.item()


class CommunicatingAgent:
    """
    Agent with communication capabilities

    Wraps base agent with communication modules
    """

    def __init__(self,
                 agent,
                 agent_id: int,
                 message_dim: int = 8,
                 influence_dim: int = 32):
        """
        Args:
            agent: Base agent (FullAutopoieticAgent)
            agent_id: Unique agent ID
            message_dim: Dimension of messages
            influence_dim: Dimension of influence from messages
        """
        self.agent = agent
        self.id = agent_id

        # Communication modules
        state_dim = 128  # Assume agent has 128-dim internal state
        self.encoder = MessageEncoder(state_dim, message_dim)
        self.decoder = MessageDecoder(message_dim, influence_dim)
        self.attention = MessageAttention(state_dim, message_dim)

        # Communication history
        self.messages_sent = deque(maxlen=100)
        self.messages_received = deque(maxlen=100)
        self.message_influences = []

        # Communication strategy (evolved)
        self.communication_threshold = 0.7  # Coherence threshold for sending
        self.energy_cost_per_message = 0.01

        # Statistics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.communication_benefit = 0.0

    def decide_to_communicate(self) -> bool:
        """
        Decide whether to send a message

        Based on:
        - Internal coherence (only communicate if organized)
        - Energy availability (communication costs energy)
        - Random exploration
        """
        # Check energy
        energy = getattr(self.agent, 'energy', 1.0)
        if energy < 0.3:  # Too low
            return False

        # Check coherence
        if hasattr(self.agent, 'compute_coherence'):
            coherence = self.agent.compute_coherence()['composite']
        else:
            coherence = 0.5

        # Decide
        if coherence > self.communication_threshold:
            # High coherence - communicate with high probability
            return np.random.random() < 0.8
        else:
            # Low coherence - rare communication
            return np.random.random() < 0.1

    def create_message(self, channel: str = 'local') -> Dict:
        """
        Create message from current state

        Args:
            channel: Communication channel ('broadcast', 'local', 'directed')

        Returns:
            Message dictionary
        """
        # Extract internal state
        if hasattr(self.agent, 'state'):
            state = self.agent.state
        else:
            # Fallback: use position, energy, material
            state = np.zeros(128)
            if hasattr(self.agent, 'x') and hasattr(self.agent, 'y'):
                state[:2] = [self.agent.x, self.agent.y]
            if hasattr(self.agent, 'energy'):
                state[2] = self.agent.energy
            if hasattr(self.agent, 'material'):
                state[3] = self.agent.material

        # Encode state into message
        signal = self.encoder.encode(state)

        message = {
            'sender_id': self.id,
            'signal': signal,
            'channel': channel,
            'timestamp': getattr(self.agent, 'age', 0)
        }

        self.messages_sent.append(message)
        self.total_messages_sent += 1

        # Energy cost
        if hasattr(self.agent, 'energy'):
            self.agent.energy = max(0, self.agent.energy - self.energy_cost_per_message)

        return message

    def process_messages(self, messages: List[Dict]) -> Optional[np.ndarray]:
        """
        Process received messages

        Args:
            messages: List of message dictionaries

        Returns:
            Combined influence from messages (or None)
        """
        if not messages:
            return None

        # Get agent state
        if hasattr(self.agent, 'state'):
            state = self.agent.state
        else:
            state = np.zeros(128)

        # Compute attention for each message
        attentions = []
        influences = []

        for msg in messages:
            # Compute attention
            attention = self.attention.compute_attention(state, msg['signal'])
            attentions.append(attention)

            # Decode message
            influence = self.decoder.decode(msg['signal'])
            influences.append(influence)

        # Normalize attention (softmax)
        attentions = np.array(attentions)
        attentions = np.exp(attentions) / (np.exp(attentions).sum() + 1e-8)

        # Weighted combination
        combined_influence = sum(
            w * inf for w, inf in zip(attentions, influences)
        )

        self.message_influences.append(combined_influence)
        self.total_messages_received += len(messages)

        return combined_influence

    def get_statistics(self) -> Dict:
        """Get communication statistics"""
        return {
            'messages_sent': self.total_messages_sent,
            'messages_received': self.total_messages_received,
            'communication_benefit': self.communication_benefit,
            'avg_influence': np.mean([np.linalg.norm(inf) for inf in self.message_influences])
                            if self.message_influences else 0.0
        }


class CommunicationManager:
    """
    Manages message passing between agents

    Handles different communication channels and message delivery
    """

    def __init__(self, local_radius: float = 5.0):
        """
        Args:
            local_radius: Radius for local communication
        """
        self.local_radius = local_radius

        # Message buffers
        self.broadcast_messages = []
        self.local_messages = []

        # Statistics
        self.total_messages = 0
        self.messages_per_channel = defaultdict(int)

    def step(self, communicating_agents: List[CommunicatingAgent]):
        """
        Execute one communication step

        Args:
            communicating_agents: List of CommunicatingAgent objects
        """
        # Clear buffers
        self.broadcast_messages.clear()
        self.local_messages.clear()

        # 1. Collect messages from agents
        for comm_agent in communicating_agents:
            if comm_agent.decide_to_communicate():
                # Decide channel (mostly local, occasionally broadcast)
                channel = 'local' if np.random.random() < 0.9 else 'broadcast'

                message = comm_agent.create_message(channel=channel)

                if channel == 'broadcast':
                    self.broadcast_messages.append(message)
                else:
                    sender_pos = np.array([comm_agent.agent.x, comm_agent.agent.y])
                    self.local_messages.append((message, sender_pos))

                self.messages_per_channel[channel] += 1

        # 2. Distribute messages
        for comm_agent in communicating_agents:
            received_messages = []

            # Receive broadcast messages
            for msg in self.broadcast_messages:
                if msg['sender_id'] != comm_agent.id:
                    received_messages.append(msg)

            # Receive local messages
            agent_pos = np.array([comm_agent.agent.x, comm_agent.agent.y])
            for msg, sender_pos in self.local_messages:
                if msg['sender_id'] == comm_agent.id:
                    continue

                # Check distance
                distance = np.linalg.norm(agent_pos - sender_pos)
                if distance <= self.local_radius:
                    received_messages.append(msg)

            # Process received messages
            if received_messages:
                influence = comm_agent.process_messages(received_messages)

                # Apply influence to agent (modify next action)
                # In full implementation, would integrate with agent's decision-making
                if influence is not None and hasattr(comm_agent.agent, 'state'):
                    # Pad influence to match state dimensions
                    state_len = len(comm_agent.agent.state)
                    if len(influence) < state_len:
                        padded_influence = np.zeros(state_len)
                        padded_influence[:len(influence)] = influence
                    else:
                        padded_influence = influence[:state_len]

                    # Add small influence to state
                    comm_agent.agent.state = comm_agent.agent.state + 0.1 * padded_influence

        self.total_messages += len(self.broadcast_messages) + len(self.local_messages)

    def get_statistics(self) -> Dict:
        """Get communication statistics"""
        return {
            'total_messages': self.total_messages,
            'broadcast_messages': self.messages_per_channel['broadcast'],
            'local_messages': self.messages_per_channel['local']
        }


class MessageAnalyzer:
    """
    Analyzes emergent communication protocols

    Tracks message statistics and protocol emergence
    """

    def __init__(self):
        """Initialize analyzer"""
        # Signal clustering
        self.all_signals = []

        # Signal-context pairs (for convention analysis)
        self.signal_contexts = []

        # Statistics
        self.signal_diversity = 0.0
        self.signal_stability = 0.0

    def analyze_message(self, message: Dict, context: Dict):
        """
        Analyze a message

        Args:
            message: Message dictionary
            context: Context when message was sent (situation, coherence, etc.)
        """
        self.all_signals.append(message['signal'])
        self.signal_contexts.append((message['signal'], context))

    def compute_diversity(self) -> float:
        """
        Compute message diversity (entropy)

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(self.all_signals) < 10:
            return 0.0

        # Cluster signals (simple k-means)
        from sklearn.cluster import KMeans

        signals = np.array(self.all_signals[-1000:])  # Last 1000 messages
        n_clusters = min(20, len(signals) // 10)

        if n_clusters < 2:
            return 0.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(signals)

        # Compute entropy
        counts = np.bincount(labels)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-8))

        # Normalize by max entropy
        max_entropy = np.log2(n_clusters)
        diversity = entropy / max_entropy

        self.signal_diversity = diversity
        return diversity

    def compute_stability(self) -> float:
        """
        Compute signal stability (consistency over time)

        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self.all_signals) < 100:
            return 0.0

        # Compare recent signals to earlier signals
        recent = np.array(self.all_signals[-100:])
        earlier = np.array(self.all_signals[-200:-100])

        # Mean distance between recent and earlier
        distances = []
        for r in recent:
            min_dist = min(np.linalg.norm(r - e) for e in earlier)
            distances.append(min_dist)

        # Stability = 1 - normalized_distance
        avg_distance = np.mean(distances)
        max_distance = 2.0  # Assuming signals in [-1, 1], max distance ~2
        stability = 1.0 - min(avg_distance / max_distance, 1.0)

        self.signal_stability = stability
        return stability

    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        diversity = self.compute_diversity()
        stability = self.compute_stability()

        return {
            'total_signals_analyzed': len(self.all_signals),
            'signal_diversity': diversity,
            'signal_stability': stability
        }
