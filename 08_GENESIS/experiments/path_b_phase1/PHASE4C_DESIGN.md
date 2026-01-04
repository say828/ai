# GENESIS Phase 4C: Emergent Communication Design

**Date:** 2026-01-04
**Status:** 🚧 In Design

## Overview

Phase 4C implements **emergent communication** - the ability for agents to develop language-like protocols for coordination and information sharing.

### Key Innovation

Instead of hand-coding communication, agents:
1. **Evolve** communication protocols through interaction
2. **Discover** useful signals/messages automatically
3. **Coordinate** behavior through learned language
4. **Transfer** knowledge via communication

## Motivation

In nature, communication evolves when:
- **Coordination benefits** outweigh costs
- **Information sharing** provides fitness advantage
- **Social learning** accelerates adaptation

Examples:
- Bee waggle dance (resource location)
- Ant pheromone trails (path finding)
- Bird alarm calls (predator warning)
- Dolphin signature whistles (identity)

## Architecture

### 1. Communication Channel

**Message Structure:**
```python
message = {
    'sender_id': int,
    'signal': np.ndarray,  # Continuous vector (e.g., 8-dim)
    'channel': str,  # 'broadcast', 'local', 'directed'
    'timestamp': int
}
```

**Channels:**
- **Broadcast:** All agents receive
- **Local:** Only nearby agents (radius-based)
- **Directed:** Specific recipient

### 2. Communication Components

#### A. Message Encoder (Sender)
Converts internal state → message signal

```python
class MessageEncoder(nn.Module):
    def __init__(self, state_dim=128, message_dim=8):
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, message_dim),
            nn.Tanh()  # Bounded signals
        )

    def encode(self, internal_state):
        return self.encoder(internal_state)
```

#### B. Message Decoder (Receiver)
Converts received message → internal influence

```python
class MessageDecoder(nn.Module):
    def __init__(self, message_dim=8, influence_dim=32):
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, 32),
            nn.ReLU(),
            nn.Linear(32, influence_dim),
            nn.Tanh()
        )

    def decode(self, message):
        return self.decoder(message)
```

#### C. Attention Mechanism
Decides which messages to attend to

```python
class MessageAttention(nn.Module):
    def __init__(self, state_dim=128, message_dim=8):
        self.attention = nn.Sequential(
            nn.Linear(state_dim + message_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Attention weight
        )

    def compute_attention(self, state, message):
        combined = torch.cat([state, message])
        return self.attention(combined)
```

### 3. Communication Evolution

**Fitness Bonus for Communication:**
```python
def communication_fitness_bonus(agent, messages_sent, messages_received):
    # Reward useful communication
    bonus = 0.0

    # 1. Information gain (receiver benefits)
    for msg in messages_received:
        if msg_helped_decision(msg):
            bonus += 0.1

    # 2. Coordination success
    if coordinated_with_others():
        bonus += 0.2

    # 3. Cost of communication (energy)
    cost = len(messages_sent) * 0.01

    return bonus - cost
```

### 4. Message Types (Emergent)

Agents might discover:
- **Alarm:** "Danger nearby"
- **Resource:** "Food here"
- **Help:** "Need assistance"
- **Identity:** "I am agent X"
- **Coordination:** "Let's cooperate"

## Implementation

### Core Classes

```python
class CommunicatingAgent:
    """Agent with communication capabilities"""
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.encoder = MessageEncoder()
        self.decoder = MessageDecoder()
        self.attention = MessageAttention()

        # Communication history
        self.messages_sent = []
        self.messages_received = []

    def decide_to_communicate(self, state):
        """Decide whether to send message"""
        # Simple: communicate if coherence high and energy available
        if self.coherence > 0.7 and self.energy > 0.5:
            return True
        return False

    def create_message(self, state):
        """Generate message from current state"""
        signal = self.encoder.encode(state)
        return {
            'sender_id': self.id,
            'signal': signal,
            'channel': 'local',
            'timestamp': self.age
        }

    def process_messages(self, messages):
        """Process received messages"""
        if not messages:
            return None

        # Compute attention for each message
        attentions = []
        for msg in messages:
            attention = self.attention.compute_attention(
                self.state, msg['signal']
            )
            attentions.append(attention)

        # Weighted combination
        weights = softmax(attentions)
        combined_influence = sum(
            w * self.decoder.decode(msg['signal'])
            for w, msg in zip(weights, messages)
        )

        return combined_influence


class CommunicationManager:
    """Manages message passing between agents"""
    def __init__(self, local_radius=5.0):
        self.local_radius = local_radius
        self.message_buffer = []

    def broadcast_message(self, message, agents):
        """Send message to all agents"""
        for agent in agents:
            if agent.id != message['sender_id']:
                agent.messages_received.append(message)

    def local_broadcast(self, message, sender_pos, agents):
        """Send message to nearby agents"""
        for agent in agents:
            if agent.id == message['sender_id']:
                continue

            distance = np.linalg.norm(agent.position - sender_pos)
            if distance <= self.local_radius:
                agent.messages_received.append(message)

    def step(self, agents):
        """Process one communication step"""
        self.message_buffer.clear()

        # 1. Collect messages from agents
        for agent in agents:
            if agent.decide_to_communicate(agent.state):
                msg = agent.create_message(agent.state)
                self.message_buffer.append((msg, agent.position))

        # 2. Distribute messages
        for msg, sender_pos in self.message_buffer:
            if msg['channel'] == 'broadcast':
                self.broadcast_message(msg, agents)
            elif msg['channel'] == 'local':
                self.local_broadcast(msg, sender_pos, agents)
```

### Integration with GENESIS

```python
class Phase4C_CommunicationManager(Phase4B_OpenEndedManager):
    """Phase 4C: Adds emergent communication"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Wrap agents with communication
        self.comm_manager = CommunicationManager()
        self.communicating_agents = {}

        for agent in self.agents:
            self.communicating_agents[agent.id] = CommunicatingAgent(agent)

    def step(self):
        # Phase 4B step
        stats = super().step()

        # Communication step
        self.comm_manager.step(list(self.communicating_agents.values()))

        # Process messages and influence behavior
        for agent_id, comm_agent in self.communicating_agents.items():
            if comm_agent.messages_received:
                influence = comm_agent.process_messages(
                    comm_agent.messages_received
                )
                # Apply influence to agent's next action
                comm_agent.messages_received.clear()

        # Track stats
        stats['phase4c'] = {
            'total_messages': len(self.comm_manager.message_buffer),
            'avg_messages_per_agent': len(self.comm_manager.message_buffer) / len(self.agents)
        }

        return stats
```

## Expected Outcomes

### 1. Emergent Protocols
- Agents develop consistent signal meanings
- Different signals for different situations
- Stable communication conventions

### 2. Coordination Improvement
- Better resource sharing
- Coordinated exploration
- Group hunting/foraging
- Collective decision-making

### 3. Social Learning
- Knowledge transfer via communication
- Faster population-level learning
- Cultural transmission of strategies

## Metrics

```python
metrics = {
    'communication': {
        'total_messages_sent': 1523,
        'avg_messages_per_agent': 3.2,
        'communication_rate': 0.45,  # % of agents communicating

        'message_diversity': 0.73,  # Variety of messages (entropy)
        'signal_stability': 0.82,  # Consistency of signals over time

        'coordination_improvement': 1.34,  # Fitness boost from communication
        'information_transfer_rate': 0.56  # Knowledge spread speed
    }
}
```

## Success Criteria

| Metric | Target | Why |
|--------|--------|-----|
| Communication rate | >30% | Agents find it useful |
| Message diversity | >0.5 | Rich protocol developed |
| Signal stability | >0.7 | Conventions established |
| Coordination improvement | >1.2x | Communication helps |
| Faster learning | 2x faster | Social learning works |

## References

1. **Emergent Communication:** Lazaridou & Baroni (2020)
2. **Multi-Agent Communication:** Foerster et al. (2016)
3. **Language Evolution:** Nowak & Krakauer (1999)
4. **Signaling Games:** Lewis (1969)

---

**Next:** Implement `emergent_communication.py`
