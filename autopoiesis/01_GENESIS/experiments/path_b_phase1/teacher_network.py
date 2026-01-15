"""
GENESIS Path B: Teacher Network for Infinite Learning

Population-level knowledge preservation system that prevents knowledge loss
when individual agents die. Enables cumulative learning across generations.

Key Concept: Knowledge ≠ Individual Agents
           Knowledge = Organizational Structure of Population

The Teacher Network is the collective "memory" of the population, continuously
updated from elite agents and used to initialize new agents with accumulated
knowledge rather than random weights.
"""

import numpy as np
from typing import List, Dict, Optional
import copy


class TeacherNetwork:
    """
    Population-Level Knowledge Distillation

    The Teacher Network represents the accumulated knowledge of the entire
    population. Unlike individual agents that die and lose their learning,
    the teacher persists and continuously improves.

    Mechanism:
    1. Elite agents (top 20% by coherence) donate their genomes
    2. Teacher = Exponential Moving Average (EMA) of elite genomes
    3. New agents initialize from Teacher (NOT random!)
    4. Small mutations add variation for exploration

    Result: Each generation starts smarter than the previous generation.
            Knowledge accumulates rather than resetting.

    Mathematical Formulation:
        θ_teacher(t+1) = (1-α) θ_teacher(t) + α E[θ_elite(t)]

    where:
        - θ_teacher: Teacher's weights
        - θ_elite: Elite agents' weights
        - α: Learning rate (default 0.1)
        - E[·]: Expected value (mean) over elite population
    """

    def __init__(self,
                 state_dim: int = 128,
                 sensor_dim: int = 370,
                 action_dim: int = 5,
                 learning_rate: float = 0.1):
        """
        Initialize Teacher Network

        Args:
            state_dim: Dimensionality of internal RNN state
            sensor_dim: Dimensionality of sensory input
            action_dim: Dimensionality of action output
            learning_rate: EMA update rate (higher = faster adaptation)
        """
        self.state_dim = state_dim
        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Initialize with small random weights (He initialization)
        self.W_in = np.random.randn(state_dim, sensor_dim) * np.sqrt(2.0 / sensor_dim)
        self.W_rec = np.random.randn(state_dim, state_dim) * np.sqrt(2.0 / state_dim)
        self.W_out = np.random.randn(action_dim, state_dim) * np.sqrt(2.0 / state_dim)

        # Statistics tracking
        self.update_count = 0
        self.total_agents_learned_from = 0
        self.avg_elite_coherence_history = []
        self.knowledge_level = 0.0  # Estimated teacher knowledge level

    def distill_from_elite(self, elite_agents: List, verbose: bool = False) -> Dict:
        """
        Update Teacher from elite agents using EMA

        This is the core learning mechanism. The Teacher continuously learns
        from the best agents in the population, accumulating their knowledge.

        Args:
            elite_agents: List of top-performing agents
            verbose: Whether to print update info

        Returns:
            Statistics about the update
        """
        if not elite_agents:
            return {'updated': False, 'reason': 'no_elite_agents'}

        # Compute mean genome of elite agents
        W_in_elite = np.mean([a.W_in for a in elite_agents], axis=0)
        W_rec_elite = np.mean([a.W_rec for a in elite_agents], axis=0)
        W_out_elite = np.mean([a.W_out for a in elite_agents], axis=0)

        # Compute elite statistics
        elite_coherences = [
            a.coherence_history[-1] if a.coherence_history else 0.5
            for a in elite_agents
        ]
        avg_elite_coherence = np.mean(elite_coherences)

        # Exponential Moving Average update
        # Teacher slowly incorporates elite knowledge (stability)
        alpha = self.learning_rate
        self.W_in = (1 - alpha) * self.W_in + alpha * W_in_elite
        self.W_rec = (1 - alpha) * self.W_rec + alpha * W_rec_elite
        self.W_out = (1 - alpha) * self.W_out + alpha * W_out_elite

        # Update statistics
        self.update_count += 1
        self.total_agents_learned_from += len(elite_agents)
        self.avg_elite_coherence_history.append(avg_elite_coherence)

        # Estimate teacher knowledge level (moving average of elite coherence)
        if len(self.avg_elite_coherence_history) > 10:
            self.knowledge_level = np.mean(self.avg_elite_coherence_history[-10:])
        else:
            self.knowledge_level = avg_elite_coherence

        stats = {
            'updated': True,
            'n_elite': len(elite_agents),
            'avg_elite_coherence': avg_elite_coherence,
            'teacher_knowledge_level': self.knowledge_level,
            'update_count': self.update_count
        }

        if verbose:
            print(f"📚 Teacher Update #{self.update_count}:")
            print(f"   Elite agents: {len(elite_agents)}")
            print(f"   Avg coherence: {avg_elite_coherence:.3f}")
            print(f"   Teacher knowledge: {self.knowledge_level:.3f}")

        return stats

    def initialize_student(self, mutation_rate: float = 0.05,
                          mutation_scale: float = 0.1) -> Dict:
        """
        Create new agent genome from Teacher knowledge

        This is the KEY INNOVATION that prevents knowledge loss!
        Instead of random initialization, new agents inherit the
        accumulated knowledge of the population through the Teacher.

        Small mutations provide variation for exploration while
        preserving the core learned behaviors.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_scale: Standard deviation of mutations

        Returns:
            Genome dictionary ready for agent initialization
        """
        # Copy teacher's weights (inherited knowledge)
        genome = {
            'W_in': self.W_in.copy(),
            'W_rec': self.W_rec.copy(),
            'W_out': self.W_out.copy()
        }

        # Add small mutations for exploration
        # This creates variation while preserving learned structure
        for key in ['W_in', 'W_rec', 'W_out']:
            mutation_mask = np.random.rand(*genome[key].shape) < mutation_rate
            mutations = np.random.randn(*genome[key].shape) * mutation_scale
            genome[key] += mutation_mask * mutations

        # Add other genome parameters (not learned, but required)
        genome['b_state'] = np.zeros(self.state_dim)
        genome['b_action'] = np.zeros(self.action_dim)
        genome['coherence_sensitivity'] = np.random.uniform(0.5, 1.5)
        genome['metabolic_efficiency'] = np.random.uniform(0.8, 1.2)

        return genome

    def get_statistics(self) -> Dict:
        """
        Get comprehensive Teacher statistics

        Returns:
            Dictionary with all relevant metrics
        """
        stats = {
            'update_count': self.update_count,
            'total_agents_learned_from': self.total_agents_learned_from,
            'knowledge_level': self.knowledge_level,
            'weight_magnitudes': {
                'W_in_norm': float(np.linalg.norm(self.W_in)),
                'W_rec_norm': float(np.linalg.norm(self.W_rec)),
                'W_out_norm': float(np.linalg.norm(self.W_out))
            }
        }

        if self.avg_elite_coherence_history:
            stats['coherence_history'] = {
                'mean': float(np.mean(self.avg_elite_coherence_history)),
                'std': float(np.std(self.avg_elite_coherence_history)),
                'min': float(np.min(self.avg_elite_coherence_history)),
                'max': float(np.max(self.avg_elite_coherence_history)),
                'recent_10': [float(x) for x in self.avg_elite_coherence_history[-10:]]
            }

        return stats

    def save(self, filepath: str):
        """Save Teacher Network to file"""
        data = {
            'W_in': self.W_in.tolist(),
            'W_rec': self.W_rec.tolist(),
            'W_out': self.W_out.tolist(),
            'update_count': self.update_count,
            'total_agents_learned_from': self.total_agents_learned_from,
            'knowledge_level': self.knowledge_level,
            'coherence_history': self.avg_elite_coherence_history
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load Teacher Network from file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.W_in = np.array(data['W_in'])
        self.W_rec = np.array(data['W_rec'])
        self.W_out = np.array(data['W_out'])
        self.update_count = data['update_count']
        self.total_agents_learned_from = data['total_agents_learned_from']
        self.knowledge_level = data['knowledge_level']
        self.avg_elite_coherence_history = data['coherence_history']


class EpisodicMemory:
    """
    Phase 2 Enhanced Experience Replay Buffer

    Stores successful experiences with prioritization for:
    1. Curriculum learning (train new agents on past successes)
    2. Meta-learning (learn general patterns across experiences)
    3. Behavioral cloning (imitate successful behaviors)
    4. Critical event preservation (near-death, breakthroughs)

    Key Enhancement: Priority-based storage with temporal context
    """

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of experiences to store (10x increase)
        """
        self.capacity = capacity
        self.experiences = []
        self.priorities = []  # Priority scores (coherence + novelty + survival)
        self.timestamps = []  # When experience occurred
        self.contexts = []    # Environmental context

        # Statistics
        self.total_stored = 0
        self.critical_events = 0  # High priority events

    def store(self, experience: Dict, quality: float):
        """
        Backward compatibility - convert to store_critical_experience
        """
        self.store_critical_experience(experience, priority=quality)

    def store_critical_experience(self, experience: Dict, priority: float,
                                  context: Optional[Dict] = None):
        """
        Store experience with priority-based replacement

        Priority calculation:
        - priority = coherence_gain + novelty_bonus + survival_bonus

        High priority examples:
        - Achieved high coherence (priority +1.0)
        - Near-death survival (priority +2.0)
        - Novel pattern discovery (priority +0.5)

        Args:
            experience: Dictionary with 'observation', 'action', 'state', etc.
            priority: Priority score for this experience
            context: Additional environmental context
        """
        import time

        if len(self.experiences) >= self.capacity:
            # Remove lowest priority experience
            min_idx = np.argmin(self.priorities)
            if priority > self.priorities[min_idx]:
                self.experiences[min_idx] = copy.deepcopy(experience)
                self.priorities[min_idx] = priority
                self.timestamps[min_idx] = time.time()
                self.contexts[min_idx] = context or {}
        else:
            self.experiences.append(copy.deepcopy(experience))
            self.priorities.append(priority)
            self.timestamps.append(time.time())
            self.contexts.append(context or {})

        self.total_stored += 1
        if priority > 1.5:
            self.critical_events += 1

    def replay_for_learning(self, n: int = 128, temperature: float = 0.6) -> List[Dict]:
        """
        Prioritized Experience Replay (PER)

        Similar to human dreaming during REM sleep:
        - Replay important experiences
        - Extract patterns
        - Consolidate into long-term memory

        Args:
            n: Number of experiences to replay
            temperature: Prioritization strength (0.6 = moderate, 1.0 = strong)

        Returns:
            List of high-priority experiences
        """
        if not self.experiences:
            return []

        n = min(n, len(self.experiences))

        # Prioritized sampling with temperature
        probs = np.array(self.priorities) ** temperature
        probs = probs / probs.sum()

        indices = np.random.choice(
            len(self.experiences),
            size=n,
            replace=False,
            p=probs
        )

        return [self.experiences[i] for i in indices]

    def sample(self, n: int = 32, prioritize: bool = True) -> List[Dict]:
        """
        Sample experiences (backward compatible interface)
        """
        if prioritize:
            return self.replay_for_learning(n)
        else:
            # Uniform sampling
            if not self.experiences:
                return []
            n = min(n, len(self.experiences))
            indices = np.random.choice(len(self.experiences), size=n, replace=False)
            return [self.experiences[i] for i in indices]

    def get_recent_experiences(self, n: int = 100) -> List[Dict]:
        """Get most recent experiences regardless of priority"""
        if not self.experiences:
            return []
        n = min(n, len(self.experiences))
        return self.experiences[-n:]

    def get_statistics(self) -> Dict:
        """Get comprehensive memory statistics"""
        if not self.priorities:
            return {'size': 0}

        return {
            'size': len(self.experiences),
            'capacity': self.capacity,
            'utilization': len(self.experiences) / self.capacity,
            'avg_priority': float(np.mean(self.priorities)),
            'max_priority': float(np.max(self.priorities)),
            'min_priority': float(np.min(self.priorities)),
            'total_stored': self.total_stored,
            'critical_events': self.critical_events,
            'priority_distribution': {
                'p50': float(np.percentile(self.priorities, 50)),
                'p75': float(np.percentile(self.priorities, 75)),
                'p90': float(np.percentile(self.priorities, 90)),
                'p95': float(np.percentile(self.priorities, 95))
            }
        }
