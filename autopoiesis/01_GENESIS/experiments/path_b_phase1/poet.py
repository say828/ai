"""
GENESIS Phase 4B: POET (Paired Open-Ended Trailblazer)

Coevolves agents and environments together for open-ended learning.

Key insight: Intelligence doesn't evolve in a vacuum - it evolves in response to
challenges. POET generates a curriculum of increasingly challenging environments
automatically, while agents evolve to solve them. This creates an open-ended
evolutionary arms race.

References:
- Wang et al. (2019) "Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex Environments and their Solutions"
- Wang et al. (2020) "Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
from collections import deque


class EnvironmentGenerator:
    """
    Generates environment variants with controllable difficulty

    Creates new environments by mutating existing ones
    """

    def __init__(self,
                 base_env_config: Dict,
                 mutation_rate: float = 0.3,
                 difficulty_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Args:
            base_env_config: Base environment configuration
            mutation_rate: Probability of mutating each parameter
            difficulty_range: (min_difficulty, max_difficulty)
        """
        self.base_config = base_env_config
        self.mutation_rate = mutation_rate
        self.min_difficulty, self.max_difficulty = difficulty_range

        # Mutable parameters and their ranges
        self.mutable_params = {
            'size': (20, 100),  # Grid size
            'resource_density': (0.1, 0.9),  # Initial resource density
            'resource_growth_rate': (0.001, 0.05),  # Resource regeneration speed
            'predator_count': (0, 10),  # Number of predators
            'obstacle_density': (0.0, 0.5),  # Fraction of obstacles
            'energy_decay': (0.001, 0.01),  # Energy decay rate
            'temperature_variation': (0.0, 0.5),  # Environmental variability
        }

        # Difficulty factors (how each parameter affects difficulty)
        self.difficulty_weights = {
            'size': 0.2,  # Larger = harder (more exploration needed)
            'resource_density': -0.3,  # Less resources = harder
            'resource_growth_rate': -0.2,  # Slower growth = harder
            'predator_count': 0.3,  # More predators = harder
            'obstacle_density': 0.2,  # More obstacles = harder
            'energy_decay': 0.3,  # Faster decay = harder
            'temperature_variation': 0.2,  # More variation = harder
        }

        # Generation counter
        self.generation = 0

    def generate_initial(self) -> Dict:
        """Generate initial minimal environment"""
        config = copy.deepcopy(self.base_config)

        # Minimal difficulty settings
        config['size'] = 30
        config['resource_density'] = 0.7
        config['resource_growth_rate'] = 0.02
        config['predator_count'] = 0
        config['obstacle_density'] = 0.0
        config['energy_decay'] = 0.002
        config['temperature_variation'] = 0.0

        config['difficulty'] = self.compute_difficulty(config)
        config['generation'] = 0

        return config

    def mutate(self, parent_config: Dict) -> Dict:
        """
        Generate new environment by mutating parent

        Args:
            parent_config: Parent environment configuration

        Returns:
            Mutated environment configuration
        """
        child_config = copy.deepcopy(parent_config)

        # Mutate parameters
        for param, (min_val, max_val) in self.mutable_params.items():
            if np.random.random() < self.mutation_rate:
                current_value = child_config.get(param, (min_val + max_val) / 2)

                # Gaussian mutation
                sigma = (max_val - min_val) * 0.1  # 10% of range
                mutated_value = current_value + np.random.normal(0, sigma)

                # Clip to valid range
                mutated_value = np.clip(mutated_value, min_val, max_val)

                child_config[param] = mutated_value

        # Update difficulty
        child_config['difficulty'] = self.compute_difficulty(child_config)
        child_config['generation'] = parent_config.get('generation', 0) + 1

        self.generation += 1

        return child_config

    def compute_difficulty(self, config: Dict) -> float:
        """
        Compute difficulty of environment configuration

        Args:
            config: Environment configuration

        Returns:
            Difficulty score (0-1, higher = harder)
        """
        difficulty = 0.0

        for param, weight in self.difficulty_weights.items():
            if param not in config:
                continue

            # Normalize parameter to [0, 1]
            min_val, max_val = self.mutable_params[param]
            value = config[param]
            normalized = (value - min_val) / (max_val - min_val)

            # Add weighted contribution
            difficulty += weight * normalized

        # Normalize to [0, 1]
        total_weight = sum(abs(w) for w in self.difficulty_weights.values())
        difficulty = (difficulty + total_weight / 2) / total_weight

        return np.clip(difficulty, 0.0, 1.0)

    def generate_variants(self, parent_config: Dict, n: int = 5) -> List[Dict]:
        """
        Generate multiple variants of parent environment

        Args:
            parent_config: Parent configuration
            n: Number of variants to generate

        Returns:
            List of environment configurations
        """
        return [self.mutate(parent_config) for _ in range(n)]


class AgentEnvironmentPair:
    """
    Pair of agent and its environment

    The fundamental unit in POET
    """

    def __init__(self,
                 agent,
                 env_config: Dict,
                 pair_id: int):
        """
        Args:
            agent: Agent (will be deep copied)
            env_config: Environment configuration
            pair_id: Unique pair ID
        """
        self.agent = copy.deepcopy(agent)
        self.env_config = copy.deepcopy(env_config)
        self.pair_id = pair_id

        # Performance tracking
        self.fitness_history = deque(maxlen=100)
        self.generation = 0

        # Transfer history
        self.origin_pair_id = pair_id  # Where this agent came from
        self.transfers_received = 0

    def get_current_fitness(self) -> float:
        """Get current fitness (average of recent history)"""
        if not self.fitness_history:
            return 0.0
        return np.mean(self.fitness_history)

    def update_fitness(self, fitness: float):
        """Update fitness history"""
        self.fitness_history.append(fitness)
        self.generation += 1

    def get_difficulty(self) -> float:
        """Get environment difficulty"""
        return self.env_config.get('difficulty', 0.5)

    def is_solved(self, threshold: float = 0.8) -> bool:
        """Check if environment is solved (agent performs well)"""
        return self.get_current_fitness() > threshold


class POETSystem:
    """
    POET: Paired Open-Ended Trailblazer

    Coevolves agents and environments
    """

    def __init__(self,
                 base_env_config: Dict,
                 max_active_pairs: int = 20,
                 pair_selection_tournament_size: int = 5,
                 transfer_interval: int = 10,
                 solved_threshold: float = 0.8,
                 minimal_criterion_fitness: float = 0.3):
        """
        Args:
            base_env_config: Base environment configuration
            max_active_pairs: Maximum number of active pairs to maintain
            pair_selection_tournament_size: Size of tournament for pair selection
            transfer_interval: How often to attempt transfers
            solved_threshold: Fitness threshold to consider environment solved
            minimal_criterion_fitness: Minimum fitness for environment to be kept
        """
        self.base_env_config = base_env_config
        self.max_active_pairs = max_active_pairs
        self.pair_selection_tournament_size = pair_selection_tournament_size
        self.transfer_interval = transfer_interval
        self.solved_threshold = solved_threshold
        self.minimal_criterion_fitness = minimal_criterion_fitness

        # Environment generator
        self.env_generator = EnvironmentGenerator(base_env_config)

        # Active pairs
        self.active_pairs = []  # List of AgentEnvironmentPair

        # Archives
        self.solved_pairs = []  # Pairs that solved their environment
        self.all_pairs_history = []  # All pairs ever created

        # Statistics
        self.generation = 0
        self.total_pairs_created = 0
        self.total_transfers_attempted = 0
        self.total_transfers_successful = 0
        self.total_envs_generated = 0

    def initialize(self, initial_agents: List):
        """
        Initialize POET with initial agents and minimal environment

        Args:
            initial_agents: List of initial agents
        """
        # Create minimal environment
        minimal_env = self.env_generator.generate_initial()

        # Create initial pairs
        for i, agent in enumerate(initial_agents):
            pair = AgentEnvironmentPair(
                agent=agent,
                env_config=minimal_env,
                pair_id=self.total_pairs_created
            )
            self.active_pairs.append(pair)
            self.total_pairs_created += 1

    def step(self, evolution_fn: callable) -> Dict:
        """
        Execute one POET generation

        Args:
            evolution_fn: Function that evolves agents in their environments
                         Signature: evolution_fn(agent, env_config) -> (new_agent, fitness)

        Returns:
            Statistics dictionary
        """
        # 1. Evolve agents in their environments
        for pair in self.active_pairs:
            new_agent, fitness = evolution_fn(pair.agent, pair.env_config)
            pair.agent = new_agent
            pair.update_fitness(fitness)

        # 2. Generate new environment variants (periodically)
        if self.generation % 5 == 0:
            self._generate_new_pairs()

        # 3. Transfer agents between environments (periodically)
        if self.generation % self.transfer_interval == 0:
            self._attempt_transfers(evolution_fn)

        # 4. Prune unsuccessful pairs
        self._prune_pairs()

        # 5. Archive solved pairs
        self._archive_solved_pairs()

        self.generation += 1

        return self.get_statistics()

    def _generate_new_pairs(self):
        """Generate new environment variants and pair with agents"""
        if not self.active_pairs:
            return

        # Select parents for mutation
        parent_pairs = np.random.choice(
            self.active_pairs,
            size=min(3, len(self.active_pairs)),
            replace=False
        )

        for parent_pair in parent_pairs:
            # Generate environment variant
            new_env = self.env_generator.mutate(parent_pair.env_config)

            # Pair with parent's agent (will evolve separately)
            new_pair = AgentEnvironmentPair(
                agent=parent_pair.agent,
                env_config=new_env,
                pair_id=self.total_pairs_created
            )
            new_pair.origin_pair_id = parent_pair.pair_id

            self.active_pairs.append(new_pair)
            self.all_pairs_history.append(new_pair)
            self.total_pairs_created += 1
            self.total_envs_generated += 1

    def _attempt_transfers(self, evolution_fn: callable):
        """Attempt to transfer agents between environments"""
        if len(self.active_pairs) < 2:
            return

        # Try multiple random transfers
        n_transfers = min(10, len(self.active_pairs) * 2)

        for _ in range(n_transfers):
            # Select random agent and environment
            agent_pair = np.random.choice(self.active_pairs)
            env_pair = np.random.choice(self.active_pairs)

            if agent_pair == env_pair:
                continue

            # Evaluate agent in new environment
            _, fitness = evolution_fn(agent_pair.agent, env_pair.env_config)

            self.total_transfers_attempted += 1

            # Check if transfer is beneficial
            current_fitness = env_pair.get_current_fitness()

            if fitness > current_fitness + 0.1:  # 10% improvement threshold
                # Successful transfer!
                env_pair.agent = copy.deepcopy(agent_pair.agent)
                env_pair.update_fitness(fitness)
                env_pair.origin_pair_id = agent_pair.pair_id
                env_pair.transfers_received += 1

                self.total_transfers_successful += 1

    def _prune_pairs(self):
        """Remove poorly performing pairs"""
        if len(self.active_pairs) <= self.max_active_pairs:
            return

        # Compute scores for each pair
        scores = []
        for pair in self.active_pairs:
            fitness = pair.get_current_fitness()
            difficulty = pair.get_difficulty()

            # Score = fitness * difficulty (reward hard problems)
            # Also consider minimal criterion
            if fitness < self.minimal_criterion_fitness:
                score = 0.0  # Mark for removal
            else:
                score = fitness * (0.5 + 0.5 * difficulty)

            scores.append(score)

        # Keep top pairs
        scores = np.array(scores)
        keep_indices = np.argsort(scores)[-self.max_active_pairs:]

        self.active_pairs = [self.active_pairs[i] for i in keep_indices]

    def _archive_solved_pairs(self):
        """Move solved pairs to archive"""
        unsolved = []

        for pair in self.active_pairs:
            if pair.is_solved(self.solved_threshold):
                self.solved_pairs.append(pair)
            else:
                unsolved.append(pair)

        self.active_pairs = unsolved

    def get_statistics(self) -> Dict:
        """Get POET statistics"""
        if not self.active_pairs:
            return {
                'generation': self.generation,
                'active_pairs': 0,
                'solved_pairs': len(self.solved_pairs),
                'total_pairs_created': self.total_pairs_created,
                'total_envs_generated': self.total_envs_generated,
                'transfer_success_rate': 0.0,
                'avg_difficulty': 0.0,
                'max_difficulty': 0.0,
                'avg_fitness': 0.0
            }

        difficulties = [pair.get_difficulty() for pair in self.active_pairs]
        fitnesses = [pair.get_current_fitness() for pair in self.active_pairs]

        transfer_rate = (self.total_transfers_successful /
                        max(self.total_transfers_attempted, 1))

        return {
            'generation': self.generation,
            'active_pairs': len(self.active_pairs),
            'solved_pairs': len(self.solved_pairs),
            'total_pairs_created': self.total_pairs_created,
            'total_envs_generated': self.total_envs_generated,
            'total_transfers_attempted': self.total_transfers_attempted,
            'total_transfers_successful': self.total_transfers_successful,
            'transfer_success_rate': transfer_rate,
            'avg_difficulty': np.mean(difficulties),
            'max_difficulty': np.max(difficulties),
            'min_difficulty': np.min(difficulties),
            'std_difficulty': np.std(difficulties),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses)
        }

    def get_best_agent_for_difficulty(self, target_difficulty: float) -> Optional[AgentEnvironmentPair]:
        """
        Get best agent for a specific difficulty level

        Useful for curriculum learning / transfer
        """
        # Find pairs with similar difficulty
        candidates = []
        for pair in self.active_pairs + self.solved_pairs:
            diff_delta = abs(pair.get_difficulty() - target_difficulty)
            if diff_delta < 0.2:  # Within 20% of target
                candidates.append((pair, diff_delta))

        if not candidates:
            return None

        # Sort by fitness (within difficulty range)
        candidates.sort(key=lambda x: x[0].get_current_fitness(), reverse=True)

        return candidates[0][0]


def create_poet_system(base_env_config: Dict,
                      max_active_pairs: int = 20) -> POETSystem:
    """
    Convenience function to create POET system

    Args:
        base_env_config: Base environment configuration
        max_active_pairs: Maximum number of active pairs

    Returns:
        Configured POET system
    """
    return POETSystem(
        base_env_config=base_env_config,
        max_active_pairs=max_active_pairs,
        transfer_interval=10,
        solved_threshold=0.8,
        minimal_criterion_fitness=0.3
    )
