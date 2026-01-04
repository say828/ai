"""
GENESIS Phase 4B Integration: Open-Ended Learning Manager

Integrates Novelty Search, MAP-Elites, and POET with Phase 4A system.

Key capabilities:
- Behavioral diversity through novelty search
- Quality-diversity through MAP-Elites
- Automatic curriculum through POET
- Combines all three approaches for maximum open-endedness
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque

# Phase 4A
from phase4_integration import Phase4PopulationManager
from full_environment import FullALifeEnvironment

# Phase 4B
from novelty_search import NoveltySearch, BehaviorCharacterizer
from map_elites import MAPElites
from poet import POETSystem, AgentEnvironmentPair


class GENESIS_BehaviorCharacterizer:
    """
    Behavior characterization specialized for GENESIS agents

    Extracts behavioral features from agent lifetime
    """

    def __init__(self, characterization_type: str = 'composite'):
        """
        Args:
            characterization_type: Type of characterization
                - 'trajectory': Final position
                - 'resource': Resource usage pattern
                - 'reproduction': Reproduction strategy
                - 'composite': All combined
        """
        self.characterization_type = characterization_type

    def characterize_agent(self, agent) -> Dict[str, float]:
        """
        Extract behavior from agent

        Args:
            agent: FullAutopoieticAgent

        Returns:
            Behavior dictionary (dimension_name -> value)
        """
        behavior = {}

        # Position-based features
        if hasattr(agent, 'position'):
            behavior['final_x'] = agent.position[0]
            behavior['final_y'] = agent.position[1]

        # Resource usage
        if hasattr(agent, 'energy'):
            behavior['avg_energy'] = agent.energy

        if hasattr(agent, 'material'):
            behavior['avg_material'] = agent.material

        # Reproduction
        if hasattr(agent, 'total_offspring'):
            behavior['reproduction_count'] = float(agent.total_offspring)

        # Age/survival
        if hasattr(agent, 'age'):
            behavior['survival_time'] = float(agent.age)

        # Coherence (organizational quality)
        if hasattr(agent, 'compute_coherence'):
            coherence_dict = agent.compute_coherence()
            behavior['coherence'] = coherence_dict.get('composite', 0.5)

        # Movement style (if position history available)
        if hasattr(agent, 'position_history') and len(agent.position_history) > 10:
            # Convert deque to list and slice
            pos_list = list(agent.position_history)[-100:]
            positions = np.array(pos_list)
            movements = np.diff(positions, axis=0)
            speeds = np.linalg.norm(movements, axis=1)

            behavior['avg_speed'] = float(np.mean(speeds))
            behavior['exploration_radius'] = float(np.max(np.linalg.norm(
                positions - positions[0], axis=1
            )))

        return behavior

    def to_numpy(self, behavior: Dict[str, float]) -> np.ndarray:
        """Convert behavior dict to numpy array (for novelty search)"""
        # Fixed order of dimensions
        dims = ['final_x', 'final_y', 'avg_energy', 'avg_material',
                'reproduction_count', 'survival_time', 'coherence',
                'avg_speed', 'exploration_radius']

        return np.array([behavior.get(dim, 0.0) for dim in dims], dtype=np.float32)


class Phase4B_OpenEndedManager(Phase4PopulationManager):
    """
    Phase 4B: Open-Ended Learning Manager

    Extends Phase 4A with open-ended evolution capabilities
    """

    def __init__(self,
                 env: FullALifeEnvironment,
                 # Phase 4A parameters (inherited)
                 initial_pop: int = 300,
                 max_population: int = 500,
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.1,
                 min_population: int = 80,
                 enable_teacher: bool = True,
                 teacher_update_interval: int = 100,
                 teacher_learning_rate: float = 0.01,
                 phase2_enabled: bool = True,
                 episodic_capacity: int = 100000,
                 enable_semantic: bool = True,
                 enable_stigmergy: bool = True,
                 enable_meta_learning: bool = True,
                 phase3_enabled: bool = True,
                 phase4_enabled: bool = True,
                 use_advanced_teacher: bool = True,
                 use_learned_memory: bool = True,
                 use_knowledge_guidance: bool = True,
                 # Phase 4B parameters
                 phase4b_enabled: bool = True,
                 use_novelty_search: bool = True,
                 use_map_elites: bool = True,
                 use_poet: bool = False,  # POET disabled by default (complex)
                 novelty_weight: float = 0.5,  # 50% novelty, 50% fitness
                 map_elites_bins: int = 10):
        """
        Initialize Phase 4B system

        Args:
            env: Environment
            [Phase 4A parameters inherited...]
            phase4b_enabled: Enable Phase 4B features
            use_novelty_search: Use novelty search
            use_map_elites: Use MAP-Elites
            use_poet: Use POET (coevolution)
            novelty_weight: Weight for novelty (1-weight for fitness)
            map_elites_bins: Bins per dimension for MAP-Elites
        """
        # Initialize Phase 4A
        super().__init__(
            env=env,
            initial_pop=initial_pop,
            max_population=max_population,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            min_population=min_population,
            enable_teacher=enable_teacher,
            teacher_update_interval=teacher_update_interval,
            teacher_learning_rate=teacher_learning_rate,
            phase2_enabled=phase2_enabled,
            episodic_capacity=episodic_capacity,
            enable_semantic=enable_semantic,
            enable_stigmergy=enable_stigmergy,
            enable_meta_learning=enable_meta_learning,
            phase3_enabled=phase3_enabled,
            phase4_enabled=phase4_enabled,
            use_advanced_teacher=use_advanced_teacher,
            use_learned_memory=use_learned_memory,
            use_knowledge_guidance=use_knowledge_guidance
        )

        self.phase4b_enabled = phase4b_enabled

        if not self.phase4b_enabled:
            self.novelty_search = None
            self.map_elites = None
            self.poet = None
            return

        # Behavior characterizer
        self.behavior_characterizer = GENESIS_BehaviorCharacterizer('composite')

        # Novelty Search
        if use_novelty_search:
            base_characterizer = BehaviorCharacterizer('composite')
            self.novelty_search = NoveltySearch(
                behavior_characterizer=base_characterizer,
                k_nearest=15,
                combine_with_fitness=True,
                fitness_weight=1.0 - novelty_weight
            )
        else:
            self.novelty_search = None

        # MAP-Elites
        if use_map_elites:
            # Define behavioral dimensions for GENESIS
            behavior_dims = [
                'avg_energy',
                'reproduction_count',
                'coherence',
                'exploration_radius'
            ]

            self.map_elites = MAPElites(
                behavior_dims=behavior_dims,
                bins_per_dim=map_elites_bins,
                behavior_ranges=None  # Auto-detect
            )
        else:
            self.map_elites = None

        # POET (optional, complex)
        if use_poet:
            base_env_config = {
                'size': env.size,
                'resource_density': 0.5,
                'resource_growth_rate': 0.01,
                'predator_count': 0,
                'obstacle_density': 0.0,
                'energy_decay': 0.005,
                'temperature_variation': 0.0
            }

            self.poet = POETSystem(
                base_env_config=base_env_config,
                max_active_pairs=20
            )

            # Initialize POET with current population
            if self.agents:
                elite_agents = self._get_elite_agents(top_k_percent=0.1)
                self.poet.initialize(elite_agents[:5])  # Start with 5 best
        else:
            self.poet = None

        # Statistics
        self.behaviors_discovered = 0
        self.map_elites_improvements = 0

    def step(self) -> Dict:
        """Execute one step with Phase 4B enhancements"""
        if not self.phase4b_enabled:
            return super().step()

        # Phase 4A step (includes Phases 1-3)
        stats = super().step()

        # Phase 4B enhancements
        self._phase4b_step()

        # Add Phase 4B statistics
        if self.phase4b_enabled:
            stats['phase4b'] = {}

            if self.novelty_search:
                stats['phase4b']['novelty_search'] = self.novelty_search.get_statistics()

            if self.map_elites:
                stats['phase4b']['map_elites'] = self.map_elites.get_statistics()

            if self.poet:
                stats['phase4b']['poet'] = self.poet.get_statistics()

            stats['phase4b']['behaviors_discovered'] = self.behaviors_discovered
            stats['phase4b']['map_elites_improvements'] = self.map_elites_improvements

        return stats

    def _phase4b_step(self):
        """Phase 4B processing"""
        # Process agents with novelty search and MAP-Elites
        for agent in self.agents:
            # Extract behavior
            behavior_dict = self.behavior_characterizer.characterize_agent(agent)

            # Compute fitness (coherence)
            fitness = agent.compute_coherence()['composite']

            # Novelty Search
            if self.novelty_search:
                # Create agent history for novelty search
                agent_history = {
                    'positions': getattr(agent, 'position_history', []),
                    'energies': [agent.energy] if hasattr(agent, 'energy') else [],
                    'materials': [agent.material] if hasattr(agent, 'material') else [],
                    'actions': [],
                    'reproductions': getattr(agent, 'total_offspring', 0)
                }

                score, info = self.novelty_search.evaluate_agent(
                    agent_history=agent_history,
                    fitness=fitness
                )

                if info['added_to_archive']:
                    self.behaviors_discovered += 1

            # MAP-Elites
            if self.map_elites:
                result = self.map_elites.add_solution(
                    agent=agent,
                    fitness=fitness,
                    behavior=behavior_dict
                )

                if result['added'] and result['reason'] == 'improvement':
                    self.map_elites_improvements += 1

        # POET step (if enabled)
        if self.poet and self.current_step % 100 == 0:
            # POET evolution function
            def evolution_fn(agent, env_config):
                # Simplified: just evaluate agent
                # In full implementation, would evolve agent in environment
                fitness = agent.compute_coherence()['composite']
                return agent, fitness

            self.poet.step(evolution_fn)

    def get_diverse_population(self, n: int = 10) -> List:
        """
        Get diverse subset of population

        Uses MAP-Elites if available, otherwise samples randomly

        Args:
            n: Number of agents to return

        Returns:
            List of diverse agents
        """
        if self.map_elites and len(self.map_elites.archive.archive) > 0:
            # Get diverse elites from MAP-Elites
            diverse_elites = self.map_elites.get_diverse_elites(n=n)
            return [agent for agent, _, _, _ in diverse_elites]
        else:
            # Random sample
            n = min(n, len(self.agents))
            return list(np.random.choice(self.agents, size=n, replace=False))

    def get_best_agent_for_behavior(self, target_behavior: Dict[str, float]):
        """
        Find agent closest to target behavior

        Useful for transfer learning / task adaptation
        """
        if not self.agents:
            return None

        # Compute distances
        best_agent = None
        best_distance = float('inf')

        for agent in self.agents:
            agent_behavior = self.behavior_characterizer.characterize_agent(agent)

            # Euclidean distance in behavior space
            distance = 0.0
            for key in target_behavior:
                if key in agent_behavior:
                    distance += (target_behavior[key] - agent_behavior[key]) ** 2

            distance = np.sqrt(distance)

            if distance < best_distance:
                best_distance = distance
                best_agent = agent

        return best_agent


def create_phase4b_system(
        env_size: int = 50,
        initial_population: int = 300,
        phase1_enabled: bool = True,
        phase2_enabled: bool = True,
        phase3_enabled: bool = True,
        phase4_enabled: bool = True,
        phase4b_enabled: bool = True,
        use_novelty_search: bool = True,
        use_map_elites: bool = True,
        use_poet: bool = False) -> Phase4B_OpenEndedManager:
    """
    Create complete Phase 4B system

    Args:
        env_size: Environment size
        initial_population: Initial population
        phase1_enabled: Enable Phase 1
        phase2_enabled: Enable Phase 2
        phase3_enabled: Enable Phase 3
        phase4_enabled: Enable Phase 4A
        phase4b_enabled: Enable Phase 4B
        use_novelty_search: Use novelty search
        use_map_elites: Use MAP-Elites
        use_poet: Use POET

    Returns:
        Phase4B_OpenEndedManager
    """
    # Create environment
    env = FullALifeEnvironment(size=env_size)

    # Create manager
    manager = Phase4B_OpenEndedManager(
        env=env,
        initial_pop=initial_population,
        max_population=500,
        phase2_enabled=phase2_enabled,
        phase3_enabled=phase3_enabled,
        phase4_enabled=phase4_enabled,
        use_advanced_teacher=phase4_enabled,
        use_learned_memory=phase4_enabled,
        use_knowledge_guidance=phase4_enabled,
        phase4b_enabled=phase4b_enabled,
        use_novelty_search=use_novelty_search,
        use_map_elites=use_map_elites,
        use_poet=use_poet,
        novelty_weight=0.5,  # 50% novelty, 50% fitness
        map_elites_bins=10
    )

    return manager
