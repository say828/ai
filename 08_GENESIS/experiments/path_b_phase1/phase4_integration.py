"""
GENESIS Phase 4 Integration (FIXED)

Integrates all phases by extending Phase2PopulationManager
with Phase 4 components as addons.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque

# Phase 1-2 imports
from full_environment import FullALifeEnvironment
from phase2_population import Phase2PopulationManager

# Phase 3 imports
from knowledge_ingestion import KnowledgeIngestionPipeline
from query_system import QuerySystem
from universal_knowledge_graph import UniversalKnowledgeGraph

# Phase 4 imports
from advanced_teacher import AdvancedTeacherNetwork
from learned_memory import LearnedEpisodicMemory, MemoryConsolidator
from knowledge_guided_agent import KnowledgeGuidedAgent


class Phase4PopulationManager(Phase2PopulationManager):
    """
    Phase 4 Population Manager

    Extends Phase 2 with Phase 4 features as optional addons
    """

    def __init__(self,
                 env: FullALifeEnvironment,
                 # Phase 1-2 parameters (compatible with Phase2PopulationManager)
                 initial_pop: int = 300,
                 max_population: int = 500,
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.1,
                 min_population: int = 80,
                 enable_teacher: bool = True,
                 teacher_update_interval: int = 100,
                 teacher_learning_rate: float = 0.01,
                 # Phase 2 parameters
                 phase2_enabled: bool = True,
                 episodic_capacity: int = 100000,
                 enable_semantic: bool = True,
                 enable_stigmergy: bool = True,
                 enable_meta_learning: bool = True,
                 # Phase 3 parameters
                 phase3_enabled: bool = True,
                 # Phase 4 parameters
                 phase4_enabled: bool = True,
                 use_advanced_teacher: bool = True,
                 use_learned_memory: bool = True,
                 use_knowledge_guidance: bool = True):
        """
        Initialize Phase 4 system

        Args:
            env: Environment
            initial_pop: Initial population size
            max_population: Maximum population
            mutation_rate: Mutation probability
            mutation_scale: Mutation magnitude
            min_population: Minimum population
            enable_teacher: Enable teacher network
            teacher_update_interval: Teacher update frequency
            teacher_learning_rate: Teacher learning rate
            phase2_enabled: Enable Phase 2 features
            episodic_capacity: Episodic memory capacity
            enable_semantic: Enable semantic memory
            enable_stigmergy: Enable stigmergy
            enable_meta_learning: Enable meta-learning
            phase3_enabled: Enable Phase 3 features
            phase4_enabled: Enable Phase 4 features
            use_advanced_teacher: Use advanced teacher
            use_learned_memory: Use learned memory
            use_knowledge_guidance: Use knowledge guidance
        """
        # Initialize Phase 2 (which includes Phase 1)
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
            enable_meta_learning=enable_meta_learning
        )

        self.phase3_enabled = phase3_enabled
        self.phase4_enabled = phase4_enabled

        # Phase 3: Knowledge System
        if self.phase3_enabled:
            self.knowledge_ingestion = KnowledgeIngestionPipeline(quality_threshold=0.3)
            self.knowledge_graph = UniversalKnowledgeGraph()
            self.query_system = QuerySystem(self.knowledge_graph, self.knowledge_ingestion)
        else:
            self.knowledge_ingestion = None
            self.knowledge_graph = None
            self.query_system = None

        # Phase 4: Advanced Intelligence
        if self.phase4_enabled:
            # Get network shape from first agent
            if self.agents:
                agent = self.agents[0]
                # Extract shape from FullAutopoieticAgent structure
                # W_in: (input_dim, state_dim)
                # W_rec: (state_dim, state_dim)
                # W_out: (state_dim, output_dim)
                input_dim = agent.W_in.shape[0]
                state_dim = agent.W_in.shape[1]
                output_dim = agent.W_out.shape[1]
                network_shape = [input_dim, state_dim, output_dim]
            else:
                network_shape = [370, 256, 10]  # Default

            # Advanced teacher
            if use_advanced_teacher:
                self.advanced_teacher = AdvancedTeacherNetwork(
                    network_shape=network_shape,
                    exploration_lr=0.02,
                    exploitation_lr=0.01,
                    robustness_lr=0.005
                )
            else:
                self.advanced_teacher = None

            # Learned memory (replaces Phase 2 episodic memory)
            if use_learned_memory and self.phase2_enabled:
                self.learned_memory = LearnedEpisodicMemory(
                    initial_capacity=episodic_capacity,
                    min_capacity=max(episodic_capacity // 10, 10000),
                    max_capacity=episodic_capacity * 10
                )
                self.memory_consolidator = MemoryConsolidator(self.learned_memory)
                # Replace Phase 2 memory
                self.memory = self.learned_memory
            else:
                self.learned_memory = None
                self.memory_consolidator = None

            # Knowledge-guided agents
            if use_knowledge_guidance and self.phase3_enabled:
                self.knowledge_guided_agents = {}
            else:
                self.knowledge_guided_agents = None
        else:
            self.advanced_teacher = None
            self.learned_memory = None
            self.memory_consolidator = None
            self.knowledge_guided_agents = None

        # Statistics
        self.total_knowledge_queries = 0
        self.total_concepts_discovered = 0

    def step(self) -> Dict:
        """Execute one step with Phase 4 enhancements"""
        if not self.phase4_enabled:
            return super().step()

        # Use Phase 2 step as base, then add Phase 4 enhancements
        stats = super().step()

        # Phase 4 enhancements after Phase 2 step
        if self.current_step % 100 == 0:  # Every 100 steps
            self._phase4_enhancements()

        # Add Phase 4 statistics
        if self.phase3_enabled:
            stats['phase3'] = {
                'knowledge_graph': self.knowledge_graph.get_statistics(),
                'query_system': self.query_system.get_statistics() if self.query_system else {},
                'ingestion': self.knowledge_ingestion.get_statistics() if self.knowledge_ingestion else {}
            }

        if self.phase4_enabled:
            stats['phase4'] = {}

            if self.advanced_teacher:
                context = self._build_context()
                # Update advanced teacher with current elites
                elites = self._select_elites()
                self.advanced_teacher.update(elites, context)
                stats['phase4']['advanced_teacher'] = self.advanced_teacher.get_statistics()

            if self.learned_memory:
                stats['phase4']['learned_memory'] = self.learned_memory.get_statistics()

            if self.knowledge_guided_agents:
                kg_stats = {
                    'total_agents_wrapped': len(self.knowledge_guided_agents),
                    'total_queries': sum(kg.queries_made for kg in self.knowledge_guided_agents.values()),
                    'total_knowledge_used': sum(kg.knowledge_used_count for kg in self.knowledge_guided_agents.values()),
                    'total_concepts_discovered': self.total_concepts_discovered
                }
                stats['phase4']['knowledge_guidance'] = kg_stats

        return stats

    def _phase4_enhancements(self):
        """Apply Phase 4 enhancements periodically"""
        # Memory consolidation
        if self.memory_consolidator and self.current_step % 1000 == 0:
            agent_experiences = []
            for agent in self.agents:
                if len(agent.action_history) > 0:
                    coherence_dict = agent.compute_coherence()
                    coherence = coherence_dict['composite']
                    if coherence > 0.7:
                        exp = {
                            'observation': agent.sensor_history[-1] if agent.sensor_history else np.zeros(370),
                            'action': agent.action_history[-1],
                            'coherence': coherence
                        }
                        agent_experiences.append(exp)

            if agent_experiences:
                self.memory_consolidator.consolidate_population_memories(agent_experiences)
                self.learned_memory.consolidate()

    def _build_context(self) -> Dict:
        """Build context for advanced teacher"""
        coherences = [agent.compute_coherence()['composite'] for agent in self.agents]

        return {
            'step': self.current_step,
            'population_size': len(self.agents),
            'avg_coherence': np.mean(coherences) if coherences else 0.0,
            'coherence_std': np.std(coherences) if coherences else 0.0,
            'diversity': self._compute_diversity(),
            'resource_density': (self.env.energy_grid.mean() + self.env.material_grid.mean()) / 2,
            'challenge_level': 0.5,
            'coherence_history': [np.mean([a.compute_coherence()['composite'] for a in self.agents])
                                 for _ in range(min(100, self.current_step))]
        }

    def _select_elites(self):
        """Select elite agents for advanced teacher (wrapper for Phase 2 method)"""
        return self._get_elite_agents(top_k_percent=0.2)

    def _compute_diversity(self) -> float:
        """Compute population diversity"""
        if len(self.agents) < 2:
            return 0.0

        sample_size = min(50, len(self.agents))
        sampled = np.random.choice(self.agents, size=sample_size, replace=False)

        distances = []
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                dist = 0
                # Use genome dict for FullAutopoieticAgent
                for key in sampled[i].genome:
                    dist += np.sum((sampled[i].genome[key] - sampled[j].genome[key])**2)
                distances.append(np.sqrt(dist))

        return np.mean(distances) if distances else 0.0


def create_phase4_system(
        env_size: int = 50,
        initial_population: int = 300,
        phase1_enabled: bool = True,
        phase2_enabled: bool = True,
        phase3_enabled: bool = True,
        phase4_enabled: bool = True) -> Phase4PopulationManager:
    """
    Create complete Phase 4 system

    Args:
        env_size: Environment size
        initial_population: Initial population
        phase1_enabled: Enable Phase 1 features
        phase2_enabled: Enable Phase 2 features
        phase3_enabled: Enable Phase 3 features
        phase4_enabled: Enable Phase 4 features

    Returns:
        Phase4PopulationManager
    """
    # Create environment
    env = FullALifeEnvironment(size=env_size)

    # Create population manager
    manager = Phase4PopulationManager(
        env=env,
        initial_pop=initial_population,
        max_population=500,
        phase2_enabled=phase2_enabled,
        phase3_enabled=phase3_enabled,
        phase4_enabled=phase4_enabled,
        use_advanced_teacher=phase4_enabled,
        use_learned_memory=phase4_enabled,
        use_knowledge_guidance=phase4_enabled
    )

    return manager
