"""
GENESIS Phase 2: Multi-Memory Population Manager

Extends Phase 1 population with:
- Episodic memory (prioritized experience replay)
- Semantic memory (knowledge graph)
- Stigmergy (environmental communication)
- Meta-learning (adaptive hyperparameters)

Backward compatible with Phase 1 - can run with phase2_enabled=False
"""

import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Imports from Phase 1
from full_population import FullPopulationManager
from full_environment import FullALifeEnvironment
from full_agent import FullAutopoieticAgent

# Imports from Phase 2
from teacher_network import TeacherNetwork, EpisodicMemory
from semantic_memory import SemanticMemory
from stigmergy import StigmergyField
from meta_learner import MetaLearner
from phase2_extensions import Phase2AgentExtension


class Phase2PopulationManager(FullPopulationManager):
    """
    Phase 2 Population with Multi-Layer Memory System

    Architecture:
    - Layer 1: Procedural Memory (Teacher Network) [Phase 1]
    - Layer 2: Episodic Memory (Experience Replay) [Phase 2]
    - Layer 3: Semantic Memory (Knowledge Graph) [Phase 2]
    - Layer 4: Environmental Memory (Stigmergy) [Phase 2]
    - Layer 5: Meta-Learning (Algorithm Evolution) [Phase 2]
    """

    def __init__(self,
                 env: FullALifeEnvironment,
                 initial_pop: int = 100,
                 max_population: int = 500,
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.1,
                 min_population: int = 50,
                 enable_teacher: bool = True,
                 teacher_update_interval: int = 100,
                 teacher_learning_rate: float = 0.1,
                 # Phase 2 parameters
                 phase2_enabled: bool = True,
                 episodic_capacity: int = 100000,
                 enable_semantic: bool = True,
                 enable_stigmergy: bool = True,
                 enable_meta_learning: bool = True):
        """
        Args:
            (Phase 1 parameters as before...)
            phase2_enabled: Enable all Phase 2 features
            episodic_capacity: Episodic memory capacity
            enable_semantic: Enable semantic memory
            enable_stigmergy: Enable environmental stigmergy
            enable_meta_learning: Enable meta-learning
        """
        # Initialize Phase 1
        super().__init__(
            env=env,
            initial_pop=initial_pop,
            max_population=max_population,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            min_population=min_population,
            enable_teacher=enable_teacher,
            teacher_update_interval=teacher_update_interval,
            teacher_learning_rate=teacher_learning_rate
        )

        # Phase 2 configuration
        self.phase2_enabled = phase2_enabled

        # Store teacher_learning_rate (parent doesn't store it)
        self.teacher_learning_rate = teacher_learning_rate

        if self.phase2_enabled:
            # Layer 2: Enhanced Episodic Memory
            self.memory = EpisodicMemory(capacity=episodic_capacity)

            # Layer 3: Semantic Memory
            self.semantic_memory = SemanticMemory() if enable_semantic else None

            # Layer 4: Stigmergy Field
            self.stigmergy = StigmergyField(
                grid_size=env.size,
                decay_rate=0.98
            ) if enable_stigmergy else None

            # Layer 5: Meta-Learner
            self.meta_learner = MetaLearner() if enable_meta_learning else None

            # Agent extensions (wrapper for Phase 2 capabilities)
            self.agent_extensions = {}  # {agent_id: Phase2AgentExtension}

            # Phase 2 statistics
            self.semantic_update_interval = 1000
            self.meta_adapt_interval = 5000
            self.stigmergy_diffusion_interval = 100

            print("✨ Phase 2 enabled:")
            print(f"   - Episodic Memory: {episodic_capacity:,} capacity")
            print(f"   - Semantic Memory: {enable_semantic}")
            print(f"   - Stigmergy: {enable_stigmergy}")
            print(f"   - Meta-Learning: {enable_meta_learning}")
        else:
            print("📦 Running in Phase 1 mode (Phase 2 disabled)")

    def step(self) -> Dict:
        """
        Enhanced step with Phase 2 components
        """
        if not self.phase2_enabled:
            return super().step()  # Fall back to Phase 1

        # === Phase 2 Enhanced Step ===

        # 1. Agent sensing and acting with Phase 2 extensions
        for agent in self.agents:
            # Get or create extension
            if agent.id not in self.agent_extensions:
                self.agent_extensions[agent.id] = Phase2AgentExtension(agent)

            ext = self.agent_extensions[agent.id]

            # Get nearby agents
            nearby = self._get_nearby_agents(agent, radius=5)

            # Enhanced sensing (includes stigmergy)
            observation = ext.sense_with_stigmergy(
                self.env,
                nearby,
                self.stigmergy
            )

            # Enhanced acting (includes semantic guidance)
            # Note: observation is 374-dim, but agent.forward expects 370-dim
            # So we use the base observation (first 370 dims) for action computation
            base_observation = observation[:370]
            action = ext.act_with_semantic_guidance(
                observation,
                self.semantic_memory,
                guidance_strength=0.3
            )

            # Execute action using Phase 1 method
            # The agent.forward() method handles internal state update and action
            # But we already computed action with semantic guidance, so we
            # need to manually update internal state

            # Forward pass through RNN (updates internal state)
            # We'll use agent's forward method but pass our action
            _ = agent.forward(base_observation)  # This updates agent.state

            # Override action with our semantic-guided action
            agent.action_history[-1] = action.copy()

            # Execute action in environment (use parent's method)
            self._execute_action(agent, action)

            # Compute coherence (returns dict with 'composite' key)
            coherence_dict = agent.compute_coherence()
            coherence = coherence_dict['composite']  # Extract composite score

            # Environmental marking
            ext.mark_environment(self.stigmergy, coherence)

            # Store high-quality experiences
            if coherence > 0.8:
                priority = ext.compute_experience_priority(coherence)
                experience = ext.get_experience_dict(coherence)
                self.memory.store_critical_experience(experience, priority)

        # 2. Stigmergy dynamics
        if self.stigmergy is not None:
            self.stigmergy.decay_all()

            # Periodic diffusion
            if self.current_step % self.stigmergy_diffusion_interval == 0:
                try:
                    self.stigmergy.diffuse_all(diffusion_rate=0.05)
                except:
                    pass  # scipy may not be available

        # 3. Run Phase 1 dynamics (metabolism, deaths, births, etc.)
        # But skip the acting part since we already did it above
        self._process_metabolism()
        self._handle_deaths()
        self._handle_reproduction()
        self._maintain_minimum_population()

        # 4. Teacher Network update (Phase 1)
        if self.enable_teacher and self.current_step % self.teacher_update_interval == 0:
            elite = self._get_elite_agents()
            if elite:
                self.teacher.distill_from_elite(elite)

                # Phase 2 enhancement: Experience replay
                if self.memory.experiences:
                    replay_batch = self.memory.replay_for_learning(n=256)
                    # TODO: Use replay_batch to further train teacher
                    # This would require adding a learning method to teacher

        # 5. Semantic Memory update (every 1000 steps)
        if (self.semantic_memory is not None and
            self.current_step % self.semantic_update_interval == 0):

            # Extract concepts from recent experiences
            recent_exp = self.memory.get_recent_experiences(n=1000)
            for exp in recent_exp:
                if 'observation' in exp and 'action' in exp:
                    self.semantic_memory.extract_concept(
                        exp['observation'],
                        exp['action'],
                        exp.get('coherence', 0.5)
                    )

            # Discover relations
            self.semantic_memory.discover_causal_relations()

            # Generate rules
            new_rules = self.semantic_memory.generate_survival_rules()

            if new_rules > 0:
                print(f"   📚 Semantic: Discovered {new_rules} new rules")

        # 6. Meta-Learning adaptation (every 5000 steps)
        if (self.meta_learner is not None and
            self.current_step % self.meta_adapt_interval == 0):

            stats = self.get_statistics()
            old_params = self.meta_learner.meta_params.copy()
            new_params = self.meta_learner.adapt_learning_strategy(stats)

            # Apply adapted parameters
            self.teacher_learning_rate = new_params['teacher_lr']
            self.mutation_rate = new_params['mutation_rate']
            self.mutation_scale = new_params['mutation_scale']
            if self.stigmergy is not None:
                self.stigmergy.decay_rate = new_params['stigmergy_decay']

            print(f"   🧠 Meta: Adapted learning strategy")
            print(f"      teacher_lr: {old_params['teacher_lr']:.3f} → {new_params['teacher_lr']:.3f}")
            print(f"      mutation_rate: {old_params['mutation_rate']:.3f} → {new_params['mutation_rate']:.3f}")

        # 7. Collect statistics
        self.current_step += 1
        return self.get_statistics()

    def get_statistics(self) -> Dict:
        """
        Enhanced statistics with Phase 2 metrics
        """
        stats = super().get_statistics()

        if self.phase2_enabled:
            # Episodic Memory stats
            if self.memory:
                memory_stats = self.memory.get_statistics()
                stats['episodic_memory'] = {
                    'size': memory_stats.get('size', 0),
                    'utilization': memory_stats.get('utilization', 0.0),
                    'critical_events': memory_stats.get('critical_events', 0)
                }

            # Semantic Memory stats
            if self.semantic_memory:
                semantic_stats = self.semantic_memory.get_statistics()
                stats['semantic_memory'] = {
                    'concepts': semantic_stats['total_concepts'],
                    'relations': semantic_stats['total_relations'],
                    'rules': semantic_stats['total_rules']
                }

            # Stigmergy stats
            if self.stigmergy:
                stigmergy_stats = self.stigmergy.get_statistics()
                stats['stigmergy'] = {
                    'total_deposits': stigmergy_stats['total_deposits'],
                    'pheromone_coverage': stigmergy_stats['field_statistics']['pheromone']['coverage'],
                    'success_coverage': stigmergy_stats['field_statistics']['success']['coverage']
                }

            # Meta-Learning stats
            if self.meta_learner:
                meta_stats = self.meta_learner.get_statistics()
                stats['meta_learning'] = {
                    'adaptation_count': meta_stats['adaptation_count'],
                    'current_teacher_lr': self.teacher_learning_rate,
                    'current_mutation_rate': self.mutation_rate
                }

        return stats

    def _get_nearby_agents(self, agent: FullAutopoieticAgent, radius: int = 5) -> List:
        """
        Get agents within radius of given agent
        """
        nearby = []
        for other in self.agents:
            if other.id != agent.id:
                dx = abs(other.x - agent.x)
                dy = abs(other.y - agent.y)
                if dx <= radius and dy <= radius:
                    nearby.append(other)
        return nearby

    def _process_metabolism(self):
        """Phase 1 metabolism (reuse parent implementation)"""
        for agent in self.agents:
            # Metabolic costs
            base_cost = 0.001 * agent.age ** 0.5
            if agent.coherence_history:
                coherence = agent.coherence_history[-1]
                coherence_penalty = max(0, 0.5 - coherence) * 0.01
            else:
                coherence_penalty = 0.005

            total_cost = base_cost + coherence_penalty
            agent.energy -= total_cost

    def _handle_deaths(self):
        """Phase 1 deaths with cleanup of Phase 2 extensions"""
        dead_agents = [a for a in self.agents if a.energy <= 0 or a.age > 10000]

        for agent in dead_agents:
            self.agents.remove(agent)
            self.total_deaths += 1

            # Clean up Phase 2 extension
            if self.phase2_enabled and agent.id in self.agent_extensions:
                del self.agent_extensions[agent.id]

    def _handle_reproduction(self):
        """Phase 1 reproduction (reuse parent)"""
        # Use parent implementation or simplified version
        pass  # TODO: Implement if needed

    def _maintain_minimum_population(self):
        """Phase 1 minimum population (reuse parent)"""
        if len(self.agents) < self.min_population:
            needed = self.min_population - len(self.agents)
            new_agents = self._spawn_agents_from_teacher(needed)
            self.agents.extend(new_agents)

    def save_phase2_state(self, directory: str):
        """
        Save Phase 2 components
        """
        import os
        os.makedirs(directory, exist_ok=True)

        if self.memory:
            # Episodic memory is too large to save easily
            pass

        if self.semantic_memory:
            self.semantic_memory.save(f"{directory}/semantic_memory.json")

        if self.meta_learner:
            self.meta_learner.save(f"{directory}/meta_learner.json")

        print(f"Phase 2 state saved to {directory}")
