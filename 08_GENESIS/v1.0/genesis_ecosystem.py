"""
GENESIS: Ecosystem
Population of co-evolving entities

Key dynamics:
- Natural selection (viability-based)
- Symbiosis (cooperation)
- Reproduction (sexual & asexual)
- Emergence (collective intelligence)
"""

import numpy as np
from typing import List, Dict
from genesis_entity import GENESIS_Entity
from genesis_models import Genome
from genesis_environment import Environment


class GENESIS_Ecosystem:
    """
    Population of co-evolving GENESIS entities

    This is where collective intelligence emerges!
    """

    def __init__(self, initial_population: int = 20,
                 environment: Environment = None):
        # Population
        self.entities = [
            GENESIS_Entity(genome=Genome(), entity_id=i)
            for i in range(initial_population)
        ]

        self.environment = environment
        self.generation = 0

        # History
        self.population_history = []
        self.viability_history = []
        self.diversity_history = []

    def evolve_one_generation(self) -> Dict:
        """
        One generation of ecosystem evolution

        1. Each entity lives
        2. Natural selection
        3. Reproduction
        4. Environment drifts
        5. Measure emergence
        """
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        # 1. Each entity lives one step
        for entity in self.entities:
            entity.live_one_step(self.environment, self)

        # 2. Natural selection (survival of viable)
        self.selection()

        # 3. Reproduction (viable entities reproduce)
        self.reproduction()

        # 4. Environment changes
        if self.environment is not None:
            self.environment.drift()

        # 5. Measure collective properties
        stats = self.measure_emergence()

        # Record history
        self.population_history.append(len(self.entities))
        self.viability_history.append(stats['avg_viability'])
        self.diversity_history.append(stats['diversity'])

        self.generation += 1

        return stats

    def selection(self) -> None:
        """
        Natural selection based on viability

        NO explicit fitness function!
        Just: can you survive?
        """
        # Sort by viability
        self.entities.sort(key=lambda e: e.viability, reverse=True)

        initial_count = len(self.entities)

        # Bottom entities die (harshly)
        # Keep top 60%
        cutoff = int(0.6 * len(self.entities))
        cutoff = max(5, cutoff)  # At least 5 survive

        died = self.entities[cutoff:]
        self.entities = self.entities[:cutoff]

        if len(died) > 0:
            print(f"Selection: {len(died)} entities died (viability < {died[0].viability:.3f})")
            print(f"  Survivors: {len(self.entities)}/{initial_count}")
            print(f"  Best viability: {self.entities[0].viability:.3f}")

    def reproduction(self) -> None:
        """
        Viable entities reproduce

        Two modes:
        - Sexual: crossover between two entities
        - Asexual: clone with mutation
        """
        offspring = []

        for entity in self.entities:
            # Only highly viable entities reproduce
            if entity.viability > 0.6 and np.random.rand() < entity.viability:

                # Sexual reproduction (50% chance)
                if np.random.rand() < 0.5 and len(self.entities) > 1:
                    # Find partner
                    partner = np.random.choice([e for e in self.entities if e.id != entity.id])

                    # Crossover genomes
                    child_genome = entity.genome.crossover(partner.genome)

                    # Mutation
                    child_genome = child_genome.mutate(mutation_rate=0.1)

                    # Create offspring
                    child = GENESIS_Entity(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

                # Asexual reproduction
                else:
                    # Clone with mutation
                    child_genome = entity.genome.mutate(mutation_rate=0.2)

                    child = GENESIS_Entity(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

        self.entities.extend(offspring)

        if len(offspring) > 0:
            print(f"Reproduction: {len(offspring)} offspring born")
            print(f"  New population: {len(self.entities)}")

    def measure_emergence(self) -> Dict:
        """
        Measure collective intelligence

        Emergent properties that individual entities don't have
        """
        if len(self.entities) == 0:
            return {
                'avg_viability': 0,
                'diversity': 0,
                'specialization': 0,
                'collective_knowledge': 0
            }

        # 1. Average viability
        avg_viability = np.mean([e.viability for e in self.entities])

        # 2. Diversity (genetic)
        diversity = self._measure_diversity()

        # 3. Specialization (different entities good at different things)
        specialization = self._measure_specialization()

        # 4. Collective knowledge
        collective_knowledge = self._measure_collective_knowledge()

        return {
            'generation': self.generation,
            'population': len(self.entities),
            'avg_viability': avg_viability,
            'best_viability': max(e.viability for e in self.entities),
            'worst_viability': min(e.viability for e in self.entities),
            'diversity': diversity,
            'specialization': specialization,
            'collective_knowledge': collective_knowledge
        }

    def _measure_diversity(self) -> float:
        """
        Genetic diversity

        High diversity = resilient ecosystem
        Low diversity = vulnerable to change
        """
        if len(self.entities) < 2:
            return 0.0

        # Measure variance in genome parameters
        curiosities = [e.curiosity for e in self.entities]
        risk_tolerances = [e.risk_tolerance for e in self.entities]
        sociabilities = [e.sociability for e in self.entities]

        diversity = (
            np.std(curiosities) +
            np.std(risk_tolerances) +
            np.std(sociabilities)
        ) / 3.0

        return diversity

    def _measure_specialization(self) -> float:
        """
        Are entities specializing in different strategies?
        """
        if len(self.entities) < 2:
            return 0.0

        # Measure variance in capabilities
        capability_counts = [
            len(e.self_model.capabilities)
            for e in self.entities
        ]

        specialization = np.std(capability_counts) / (np.mean(capability_counts) + 0.1)

        return min(1.0, specialization)

    def _measure_collective_knowledge(self) -> float:
        """
        Total knowledge across all entities
        """
        total_capabilities = set()

        for entity in self.entities:
            total_capabilities.update(entity.self_model.capabilities)

        return len(total_capabilities)

    def _next_entity_id(self) -> int:
        """Get next entity ID"""
        if len(self.entities) == 0:
            return 0
        return max(e.id for e in self.entities) + 1

    def get_best_entity(self) -> GENESIS_Entity:
        """Get most viable entity"""
        return max(self.entities, key=lambda e: e.viability)

    def summary(self) -> str:
        """Summary of ecosystem state"""
        if len(self.entities) == 0:
            return "Ecosystem: EXTINCT"

        stats = self.measure_emergence()

        return f"""
Ecosystem Summary (Generation {self.generation}):
  Population: {len(self.entities)}
  Avg Viability: {stats['avg_viability']:.3f}
  Best Viability: {stats['best_viability']:.3f}
  Diversity: {stats['diversity']:.3f}
  Specialization: {stats['specialization']:.3f}
  Collective Knowledge: {stats['collective_knowledge']} capabilities
"""

    def __repr__(self):
        return f"Ecosystem(gen={self.generation}, pop={len(self.entities)})"


if __name__ == "__main__":
    print("Testing GENESIS Ecosystem...")

    from genesis_environment import OpenWorldEnvironment

    # Create environment
    env = OpenWorldEnvironment(size=10)

    # Create ecosystem
    ecosystem = GENESIS_Ecosystem(initial_population=10, environment=env)

    print(f"Initial: {ecosystem}")
    print(ecosystem.summary())

    # Evolve for 5 generations
    print("\nEvolving for 5 generations:")
    for gen in range(5):
        stats = ecosystem.evolve_one_generation()

        if gen % 2 == 0:
            print(f"\nGen {gen} stats:")
            print(f"  Population: {stats['population']}")
            print(f"  Avg viability: {stats['avg_viability']:.3f}")
            print(f"  Diversity: {stats['diversity']:.3f}")

    print("\n" + ecosystem.summary())

    print("\n✅ Ecosystem test complete!")
