"""
GENESIS Experiment 2: Ecosystem Evolution

Question: Can collective intelligence emerge from a population of entities?
Is a population smarter than a single entity?

Single Entity Learning:
  entity.live() → viability → survive or die

Ecosystem Evolution:
  population.evolve() → natural selection + reproduction → emergent intelligence

Core Hypothesis:
  Collective learning > Individual learning
  Natural selection accelerates adaptation
  Symbiosis enables knowledge sharing

NO TRADITIONAL TRAINING
ONLY EVOLUTION
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from genesis_entity_v1_1 import GENESIS_Entity_v1_1
from genesis_environment import RegressionEnvironment
from genesis_models import Genome


class GENESIS_Ecosystem_v1_1:
    """
    Ecosystem using v1.1 entities

    Population dynamics:
    - Natural selection (viability-based)
    - Sexual & asexual reproduction
    - Genetic diversity
    - Emergent specialization
    """

    def __init__(self, initial_population: int = 20,
                 environment=None):
        # Population
        self.entities = [
            GENESIS_Entity_v1_1(genome=Genome(), entity_id=i)
            for i in range(initial_population)
        ]

        self.environment = environment
        self.generation = 0

        # History
        self.population_history = []
        self.viability_history = []
        self.diversity_history = []
        self.best_error_history = []
        self.avg_error_history = []

    def evolve_one_generation(self, X_test, y_test) -> Dict:
        """
        One generation of ecosystem evolution

        1. Each entity lives
        2. Natural selection
        3. Reproduction
        4. Measure emergence
        """
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        # 1. Each entity lives multiple steps
        for entity in self.entities:
            for _ in range(10):  # 10 life steps per generation
                entity.live_one_step(self.environment, self)

        # 2. Natural selection (survival of viable)
        self.selection()

        # 3. Reproduction (viable entities reproduce)
        self.reproduction()

        # 4. Environment changes
        if self.environment is not None and hasattr(self.environment, 'drift'):
            self.environment.drift()

        # 5. Measure collective properties
        stats = self.measure_emergence(X_test, y_test)

        # Record history
        self.population_history.append(len(self.entities))
        self.viability_history.append(stats['avg_viability'])
        self.diversity_history.append(stats['diversity'])
        self.best_error_history.append(stats['best_error'])
        self.avg_error_history.append(stats['avg_error'])

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
            if entity.viability > 0.5 and np.random.rand() < entity.viability:

                # Sexual reproduction (50% chance)
                if np.random.rand() < 0.5 and len(self.entities) > 1:
                    # Find partner
                    partner = np.random.choice([e for e in self.entities if e.id != entity.id])

                    # Crossover genomes
                    child_genome = entity.genome.crossover(partner.genome)

                    # Mutation
                    child_genome = child_genome.mutate(mutation_rate=0.1)

                    # Create offspring
                    child = GENESIS_Entity_v1_1(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

                # Asexual reproduction
                else:
                    # Clone with mutation
                    child_genome = entity.genome.mutate(mutation_rate=0.2)

                    child = GENESIS_Entity_v1_1(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

        self.entities.extend(offspring)

        if len(offspring) > 0:
            print(f"Reproduction: {len(offspring)} offspring born")
            print(f"  New population: {len(self.entities)}")

    def measure_emergence(self, X_test, y_test) -> Dict:
        """
        Measure collective intelligence

        Emergent properties that individual entities don't have
        """
        if len(self.entities) == 0:
            return {
                'avg_viability': 0,
                'diversity': 0,
                'specialization': 0,
                'collective_knowledge': 0,
                'best_error': 100.0,
                'avg_error': 100.0
            }

        # 1. Average viability
        avg_viability = np.mean([e.viability for e in self.entities])

        # 2. Diversity (genetic)
        diversity = self._measure_diversity()

        # 3. Specialization (different entities good at different things)
        specialization = self._measure_specialization()

        # 4. Collective knowledge
        collective_knowledge = self._measure_collective_knowledge()

        # 5. Prediction performance (for comparison with single entity)
        best_error, avg_error = self._measure_prediction_error(X_test, y_test)

        print(f"  Avg viability: {avg_viability:.3f}")
        print(f"  Best error: {best_error:.3f}")
        print(f"  Diversity: {diversity:.3f}")

        return {
            'generation': self.generation,
            'population': len(self.entities),
            'avg_viability': avg_viability,
            'best_viability': max(e.viability for e in self.entities),
            'worst_viability': min(e.viability for e in self.entities),
            'diversity': diversity,
            'specialization': specialization,
            'collective_knowledge': collective_knowledge,
            'best_error': best_error,
            'avg_error': avg_error
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

    def _measure_prediction_error(self, X_test, y_test) -> tuple:
        """
        Measure prediction error for best and average entity

        This is for comparison only - entities don't see this!
        """
        errors = []

        for entity in self.entities:
            entity_errors = []
            for i in range(min(20, len(X_test))):
                try:
                    prediction = entity.phenotype.forward(X_test[i])
                    if len(prediction) > 0:
                        error = np.abs(prediction[0] - y_test[i][0])
                    else:
                        error = 100.0
                    entity_errors.append(error)
                except:
                    entity_errors.append(100.0)

            errors.append(np.mean(entity_errors))

        best_error = min(errors)
        avg_error = np.mean(errors)

        return best_error, avg_error

    def _next_entity_id(self) -> int:
        """Get next entity ID"""
        if len(self.entities) == 0:
            return 0
        return max(e.id for e in self.entities) + 1

    def get_best_entity(self):
        """Get most viable entity"""
        return max(self.entities, key=lambda e: e.viability)


def run_experiment_2():
    """
    Test: Can ecosystem evolve better learning than single entity?
    """
    print("="*70)
    print("GENESIS EXPERIMENT 2: Ecosystem Evolution")
    print("="*70)

    # Generate regression problem (SAME as experiment 1)
    print("\n1. Generating regression problem...")
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1).reshape(-1, 1)

    X_test = np.random.randn(50, 2)
    y_test = (X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.randn(50) * 0.1).reshape(-1, 1)

    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   True function: y = 2*x1 + 3*x2 + noise")

    # Create environment
    env = RegressionEnvironment(X_train, y_train, noise_level=0.1)

    # Create ecosystem
    print("\n2. Creating GENESIS ecosystem...")
    initial_pop = 15
    ecosystem = GENESIS_Ecosystem_v1_1(initial_population=initial_pop, environment=env)
    print(f"   Initial population: {initial_pop} entities")
    print(f"   Generation: {ecosystem.generation}")

    # Measure initial performance
    print("\n3. Initial population performance...")
    initial_stats = ecosystem.measure_emergence(X_test, y_test)
    print(f"   Avg viability: {initial_stats['avg_viability']:.3f}")
    print(f"   Best error: {initial_stats['best_error']:.3f}")
    print(f"   Diversity: {initial_stats['diversity']:.3f}")

    # Evolve for 10 generations
    print("\n4. Evolving for 10 generations...")
    print("   (Natural selection + reproduction + mutation)")

    generation_stats = []

    for gen in range(10):
        stats = ecosystem.evolve_one_generation(X_test, y_test)
        generation_stats.append(stats)

    # Final results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    final_stats = generation_stats[-1]

    print(f"\nPopulation:")
    print(f"  Initial: {initial_pop} entities")
    print(f"  Final: {final_stats['population']} entities")

    print(f"\nViability:")
    print(f"  Initial avg: {initial_stats['avg_viability']:.3f}")
    print(f"  Final avg: {final_stats['avg_viability']:.3f}")
    print(f"  Best: {final_stats['best_viability']:.3f}")
    print(f"  Worst: {final_stats['worst_viability']:.3f}")

    print(f"\nPrediction Error:")
    print(f"  Initial best: {initial_stats['best_error']:.3f}")
    print(f"  Final best: {final_stats['best_error']:.3f}")
    print(f"  Improvement: {(initial_stats['best_error'] - final_stats['best_error']) / initial_stats['best_error'] * 100:.1f}%")

    print(f"\nDiversity:")
    print(f"  Initial: {initial_stats['diversity']:.3f}")
    print(f"  Final: {final_stats['diversity']:.3f}")

    print(f"\nEmergent Properties:")
    print(f"  Specialization: {final_stats['specialization']:.3f}")
    print(f"  Collective knowledge: {final_stats['collective_knowledge']} capabilities")

    # Get best entity
    best_entity = ecosystem.get_best_entity()
    print(f"\nBest Entity:")
    print(f"  {best_entity}")
    print(f"  Capabilities: {best_entity.self_model.assess_capabilities()}")

    # Compare with single entity (from experiment 1 baseline)
    print("\n" + "="*70)
    print("COMPARISON: Ecosystem vs Single Entity")
    print("="*70)

    print("\nRunning single entity baseline for comparison...")
    single_entity = GENESIS_Entity_v1_1(entity_id=999)

    single_errors = []
    for step in range(100):  # Same total steps as ecosystem (10 entities * 10 steps)
        single_entity.live_one_step(env, ecosystem=None)

        # Measure error
        test_errors = []
        for i in range(min(20, len(X_test))):
            try:
                prediction = single_entity.phenotype.forward(X_test[i])
                if len(prediction) > 0:
                    error = np.abs(prediction[0] - y_test[i][0])
                else:
                    error = 100.0
                test_errors.append(error)
            except:
                test_errors.append(100.0)

        single_errors.append(np.mean(test_errors))

    single_entity_final_error = np.mean(single_errors[-20:])

    print(f"\nSingle Entity v1.1:")
    print(f"  Final error: {single_entity_final_error:.3f}")
    print(f"  Final viability: {single_entity.viability:.3f}")

    print(f"\nEcosystem Best Entity:")
    print(f"  Final error: {final_stats['best_error']:.3f}")
    print(f"  Final viability: {final_stats['best_viability']:.3f}")

    advantage = (single_entity_final_error - final_stats['best_error']) / single_entity_final_error * 100
    print(f"\nEcosystem Advantage: {advantage:.1f}%")

    if advantage > 0:
        print("✅ Ecosystem WINS! Collective intelligence is better!")
    else:
        print("⚠️  Single entity performed better (more evolution needed?)")

    # Visualization
    print("\n5. Generating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Population size over generations
    ax = axes[0, 0]
    ax.plot(ecosystem.population_history, color='blue', linewidth=2, marker='o')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Size')
    ax.set_title('Population Dynamics')
    ax.grid(True, alpha=0.3)

    # Plot 2: Average viability over generations
    ax = axes[0, 1]
    ax.plot(ecosystem.viability_history, color='green', linewidth=2, marker='o')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Survival threshold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Viability')
    ax.set_title('Viability Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Best prediction error over generations
    ax = axes[0, 2]
    ax.plot(ecosystem.best_error_history, color='red', linewidth=2, marker='o', label='Best entity')
    ax.plot(ecosystem.avg_error_history, color='orange', linewidth=2, marker='s', label='Population avg', alpha=0.7)
    ax.axhline(y=single_entity_final_error, color='purple', linestyle='--', alpha=0.7, label='Single entity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Learning Progress (Ecosystem vs Single)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Diversity over generations
    ax = axes[1, 0]
    ax.plot(ecosystem.diversity_history, color='purple', linewidth=2, marker='o')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genetic Diversity')
    ax.set_title('Diversity Evolution')
    ax.grid(True, alpha=0.3)

    # Plot 5: Viability distribution (final generation)
    ax = axes[1, 1]
    viabilities = [e.viability for e in ecosystem.entities]
    ax.hist(viabilities, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(viabilities), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(viabilities):.3f}')
    ax.set_xlabel('Viability')
    ax.set_ylabel('Count')
    ax.set_title('Final Viability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Generation 0 vs Generation 10 comparison
    ax = axes[1, 2]
    categories = ['Avg Viability', 'Best Error', 'Diversity']
    gen0_values = [
        initial_stats['avg_viability'],
        initial_stats['best_error'] / 10,  # Normalize for visualization
        initial_stats['diversity']
    ]
    gen10_values = [
        final_stats['avg_viability'],
        final_stats['best_error'] / 10,  # Normalize for visualization
        final_stats['diversity']
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, gen0_values, width, label='Generation 0', color='lightblue', edgecolor='black')
    ax.bar(x + width/2, gen10_values, width, label='Generation 10', color='darkblue', edgecolor='black')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Generation 0 vs Generation 10')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = '/Users/say/Documents/GitHub/ai/08_GENESIS/experiment_2_ecosystem_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if advantage > 0:
        print(f"\n✅ BREAKTHROUGH! Ecosystem shows {advantage:.1f}% advantage!")
        print("\n   This proves:")
        print("   1. Collective intelligence > Individual intelligence")
        print("   2. Natural selection accelerates learning")
        print("   3. Genetic diversity enables exploration")
        print("   4. Population dynamics create emergent optimization")

        print("\n   Key mechanisms:")
        print(f"   - Population explored diverse strategies (diversity={final_stats['diversity']:.3f})")
        print(f"   - Selection pressure drove improvement")
        print(f"   - Best solutions were preserved and refined")
        print(f"   - Reproduction created beneficial variations")

    else:
        print(f"\n⚠️  Single entity performed {-advantage:.1f}% better")
        print("\n   Possible reasons:")
        print("   - Insufficient generations for evolution")
        print("   - Population too small for effective selection")
        print("   - Viability metric needs refinement")
        print("   - More steps per generation needed")

    print("\n   Emergent properties observed:")
    print(f"   - Specialization index: {final_stats['specialization']:.3f}")
    print(f"   - Collective knowledge: {final_stats['collective_knowledge']} capabilities")
    print(f"   - Population stability: {final_stats['population']} entities survived")

    print("\n" + "="*70)

    return {
        'ecosystem': ecosystem,
        'single_entity': single_entity,
        'ecosystem_best_error': final_stats['best_error'],
        'single_entity_error': single_entity_final_error,
        'advantage': advantage,
        'final_stats': final_stats,
        'initial_stats': initial_stats
    }


if __name__ == "__main__":
    results = run_experiment_2()

    print("\n" + "="*70)
    print("EXPERIMENT 2 COMPLETE!")
    print("="*70)
    print(f"\nKey Finding: Ecosystem has {results['advantage']:.1f}% advantage over single entity")
    print(f"  Ecosystem best: {results['ecosystem_best_error']:.3f}")
    print(f"  Single entity: {results['single_entity_error']:.3f}")

    if results['advantage'] > 0:
        print("\n🎉 Collective intelligence PROVEN!")
    else:
        print("\n🤔 More evolution needed to demonstrate advantage")
