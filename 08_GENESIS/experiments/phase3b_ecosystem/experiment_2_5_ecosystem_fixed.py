"""
GENESIS Experiment 2.5: Ecosystem Evolution (FIXED)

Critical Fix: Connect entity predictions to environment feedback!

Previous problem: Entity actions didn't include actual predictions
Fix: Make entities compute predictions and get real feedback

This is the REAL test of collective intelligence.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from genesis_entity_v1_1 import GENESIS_Entity_v1_1, Phenotype_v1_1
from genesis_environment import RegressionEnvironment
from genesis_models import Genome, Experience


class FixedEntity(GENESIS_Entity_v1_1):
    """
    Fixed entity that actually makes predictions!
    """

    def live_one_step(self, environment, ecosystem=None) -> float:
        """One cycle with ACTUAL prediction"""
        self.age += 1

        # 1. PERCEIVE - Get task data
        perception = self.perceive(environment)

        # 2. REFLECT
        self_awareness = self.reflect()

        # 3. INTEND
        self.intentions = self.generate_intentions(
            perception,
            self_awareness,
            ecosystem
        )

        # 4. ACT - CRITICAL FIX: Actually make prediction!
        if len(self.intentions) > 0 and self.intentions[0].type == 'survival':
            # Get input from environment
            probe_response = environment.probe({'type': 'random_sample'})
            input_data = probe_response['input']

            # MAKE ACTUAL PREDICTION
            prediction = self.phenotype.forward(input_data)

            # Send prediction to environment
            action = {
                'type': 'predict',
                'input': input_data,
                'prediction': prediction[0] if len(prediction) > 0 else 0.0,
                'intention': 'survival'
            }

            # Get consequence with feedback!
            consequence = environment.apply(action)

            # Also observe consequence
            observation = environment.observe_consequence()
            consequence.update(observation)

        else:
            action = {'type': 'idle'}
            consequence = {'idle': True, 'success': False}

        # Store feedback (CRITICAL!)
        if 'viability_contribution' in consequence:
            self.recent_feedback.append(consequence['viability_contribution'])
            if len(self.recent_feedback) > 20:
                self.recent_feedback.pop(0)

        # 5. INTEGRATE
        self.integrate_experience(perception, action, consequence)

        # 6. INTERACT
        if ecosystem is not None:
            self.interact_with_ecosystem(ecosystem)

        # 7. EVALUATE (now with real feedback!)
        self.viability = self.assess_viability(environment, ecosystem)
        self.viability_history.append(self.viability)

        # 8. EVOLVE (less frequent)
        if self.should_metamorphose():
            self.metamorphose()

        return self.viability

    def should_metamorphose(self) -> bool:
        """Much less frequent metamorphosis"""
        # Only if really stuck
        if self.viability < 0.15:
            return True

        # Very rare random
        if np.random.rand() < 0.001:  # Was 0.005, now 0.001
            return True

        return False


class FixedEcosystem:
    """Ecosystem with fixed entities"""

    def __init__(self, initial_population: int = 20, environment=None):
        self.entities = [
            FixedEntity(genome=Genome(), entity_id=i)
            for i in range(initial_population)
        ]

        self.environment = environment
        self.generation = 0

        self.population_history = []
        self.viability_history = []
        self.diversity_history = []
        self.best_error_history = []
        self.avg_error_history = []
        self.ensemble_error_history = []

    def evolve_one_generation(self, X_test, y_test) -> Dict:
        """One generation of evolution"""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        # Each entity lives
        for entity in self.entities:
            for _ in range(10):  # 10 steps per generation
                entity.live_one_step(self.environment, self)

        # Selection
        self.selection()

        # Reproduction
        self.reproduction()

        # Measure
        stats = self.measure_emergence(X_test, y_test)

        # Record
        self.population_history.append(len(self.entities))
        self.viability_history.append(stats['avg_viability'])
        self.diversity_history.append(stats['diversity'])
        self.best_error_history.append(stats['best_error'])
        self.avg_error_history.append(stats['avg_error'])
        self.ensemble_error_history.append(stats['ensemble_error'])

        self.generation += 1

        return stats

    def selection(self) -> None:
        """Natural selection"""
        self.entities.sort(key=lambda e: e.viability, reverse=True)

        initial_count = len(self.entities)

        cutoff = int(0.6 * len(self.entities))
        cutoff = max(5, cutoff)

        died = self.entities[cutoff:]
        self.entities = self.entities[:cutoff]

        if len(died) > 0:
            print(f"Selection: {len(died)} died (viability < {died[0].viability:.3f})")
            print(f"  Survivors: {len(self.entities)}/{initial_count}")

    def reproduction(self) -> None:
        """Reproduction"""
        offspring = []

        for entity in self.entities:
            # Higher threshold
            if entity.viability > 0.6 and np.random.rand() < entity.viability:

                if np.random.rand() < 0.5 and len(self.entities) > 1:
                    partner = np.random.choice([e for e in self.entities if e.id != entity.id])
                    child_genome = entity.genome.crossover(partner.genome)
                    child_genome = child_genome.mutate(mutation_rate=0.1)

                    child = FixedEntity(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

                else:
                    child_genome = entity.genome.mutate(mutation_rate=0.2)

                    child = FixedEntity(
                        genome=child_genome,
                        entity_id=self._next_entity_id()
                    )
                    offspring.append(child)

        self.entities.extend(offspring)

        if len(offspring) > 0:
            print(f"Reproduction: {len(offspring)} offspring")

    def measure_emergence(self, X_test, y_test) -> Dict:
        """Measure collective properties"""
        if len(self.entities) == 0:
            return {
                'avg_viability': 0,
                'diversity': 0,
                'best_error': 100.0,
                'avg_error': 100.0,
                'ensemble_error': 100.0
            }

        avg_viability = np.mean([e.viability for e in self.entities])
        diversity = self._measure_diversity()

        # Prediction performance
        best_error, avg_error = self._measure_prediction_error(X_test, y_test)

        # NEW: Ensemble prediction (collective intelligence!)
        ensemble_error = self._measure_ensemble_error(X_test, y_test)

        print(f"  Viability: {avg_viability:.3f}")
        print(f"  Best error: {best_error:.3f}")
        print(f"  Ensemble error: {ensemble_error:.3f}")

        return {
            'generation': self.generation,
            'population': len(self.entities),
            'avg_viability': avg_viability,
            'best_viability': max(e.viability for e in self.entities),
            'diversity': diversity,
            'best_error': best_error,
            'avg_error': avg_error,
            'ensemble_error': ensemble_error
        }

    def _measure_diversity(self) -> float:
        """Genetic diversity"""
        if len(self.entities) < 2:
            return 0.0

        curiosities = [e.curiosity for e in self.entities]
        risk_tolerances = [e.risk_tolerance for e in self.entities]
        sociabilities = [e.sociability for e in self.entities]

        diversity = (
            np.std(curiosities) +
            np.std(risk_tolerances) +
            np.std(sociabilities)
        ) / 3.0

        return diversity

    def _measure_prediction_error(self, X_test, y_test) -> tuple:
        """Individual prediction errors"""
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

    def _measure_ensemble_error(self, X_test, y_test) -> float:
        """
        Ensemble prediction: Average of top entities

        This is TRUE collective intelligence!
        """
        # Use top 50% of entities
        top_k = max(1, len(self.entities) // 2)
        top_entities = sorted(self.entities, key=lambda e: e.viability, reverse=True)[:top_k]

        ensemble_errors = []

        for i in range(min(20, len(X_test))):
            predictions = []

            for entity in top_entities:
                try:
                    pred = entity.phenotype.forward(X_test[i])
                    if len(pred) > 0:
                        predictions.append(pred[0])
                except:
                    pass

            if len(predictions) > 0:
                ensemble_pred = np.mean(predictions)
                error = np.abs(ensemble_pred - y_test[i][0])
                ensemble_errors.append(error)

        return np.mean(ensemble_errors) if ensemble_errors else 100.0

    def _next_entity_id(self) -> int:
        if len(self.entities) == 0:
            return 0
        return max(e.id for e in self.entities) + 1

    def get_best_entity(self):
        return max(self.entities, key=lambda e: e.viability)


def run_experiment_2_5():
    """
    Fixed experiment: Entities actually learn!
    """
    print("="*70)
    print("GENESIS EXPERIMENT 2.5: Ecosystem Evolution (FIXED)")
    print("="*70)

    # Same problem
    print("\n1. Generating regression problem...")
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1).reshape(-1, 1)

    X_test = np.random.randn(50, 2)
    y_test = (X_test[:, 0] * 2 + X_test[:, 1] * 3 + np.random.randn(50) * 0.1).reshape(-1, 1)

    print(f"   Function: y = 2*x1 + 3*x2 + noise")

    # Environment
    env = RegressionEnvironment(X_train, y_train, noise_level=0.1)

    # Ecosystem
    print("\n2. Creating fixed ecosystem...")
    initial_pop = 15
    ecosystem = FixedEcosystem(initial_population=initial_pop, environment=env)

    # Initial
    print("\n3. Initial performance...")
    initial_stats = ecosystem.measure_emergence(X_test, y_test)

    # Evolve
    print("\n4. Evolving for 15 generations...")

    generation_stats = []

    for gen in range(15):
        stats = ecosystem.evolve_one_generation(X_test, y_test)
        generation_stats.append(stats)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    final_stats = generation_stats[-1]

    print(f"\nViability:")
    print(f"  Initial: {initial_stats['avg_viability']:.3f}")
    print(f"  Final: {final_stats['avg_viability']:.3f}")

    print(f"\nPrediction Error:")
    print(f"  Initial best: {initial_stats['best_error']:.3f}")
    print(f"  Final best: {final_stats['best_error']:.3f}")
    print(f"  Improvement: {(initial_stats['best_error'] - final_stats['best_error']) / initial_stats['best_error'] * 100:.1f}%")

    print(f"\n  Initial ensemble: {initial_stats['ensemble_error']:.3f}")
    print(f"  Final ensemble: {final_stats['ensemble_error']:.3f}")
    print(f"  Improvement: {(initial_stats['ensemble_error'] - final_stats['ensemble_error']) / initial_stats['ensemble_error'] * 100:.1f}%")

    # Compare with single entity
    print("\n" + "="*70)
    print("COMPARISON: Ecosystem vs Single Entity")
    print("="*70)

    print("\nRunning single entity...")
    single_entity = FixedEntity(entity_id=999)

    for step in range(150):  # 15 gen * 10 steps
        single_entity.live_one_step(env, ecosystem=None)

    # Measure single entity
    single_errors = []
    for i in range(min(20, len(X_test))):
        try:
            prediction = single_entity.phenotype.forward(X_test[i])
            if len(prediction) > 0:
                error = np.abs(prediction[0] - y_test[i][0])
            else:
                error = 100.0
            single_errors.append(error)
        except:
            single_errors.append(100.0)

    single_error = np.mean(single_errors)

    print(f"\nSingle Entity: {single_error:.3f}")
    print(f"Ecosystem Best: {final_stats['best_error']:.3f}")
    print(f"Ecosystem Ensemble: {final_stats['ensemble_error']:.3f}")

    best_advantage = (single_error - final_stats['best_error']) / single_error * 100
    ensemble_advantage = (single_error - final_stats['ensemble_error']) / single_error * 100

    print(f"\nBest entity advantage: {best_advantage:.1f}%")
    print(f"Ensemble advantage: {ensemble_advantage:.1f}%")

    if ensemble_advantage > 0:
        print("\n✅ ECOSYSTEM WINS! Collective intelligence proven!")
    else:
        print("\n⚠️  Still need improvement")

    # Visualization
    print("\n5. Generating visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Population
    ax = axes[0, 0]
    ax.plot(ecosystem.population_history, 'o-', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics (Fixed)')
    ax.grid(True, alpha=0.3)

    # Viability
    ax = axes[0, 1]
    ax.plot(ecosystem.viability_history, 'o-', linewidth=2, color='green')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Viability')
    ax.set_title('Viability Evolution (Should Increase!)')
    ax.grid(True, alpha=0.3)

    # Error comparison (KEY!)
    ax = axes[0, 2]
    ax.plot(ecosystem.best_error_history, 'o-', linewidth=2, label='Best entity', color='red')
    ax.plot(ecosystem.ensemble_error_history, 's-', linewidth=2, label='Ensemble', color='blue')
    ax.axhline(y=single_error, linestyle='--', color='purple', label='Single entity', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Learning Progress (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Diversity
    ax = axes[1, 0]
    ax.plot(ecosystem.diversity_history, 'o-', linewidth=2, color='purple')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.set_title('Genetic Diversity')
    ax.grid(True, alpha=0.3)

    # Feedback samples (from best entity)
    ax = axes[1, 1]
    best_entity = ecosystem.get_best_entity()
    if len(best_entity.recent_feedback) > 0:
        ax.plot(best_entity.recent_feedback, 'o-', linewidth=2, color='orange')
        ax.set_xlabel('Recent Steps')
        ax.set_ylabel('Viability Contribution')
        ax.set_title('Environment Feedback (Should Work Now!)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No feedback', ha='center', va='center')

    # Final comparison
    ax = axes[1, 2]
    methods = ['Single\nEntity', 'Ecosystem\nBest', 'Ecosystem\nEnsemble']
    errors = [single_error, final_stats['best_error'], final_stats['ensemble_error']]
    colors = ['purple', 'red', 'blue']

    bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Final Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()

    save_path = '/Users/say/Documents/GitHub/ai/08_GENESIS/experiment_2_5_fixed_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if ensemble_advantage > 5:
        print(f"\n✅ SUCCESS! Ecosystem shows {ensemble_advantage:.1f}% advantage!")
        print("\n   Collective intelligence mechanisms:")
        print("   1. Diversity enables exploration")
        print("   2. Selection preserves good solutions")
        print("   3. Ensemble leverages population wisdom")
        print("   4. Evolution improves over generations")
    elif best_advantage > 5:
        print(f"\n✅ PARTIAL SUCCESS! Best entity {best_advantage:.1f}% better!")
        print("   Evolution works, but ensemble needs improvement")
    else:
        print("\n⚠️  Learning observed but advantage marginal")
        print("   May need more generations or better viability metric")

    print("\n" + "="*70)

    return {
        'ecosystem': ecosystem,
        'single_entity': single_entity,
        'ensemble_error': final_stats['ensemble_error'],
        'best_error': final_stats['best_error'],
        'single_error': single_error,
        'ensemble_advantage': ensemble_advantage,
        'best_advantage': best_advantage
    }


if __name__ == "__main__":
    results = run_experiment_2_5()

    print("\n" + "="*70)
    print("EXPERIMENT 2.5 COMPLETE!")
    print("="*70)
    print(f"\nEnsemble advantage: {results['ensemble_advantage']:.1f}%")
    print(f"Best entity advantage: {results['best_advantage']:.1f}%")

    if results['ensemble_advantage'] > 0:
        print("\n🎉 Collective intelligence WORKS!")
    else:
        print("\n🤔 Still debugging...")
