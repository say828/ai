"""
GENESIS: Ultimate Paradigm Comparison
Author: GENESIS Project
Date: 2026-01-04

근본적 질문에 대한 답:
    "What is the difference between autopoietic learning and ML?"

비교 차원:
    1. Objective: External vs Internal
    2. Mechanism: Optimization vs Organization
    3. Criterion: Loss/Reward vs Coherence
    4. Causality: Linear vs Circular
    5. Adaptation: Parameter tuning vs Structural drift

실험:
    - Autopoietic Population
    - Supervised Learning (SGD)
    - Reinforcement Learning
    - Hebbian Learning
    - Random Baseline

측정:
    - Survival rate
    - Organizational coherence
    - Adaptation capacity
    - Structural changes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from autopoietic_population import (
    AutopoeticPopulation,
    PerturbationField
)


class MLPopulation:
    """
    ML 기반 개체군 (비교용)

    차이점:
        - External fitness function
        - Gradient-based optimization
        - Fixed structure
    """

    def __init__(self, learning_type: str, population_size: int = 10):
        """
        Args:
            learning_type: 'supervised', 'rl', 'hebbian', 'random'
            population_size: 개체 수
        """
        self.learning_type = learning_type
        self.entities = []

        # 개체 생성
        for i in range(population_size):
            entity = {
                'W': np.random.randn(20, 20) * 0.2,
                'state': np.zeros(20),
                'fitness': 0.5,
                'age': 0
            }
            self.entities.append(entity)

        self.generation = 0
        self.avg_fitness_history = deque(maxlen=1000)

    def step(self, perturbation_field: PerturbationField) -> dict:
        """한 스텝 실행"""
        self.generation += 1

        # 행동 생성
        actions = []
        for entity in self.entities:
            # 내부 역학
            entity['state'] = np.tanh(np.dot(entity['W'], entity['state']))
            action = entity['state'][:3]
            actions.append(action)

        # 교란 받기
        perturbations = perturbation_field.step(actions)

        # 학습 및 fitness 평가
        fitnesses = []

        for i, entity in enumerate(self.entities):
            perturbation = perturbations[i] if i < len(perturbations) else np.random.randn(20) * 0.3

            # "fitness" = 얼마나 교란을 잘 보상하는가 (외부 기준!)
            compensation_quality = -np.linalg.norm(entity['state'] + perturbation)
            fitness = 1.0 / (1.0 + abs(compensation_quality))

            # 학습
            if self.learning_type == 'supervised':
                # Target = 교란의 반대 (외부 목표!)
                target = -perturbation
                error = entity['state'] - target[:20]
                entity['W'] -= 0.01 * np.outer(entity['state'], error)

            elif self.learning_type == 'rl':
                # Reward = fitness (외부 보상!)
                if fitness > entity['fitness']:
                    entity['W'] += 0.01 * np.outer(entity['state'], entity['state'])

            elif self.learning_type == 'hebbian':
                # 상관관계 기반
                entity['W'] += 0.01 * np.outer(entity['state'], entity['state'])

            # Random은 학습 안 함

            entity['fitness'] = fitness
            entity['age'] += 1
            fitnesses.append(fitness)

        self.avg_fitness_history.append(np.mean(fitnesses))

        return {
            'generation': self.generation,
            'avg_fitness': np.mean(fitnesses)
        }

    def get_summary(self) -> dict:
        return {
            'generation': self.generation,
            'avg_fitness': np.mean(list(self.avg_fitness_history)) if len(self.avg_fitness_history) > 0 else 0
        }


def run_ultimate_comparison(n_generations: int = 500) -> dict:
    """
    최종 패러다임 비교

    Returns:
        results: 모든 패러다임 결과
    """
    print("=" * 70)
    print("ULTIMATE PARADIGM COMPARISON")
    print("=" * 70)
    print("\nQuestion: What makes Autopoiesis fundamentally different?")
    print("Answer: We'll find out...\n")

    paradigms = [
        ('Autopoietic', 'autopoietic'),
        ('Supervised (SGD)', 'supervised'),
        ('Reinforcement Learning', 'rl'),
        ('Hebbian Learning', 'hebbian'),
        ('Random Baseline', 'random')
    ]

    results = {}

    for name, ptype in paradigms:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        # 교란장 생성
        field = PerturbationField(field_size=20, turbulence=0.3)

        # 개체군 생성
        if ptype == 'autopoietic':
            population = AutopoeticPopulation(
                initial_population=10,
                max_population=30,
                reproduction_threshold=0.7,
                mutation_rate=0.1
            )
        else:
            population = MLPopulation(ptype, population_size=10)

        # 진화/학습
        history = []

        for gen in range(n_generations):
            stats = population.step(field)

            if ptype == 'autopoietic':
                history.append({
                    'gen': gen,
                    'population': stats['population'],
                    'coherence': stats['avg_coherence'],
                    'fitness': stats['avg_fitness']
                })

                if gen % 100 == 0:
                    print(f"  Gen {gen:3d} | Pop: {stats['population']:2d} | "
                          f"Coherence: {stats['avg_coherence']:.3f} | "
                          f"Fitness: {stats['avg_fitness']:.3f}")
            else:
                history.append({
                    'gen': gen,
                    'fitness': stats['avg_fitness']
                })

                if gen % 100 == 0:
                    print(f"  Gen {gen:3d} | Fitness: {stats['avg_fitness']:.3f}")

        # 결과 저장
        results[name] = {
            'type': ptype,
            'history': history,
            'summary': population.get_summary()
        }

    return results


def analyze_paradigm_differences(results: dict) -> dict:
    """패러다임 차이 분석"""

    print(f"\n{'='*70}")
    print("PARADIGM ANALYSIS")
    print(f"{'='*70}\n")

    analysis = {}

    # 비교 표
    print(f"{'Paradigm':<30} | {'Final Metric':>15} | {'Key Difference':>20}")
    print("-" * 70)

    for name, data in results.items():
        if data['type'] == 'autopoietic':
            final_metric = data['summary']['avg_coherence']
            metric_name = "Coherence"
            key_diff = "Internal organization"
        else:
            final_metric = data['summary']['avg_fitness']
            metric_name = "Fitness"
            key_diff = "External optimization"

        print(f"{name:<30} | {metric_name}: {final_metric:>7.3f} | {key_diff:>20}")

        analysis[name] = {
            'final_metric': final_metric,
            'metric_name': metric_name,
            'key_difference': key_diff
        }

    # 근본적 차이 설명
    print(f"\n{'='*70}")
    print("FUNDAMENTAL DIFFERENCES")
    print(f"{'='*70}\n")

    print("Autopoietic vs All ML Paradigms:\n")

    comparison_table = """
    ┌─────────────────────┬────────────────────┬──────────────────────┐
    │ Dimension           │ ML Paradigms       │ Autopoietic          │
    ├─────────────────────┼────────────────────┼──────────────────────┤
    │ Objective           │ External (loss/R)  │ Internal (coherence) │
    │ Mechanism           │ Optimization       │ Organization         │
    │ Learning            │ Gradient/Hebbian   │ Structural drift     │
    │ Criterion           │ Performance        │ Self-maintenance     │
    │ Causality           │ Linear (I→O→L)    │ Circular (closure)   │
    │ Structure           │ Fixed architecture │ Mutable topology     │
    │ Goal                │ Predefined         │ Self-generated       │
    │ Evaluation          │ External metric    │ Intrinsic coherence  │
    └─────────────────────┴────────────────────┴──────────────────────┘
    """

    print(comparison_table)

    return analysis


def plot_ultimate_comparison(results: dict,
                            save_path: str = '../../results/ultimate_comparison.png'):
    """최종 비교 시각화"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    colors = {
        'Autopoietic': '#2E86AB',
        'Supervised (SGD)': '#A23B72',
        'Reinforcement Learning': '#F18F01',
        'Hebbian Learning': '#C73E1D',
        'Random Baseline': '#6C757D'
    }

    # Plot 1: Fitness/Coherence Over Time
    ax1 = fig.add_subplot(gs[0, :])

    for name, data in results.items():
        history = data['history']
        gens = [h['gen'] for h in history]

        if data['type'] == 'autopoietic':
            metrics = [h['coherence'] for h in history]
            label = f"{name} (Coherence)"
        else:
            metrics = [h['fitness'] for h in history]
            label = f"{name} (Fitness)"

        ax1.plot(gens, metrics, label=label, color=colors[name],
                linewidth=2, alpha=0.8)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title('Evolution of Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Population Size (Autopoietic only)
    ax2 = fig.add_subplot(gs[1, 0])

    auto_data = results['Autopoietic']
    gens = [h['gen'] for h in auto_data['history']]
    pops = [h['population'] for h in auto_data['history']]

    ax2.plot(gens, pops, color=colors['Autopoietic'], linewidth=2)
    ax2.fill_between(gens, pops, alpha=0.3, color=colors['Autopoietic'])
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Population Size', fontsize=11)
    ax2.set_title('Autopoietic Population Dynamics', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Plot 3: Final Metrics Comparison
    ax3 = fig.add_subplot(gs[1, 1])

    paradigm_names = list(results.keys())
    final_values = []

    for name in paradigm_names:
        data = results[name]
        if data['type'] == 'autopoietic':
            final_values.append(data['summary']['avg_coherence'])
        else:
            final_values.append(data['summary']['avg_fitness'])

    bars = ax3.bar(range(len(paradigm_names)), final_values,
                   color=[colors[n] for n in paradigm_names], alpha=0.7)

    # Autopoietic 강조
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(3)

    ax3.set_xticks(range(len(paradigm_names)))
    ax3.set_xticklabels([n.replace(' (SGD)', '').replace(' Learning', '')
                         for n in paradigm_names],
                        rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel('Final Metric Value', fontsize=11)
    ax3.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Paradigm Characteristics (텍스트)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    characteristics_text = """
    PARADIGM CHARACTERISTICS

    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║  AUTOPOIETIC (GENESIS)                                                ║
    ║  ✓ Internal coherence (NO external objective)                         ║
    ║  ✓ Circular causality (organization produces itself)                  ║
    ║  ✓ Structural drift (NO gradient descent)                             ║
    ║  ✓ Self-generated norms (autonomous)                                  ║
    ║                                                                        ║
    ║  ALL ML PARADIGMS (Supervised, RL, Hebbian)                           ║
    ║  ✗ External optimization (loss/reward minimization)                   ║
    ║  ✗ Linear causality (input → process → output → learn)               ║
    ║  ✗ Parameter optimization (gradient/correlation)                      ║
    ║  ✗ Predefined goals (fitness function)                                ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝

    KEY INSIGHT:
    Autopoietic learning is not "better" or "worse" than ML.
    It is FUNDAMENTALLY DIFFERENT - a different kind of intelligence.

    ML optimizes external objectives → Performance
    Autopoiesis maintains internal organization → Viability
    """

    ax4.text(0.5, 0.5, characteristics_text, fontsize=10, ha='center', va='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

    plt.suptitle('ULTIMATE PARADIGM COMPARISON: Autopoiesis vs ML',
                fontsize=16, fontweight='bold')

    # 저장
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              ULTIMATE PARADIGM COMPARISON                        ║
    ║                                                                   ║
    ║  Question: What makes GENESIS fundamentally different from ML?   ║
    ║                                                                   ║
    ║  Autopoietic vs Supervised vs RL vs Hebbian vs Random           ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # 실험 실행
    results = run_ultimate_comparison(n_generations=500)

    # 분석
    analysis = analyze_paradigm_differences(results)

    # 시각화
    plot_ultimate_comparison(results)

    print("\n" + "=" * 70)
    print("ULTIMATE COMPARISON COMPLETE!")
    print("=" * 70)

    print("\n🎯 FINAL ANSWER:")
    print("\n  GENESIS (Autopoietic) is not 'better ML'.")
    print("  It is a DIFFERENT KIND of system:")
    print("    - From optimization to organization")
    print("    - From external goals to intrinsic viability")
    print("    - From parameter tuning to structural evolution")
    print("\n  This is the paradigm shift we sought.")
