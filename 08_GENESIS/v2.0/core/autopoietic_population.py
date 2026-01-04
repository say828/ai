"""
GENESIS: Autopoietic Population System
Author: GENESIS Project
Date: 2026-01-04

진화적 자기생성:
    단일 entity 학습이 아니라
    개체군 수준의 조직적 역학

핵심:
    - 개체는 조직 유지 (autopoiesis)
    - 개체군은 진화 (selection)
    - 학습은 개체 + 진화의 조합
    - NO fitness function (coherence = intrinsic viability)
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autopoietic_entity import AutopoeticEntity


class PerturbationField:
    """
    교란장 (Perturbation Field)

    NOT: 최적화할 목표가 있는 환경
    BUT: 조직을 교란하는 역학계
    """

    def __init__(self,
                 field_size: int = 20,
                 turbulence: float = 0.5,
                 seed: int = 42):
        """
        Args:
            field_size: 장의 크기
            turbulence: 난류 강도
            seed: 랜덤 시드
        """
        np.random.seed(seed)

        self.field_size = field_size
        self.turbulence = turbulence

        # 장의 상태 (역학계)
        self.field_state = np.random.randn(field_size) * 0.5

        # 장의 역학 파라미터
        self.field_W = np.random.randn(field_size, field_size) * 0.1

        self.step_count = 0

        print(f"PerturbationField created:")
        print(f"  Field size: {field_size}")
        print(f"  Turbulence: {turbulence}")

    def step(self, entity_actions: List[np.ndarray]) -> List[np.ndarray]:
        """
        장 진화 + Entity 교란 생성

        Args:
            entity_actions: 모든 entity들의 행동

        Returns:
            perturbations: 각 entity에 대한 교란
        """
        self.step_count += 1

        # 1. Entity 행동이 장에 영향
        if len(entity_actions) > 0:
            # 모든 행동의 평균적 효과
            avg_action = np.mean([a[:min(len(a), self.field_size)] for a in entity_actions], axis=0)
            action_effect = np.zeros(self.field_size)
            action_effect[:len(avg_action)] = avg_action * 0.1
            self.field_state += action_effect

        # 2. 장의 내부 역학
        field_dynamics = np.tanh(np.dot(self.field_W, self.field_state))
        self.field_state = 0.9 * self.field_state + 0.1 * field_dynamics

        # 3. 난류 추가
        noise = np.random.randn(self.field_size) * self.turbulence
        self.field_state += noise

        # 4. 각 entity에 대한 교란 생성
        perturbations = []
        for i in range(len(entity_actions)):
            # 각 entity는 장의 다른 부분에서 교란 받음
            offset = (i * 3) % self.field_size
            perturbation = np.roll(self.field_state, offset) + np.random.randn(self.field_size) * 0.1
            perturbations.append(perturbation)

        return perturbations


class AutopoeticPopulation:
    """
    자기생성 개체군

    핵심:
        - 개체는 조직 유지
        - 일관성 높으면 번식
        - 일관성 낮으면 죽음
        - 진화적 드리프트
    """

    def __init__(self,
                 initial_population: int = 10,
                 max_population: int = 30,
                 reproduction_threshold: float = 0.7,
                 mutation_rate: float = 0.1):
        """
        Args:
            initial_population: 초기 개체 수
            max_population: 최대 개체 수
            reproduction_threshold: 번식 적합도 임계값
            mutation_rate: 변이율
        """
        self.max_population = max_population
        self.reproduction_threshold = reproduction_threshold
        self.mutation_rate = mutation_rate

        # 초기 개체군
        self.entities: List[AutopoeticEntity] = []
        for i in range(initial_population):
            entity = AutopoeticEntity(
                n_internal_units=20,
                connectivity=0.3,
                plasticity_rate=0.02,
                coherence_threshold=0.25
            )
            self.entities.append(entity)

        # 통계
        self.generation = 0
        self.total_births = initial_population
        self.total_deaths = 0

        self.population_history = deque(maxlen=1000)
        self.avg_coherence_history = deque(maxlen=1000)
        self.avg_fitness_history = deque(maxlen=1000)

        print(f"\nAutopoeticPopulation created:")
        print(f"  Initial population: {initial_population}")
        print(f"  Max population: {max_population}")
        print(f"  Reproduction threshold: {reproduction_threshold}")

    def step(self, perturbation_field: PerturbationField) -> Dict:
        """
        개체군 한 스텝 진화

        Returns:
            stats: 개체군 통계
        """
        self.generation += 1

        # 1. 모든 entity 행동 수집
        actions = []
        for entity in self.entities:
            if entity.is_alive:
                action = entity.dynamics.get_output()
                actions.append(action)

        # 2. 교란장에서 교란 생성
        perturbations = perturbation_field.step(actions)

        # 3. 각 entity 생존 스텝
        living_entities = []
        coherences = []
        fitnesses = []

        for i, entity in enumerate(self.entities):
            if not entity.is_alive:
                continue

            # 교란 받기
            perturbation = perturbations[i] if i < len(perturbations) else np.random.randn(20) * 0.3

            # 생존 스텝
            result = entity.live_one_step(perturbation)

            if result['is_alive']:
                living_entities.append(entity)
                coherences.append(result['coherence']['composite'])
                fitnesses.append(entity.get_fitness())
            else:
                self.total_deaths += 1

        self.entities = living_entities

        # 4. 번식 (높은 fitness)
        if len(self.entities) < self.max_population:
            for entity in list(self.entities):
                fitness = entity.get_fitness()

                if fitness > self.reproduction_threshold:
                    # 번식 확률
                    if np.random.rand() < 0.1:  # 10% 확률
                        offspring = entity.reproduce(self.mutation_rate)
                        self.entities.append(offspring)
                        self.total_births += 1

                        if len(self.entities) >= self.max_population:
                            break

        # 5. 통계 기록
        self.population_history.append(len(self.entities))

        if len(coherences) > 0:
            self.avg_coherence_history.append(np.mean(coherences))
            self.avg_fitness_history.append(np.mean(fitnesses))
        else:
            self.avg_coherence_history.append(0)
            self.avg_fitness_history.append(0)

        return {
            'generation': self.generation,
            'population': len(self.entities),
            'avg_coherence': np.mean(coherences) if len(coherences) > 0 else 0,
            'avg_fitness': np.mean(fitnesses) if len(fitnesses) > 0 else 0,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths
        }

    def get_summary(self) -> Dict:
        """개체군 요약"""
        return {
            'generation': self.generation,
            'current_population': len(self.entities),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'avg_coherence': np.mean(list(self.avg_coherence_history)) if len(self.avg_coherence_history) > 0 else 0,
            'avg_fitness': np.mean(list(self.avg_fitness_history)) if len(self.avg_fitness_history) > 0 else 0
        }


def run_population_evolution(n_generations: int = 500,
                             initial_population: int = 10) -> Dict:
    """
    개체군 진화 실험

    Args:
        n_generations: 세대 수
        initial_population: 초기 개체 수

    Returns:
        results: 실험 결과
    """
    print("=" * 70)
    print("AUTOPOIETIC POPULATION EVOLUTION")
    print("=" * 70)

    # 교란장 생성
    field = PerturbationField(field_size=20, turbulence=0.3)

    # 개체군 생성
    population = AutopoeticPopulation(
        initial_population=initial_population,
        max_population=30,
        reproduction_threshold=0.7,
        mutation_rate=0.1
    )

    print(f"\n{'='*70}")
    print(f"Evolving for {n_generations} generations...")
    print(f"{'='*70}\n")

    # 진화 실행
    for gen in range(n_generations):
        stats = population.step(field)

        if gen % 50 == 0 or gen == n_generations - 1:
            print(f"Gen {stats['generation']:4d} | "
                  f"Pop: {stats['population']:2d} | "
                  f"Coherence: {stats['avg_coherence']:.3f} | "
                  f"Fitness: {stats['avg_fitness']:.3f} | "
                  f"Births: {stats['total_births']:3d} | "
                  f"Deaths: {stats['total_deaths']:3d}")

        # 멸종 체크
        if stats['population'] == 0:
            print(f"\n💀 Population extinct at generation {gen}")
            break

    # 최종 요약
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    summary = population.get_summary()
    print(f"\n**Population Dynamics**:")
    print(f"  Final population: {summary['current_population']}")
    print(f"  Total births: {summary['total_births']}")
    print(f"  Total deaths: {summary['total_deaths']}")
    print(f"  Net growth: {summary['total_births'] - summary['total_deaths']}")

    print(f"\n**Evolution**:")
    print(f"  Avg coherence: {summary['avg_coherence']:.3f}")
    print(f"  Avg fitness: {summary['avg_fitness']:.3f}")

    print(f"\n**Paradigm**:")
    print(f"  ✓ NO external fitness function")
    print(f"  ✓ Coherence = intrinsic viability")
    print(f"  ✓ Evolution through organizational selection")
    print(f"  ✓ True autopoietic dynamics")

    return {
        'population_history': list(population.population_history),
        'coherence_history': list(population.avg_coherence_history),
        'fitness_history': list(population.avg_fitness_history),
        'summary': summary
    }


def plot_evolution(results: Dict, save_path: str = '../../results/autopoietic_evolution.png'):
    """진화 결과 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Population over time
    ax = axes[0, 0]
    ax.plot(results['population_history'], linewidth=2, color='#2E86AB')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Population Size', fontsize=11)
    ax.set_title('Population Dynamics', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # Plot 2: Average Coherence
    ax = axes[0, 1]
    ax.plot(results['coherence_history'], linewidth=2, color='#A23B72')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Average Coherence', fontsize=11)
    ax.set_title('Organizational Coherence Evolution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Average Fitness
    ax = axes[1, 0]
    ax.plot(results['fitness_history'], linewidth=2, color='#F18F01')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Average Fitness', fontsize=11)
    ax.set_title('Fitness Evolution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary = results['summary']
    summary_text = "EVOLUTION SUMMARY\n\n"
    summary_text += f"Generations: {summary['generation']}\n"
    summary_text += f"Final Population: {summary['current_population']}\n"
    summary_text += f"Total Births: {summary['total_births']}\n"
    summary_text += f"Total Deaths: {summary['total_deaths']}\n"
    summary_text += f"Avg Coherence: {summary['avg_coherence']:.3f}\n"
    summary_text += f"Avg Fitness: {summary['avg_fitness']:.3f}\n\n"
    summary_text += "PARADIGM:\n"
    summary_text += "✓ Autopoietic entities\n"
    summary_text += "✓ No external objectives\n"
    summary_text += "✓ Organizational selection\n"
    summary_text += "✓ Intrinsic viability"

    ax.text(0.5, 0.5, summary_text, fontsize=11, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Autopoietic Population Evolution', fontsize=14, fontweight='bold')

    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║           AUTOPOIETIC POPULATION EVOLUTION                       ║
    ║                                                                   ║
    ║  Evolution through Organizational Selection                      ║
    ║  NO external fitness function                                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # 진화 실험 실행
    results = run_population_evolution(n_generations=500, initial_population=10)

    # 시각화
    plot_evolution(results)

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE!")
    print("=" * 70)
    print("\n💡 This is the true paradigm shift:")
    print("   From optimization to organization")
    print("   From external goals to intrinsic viability")
    print("   From learning algorithms to autopoietic dynamics")
