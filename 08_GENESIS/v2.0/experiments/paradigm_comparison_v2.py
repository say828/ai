"""
GENESIS Paradigm Comparison v2 (Ultrathink Edition)
Author: GENESIS Project
Date: 2026-01-03

핵심 개선사항:
    1. TRUE Viability Entity 추가 (vs v1.1 Hebbian)
    2. 가혹한 환경 (높은 소모, 낮은 보상)
    3. 진정한 생존 압력 (죽음 가능)
    4. 의미있는 차별화

비교 대상:
    1. True Viability (GENESIS) - 예측 + 항상성 + 알로스타시스
    2. Pure Viability (v1.1) - Hebbian + 탐색
    3. Supervised Learning - Gradient descent
    4. Reinforcement Learning - Policy gradient
    5. Random Baseline - No learning

예상 결과:
    - True Viability가 더 오래 생존 (예측 능력)
    - Pure Viability는 적응적이지만 예측 없음
    - Supervised는 ground truth 의존
    - RL은 샘플 효율 낮음
    - Random은 빠르게 죽음
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from true_viability_entity import TrueViabilityEntity
from pure_viability_entity import PureViabilityEntity
from pure_viability_environment import ResourceEnvironment


class SupervisedEntity:
    """Supervised Learning (표준 ML)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=16, initial_energy=5.0):
        self.state_size = state_size
        self.action_size = action_size
        self.is_alive = True
        self.energy = initial_energy
        self.age = 0

        # 네트워크
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1

        self.energy_history = []

    def forward(self, state):
        hidden = np.tanh(np.dot(state, self.W1))
        action = np.tanh(np.dot(hidden, self.W2))
        return action

    def live_one_step(self, environment):
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()
        action = self.forward(state)

        next_state, energy_change, done, info = environment.step(action.flatten()[0])

        # Supervised learning: 정답 사용!
        target = info['optimal_action']
        error = action.flatten()[0] - target

        # Gradient descent
        learning_rate = 0.01
        grad_action = error
        grad_W2 = np.outer(np.tanh(np.dot(state, self.W1)), grad_action)
        self.W2 -= learning_rate * grad_W2

        # 에너지 업데이트
        self.energy += energy_change
        self.energy_history.append(self.energy)

        if self.energy <= 0:
            self.is_alive = False

        return {
            'is_alive': self.is_alive,
            'age': self.age,
            'energy': self.energy,
            'energy_change': energy_change
        }


class RLEntity:
    """Reinforcement Learning (REINFORCE)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=16, initial_energy=5.0):
        self.state_size = state_size
        self.action_size = action_size
        self.is_alive = True
        self.energy = initial_energy
        self.age = 0

        # Policy network
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1

        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        self.energy_history = []

    def forward(self, state):
        hidden = np.tanh(np.dot(state, self.W1))
        action_mean = np.tanh(np.dot(hidden, self.W2))
        # Stochastic policy
        action = action_mean + np.random.randn(*action_mean.shape) * 0.1
        return action

    def live_one_step(self, environment):
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()
        action = self.forward(state)

        next_state, energy_change, done, info = environment.step(action.flatten()[0])

        # Store transition
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(energy_change)

        # Policy gradient update (every 10 steps)
        if len(self.episode_rewards) >= 10:
            self._policy_gradient_update()
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []

        # 에너지 업데이트
        self.energy += energy_change
        self.energy_history.append(self.energy)

        if self.energy <= 0:
            self.is_alive = False

        return {
            'is_alive': self.is_alive,
            'age': self.age,
            'energy': self.energy,
            'energy_change': energy_change
        }

    def _policy_gradient_update(self):
        """REINFORCE algorithm"""
        if len(self.episode_rewards) == 0:
            return

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Policy gradient
        learning_rate = 0.001
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            hidden = np.tanh(np.dot(state, self.W1))
            grad_W2 = np.outer(hidden, action) * G
            self.W2 += learning_rate * grad_W2


class RandomEntity:
    """Random Baseline (no learning)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=16, initial_energy=5.0):
        self.is_alive = True
        self.energy = initial_energy
        self.age = 0
        self.energy_history = []

    def live_one_step(self, environment):
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()

        # Random action
        action = np.random.randn() * 2.0

        next_state, energy_change, done, info = environment.step(action)

        # 에너지 업데이트
        self.energy += energy_change
        self.energy_history.append(self.energy)

        if self.energy <= 0:
            self.is_alive = False

        return {
            'is_alive': self.is_alive,
            'age': self.age,
            'energy': self.energy,
            'energy_change': energy_change
        }


def run_single_trial(paradigm_name, entity, env, n_steps=500):
    """단일 시행 실행"""
    env.reset()

    results = {
        'survival_steps': 0,
        'final_energy': 0.0,
        'energy_history': [],
        'died': False
    }

    for step in range(n_steps):
        result = entity.live_one_step(env)

        if not result['is_alive']:
            results['survival_steps'] = result['age']
            results['final_energy'] = result['energy']
            results['died'] = True
            break

        results['energy_history'].append(result['energy'])
        results['survival_steps'] = result['age']
        results['final_energy'] = result['energy']

    return results


def run_paradigm_comparison(n_trials=10, n_steps=500, harsh_mode=True):
    """
    전체 패러다임 비교 실험

    Args:
        n_trials: 시행 횟수
        n_steps: 최대 스텝 수
        harsh_mode: True면 가혹한 환경
    """
    print("=" * 70)
    print("PARADIGM COMPARISON v2 (ULTRATHINK EDITION)")
    print("=" * 70)

    # 환경 설정
    if harsh_mode:
        print("\n🔥 HARSH ENVIRONMENT MODE 🔥")
        print("  Energy cost: 0.2 per step (high)")
        print("  Reward scale: 0.3 (low)")
        print("  Function: nonlinear (complex)")
        print("  Initial energy: 5.0 (limited)")
        env_kwargs = {
            'input_dim': 5,
            'function_type': 'nonlinear',
            'energy_reward_scale': 0.3,
            'energy_cost_per_step': 0.2
        }
    else:
        print("\nStandard Environment")
        env_kwargs = {
            'input_dim': 5,
            'function_type': 'linear',
            'energy_reward_scale': 1.0,
            'energy_cost_per_step': 0.05
        }

    print(f"\nTrials: {n_trials}")
    print(f"Max steps: {n_steps}")

    # 패러다임 정의
    paradigms = [
        ('True Viability (GENESIS v2)', lambda: TrueViabilityEntity(
            state_size=5, action_size=1, hidden_size=32, initial_energy=5.0
        )),
        ('Pure Viability (v1.1)', lambda: PureViabilityEntity(
            input_size=5, hidden_size=16, output_size=1, initial_energy=5.0
        )),
        ('Supervised Learning (SGD)', lambda: SupervisedEntity(
            state_size=5, action_size=1, hidden_size=16, initial_energy=5.0
        )),
        ('Reinforcement Learning', lambda: RLEntity(
            state_size=5, action_size=1, hidden_size=16, initial_energy=5.0
        )),
        ('Random Baseline', lambda: RandomEntity(
            state_size=5, action_size=1, hidden_size=16, initial_energy=5.0
        ))
    ]

    # 결과 저장
    all_results = defaultdict(list)

    # 각 패러다임 실험
    for paradigm_name, entity_factory in paradigms:
        print(f"\n{'='*70}")
        print(f"Testing: {paradigm_name}")
        print(f"{'='*70}")

        for trial in range(n_trials):
            # 새 환경과 entity 생성
            env = ResourceEnvironment(**env_kwargs, seed=42 + trial)
            entity = entity_factory()

            # 시행 실행
            result = run_single_trial(paradigm_name, entity, env, n_steps)

            all_results[paradigm_name].append(result)

            # 진행 상황 출력
            if result['died']:
                print(f"  Trial {trial+1}/{n_trials}: 💀 Died at step {result['survival_steps']} "
                      f"(energy: {result['final_energy']:.2f})")
            else:
                print(f"  Trial {trial+1}/{n_trials}: ✓ Survived {result['survival_steps']} steps "
                      f"(energy: {result['final_energy']:.2f})")

    # 결과 분석
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # 헤더
    print(f"{'Paradigm':<35} | {'Avg Survival':>15} | {'Avg Final Energy':>18} | {'Death Rate':>12}")
    print("-" * 95)

    summary_stats = {}

    for paradigm_name in [p[0] for p in paradigms]:
        results = all_results[paradigm_name]

        survival_steps = [r['survival_steps'] for r in results]
        final_energies = [r['final_energy'] for r in results]
        death_count = sum([r['died'] for r in results])
        death_rate = death_count / len(results) * 100

        avg_survival = np.mean(survival_steps)
        std_survival = np.std(survival_steps)
        avg_energy = np.mean(final_energies)
        std_energy = np.std(final_energies)

        summary_stats[paradigm_name] = {
            'avg_survival': avg_survival,
            'std_survival': std_survival,
            'avg_energy': avg_energy,
            'std_energy': std_energy,
            'death_rate': death_rate,
            'all_results': results
        }

        print(f"{paradigm_name:<35} | {avg_survival:7.1f} ± {std_survival:5.1f} | "
              f"{avg_energy:8.2f} ± {std_energy:6.2f} | {death_rate:10.1f}%")

    return summary_stats


def plot_comparison(summary_stats, save_path='../../results/paradigm_comparison_v2.png'):
    """비교 결과 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    paradigm_names = list(summary_stats.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D']

    # Plot 1: Survival Steps
    ax = axes[0, 0]
    avg_survivals = [summary_stats[p]['avg_survival'] for p in paradigm_names]
    std_survivals = [summary_stats[p]['std_survival'] for p in paradigm_names]

    bars = ax.bar(range(len(paradigm_names)), avg_survivals,
                   yerr=std_survivals, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(range(len(paradigm_names)))
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names],
                        rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Average Survival Steps', fontsize=11)
    ax.set_title('Survival Capacity', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Final Energy
    ax = axes[0, 1]
    avg_energies = [summary_stats[p]['avg_energy'] for p in paradigm_names]
    std_energies = [summary_stats[p]['std_energy'] for p in paradigm_names]

    bars = ax.bar(range(len(paradigm_names)), avg_energies,
                   yerr=std_energies, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(range(len(paradigm_names)))
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names],
                        rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Average Final Energy', fontsize=11)
    ax.set_title('Energy Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='Initial Energy')
    ax.legend(fontsize=9)

    # Plot 3: Death Rate
    ax = axes[1, 0]
    death_rates = [summary_stats[p]['death_rate'] for p in paradigm_names]

    bars = ax.bar(range(len(paradigm_names)), death_rates, color=colors, alpha=0.7)
    ax.set_xticks(range(len(paradigm_names)))
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names],
                        rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Death Rate (%)', fontsize=11)
    ax.set_title('Mortality Risk', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Energy Trajectories (sample from first trial)
    ax = axes[1, 1]
    for i, paradigm_name in enumerate(paradigm_names):
        first_trial = summary_stats[paradigm_name]['all_results'][0]
        if len(first_trial['energy_history']) > 0:
            ax.plot(first_trial['energy_history'], label=paradigm_name.split('(')[0].strip(),
                   color=colors[i], alpha=0.8, linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Energy Level', fontsize=11)
    ax.set_title('Energy Trajectories (Sample Trial)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         PARADIGM COMPARISON v2 (ULTRATHINK EDITION)              ║
    ║                                                                   ║
    ║  True Viability vs Pure Viability vs Supervised vs RL vs Random  ║
    ║  Testing under HARSH environmental conditions                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # 가혹한 환경에서 실험
    summary_stats = run_paradigm_comparison(n_trials=10, n_steps=500, harsh_mode=True)

    # 결과 시각화
    plot_comparison(summary_stats)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)

    # 핵심 발견 출력
    print("\n🔍 KEY FINDINGS:")
    paradigm_names = list(summary_stats.keys())

    # 최고 생존
    best_survival = max(paradigm_names,
                       key=lambda p: summary_stats[p]['avg_survival'])
    print(f"\n✓ Best Survival: {best_survival}")
    print(f"  Avg steps: {summary_stats[best_survival]['avg_survival']:.1f}")

    # 최저 사망률
    best_survival_rate = min(paradigm_names,
                            key=lambda p: summary_stats[p]['death_rate'])
    print(f"\n✓ Lowest Mortality: {best_survival_rate}")
    print(f"  Death rate: {summary_stats[best_survival_rate]['death_rate']:.1f}%")

    # 최고 에너지
    best_energy = max(paradigm_names,
                     key=lambda p: summary_stats[p]['avg_energy'])
    print(f"\n✓ Best Energy Management: {best_energy}")
    print(f"  Avg final energy: {summary_stats[best_energy]['avg_energy']:.2f}")
