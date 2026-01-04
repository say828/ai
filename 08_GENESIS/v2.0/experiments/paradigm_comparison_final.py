"""
GENESIS Final Paradigm Comparison: Viability Redefined
Author: GENESIS Project
Date: 2026-01-03

핵심 혁신:
    기존: 생존 시간 (survival time) 측정
    신규: 학습 궤적 (learning trajectory) 측정

    Viability = 지속적 개선 능력 (capacity for sustained improvement)

비교 차원:
    1. Learning Speed: 얼마나 빨리 개선하는가?
    2. Final Performance: 최종 성능 수준
    3. Stability: 성능의 안정성
    4. Adaptability: 새로운 조건에 적응
    5. Growth Trajectory: 시간에 따른 개선 추세

실험 설계:
    - 20 에피소드 (각 100 스텝)
    - 에피소드 간 학습 유지
    - 5가지 패러다임 비교
    - 죽음이 아니라 개선에 초점
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from viability_redefined import (
    ViabilityDrivenEntity,
    MultiEpisodeEnvironment,
    ViabilityMetrics
)


class SupervisedLearner:
    """Supervised Learning (표준 ML)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=32):
        self.state_size = state_size
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1
        self.episode_performances = []
        self.total_episodes = 0

    def forward(self, state):
        hidden = np.tanh(np.dot(state.flatten(), self.W1))
        action = np.tanh(np.dot(hidden, self.W2))
        return action

    def live_episode(self, environment):
        state = environment.reset_episode()
        episode_energy = 10.0
        episode_rewards = []

        for step in range(environment.episode_length):
            action = self.forward(state)
            next_state, energy_change, done, info = environment.step(action.flatten()[0])

            # Supervised learning: 정답 사용
            target = info['optimal_action']
            error = action.flatten()[0] - target

            # Gradient descent
            learning_rate = 0.01
            hidden = np.tanh(np.dot(state.flatten(), self.W1))
            grad_W2 = np.outer(hidden, error)
            self.W2 -= learning_rate * grad_W2

            episode_energy += energy_change
            episode_rewards.append(energy_change)
            state = next_state

            if done:
                break

        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        self.episode_performances.append(avg_reward)
        self.total_episodes += 1

        return {
            'episode': self.total_episodes,
            'avg_reward': avg_reward,
            'final_energy': episode_energy
        }

    def get_summary(self):
        return {
            'total_episodes': self.total_episodes,
            'episode_performances': self.episode_performances
        }


class HebbianLearner:
    """Pure Hebbian Learning"""

    def __init__(self, state_size=5, action_size=1, hidden_size=32):
        self.state_size = state_size
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1
        self.episode_performances = []
        self.total_episodes = 0

    def forward(self, state):
        hidden = np.tanh(np.dot(state.flatten(), self.W1))
        action = np.tanh(np.dot(hidden, self.W2))
        return action

    def live_episode(self, environment):
        state = environment.reset_episode()
        episode_energy = 10.0
        episode_rewards = []

        for step in range(environment.episode_length):
            action = self.forward(state)
            next_state, energy_change, done, info = environment.step(action.flatten()[0])

            # Hebbian learning: 성공시 강화
            if energy_change > 0:
                learning_rate = 0.01
                hidden = np.tanh(np.dot(state.flatten(), self.W1))
                self.W2 += learning_rate * np.outer(hidden, action)
                self.W1 += learning_rate * 0.5 * np.outer(state.flatten(), hidden)

            episode_energy += energy_change
            episode_rewards.append(energy_change)
            state = next_state

            if done:
                break

        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        self.episode_performances.append(avg_reward)
        self.total_episodes += 1

        return {
            'episode': self.total_episodes,
            'avg_reward': avg_reward,
            'final_energy': episode_energy
        }

    def get_summary(self):
        return {
            'total_episodes': self.total_episodes,
            'episode_performances': self.episode_performances
        }


class RLLearner:
    """Reinforcement Learning (Policy Gradient)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=32):
        self.state_size = state_size
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1
        self.episode_performances = []
        self.total_episodes = 0

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards_buffer = []

    def forward(self, state):
        hidden = np.tanh(np.dot(state.flatten(), self.W1))
        action = np.tanh(np.dot(hidden, self.W2))
        return action + np.random.randn(*action.shape) * 0.1  # Stochastic

    def live_episode(self, environment):
        state = environment.reset_episode()
        episode_energy = 10.0
        episode_rewards = []

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards_buffer = []

        for step in range(environment.episode_length):
            action = self.forward(state)
            next_state, energy_change, done, info = environment.step(action.flatten()[0])

            self.episode_states.append(state)
            self.episode_actions.append(action)
            self.episode_rewards_buffer.append(energy_change)

            episode_energy += energy_change
            episode_rewards.append(energy_change)
            state = next_state

            if done:
                break

        # Policy gradient update (end of episode)
        self._policy_gradient_update()

        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        self.episode_performances.append(avg_reward)
        self.total_episodes += 1

        return {
            'episode': self.total_episodes,
            'avg_reward': avg_reward,
            'final_energy': episode_energy
        }

    def _policy_gradient_update(self):
        if len(self.episode_rewards_buffer) == 0:
            return

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.episode_rewards_buffer):
            G = r + 0.99 * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update policy
        learning_rate = 0.001
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            hidden = np.tanh(np.dot(state.flatten(), self.W1))
            grad_W2 = np.outer(hidden, action) * G
            self.W2 += learning_rate * grad_W2

    def get_summary(self):
        return {
            'total_episodes': self.total_episodes,
            'episode_performances': self.episode_performances
        }


class RandomBaseline:
    """Random Actions (no learning)"""

    def __init__(self, state_size=5, action_size=1, hidden_size=32):
        self.episode_performances = []
        self.total_episodes = 0

    def live_episode(self, environment):
        state = environment.reset_episode()
        episode_energy = 10.0
        episode_rewards = []

        for step in range(environment.episode_length):
            action = np.random.randn() * 2.0
            next_state, energy_change, done, info = environment.step(action)

            episode_energy += energy_change
            episode_rewards.append(energy_change)
            state = next_state

            if done:
                break

        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        self.episode_performances.append(avg_reward)
        self.total_episodes += 1

        return {
            'episode': self.total_episodes,
            'avg_reward': avg_reward,
            'final_energy': episode_energy
        }

    def get_summary(self):
        return {
            'total_episodes': self.total_episodes,
            'episode_performances': self.episode_performances
        }


def run_paradigm_comparison(n_trials=5, n_episodes=20):
    """
    최종 패러다임 비교

    측정 지표:
        - Learning speed: 첫 5 에피소드 vs 마지막 5 에피소드
        - Final performance: 마지막 5 에피소드 평균
        - Stability: 성능의 표준편차
        - Improvement rate: 학습 곡선 기울기
    """
    print("=" * 70)
    print("FINAL PARADIGM COMPARISON: VIABILITY REDEFINED")
    print("=" * 70)
    print(f"\nExperimental Setup:")
    print(f"  Trials: {n_trials}")
    print(f"  Episodes per trial: {n_episodes}")
    print(f"  Steps per episode: 100")
    print(f"  Focus: Learning trajectory, NOT survival time")

    # 패러다임 정의
    paradigms = [
        ('Viability-Driven (GENESIS)', lambda: ViabilityDrivenEntity(
            state_size=5, action_size=1, hidden_size=32
        )),
        ('Supervised Learning (SGD)', lambda: SupervisedLearner(
            state_size=5, action_size=1, hidden_size=32
        )),
        ('Hebbian Learning', lambda: HebbianLearner(
            state_size=5, action_size=1, hidden_size=32
        )),
        ('Reinforcement Learning', lambda: RLLearner(
            state_size=5, action_size=1, hidden_size=32
        )),
        ('Random Baseline', lambda: RandomBaseline(
            state_size=5, action_size=1, hidden_size=32
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
            # 환경 생성
            env = MultiEpisodeEnvironment(
                input_dim=5,
                episode_length=100,
                energy_cost=0.1,
                reward_scale=0.5,
                seed=42 + trial
            )

            # Entity 생성
            entity = entity_factory()

            # 에피소드 실행
            trial_performances = []

            for episode in range(n_episodes):
                result = entity.live_episode(env)
                trial_performances.append(result['avg_reward'])

            all_results[paradigm_name].append(trial_performances)

            # 진행 상황
            initial_perf = np.mean(trial_performances[:5])
            final_perf = np.mean(trial_performances[-5:])
            improvement = ((final_perf - initial_perf) / (abs(initial_perf) + 1e-8)) * 100

            print(f"  Trial {trial+1}/{n_trials}: Initial={initial_perf:+.3f}, "
                  f"Final={final_perf:+.3f}, Improvement={improvement:+.1f}%")

    # 결과 분석
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # 헤더
    print(f"{'Paradigm':<30} | {'Initial':>10} | {'Final':>10} | {'Improve':>10} | {'Stability':>10}")
    print("-" * 85)

    summary_stats = {}

    for paradigm_name in [p[0] for p in paradigms]:
        trials_data = all_results[paradigm_name]

        # 초기 vs 최종 성능
        initial_perfs = [np.mean(trial[:5]) for trial in trials_data]
        final_perfs = [np.mean(trial[-5:]) for trial in trials_data]

        avg_initial = np.mean(initial_perfs)
        avg_final = np.mean(final_perfs)

        # 개선률
        improvements = []
        for trial in trials_data:
            initial = np.mean(trial[:5])
            final = np.mean(trial[-5:])
            improvement = ((final - initial) / (abs(initial) + 1e-8)) * 100
            improvements.append(improvement)
        avg_improvement = np.mean(improvements)

        # 안정성 (낮을수록 안정적)
        all_perfs = [perf for trial in trials_data for perf in trial]
        stability = np.std(all_perfs)

        summary_stats[paradigm_name] = {
            'avg_initial': avg_initial,
            'avg_final': avg_final,
            'avg_improvement': avg_improvement,
            'stability': stability,
            'all_trials': trials_data
        }

        print(f"{paradigm_name:<30} | {avg_initial:+10.3f} | {avg_final:+10.3f} | "
              f"{avg_improvement:+9.1f}% | {stability:10.3f}")

    return summary_stats


def plot_comparison(summary_stats, save_path='../../results/paradigm_comparison_final.png'):
    """최종 비교 결과 시각화"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    paradigm_names = list(summary_stats.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D']

    # Plot 1: Learning Curves (모든 패러다임)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, paradigm_name in enumerate(paradigm_names):
        trials = summary_stats[paradigm_name]['all_trials']
        # 평균 학습 곡선
        avg_curve = np.mean(trials, axis=0)
        std_curve = np.std(trials, axis=0)

        episodes = np.arange(len(avg_curve))
        ax1.plot(episodes, avg_curve, label=paradigm_name.split('(')[0].strip(),
                color=colors[i], linewidth=2, alpha=0.8)
        ax1.fill_between(episodes, avg_curve - std_curve, avg_curve + std_curve,
                         color=colors[i], alpha=0.2)

    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Average Reward per Step', fontsize=11)
    ax1.set_title('Learning Curves (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Initial vs Final Performance
    ax2 = fig.add_subplot(gs[0, 2])
    initials = [summary_stats[p]['avg_initial'] for p in paradigm_names]
    finals = [summary_stats[p]['avg_final'] for p in paradigm_names]

    x = np.arange(len(paradigm_names))
    width = 0.35

    ax2.bar(x - width/2, initials, width, label='Initial (Ep 1-5)',
            color=[c + '80' for c in colors], edgecolor=colors)
    ax2.bar(x + width/2, finals, width, label='Final (Ep 16-20)',
            color=colors, edgecolor='black', linewidth=1.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names],
                         rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('Avg Reward', fontsize=10)
    ax2.set_title('Initial vs Final', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 3: Improvement Rate
    ax3 = fig.add_subplot(gs[1, 0])
    improvements = [summary_stats[p]['avg_improvement'] for p in paradigm_names]

    bars = ax3.barh(range(len(paradigm_names)), improvements, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(paradigm_names)))
    ax3.set_yticklabels([p.split('(')[0].strip() for p in paradigm_names], fontsize=9)
    ax3.set_xlabel('Improvement (%)', fontsize=10)
    ax3.set_title('Learning Improvement', fontsize=11, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Color bars based on value
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        if val > 0:
            bar.set_color('#2E7D32')  # Green for positive
        else:
            bar.set_color('#C62828')  # Red for negative

    # Plot 4: Stability (낮을수록 좋음)
    ax4 = fig.add_subplot(gs[1, 1])
    stabilities = [summary_stats[p]['stability'] for p in paradigm_names]

    bars = ax4.bar(range(len(paradigm_names)), stabilities, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(paradigm_names)))
    ax4.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names],
                         rotation=15, ha='right', fontsize=8)
    ax4.set_ylabel('Std Dev', fontsize=10)
    ax4.set_title('Performance Stability', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Learning Trajectory (first trial examples)
    ax5 = fig.add_subplot(gs[1, 2])
    for i, paradigm_name in enumerate(paradigm_names):
        first_trial = summary_stats[paradigm_name]['all_trials'][0]
        # Cumulative sum to show trajectory
        cumulative = np.cumsum(first_trial)
        ax5.plot(cumulative, label=paradigm_name.split('(')[0].strip(),
                color=colors[i], linewidth=2, alpha=0.8)

    ax5.set_xlabel('Episode', fontsize=10)
    ax5.set_ylabel('Cumulative Reward', fontsize=10)
    ax5.set_title('Growth Trajectory', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(alpha=0.3)

    # Plot 6: Ranking Summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # 순위 계산
    rankings = []
    for paradigm_name in paradigm_names:
        final = summary_stats[paradigm_name]['avg_final']
        improvement = summary_stats[paradigm_name]['avg_improvement']
        stability = -summary_stats[paradigm_name]['stability']  # 낮을수록 좋음

        # 종합 점수
        score = 0.5 * final + 0.3 * (improvement / 100) + 0.2 * stability
        rankings.append((paradigm_name, score, final, improvement, stability))

    rankings.sort(key=lambda x: x[1], reverse=True)

    # 텍스트로 순위 표시
    summary_text = "OVERALL RANKING:\n\n"
    for rank, (name, score, final, improvement, stability) in enumerate(rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        summary_text += f"{medal} {name}\n"
        summary_text += f"   Score: {score:.3f} | Final: {final:+.3f} | Improve: {improvement:+.1f}% | Stability: {-stability:.3f}\n\n"

    ax6.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('PARADIGM COMPARISON: Viability Redefined', fontsize=14, fontweight='bold', y=0.995)

    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         FINAL PARADIGM COMPARISON: VIABILITY REDEFINED           ║
    ║                                                                   ║
    ║  Viability = Capacity for Sustained Improvement                  ║
    ║  NOT survival time, but LEARNING TRAJECTORY                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # 실험 실행
    summary_stats = run_paradigm_comparison(n_trials=5, n_episodes=20)

    # 결과 시각화
    plot_comparison(summary_stats)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)

    # 최고 성능 찾기
    best_paradigm = max(summary_stats.keys(),
                       key=lambda p: summary_stats[p]['avg_improvement'])
    best_improvement = summary_stats[best_paradigm]['avg_improvement']

    print(f"\n🏆 BEST LEARNER: {best_paradigm}")
    print(f"   Improvement: {best_improvement:+.1f}%")

    print(f"\n💡 KEY INSIGHT:")
    print(f"   Viability is NOT about how long you survive,")
    print(f"   but about how FAST you improve!")
