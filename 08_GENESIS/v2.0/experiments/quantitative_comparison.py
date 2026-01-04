"""
GENESIS: Rigorous Quantitative Paradigm Comparison
Author: GENESIS Project
Date: 2026-01-04

정량적 비교 메트릭:
    1. Survival Rate: 생존율 (%)
    2. Final Performance: 최종 성능
    3. Learning Speed: 학습 속도 (첫 50% 도달 시간)
    4. Stability: 성능 안정성 (std dev)
    5. Sample Efficiency: 샘플 효율성
    6. Robustness: 환경 변화 대응
    7. Adaptability: 새 조건 적응

통계적 검증:
    - Mean ± Std (N=10 trials)
    - Cohen's d (effect size)
    - Statistical significance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy import stats

from autopoietic_population import AutopoeticPopulation, PerturbationField
from viability_redefined import ViabilityDrivenEntity, MultiEpisodeEnvironment


class QuantitativeExperiment:
    """정량적 비교 실험"""

    def __init__(self, n_trials: int = 10):
        self.n_trials = n_trials
        self.results = defaultdict(list)

    def run_autopoietic_trial(self, n_steps: int = 200) -> dict:
        """Autopoietic 시행"""
        field = PerturbationField(field_size=20, turbulence=0.3)
        population = AutopoeticPopulation(
            initial_population=5,
            max_population=20,
            reproduction_threshold=0.7,
            mutation_rate=0.1
        )

        coherence_history = []
        population_history = []
        structural_changes = []

        for step in range(n_steps):
            stats = population.step(field)
            coherence_history.append(stats['avg_coherence'])
            population_history.append(stats['population'])

            # Count structural changes
            total_changes = sum([e.plasticity.structural_changes
                               for e in population.entities if e.is_alive])
            structural_changes.append(total_changes)

            if stats['population'] == 0:
                break

        return {
            'paradigm': 'Autopoietic',
            'final_performance': coherence_history[-1] if coherence_history else 0,
            'mean_performance': np.mean(coherence_history) if coherence_history else 0,
            'std_performance': np.std(coherence_history) if coherence_history else 0,
            'survival_rate': population_history[-1] / 5.0,  # Normalized by initial
            'learning_speed': self._compute_learning_speed(coherence_history),
            'sample_efficiency': len(coherence_history) / max(coherence_history[-1], 0.01) if coherence_history else 0,
            'adaptability': self._compute_adaptability(coherence_history),
            'structural_changes': structural_changes[-1] if structural_changes else 0,
            'history': coherence_history
        }

    def run_ml_trial(self, paradigm: str, n_steps: int = 200) -> dict:
        """ML 패러다임 시행"""
        env = MultiEpisodeEnvironment(
            input_dim=5,
            episode_length=100,
            energy_cost=0.1,
            reward_scale=0.5
        )

        if paradigm == 'Supervised':
            entity = self._create_supervised_entity()
        elif paradigm == 'RL':
            entity = self._create_rl_entity()
        elif paradigm == 'Hebbian':
            entity = self._create_hebbian_entity()
        elif paradigm == 'Random':
            entity = self._create_random_entity()
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

        performance_history = []
        energy_history = []

        n_episodes = n_steps // 100

        for episode in range(n_episodes):
            result = entity.live_episode(env)
            performance_history.append(result['avg_reward'])
            energy_history.append(result['final_energy'])

            if not result.get('is_alive', True):
                break

        # Normalize to 0-1
        if len(performance_history) > 0:
            perf_normalized = [(p + 0.2) / 0.4 for p in performance_history]  # Roughly normalize
        else:
            perf_normalized = [0]

        return {
            'paradigm': paradigm,
            'final_performance': perf_normalized[-1] if perf_normalized else 0,
            'mean_performance': np.mean(perf_normalized) if perf_normalized else 0,
            'std_performance': np.std(perf_normalized) if perf_normalized else 0,
            'survival_rate': 1.0 if energy_history[-1] > 0 else 0.0,
            'learning_speed': self._compute_learning_speed(perf_normalized),
            'sample_efficiency': len(perf_normalized) / max(perf_normalized[-1], 0.01) if perf_normalized else 0,
            'adaptability': self._compute_adaptability(perf_normalized),
            'structural_changes': 0,  # ML has no structural changes
            'history': perf_normalized
        }

    def _create_supervised_entity(self):
        """Supervised learning entity"""
        class SupervisedEntity:
            def __init__(self):
                self.W1 = np.random.randn(5, 16) * 0.1
                self.W2 = np.random.randn(16, 1) * 0.1
                self.episode_performances = []

            def forward(self, state):
                hidden = np.tanh(np.dot(state.flatten(), self.W1))
                return np.tanh(np.dot(hidden, self.W2))

            def live_episode(self, environment):
                state = environment.reset_episode()
                episode_energy = 10.0
                episode_rewards = []

                for _ in range(environment.episode_length):
                    action = self.forward(state)
                    next_state, energy_change, done, info = environment.step(action.flatten()[0])

                    # Supervised: use ground truth
                    target = info['optimal_action']
                    error = action.flatten()[0] - target
                    hidden = np.tanh(np.dot(state.flatten(), self.W1))
                    self.W2 -= 0.01 * np.outer(hidden, [error])

                    episode_energy += energy_change
                    episode_rewards.append(energy_change)
                    state = next_state
                    if done: break

                self.episode_performances.append(np.mean(episode_rewards) if episode_rewards else 0)
                return {
                    'avg_reward': self.episode_performances[-1],
                    'final_energy': episode_energy,
                    'is_alive': episode_energy > 0
                }

        return SupervisedEntity()

    def _create_rl_entity(self):
        """RL entity"""
        class RLEntity:
            def __init__(self):
                self.W1 = np.random.randn(5, 16) * 0.1
                self.W2 = np.random.randn(16, 1) * 0.1
                self.episode_performances = []

            def forward(self, state):
                hidden = np.tanh(np.dot(state.flatten(), self.W1))
                return np.tanh(np.dot(hidden, self.W2)) + np.random.randn(1, 1) * 0.1

            def live_episode(self, environment):
                state = environment.reset_episode()
                episode_energy = 10.0
                episode_rewards = []
                states, actions, rewards = [], [], []

                for _ in range(environment.episode_length):
                    action = self.forward(state)
                    next_state, energy_change, done, info = environment.step(action.flatten()[0])

                    states.append(state)
                    actions.append(action)
                    rewards.append(energy_change)

                    episode_energy += energy_change
                    episode_rewards.append(energy_change)
                    state = next_state
                    if done: break

                # Policy gradient
                if len(rewards) > 0:
                    returns = []
                    G = 0
                    for r in reversed(rewards):
                        G = r + 0.99 * G
                        returns.insert(0, G)
                    returns = np.array(returns)
                    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

                    for s, a, R in zip(states[:len(returns)], actions[:len(returns)], returns):
                        hidden = np.tanh(np.dot(s.flatten(), self.W1))
                        self.W2 += 0.001 * np.outer(hidden, a) * R

                self.episode_performances.append(np.mean(episode_rewards) if episode_rewards else 0)
                return {
                    'avg_reward': self.episode_performances[-1],
                    'final_energy': episode_energy,
                    'is_alive': episode_energy > 0
                }

        return RLEntity()

    def _create_hebbian_entity(self):
        """Hebbian entity"""
        class HebbianEntity:
            def __init__(self):
                self.W1 = np.random.randn(5, 16) * 0.1
                self.W2 = np.random.randn(16, 1) * 0.1
                self.episode_performances = []

            def forward(self, state):
                hidden = np.tanh(np.dot(state.flatten(), self.W1))
                return np.tanh(np.dot(hidden, self.W2))

            def live_episode(self, environment):
                state = environment.reset_episode()
                episode_energy = 10.0
                episode_rewards = []

                for _ in range(environment.episode_length):
                    action = self.forward(state)
                    next_state, energy_change, done, info = environment.step(action.flatten()[0])

                    # Hebbian
                    if energy_change > 0:
                        hidden = np.tanh(np.dot(state.flatten(), self.W1))
                        self.W2 += 0.01 * np.outer(hidden, action)
                        self.W1 += 0.005 * np.outer(state.flatten(), hidden)

                    episode_energy += energy_change
                    episode_rewards.append(energy_change)
                    state = next_state
                    if done: break

                self.episode_performances.append(np.mean(episode_rewards) if episode_rewards else 0)
                return {
                    'avg_reward': self.episode_performances[-1],
                    'final_energy': episode_energy,
                    'is_alive': episode_energy > 0
                }

        return HebbianEntity()

    def _create_random_entity(self):
        """Random baseline"""
        class RandomEntity:
            def __init__(self):
                self.episode_performances = []

            def live_episode(self, environment):
                state = environment.reset_episode()
                episode_energy = 10.0
                episode_rewards = []

                for _ in range(environment.episode_length):
                    action = np.random.randn() * 2.0
                    next_state, energy_change, done, info = environment.step(action)

                    episode_energy += energy_change
                    episode_rewards.append(energy_change)
                    state = next_state
                    if done: break

                self.episode_performances.append(np.mean(episode_rewards) if episode_rewards else 0)
                return {
                    'avg_reward': self.episode_performances[-1],
                    'final_energy': episode_energy,
                    'is_alive': episode_energy > 0
                }

        return RandomEntity()

    def _compute_learning_speed(self, history: list) -> float:
        """학습 속도: 50% 최종 성능 도달 시간"""
        if len(history) < 2:
            return 999

        final_perf = history[-1]
        target = final_perf * 0.5

        for i, perf in enumerate(history):
            if perf >= target:
                return i

        return len(history)

    def _compute_adaptability(self, history: list) -> float:
        """적응성: 성능 개선률"""
        if len(history) < 10:
            return 0

        initial = np.mean(history[:5])
        final = np.mean(history[-5:])

        if abs(initial) < 1e-8:
            return 0

        improvement = (final - initial) / (abs(initial) + 1e-8)
        return float(np.clip(improvement, -1, 1))

    def run_full_comparison(self) -> dict:
        """전체 비교 실험"""
        print("=" * 70)
        print("RIGOROUS QUANTITATIVE COMPARISON")
        print("=" * 70)
        print(f"\nRunning {self.n_trials} trials per paradigm...\n")

        paradigms = ['Autopoietic', 'Supervised', 'RL', 'Hebbian', 'Random']
        all_results = defaultdict(list)

        for paradigm in paradigms:
            print(f"\n{'='*70}")
            print(f"Testing: {paradigm}")
            print(f"{'='*70}")

            for trial in range(self.n_trials):
                if paradigm == 'Autopoietic':
                    result = self.run_autopoietic_trial(n_steps=200)
                else:
                    result = self.run_ml_trial(paradigm, n_steps=200)

                all_results[paradigm].append(result)

                print(f"  Trial {trial+1}/{self.n_trials}: "
                      f"Final Perf={result['final_performance']:.3f}, "
                      f"Speed={result['learning_speed']:.0f}, "
                      f"Adapt={result['adaptability']:+.2f}")

        return self._aggregate_results(all_results)

    def _aggregate_results(self, all_results: dict) -> dict:
        """결과 집계 및 통계"""
        aggregated = {}

        metrics = ['final_performance', 'mean_performance', 'std_performance',
                  'survival_rate', 'learning_speed', 'sample_efficiency',
                  'adaptability', 'structural_changes']

        for paradigm, trials in all_results.items():
            aggregated[paradigm] = {}

            for metric in metrics:
                values = [trial[metric] for trial in trials]
                aggregated[paradigm][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }

            # Average history
            histories = [trial['history'] for trial in trials]
            max_len = max(len(h) for h in histories)

            # Pad histories
            padded = [h + [h[-1]] * (max_len - len(h)) if len(h) > 0 else [0] * max_len
                     for h in histories]
            aggregated[paradigm]['avg_history'] = np.mean(padded, axis=0)
            aggregated[paradigm]['std_history'] = np.std(padded, axis=0)

        return aggregated

    def compute_effect_sizes(self, results: dict) -> dict:
        """Cohen's d 계산"""
        effect_sizes = {}

        baseline = 'Random'
        metrics = ['final_performance', 'learning_speed', 'adaptability']

        for paradigm in results.keys():
            if paradigm == baseline:
                continue

            effect_sizes[paradigm] = {}

            for metric in metrics:
                values1 = results[paradigm][metric]['values']
                values2 = results[baseline][metric]['values']

                mean1 = np.mean(values1)
                mean2 = np.mean(values2)
                std_pooled = np.sqrt((np.var(values1) + np.var(values2)) / 2)

                if std_pooled > 1e-8:
                    cohens_d = (mean1 - mean2) / std_pooled
                else:
                    cohens_d = 0

                effect_sizes[paradigm][metric] = cohens_d

        return effect_sizes

    def statistical_tests(self, results: dict) -> dict:
        """통계적 유의성 검정"""
        tests = {}

        baseline = 'Random'
        metrics = ['final_performance', 'adaptability']

        for paradigm in results.keys():
            if paradigm == baseline:
                continue

            tests[paradigm] = {}

            for metric in metrics:
                values1 = results[paradigm][metric]['values']
                values2 = results[baseline][metric]['values']

                # t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)

                tests[paradigm][metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        return tests


def print_quantitative_results(results: dict, effect_sizes: dict, stat_tests: dict):
    """정량적 결과 출력"""
    print(f"\n{'='*70}")
    print("QUANTITATIVE RESULTS")
    print(f"{'='*70}\n")

    # Table 1: Performance Metrics
    print("Table 1: Performance Metrics (Mean ± Std)")
    print("-" * 100)
    print(f"{'Paradigm':<20} | {'Final Perf':>15} | {'Mean Perf':>15} | {'Survival':>12} | {'Adaptability':>15}")
    print("-" * 100)

    for paradigm, data in results.items():
        final = data['final_performance']
        mean_p = data['mean_performance']
        survival = data['survival_rate']
        adapt = data['adaptability']

        print(f"{paradigm:<20} | "
              f"{final['mean']:>6.3f} ± {final['std']:>5.3f} | "
              f"{mean_p['mean']:>6.3f} ± {mean_p['std']:>5.3f} | "
              f"{survival['mean']*100:>9.1f}% | "
              f"{adapt['mean']:>+6.2f} ± {adapt['std']:>5.2f}")

    # Table 2: Learning Efficiency
    print(f"\n\nTable 2: Learning Efficiency")
    print("-" * 80)
    print(f"{'Paradigm':<20} | {'Speed (steps)':>18} | {'Sample Efficiency':>20} | {'Struct Changes':>15}")
    print("-" * 80)

    for paradigm, data in results.items():
        speed = data['learning_speed']
        efficiency = data['sample_efficiency']
        changes = data['structural_changes']

        print(f"{paradigm:<20} | "
              f"{speed['mean']:>8.1f} ± {speed['std']:>6.1f} | "
              f"{efficiency['mean']:>9.2f} ± {efficiency['std']:>7.2f} | "
              f"{changes['mean']:>12.1f}")

    # Table 3: Effect Sizes (Cohen's d)
    print(f"\n\nTable 3: Effect Sizes vs Random Baseline (Cohen's d)")
    print("-" * 70)
    print(f"{'Paradigm':<20} | {'Final Perf':>15} | {'Speed':>15} | {'Adaptability':>15}")
    print("-" * 70)

    for paradigm, effects in effect_sizes.items():
        print(f"{paradigm:<20} | "
              f"{effects['final_performance']:>+15.2f} | "
              f"{effects['learning_speed']:>+15.2f} | "
              f"{effects['adaptability']:>+15.2f}")

    # Table 4: Statistical Significance
    print(f"\n\nTable 4: Statistical Significance (t-test vs Random)")
    print("-" * 80)
    print(f"{'Paradigm':<20} | {'Metric':<20} | {'p-value':>12} | {'Significant?':>15}")
    print("-" * 80)

    for paradigm, tests in stat_tests.items():
        for metric, test_result in tests.items():
            sig_str = "✓ YES" if test_result['significant'] else "  No"
            print(f"{paradigm:<20} | {metric:<20} | "
                  f"{test_result['p_value']:>12.4f} | {sig_str:>15}")


def plot_quantitative_comparison(results: dict,
                                 save_path: str = '../../results/quantitative_comparison.png'):
    """정량적 비교 시각화"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    paradigms = list(results.keys())
    colors = {
        'Autopoietic': '#2E86AB',
        'Supervised': '#A23B72',
        'RL': '#F18F01',
        'Hebbian': '#C73E1D',
        'Random': '#6C757D'
    }

    # Plot 1: Learning Curves with Error Bars
    ax1 = fig.add_subplot(gs[0, :2])
    for paradigm in paradigms:
        mean_hist = results[paradigm]['avg_history']
        std_hist = results[paradigm]['std_history']
        x = np.arange(len(mean_hist))

        ax1.plot(x, mean_hist, label=paradigm, color=colors[paradigm],
                linewidth=2.5, alpha=0.9)
        ax1.fill_between(x, mean_hist - std_hist, mean_hist + std_hist,
                         color=colors[paradigm], alpha=0.2)

    ax1.set_xlabel('Time Steps / Episodes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (Normalized)', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Curves (Mean ± Std, N=10)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1])

    # Plot 2: Final Performance
    ax2 = fig.add_subplot(gs[0, 2])
    means = [results[p]['final_performance']['mean'] for p in paradigms]
    stds = [results[p]['final_performance']['std'] for p in paradigms]

    bars = ax2.bar(range(len(paradigms)), means, yerr=stds,
                   color=[colors[p] for p in paradigms], alpha=0.7,
                   capsize=5, error_kw={'linewidth': 2})
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(3)

    ax2.set_xticks(range(len(paradigms)))
    ax2.set_xticklabels([p.replace(' ', '\n') for p in paradigms],
                        fontsize=9)
    ax2.set_ylabel('Final Performance', fontsize=11, fontweight='bold')
    ax2.set_title('Final Performance\n(Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Learning Speed
    ax3 = fig.add_subplot(gs[1, 0])
    means = [results[p]['learning_speed']['mean'] for p in paradigms]
    stds = [results[p]['learning_speed']['std'] for p in paradigms]

    bars = ax3.barh(range(len(paradigms)), means, xerr=stds,
                    color=[colors[p] for p in paradigms], alpha=0.7,
                    capsize=5, error_kw={'linewidth': 2})

    ax3.set_yticks(range(len(paradigms)))
    ax3.set_yticklabels(paradigms, fontsize=9)
    ax3.set_xlabel('Steps to 50% Performance', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Speed\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_xaxis()

    # Plot 4: Adaptability
    ax4 = fig.add_subplot(gs[1, 1])
    means = [results[p]['adaptability']['mean'] for p in paradigms]
    stds = [results[p]['adaptability']['std'] for p in paradigms]

    bars = ax4.bar(range(len(paradigms)), means, yerr=stds,
                   color=[colors[p] for p in paradigms], alpha=0.7,
                   capsize=5, error_kw={'linewidth': 2})

    # Color bars by value
    for bar, mean in zip(bars, means):
        if mean > 0:
            bar.set_color('#2E7D32')  # Green
        else:
            bar.set_color('#C62828')  # Red

    ax4.set_xticks(range(len(paradigms)))
    ax4.set_xticklabels([p.replace(' ', '\n') for p in paradigms],
                        fontsize=9)
    ax4.set_ylabel('Improvement Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Adaptability\n(Initial→Final)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Plot 5: Survival Rate
    ax5 = fig.add_subplot(gs[1, 2])
    means = [results[p]['survival_rate']['mean'] * 100 for p in paradigms]

    bars = ax5.bar(range(len(paradigms)), means,
                   color=[colors[p] for p in paradigms], alpha=0.7)

    ax5.set_xticks(range(len(paradigms)))
    ax5.set_xticklabels([p.replace(' ', '\n') for p in paradigms],
                        fontsize=9)
    ax5.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Survival Rate', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim([0, 100])

    # Plot 6: Performance Stability
    ax6 = fig.add_subplot(gs[2, 0])
    stds = [results[p]['mean_performance']['std'] for p in paradigms]

    bars = ax6.barh(range(len(paradigms)), stds,
                    color=[colors[p] for p in paradigms], alpha=0.7)

    ax6.set_yticks(range(len(paradigms)))
    ax6.set_yticklabels(paradigms, fontsize=9)
    ax6.set_xlabel('Std Dev', fontsize=11, fontweight='bold')
    ax6.set_title('Performance Stability\n(Lower is More Stable)', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Plot 7: Box plots
    ax7 = fig.add_subplot(gs[2, 1:])

    data_to_plot = [results[p]['final_performance']['values'] for p in paradigms]
    bp = ax7.boxplot(data_to_plot, labels=[p.replace(' ', '\n') for p in paradigms],
                     patch_artist=True, widths=0.6)

    for patch, paradigm in zip(bp['boxes'], paradigms):
        patch.set_facecolor(colors[paradigm])
        patch.set_alpha(0.7)

    ax7.set_ylabel('Final Performance', fontsize=11, fontweight='bold')
    ax7.set_title('Performance Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim([0, 1])

    plt.suptitle('QUANTITATIVE PARADIGM COMPARISON (N=10 trials per paradigm)',
                fontsize=16, fontweight='bold')

    # Save
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         RIGOROUS QUANTITATIVE PARADIGM COMPARISON                ║
    ║                                                                   ║
    ║  Statistical Analysis with N=10 trials per paradigm              ║
    ║  Mean ± Std, Cohen's d, t-tests                                  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Run experiment
    experiment = QuantitativeExperiment(n_trials=10)
    results = experiment.run_full_comparison()

    # Statistical analysis
    effect_sizes = experiment.compute_effect_sizes(results)
    stat_tests = experiment.statistical_tests(results)

    # Print results
    print_quantitative_results(results, effect_sizes, stat_tests)

    # Plot
    plot_quantitative_comparison(results)

    print("\n" + "=" * 70)
    print("QUANTITATIVE COMPARISON COMPLETE!")
    print("=" * 70)
