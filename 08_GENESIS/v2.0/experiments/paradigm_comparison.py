"""
Learning Paradigm Comparison Experiment
Author: GENESIS Project
Date: 2026-01-03

목적:
    5가지 학습 패러다임을 동일한 환경에서 실증적으로 비교

Paradigms:
    1. Pure Viability (GENESIS) - Hebbian + Energy-driven
    2. Supervised Learning (SGD) - Gradient descent
    3. Reinforcement Learning - Policy gradient
    4. Hebbian Only (v1.1) - Correlation-based
    5. Random Baseline - No learning

Metrics:
    - Survival rate (생존율)
    - Energy efficiency (에너지 효율)
    - Sample efficiency (샘플 효율성)
    - Adaptation speed (적응 속도)
    - Robustness to noise (노이즈 강건성)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from pure_viability_environment import ResourceEnvironment
from pure_viability_entity import PureViabilityEntity, SimpleNeuralModule


# ============================================
# Paradigm 1: Pure Viability (GENESIS)
# ============================================
# Already implemented in PureViabilityEntity!


# ============================================
# Paradigm 2: Supervised Learning (SGD)
# ============================================

class SupervisedEntity:
    """전통적 Supervised Learning with gradient descent"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.001, initial_energy: float = 10.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 파라미터 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

        # 에너지 (비교를 위해)
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True
        self.age = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """순전파"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Hidden layer
        self.last_x = x
        self.last_h = np.tanh(np.dot(x, self.W1) + self.b1)

        # Output layer
        output = np.dot(self.last_h, self.W2) + self.b2

        return output

    def live_one_step(self, environment) -> Dict:
        """한 스텝 생존 (Supervised 방식)"""
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()

        # 예측
        prediction = self.forward(state)
        action = float(prediction.flatten()[0])

        # 환경 상호작용
        next_state, energy_change, done, info = environment.step(action)

        # **핵심 차이: Ground truth 사용!**
        target = info['optimal_action']  # ← Supervised는 이걸 받음!
        error = prediction - target

        # Gradient descent (backprop)
        grad_W2 = np.dot(self.last_h.T, error) / self.last_x.shape[0]
        grad_b2 = np.mean(error, axis=0)

        grad_h = np.dot(error, self.W2.T)
        grad_h *= (1 - self.last_h ** 2)  # tanh derivative

        grad_W1 = np.dot(self.last_x.T, grad_h) / self.last_x.shape[0]
        grad_b1 = np.mean(grad_h, axis=0)

        # Gradient clipping
        grad_W1 = np.clip(grad_W1, -1.0, 1.0)
        grad_W2 = np.clip(grad_W2, -1.0, 1.0)

        # 파라미터 업데이트
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2

        # 에너지 업데이트
        self.energy += energy_change

        if self.energy <= 0:
            self.is_alive = False

        return {
            'action': action,
            'energy_change': energy_change,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'age': self.age
        }


# ============================================
# Paradigm 3: Reinforcement Learning
# ============================================

class RLEntity:
    """Policy Gradient (REINFORCE)"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.01, initial_energy: float = 10.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 파라미터
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

        # RL 메모리
        self.trajectory = []  # (state, action, reward)

        # 에너지
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True
        self.age = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """순전파"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        self.last_x = x
        self.last_h = np.tanh(np.dot(x, self.W1) + self.b1)
        output = np.dot(self.last_h, self.W2) + self.b2

        return output

    def live_one_step(self, environment) -> Dict:
        """한 스텝 생존 (RL 방식)"""
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()

        # 행동 생성
        mean = self.forward(state)
        action = float(mean.flatten()[0]) + np.random.randn() * 0.1  # Exploration

        # 환경 상호작용
        next_state, energy_change, done, info = environment.step(action)

        # Trajectory 저장
        self.trajectory.append((state, action, energy_change))

        # 주기적으로 policy update (10 steps마다)
        if len(self.trajectory) >= 10:
            self._policy_gradient_update()
            self.trajectory = []

        # 에너지 업데이트
        self.energy += energy_change

        if self.energy <= 0:
            self.is_alive = False

        return {
            'action': action,
            'energy_change': energy_change,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'age': self.age
        }

    def _policy_gradient_update(self):
        """Policy gradient 업데이트"""
        if len(self.trajectory) == 0:
            return

        # Compute returns
        returns = []
        R = 0
        for (state, action, reward) in reversed(self.trajectory):
            R = reward + 0.99 * R
            returns.insert(0, R)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Policy gradient
        for (state, action, reward), R in zip(self.trajectory, returns):
            # Forward
            mean = self.forward(state)

            # Gradient (간단한 버전)
            grad_output = (action - mean) * R

            # Backprop (간략화)
            grad_W2 = np.dot(self.last_h.T, grad_output)
            grad_h = np.dot(grad_output, self.W2.T) * (1 - self.last_h ** 2)
            grad_W1 = np.dot(self.last_x.T, grad_h)

            # Clipping
            grad_W1 = np.clip(grad_W1, -1.0, 1.0)
            grad_W2 = np.clip(grad_W2, -1.0, 1.0)

            # Update
            self.W1 += self.learning_rate * grad_W1 * 0.1
            self.W2 += self.learning_rate * grad_W2 * 0.1


# ============================================
# Paradigm 4: Hebbian Only (v1.1 style)
# ============================================

class HebbianOnlyEntity:
    """Pure Hebbian learning (no viability assessment)"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hebbian_lr: float = 0.01, initial_energy: float = 10.0):
        self.encoder = SimpleNeuralModule(input_size, hidden_size, "encoder")
        self.decoder = SimpleNeuralModule(hidden_size, output_size, "decoder")

        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True
        self.age = 0

        self.hebbian_lr = hebbian_lr

    def forward(self, state: np.ndarray) -> np.ndarray:
        hidden = self.encoder.forward(state)
        action = self.decoder.forward(hidden)
        return action

    def live_one_step(self, environment) -> Dict:
        """한 스텝 생존 (Hebbian only)"""
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1
        state = environment.get_state()

        action = self.forward(state)
        action_scalar = float(action.flatten()[0])

        next_state, energy_change, done, info = environment.step(action_scalar)

        # Hebbian update (단순히 에너지 증가 = 성공)
        success = (energy_change > 0)
        self.encoder.hebbian_update(success, self.hebbian_lr)
        self.decoder.hebbian_update(success, self.hebbian_lr)

        self.energy += energy_change

        if self.energy <= 0:
            self.is_alive = False

        return {
            'action': action_scalar,
            'energy_change': energy_change,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'age': self.age
        }


# ============================================
# Paradigm 5: Random Baseline
# ============================================

class RandomEntity:
    """랜덤 행동 (학습 없음)"""

    def __init__(self, initial_energy: float = 10.0):
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True
        self.age = 0

    def live_one_step(self, environment) -> Dict:
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1

        # 랜덤 행동
        action = np.random.randn() * 2.0

        next_state, energy_change, done, info = environment.step(action)

        self.energy += energy_change

        if self.energy <= 0:
            self.is_alive = False

        return {
            'action': action,
            'energy_change': energy_change,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'age': self.age
        }


# ============================================
# Experiment Runner
# ============================================

def run_paradigm_experiment(paradigm_name: str, entity, env, n_steps: int = 200) -> Dict:
    """
    단일 패러다임 실험 실행

    Returns:
        results: {
            'paradigm': str,
            'survival_steps': int,
            'final_energy': float,
            'avg_energy': float,
            'energy_gain_rate': float,
            'energy_history': List[float]
        }
    """
    print(f"\n[{paradigm_name}] Starting...")

    env.reset()
    energy_history = []

    for step in range(n_steps):
        result = entity.live_one_step(env)

        if not result['is_alive']:
            print(f"  Died at step {step}")
            break

        energy_history.append(result['energy'])

    survival_steps = len(energy_history)
    final_energy = energy_history[-1] if len(energy_history) > 0 else 0
    avg_energy = np.mean(energy_history) if len(energy_history) > 0 else 0
    energy_gain_rate = (final_energy - entity.initial_energy) / max(survival_steps, 1)

    print(f"  Survived: {survival_steps}/{n_steps} steps")
    print(f"  Final energy: {final_energy:.2f}")
    print(f"  Avg energy: {avg_energy:.2f}")
    print(f"  Energy gain rate: {energy_gain_rate:+.3f}/step")

    return {
        'paradigm': paradigm_name,
        'survival_steps': survival_steps,
        'final_energy': final_energy,
        'avg_energy': avg_energy,
        'energy_gain_rate': energy_gain_rate,
        'energy_history': energy_history
    }


def run_full_comparison(n_trials: int = 5, n_steps: int = 200):
    """
    전체 패러다임 비교 실험

    Args:
        n_trials: 각 패러다임당 trial 수
        n_steps: 각 trial의 스텝 수
    """
    print("=" * 70)
    print("PARADIGM COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Trials per paradigm: {n_trials}")
    print(f"  Steps per trial: {n_steps}")
    print(f"  Environment: ResourceEnvironment (linear)")

    all_results = []

    paradigms = [
        ('Pure Viability (GENESIS)', lambda: PureViabilityEntity(
            input_size=5, hidden_size=16, output_size=1, initial_energy=10.0, hebbian_lr=0.01)),
        ('Supervised Learning (SGD)', lambda: SupervisedEntity(
            input_size=5, hidden_size=16, output_size=1, learning_rate=0.001, initial_energy=10.0)),
        ('Reinforcement Learning', lambda: RLEntity(
            input_size=5, hidden_size=16, output_size=1, learning_rate=0.01, initial_energy=10.0)),
        ('Hebbian Only (v1.1)', lambda: HebbianOnlyEntity(
            input_size=5, hidden_size=16, output_size=1, hebbian_lr=0.01, initial_energy=10.0)),
        ('Random Baseline', lambda: RandomEntity(initial_energy=10.0))
    ]

    for paradigm_name, entity_factory in paradigms:
        print(f"\n{'='*70}")
        print(f"Testing: {paradigm_name}")
        print(f"{'='*70}")

        paradigm_results = []

        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}")

            # 환경 생성 (매 trial마다 새로운 환경)
            env = ResourceEnvironment(
                input_dim=5,
                function_type='linear',
                energy_cost_per_step=0.05,
                seed=42 + trial
            )

            # Entity 생성
            entity = entity_factory()

            # 실험 실행
            result = run_paradigm_experiment(paradigm_name, entity, env, n_steps)
            paradigm_results.append(result)

        all_results.append({
            'paradigm': paradigm_name,
            'trials': paradigm_results
        })

    return all_results


def analyze_and_plot_results(all_results: List[Dict], save_dir: str = '../../results'):
    """결과 분석 및 시각화"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # 각 패러다임별 통계
    summary = []

    for paradigm_data in all_results:
        paradigm_name = paradigm_data['paradigm']
        trials = paradigm_data['trials']

        survival_steps = [t['survival_steps'] for t in trials]
        final_energies = [t['final_energy'] for t in trials]
        avg_energies = [t['avg_energy'] for t in trials]
        energy_gain_rates = [t['energy_gain_rate'] for t in trials]

        summary.append({
            'paradigm': paradigm_name,
            'avg_survival': np.mean(survival_steps),
            'std_survival': np.std(survival_steps),
            'avg_final_energy': np.mean(final_energies),
            'std_final_energy': np.std(final_energies),
            'avg_energy_gain_rate': np.mean(energy_gain_rates),
            'std_energy_gain_rate': np.std(energy_gain_rates)
        })

    # 표 출력
    print(f"{'Paradigm':<30} | {'Survival':<15} | {'Final Energy':<15} | {'Gain Rate':<15}")
    print("-" * 90)

    for s in summary:
        print(f"{s['paradigm']:<30} | "
              f"{s['avg_survival']:6.1f} ± {s['std_survival']:5.1f} | "
              f"{s['avg_final_energy']:6.2f} ± {s['std_final_energy']:5.2f} | "
              f"{s['avg_energy_gain_rate']:+6.3f} ± {s['std_energy_gain_rate']:5.3f}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    paradigm_names = [s['paradigm'] for s in summary]
    x_pos = np.arange(len(paradigm_names))

    # Plot 1: Survival Steps
    ax = axes[0]
    survival_means = [s['avg_survival'] for s in summary]
    survival_stds = [s['std_survival'] for s in summary]
    ax.bar(x_pos, survival_means, yerr=survival_stds, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names], rotation=45, ha='right')
    ax.set_ylabel('Survival Steps')
    ax.set_title('Survival Duration')
    ax.grid(True, alpha=0.3)

    # Plot 2: Final Energy
    ax = axes[1]
    energy_means = [s['avg_final_energy'] for s in summary]
    energy_stds = [s['std_final_energy'] for s in summary]
    ax.bar(x_pos, energy_means, yerr=energy_stds, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names], rotation=45, ha='right')
    ax.set_ylabel('Final Energy')
    ax.set_title('Final Energy Level')
    ax.grid(True, alpha=0.3)

    # Plot 3: Energy Gain Rate
    ax = axes[2]
    gain_means = [s['avg_energy_gain_rate'] for s in summary]
    gain_stds = [s['std_energy_gain_rate'] for s in summary]
    colors = ['green' if g > 0 else 'red' for g in gain_means]
    ax.bar(x_pos, gain_means, yerr=gain_stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.split('(')[0].strip() for p in paradigm_names], rotation=45, ha='right')
    ax.set_ylabel('Energy Gain Rate (per step)')
    ax.set_title('Learning Efficiency')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'paradigm_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")

    return summary


if __name__ == "__main__":
    # 실험 실행
    results = run_full_comparison(n_trials=5, n_steps=200)

    # 분석 및 시각화
    summary = analyze_and_plot_results(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
