"""
Viability Redefined: The Fundamental Reconceptualization
Author: GENESIS Project
Date: 2026-01-03

핵심 통찰:
    Viability는 생존 시간이 아니라 생존 능력의 궤적이다.

    잘못된 정의: Viability = Current State (alive/dead)
    올바른 정의: Viability = Capacity for Sustained Existence

진정한 생존력의 5가지 차원:
    1. Sustainability: 현재 상태를 유지할 수 있는가?
    2. Adaptability: 새로운 도전에 적응할 수 있는가?
    3. Anticipation: 미래를 예측하고 준비하는가?
    4. Resilience: 실패에서 회복할 수 있는가?
    5. Growth: 시간에 따라 개선되고 있는가?

새로운 실험 패러다임:
    기존: Single life until death (한 번 살다 죽음)
    신규: Multi-episode learning (여러 에피소드에서 학습)

    초점: 생존 시간 → 학습 곡선의 기울기
         에너지 수준 → 에너지 획득 효율 개선률
         죽음 회피 → 각 생애마다 더 나아지는가?
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import uuid


class ViabilityMetrics:
    """
    생존력의 다차원 평가 시스템

    단순히 "살아있음"이 아니라 생존 능력의 질을 측정
    """

    def __init__(self):
        # 5가지 핵심 차원
        self.sustainability = 0.5  # 현재 상태 유지 가능성
        self.adaptability = 0.5    # 적응 능력
        self.anticipation = 0.5    # 예측 능력
        self.resilience = 0.5      # 회복 능력
        self.growth = 0.5          # 성장 가능성

        # 역사적 기록
        self.history = {
            'sustainability': deque(maxlen=100),
            'adaptability': deque(maxlen=100),
            'anticipation': deque(maxlen=100),
            'resilience': deque(maxlen=100),
            'growth': deque(maxlen=100)
        }

    def update_sustainability(self, energy_level: float, energy_trend: float):
        """
        지속가능성: 현재 자원 수준과 트렌드

        Args:
            energy_level: 정규화된 에너지 (0~1+)
            energy_trend: 에너지 변화율 (-1~1)
        """
        # 에너지 수준 + 트렌드 고려
        self.sustainability = 0.6 * np.clip(energy_level, 0, 1) + 0.4 * (energy_trend + 1) / 2
        self.history['sustainability'].append(self.sustainability)

    def update_adaptability(self, learning_rate: float, exploration_rate: float):
        """
        적응성: 얼마나 빨리 학습하고 탐색하는가

        Args:
            learning_rate: 최근 성능 개선 속도
            exploration_rate: 탐색 행동 비율
        """
        # 학습 속도 + 탐색 의지
        self.adaptability = 0.7 * learning_rate + 0.3 * exploration_rate
        self.history['adaptability'].append(self.adaptability)

    def update_anticipation(self, prediction_accuracy: float, planning_horizon: int):
        """
        예측력: 미래를 얼마나 정확하게 예측하는가

        Args:
            prediction_accuracy: 예측 정확도 (0~1)
            planning_horizon: 예측 시간 범위 (steps)
        """
        # 정확도 + 시간 범위
        horizon_score = np.clip(planning_horizon / 20.0, 0, 1)  # 20 스텝이 max
        self.anticipation = 0.7 * prediction_accuracy + 0.3 * horizon_score
        self.history['anticipation'].append(self.anticipation)

    def update_resilience(self, recovery_rate: float, failure_tolerance: float):
        """
        회복력: 실패 후 얼마나 빨리 회복하는가

        Args:
            recovery_rate: 에너지 회복 속도
            failure_tolerance: 실패 견딤 정도
        """
        # 회복 속도 + 실패 내성
        self.resilience = 0.6 * recovery_rate + 0.4 * failure_tolerance
        self.history['resilience'].append(self.resilience)

    def update_growth(self, performance_trajectory: List[float]):
        """
        성장성: 시간에 따라 개선되고 있는가

        Args:
            performance_trajectory: 최근 성능 기록
        """
        if len(performance_trajectory) < 10:
            self.growth = 0.5
        else:
            # 선형 회귀로 추세 계산
            x = np.arange(len(performance_trajectory))
            y = np.array(performance_trajectory)

            # 기울기 계산
            slope = np.polyfit(x, y, 1)[0]

            # 정규화 (-1~1 → 0~1)
            self.growth = np.clip((slope + 0.1) / 0.2, 0, 1)

        self.history['growth'].append(self.growth)

    def compute_composite_viability(self) -> float:
        """
        종합 생존력 점수

        가중치:
            - Sustainability: 20% (현재 상태)
            - Adaptability: 25% (변화 능력)
            - Anticipation: 25% (예측 능력)
            - Resilience: 15% (회복 능력)
            - Growth: 15% (발전 가능성)
        """
        composite = (
            0.20 * self.sustainability +
            0.25 * self.adaptability +
            0.25 * self.anticipation +
            0.15 * self.resilience +
            0.15 * self.growth
        )

        return float(np.clip(composite, 0, 1))

    def get_trajectory_score(self) -> float:
        """
        궤적 점수: 시간에 따른 생존력 개선도

        Returns:
            개선률 (-1~1, 양수 = 개선 중)
        """
        trajectories = []

        for dimension, history in self.history.items():
            if len(history) >= 20:
                recent = np.mean(list(history)[-10:])
                older = np.mean(list(history)[-20:-10])
                improvement = (recent - older) / (older + 1e-8)
                trajectories.append(improvement)

        if len(trajectories) == 0:
            return 0.0

        return float(np.mean(trajectories))

    def get_summary(self) -> Dict:
        """생존력 차원별 요약"""
        return {
            'sustainability': self.sustainability,
            'adaptability': self.adaptability,
            'anticipation': self.anticipation,
            'resilience': self.resilience,
            'growth': self.growth,
            'composite': self.compute_composite_viability(),
            'trajectory': self.get_trajectory_score()
        }


class MultiEpisodeEnvironment:
    """
    다중 에피소드 환경

    핵심: 한 번 죽고 끝이 아니라, 여러 생애를 거치며 학습
    """

    def __init__(self,
                 input_dim: int = 5,
                 episode_length: int = 100,
                 energy_cost: float = 0.1,
                 reward_scale: float = 0.5,
                 seed: int = 42):
        """
        Args:
            input_dim: 상태 차원
            episode_length: 에피소드 길이
            energy_cost: 에너지 소모
            reward_scale: 보상 스케일
        """
        np.random.seed(seed)

        self.input_dim = input_dim
        self.episode_length = episode_length
        self.energy_cost = energy_cost
        self.reward_scale = reward_scale

        # 숨겨진 함수 (여러 에피소드 간 고정)
        self.W_hidden = np.random.randn(input_dim) * 1.5

        # 에피소드 상태
        self.current_state = None
        self.episode_step = 0
        self.episode_count = 0

        print(f"MultiEpisodeEnvironment initialized:")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Energy cost: {energy_cost}/step")
        print(f"  Reward scale: {reward_scale}")

    def reset_episode(self) -> np.ndarray:
        """새 에피소드 시작"""
        self.current_state = np.random.randn(self.input_dim)
        self.episode_step = 0
        self.episode_count += 1
        return self.current_state.copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 스텝

        Returns:
            next_state: 다음 상태
            energy_change: 에너지 변화
            done: 에피소드 종료
            info: 추가 정보
        """
        self.episode_step += 1

        # 최적 행동 (entity는 모름)
        optimal_action = np.dot(np.tanh(self.current_state), self.W_hidden)

        # 행동 품질
        action_quality = -np.abs(action - optimal_action)

        # 에너지 변화
        if action_quality > -1.0:
            energy_reward = self.reward_scale * np.exp(action_quality)
        else:
            energy_reward = 0.0

        energy_change = energy_reward - self.energy_cost

        # 다음 상태
        next_state = np.random.randn(self.input_dim)
        self.current_state = next_state

        # 에피소드 종료 (시간 제한)
        done = (self.episode_step >= self.episode_length)

        info = {
            'optimal_action': optimal_action,
            'action_quality': action_quality,
            'energy_reward': energy_reward,
            'episode': self.episode_count,
            'step': self.episode_step
        }

        return next_state, energy_change, done, info

    def get_state(self) -> np.ndarray:
        """현재 상태"""
        return self.current_state.copy()


class ViabilityDrivenEntity:
    """
    진정한 생존력 기반 Entity

    핵심 차이:
        - 죽음이 아니라 학습 궤적에 초점
        - 다중 에피소드에서 개선
        - 생존력의 5가지 차원 추적
    """

    def __init__(self,
                 state_size: int = 5,
                 action_size: int = 1,
                 hidden_size: int = 32,
                 entity_id: Optional[str] = None):
        """
        Args:
            state_size: 상태 차원
            action_size: 행동 차원
            hidden_size: 은닉층 크기
        """
        self.id = entity_id or str(uuid.uuid4())[:8]
        self.state_size = state_size
        self.action_size = action_size

        # 신경망
        self.W1 = np.random.randn(state_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, action_size) * 0.1

        # 생존력 평가 시스템
        self.viability_metrics = ViabilityMetrics()

        # 학습 기록
        self.episode_performances = []  # 각 에피소드 성능
        self.episode_energies = []      # 각 에피소드 최종 에너지
        self.prediction_errors = deque(maxlen=50)

        # 현재 에피소드 기록
        self.current_episode_energy = 0.0
        self.current_episode_rewards = []
        self.current_episode_actions = []

        # 예측 모듈
        self.predicted_states = deque(maxlen=20)
        self.actual_states = deque(maxlen=20)

        # 통계
        self.total_episodes = 0
        self.total_steps = 0

        print(f"ViabilityDrivenEntity created: id={self.id}")
        print(f"  Focus: Learning trajectory, not survival time")
        print(f"  Metrics: 5-dimensional viability assessment")

    def forward(self, state: np.ndarray) -> np.ndarray:
        """행동 생성"""
        x = state.flatten()
        hidden = np.tanh(np.dot(x, self.W1))
        action = np.tanh(np.dot(hidden, self.W2))
        return action

    def live_episode(self, environment: MultiEpisodeEnvironment) -> Dict:
        """
        한 에피소드 생존

        핵심: 에피소드 내에서 학습하고, 에피소드 간 개선
        """
        state = environment.reset_episode()

        episode_energy = 10.0  # 에피소드 초기 에너지
        episode_rewards = []
        episode_steps = 0

        # 에피소드 실행
        for step in range(environment.episode_length):
            # 행동 생성
            action = self.forward(state)

            # 환경 상호작용
            next_state, energy_change, done, info = environment.step(action.flatten()[0])

            # 에너지 업데이트
            episode_energy += energy_change
            episode_rewards.append(energy_change)
            episode_steps += 1
            self.total_steps += 1

            # 예측 기록
            self.actual_states.append(next_state)

            # 학습 (Hebbian)
            if energy_change > 0:
                # 성공 → 강화
                learning_rate = 0.01
                hidden = np.tanh(np.dot(state.flatten(), self.W1))
                self.W2 += learning_rate * np.outer(hidden, action)
                self.W1 += learning_rate * 0.5 * np.outer(state.flatten(), hidden)

            # 상태 전이
            state = next_state

            # 죽음 또는 에피소드 종료
            if episode_energy <= 0 or done:
                break

        # 에피소드 성능 기록
        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        self.episode_performances.append(avg_reward)
        self.episode_energies.append(episode_energy)
        self.total_episodes += 1

        # 생존력 업데이트
        self._update_viability_metrics(episode_energy, episode_rewards)

        return {
            'episode': self.total_episodes,
            'steps': episode_steps,
            'final_energy': episode_energy,
            'avg_reward': avg_reward,
            'viability': self.viability_metrics.compute_composite_viability(),
            'viability_trajectory': self.viability_metrics.get_trajectory_score()
        }

    def _update_viability_metrics(self, final_energy: float, episode_rewards: List[float]):
        """생존력 5차원 업데이트"""

        # 1. Sustainability
        energy_level = final_energy / 10.0  # 정규화
        if len(self.episode_energies) >= 2:
            energy_trend = (self.episode_energies[-1] - self.episode_energies[-2]) / 10.0
        else:
            energy_trend = 0.0
        self.viability_metrics.update_sustainability(energy_level, energy_trend)

        # 2. Adaptability
        if len(self.episode_performances) >= 10:
            recent_perf = np.mean(self.episode_performances[-5:])
            older_perf = np.mean(self.episode_performances[-10:-5])
            learning_rate = np.clip((recent_perf - older_perf) / (abs(older_perf) + 1e-8), 0, 1)
        else:
            learning_rate = 0.5
        exploration_rate = 0.5  # Placeholder
        self.viability_metrics.update_adaptability(learning_rate, exploration_rate)

        # 3. Anticipation
        if len(self.predicted_states) >= 10 and len(self.actual_states) >= 10:
            # 예측 정확도 계산 (간소화)
            prediction_accuracy = 0.5  # Placeholder
        else:
            prediction_accuracy = 0.3
        planning_horizon = min(len(self.predicted_states), 20)
        self.viability_metrics.update_anticipation(prediction_accuracy, planning_horizon)

        # 4. Resilience
        if len(episode_rewards) > 0:
            negative_rewards = [r for r in episode_rewards if r < 0]
            if len(negative_rewards) > 0:
                recovery_rate = 1.0 - (len(negative_rewards) / len(episode_rewards))
            else:
                recovery_rate = 1.0
        else:
            recovery_rate = 0.5
        failure_tolerance = 0.5  # Placeholder
        self.viability_metrics.update_resilience(recovery_rate, failure_tolerance)

        # 5. Growth
        if len(self.episode_performances) >= 10:
            self.viability_metrics.update_growth(self.episode_performances[-20:])
        else:
            self.viability_metrics.update_growth([])

    def get_summary(self) -> Dict:
        """Entity 요약"""
        viability_summary = self.viability_metrics.get_summary()

        return {
            'id': self.id,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_episode_performance': np.mean(self.episode_performances) if len(self.episode_performances) > 0 else 0.0,
            'avg_final_energy': np.mean(self.episode_energies) if len(self.episode_energies) > 0 else 0.0,
            'viability': viability_summary
        }


# =======================
# Testing
# =======================

if __name__ == "__main__":
    print("=" * 70)
    print("Viability Redefined: Multi-Episode Learning Test")
    print("=" * 70)

    # 환경 생성
    env = MultiEpisodeEnvironment(
        input_dim=5,
        episode_length=100,
        energy_cost=0.1,
        reward_scale=0.5
    )

    # Entity 생성
    entity = ViabilityDrivenEntity(
        state_size=5,
        action_size=1,
        hidden_size=32
    )

    # 다중 에피소드 실행
    print(f"\n{'='*70}")
    print("Multi-Episode Learning")
    print(f"{'='*70}\n")

    n_episodes = 20

    for episode in range(n_episodes):
        result = entity.live_episode(env)

        if episode % 5 == 0 or episode == n_episodes - 1:
            print(f"Episode {result['episode']:2d} | "
                  f"Steps: {result['steps']:3d} | "
                  f"Energy: {result['final_energy']:6.2f} | "
                  f"Viability: {result['viability']:.3f} | "
                  f"Trajectory: {result['viability_trajectory']:+.3f}")

    # 최종 요약
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    summary = entity.get_summary()
    print(f"\n**Overall Performance**:")
    print(f"  Total episodes: {summary['total_episodes']}")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Avg episode performance: {summary['avg_episode_performance']:.3f}")
    print(f"  Avg final energy: {summary['avg_final_energy']:.2f}")

    viability = summary['viability']
    print(f"\n**Viability (5 Dimensions)**:")
    print(f"  Sustainability:  {viability['sustainability']:.3f}")
    print(f"  Adaptability:    {viability['adaptability']:.3f}")
    print(f"  Anticipation:    {viability['anticipation']:.3f}")
    print(f"  Resilience:      {viability['resilience']:.3f}")
    print(f"  Growth:          {viability['growth']:.3f}")
    print(f"  → Composite:     {viability['composite']:.3f}")
    print(f"  → Trajectory:    {viability['trajectory']:+.3f}")

    print(f"\n**Key Innovation**:")
    print(f"  ✓ Multi-episode learning (not single death)")
    print(f"  ✓ 5-dimensional viability assessment")
    print(f"  ✓ Learning trajectory focus (not survival time)")
    print(f"  ✓ Growth rate measurement (improvement over time)")

    print("\n" + "=" * 70)
    print("Viability = Capacity for sustained improvement!")
    print("=" * 70)
