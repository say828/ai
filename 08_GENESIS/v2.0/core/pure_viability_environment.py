"""
Pure Viability Environment for GENESIS
Author: GENESIS Project
Date: 2026-01-03

핵심 원칙:
    - NO ground truth targets (정답 없음)
    - NO supervised signals (지도 신호 없음)
    - Only survival/death (오직 생존/죽음만)

Environment는 다음만 제공:
    1. State observations (상태 관찰)
    2. Survival signals (생존 신호 = energy/resources)
    3. Death when energy depletes (에너지 고갈시 죽음)

Entity는 반드시:
    - 스스로 행동 생성
    - 결과로부터 학습 (생존 = 성공, 죽음 = 실패)
    - 명시적 에러 신호 없이 적응
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ResourceEnvironment:
    """
    자원 기반 생존 환경

    Mechanics:
        - 환경에 자원(resources)이 분포
        - Entity는 행동(action)으로 자원 획득
        - 올바른 행동 → 에너지 증가 (생존)
        - 잘못된 행동 → 에너지 감소 (죽음으로 이동)
        - Entity는 "정답"을 모름, 오직 생존 결과만 관찰

    Task: Function Approximation (but entity doesn't know!)
        - 숨겨진 함수: y = f(x)
        - Entity의 행동이 f(x)에 가까우면 에너지 획득
        - 하지만 environment는 절대 f(x) 값을 알려주지 않음!
        - 오직 "에너지 변화"만 관찰 가능
    """

    def __init__(self,
                 input_dim: int = 10,
                 function_type: str = 'linear',
                 energy_reward_scale: float = 1.0,
                 energy_cost_per_step: float = 0.1,
                 noise_level: float = 0.1,
                 seed: int = 42):
        """
        Args:
            input_dim: 입력 차원
            function_type: 숨겨진 함수 타입 ('linear', 'quadratic', 'nonlinear')
            energy_reward_scale: 에너지 보상 스케일
            energy_cost_per_step: 매 스텝 에너지 소모
            noise_level: 환경 노이즈
            seed: 랜덤 시드
        """
        np.random.seed(seed)

        self.input_dim = input_dim
        self.function_type = function_type
        self.energy_reward_scale = energy_reward_scale
        self.energy_cost_per_step = energy_cost_per_step
        self.noise_level = noise_level

        # 숨겨진 함수 (entity는 이걸 모름!)
        self._initialize_hidden_function()

        # 현재 상태
        self.current_state = None
        self.step_count = 0

        print(f"ResourceEnvironment initialized:")
        print(f"  Function: {function_type}")
        print(f"  Input dim: {input_dim}")
        print(f"  Energy reward scale: {energy_reward_scale}")
        print(f"  Energy cost per step: {energy_cost_per_step}")

    def _initialize_hidden_function(self):
        """숨겨진 함수 초기화 (entity는 접근 불가!)"""
        if self.function_type == 'linear':
            # y = W·x
            self.W_hidden = np.random.randn(self.input_dim) * 2.0
        elif self.function_type == 'quadratic':
            # y = W·x²
            self.W_hidden = np.random.randn(self.input_dim) * 1.0
        elif self.function_type == 'nonlinear':
            # y = W·tanh(x)
            self.W_hidden = np.random.randn(self.input_dim) * 1.5
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")

    def _compute_hidden_output(self, state: np.ndarray) -> float:
        """
        숨겨진 함수 계산 (entity는 이 값을 절대 받지 못함!)

        이것이 전통적 supervised learning과의 핵심 차이:
        - Supervised: target = f(x), loss = (prediction - target)²
        - Viability: 오직 survival signal만 (에너지 변화)
        """
        if self.function_type == 'linear':
            output = np.dot(state, self.W_hidden)
        elif self.function_type == 'quadratic':
            output = np.dot(state**2, self.W_hidden)
        elif self.function_type == 'nonlinear':
            output = np.dot(np.tanh(state), self.W_hidden)

        # 노이즈 추가 (환경의 불확실성)
        output += np.random.randn() * self.noise_level

        return output

    def reset(self) -> np.ndarray:
        """
        환경 리셋

        Returns:
            state: 초기 상태 관찰
        """
        self.current_state = np.random.randn(self.input_dim)
        self.step_count = 0
        return self.current_state.copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경과 상호작용

        핵심: Ground truth를 주지 않음!

        Args:
            action: Entity의 행동 (스칼라 출력)

        Returns:
            next_state: 다음 상태 관찰
            energy_change: 에너지 변화 (survival signal)
            done: 에피소드 종료 여부
            info: 추가 정보 (디버깅용, entity는 접근 불가)
        """
        self.step_count += 1

        # 1. 숨겨진 "최적 행동" 계산 (entity는 모름!)
        optimal_action = self._compute_hidden_output(self.current_state)

        # 2. Entity의 행동이 얼마나 "좋았는지" 평가
        #    하지만 이걸 직접적으로 알려주지 않음!
        action_quality = -np.abs(action - optimal_action)

        # 3. 에너지 변화 계산
        #    좋은 행동 → 에너지 증가
        #    나쁜 행동 → 에너지 감소 (기본 소모만)
        if action_quality > -1.0:  # 괜찮은 행동
            energy_reward = self.energy_reward_scale * np.exp(action_quality)
        else:  # 나쁜 행동
            energy_reward = 0.0

        energy_change = energy_reward - self.energy_cost_per_step

        # 4. 다음 상태 생성
        next_state = np.random.randn(self.input_dim)
        self.current_state = next_state

        # 5. 종료 조건 (없음, entity가 죽을 때까지 계속)
        done = False

        # 6. 디버깅 정보 (entity는 접근 불가!)
        info = {
            'optimal_action': optimal_action,
            'action_quality': action_quality,
            'energy_reward': energy_reward,
            'step': self.step_count
        }

        return next_state, energy_change, done, info

    def get_state(self) -> np.ndarray:
        """현재 상태 관찰"""
        return self.current_state.copy()


class ForagingEnvironment:
    """
    먹이 찾기 환경 (더 생물학적)

    Mechanics:
        - 2D 공간에 먹이가 분포
        - Entity는 위치 이동
        - 먹이 근처 → 에너지 획득
        - 먹이 먼 곳 → 에너지 소모만
        - Entity는 "먹이 위치"를 모름, 탐색해야 함
    """

    def __init__(self,
                 space_size: int = 10,
                 n_food_sources: int = 3,
                 energy_cost_per_step: float = 0.1,
                 food_energy: float = 1.0,
                 seed: int = 42):
        """
        Args:
            space_size: 2D 공간 크기
            n_food_sources: 먹이 개수
            energy_cost_per_step: 이동 에너지 소모
            food_energy: 먹이 에너지 값
            seed: 랜덤 시드
        """
        np.random.seed(seed)

        self.space_size = space_size
        self.n_food_sources = n_food_sources
        self.energy_cost_per_step = energy_cost_per_step
        self.food_energy = food_energy

        # 먹이 위치 (entity는 모름!)
        self.food_positions = np.random.rand(n_food_sources, 2) * space_size

        # Entity 위치
        self.entity_position = None
        self.step_count = 0

        print(f"ForagingEnvironment initialized:")
        print(f"  Space: {space_size}x{space_size}")
        print(f"  Food sources: {n_food_sources}")
        print(f"  Energy cost: {energy_cost_per_step}")

    def reset(self) -> np.ndarray:
        """환경 리셋"""
        # Entity 랜덤 위치
        self.entity_position = np.random.rand(2) * self.space_size
        self.step_count = 0

        # 상태: entity 위치 (먹이 위치는 모름!)
        state = self._get_observation()
        return state

    def _get_observation(self) -> np.ndarray:
        """
        상태 관찰

        Entity는 다음만 볼 수 있음:
        - 자신의 위치 (normalized)
        - 가장 가까운 먹이까지의 거리 (하지만 방향은 모름!)
        """
        # 자신의 위치
        normalized_position = self.entity_position / self.space_size

        # 가장 가까운 먹이까지 거리 (방향은 모름!)
        distances = np.linalg.norm(self.food_positions - self.entity_position, axis=1)
        min_distance = np.min(distances) / self.space_size  # normalized

        # 관찰: [pos_x, pos_y, nearest_food_distance]
        observation = np.array([
            normalized_position[0],
            normalized_position[1],
            min_distance
        ])

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경과 상호작용

        Args:
            action: [delta_x, delta_y] 이동 방향

        Returns:
            observation: 다음 상태 관찰
            energy_change: 에너지 변화
            done: 종료 여부
            info: 디버깅 정보
        """
        self.step_count += 1

        # 1. 이동 (action은 -1~1로 정규화되어 있다고 가정)
        movement = np.clip(action, -1.0, 1.0) * 0.5  # 최대 0.5 단위 이동
        new_position = self.entity_position + movement

        # 경계 체크
        new_position = np.clip(new_position, 0, self.space_size)
        self.entity_position = new_position

        # 2. 먹이 근처 확인
        distances = np.linalg.norm(self.food_positions - self.entity_position, axis=1)
        min_distance = np.min(distances)

        # 3. 에너지 변화
        if min_distance < 1.0:  # 먹이 근처 (1.0 이내)
            # 거리에 반비례하게 에너지 획득
            energy_reward = self.food_energy * (1.0 - min_distance)
        else:
            energy_reward = 0.0

        energy_change = energy_reward - self.energy_cost_per_step

        # 4. 다음 관찰
        observation = self._get_observation()

        # 5. 종료 조건 (없음)
        done = False

        # 6. 디버깅 정보
        info = {
            'position': self.entity_position.copy(),
            'nearest_food_distance': min_distance,
            'energy_reward': energy_reward,
            'step': self.step_count
        }

        return observation, energy_change, done, info


# =======================
# Testing
# =======================

if __name__ == "__main__":
    print("=" * 70)
    print("Pure Viability Environment Test")
    print("=" * 70)

    # Test 1: Resource Environment
    print("\n[Test 1] ResourceEnvironment")
    env = ResourceEnvironment(input_dim=5, function_type='linear')

    state = env.reset()
    print(f"Initial state: {state[:3]}...")

    for i in range(5):
        # Random action (entity doesn't know optimal)
        action = np.random.randn() * 5.0
        next_state, energy_change, done, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  Action: {action:.3f}")
        print(f"  Energy change: {energy_change:+.3f}")
        print(f"  (Hidden optimal: {info['optimal_action']:.3f})")
        print(f"  (Hidden quality: {info['action_quality']:.3f})")

    # Test 2: Foraging Environment
    print("\n[Test 2] ForagingEnvironment")
    env2 = ForagingEnvironment(space_size=10, n_food_sources=2)

    state = env2.reset()
    print(f"Initial observation: {state}")

    for i in range(5):
        # Random movement
        action = np.random.randn(2)
        next_state, energy_change, done, info = env2.step(action)

        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Position: {info['position']}")
        print(f"  Energy change: {energy_change:+.3f}")
        print(f"  Nearest food: {info['nearest_food_distance']:.2f}")

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
    print("\nKey principle: Environment NEVER provides ground truth!")
    print("Only survival signals (energy) available.")
