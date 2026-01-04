"""
Pure Viability Entity for GENESIS
Author: GENESIS Project
Date: 2026-01-03

핵심 원칙:
    - NO gradient descent (그래디언트 하강 없음)
    - NO explicit loss function (명시적 손실 함수 없음)
    - Only viability-driven learning (오직 생존력 기반 학습)

학습 메커니즘:
    1. Hebbian Learning: 성공한 경로 강화
    2. Homeostatic Regulation: 내부 균형 유지
    3. Structural Adaptation: 실패시 구조 변화
    4. Energy Management: 에너지로 생존 평가

비교:
    Standard ML: loss = (pred - target)², θ -= α·∇loss
    Pure Viability: if survived: strengthen_pathways()
                   else: adapt_structure()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import uuid
from collections import deque


class SimpleNeuralModule:
    """
    간단한 신경 모듈 (NO backprop!)

    학습 방법:
        - Hebbian: 활성화 상관관계로 가중치 강화
        - Pathway strength: 성공시 경로 강도 증가
        - Random exploration: 실패시 랜덤 변화
    """

    def __init__(self, input_size: int, output_size: int, name: str = "module"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        # 파라미터 초기화 (작은 값으로)
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros(output_size)

        # Hebbian pathway strength
        self.pathway_strength = np.ones_like(self.W)

        # 활동 기록
        self.last_input = None
        self.last_output = None
        self.last_activation = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """순전파 (NO gradient tracking)"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # 선형 변환
        z = np.dot(x, self.W) + self.b

        # 활성화 (tanh)
        output = np.tanh(z)

        # 기록 (Hebbian 학습용)
        self.last_input = x.copy()
        self.last_output = output.copy()
        self.last_activation = z.copy()

        return output

    def hebbian_update(self, success: bool, learning_rate: float = 0.01):
        """
        Hebbian 학습: "Neurons that fire together, wire together"

        Args:
            success: 최근 행동이 성공했는지
            learning_rate: 학습률
        """
        if self.last_input is None or self.last_output is None:
            return

        # Hebbian rule: ΔW = η · x · y
        hebbian_update = learning_rate * np.dot(
            self.last_input.T,
            self.last_output
        )

        if success:
            # 성공: 현재 경로 강화
            self.W += hebbian_update * self.pathway_strength
            self.pathway_strength *= 1.01  # 경로 강도 증가
        else:
            # 실패: 현재 경로 약화
            self.W -= hebbian_update * 0.5 * self.pathway_strength
            self.pathway_strength *= 0.99  # 경로 강도 감소

        # Pathway strength 범위 제한
        self.pathway_strength = np.clip(self.pathway_strength, 0.1, 10.0)

    def random_exploration(self, exploration_rate: float = 0.1):
        """
        랜덤 탐색: 실패시 구조 변화

        Args:
            exploration_rate: 탐색 강도
        """
        # 가중치에 랜덤 노이즈 추가
        noise = np.random.randn(*self.W.shape) * exploration_rate
        self.W += noise

        # Pathway strength 리셋
        self.pathway_strength = np.ones_like(self.W)


class PureViabilityEntity:
    """
    순수 생존력 기반 Entity

    핵심 차이점:
        - Ground truth 사용 안 함
        - Gradient 계산 안 함
        - 오직 생존/죽음으로만 학습
    """

    def __init__(self,
                 input_size: int = 10,
                 hidden_size: int = 32,
                 output_size: int = 1,
                 initial_energy: float = 10.0,
                 hebbian_lr: float = 0.01,
                 exploration_threshold: float = 0.3,
                 entity_id: Optional[str] = None):
        """
        Args:
            input_size: 입력 차원
            hidden_size: 은닉층 크기
            output_size: 출력 차원
            initial_energy: 초기 에너지
            hebbian_lr: Hebbian 학습률
            exploration_threshold: 탐색 시작 임계값
            entity_id: Entity ID
        """
        self.id = entity_id or str(uuid.uuid4())[:8]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 신경망 구조
        self.encoder = SimpleNeuralModule(input_size, hidden_size, "encoder")
        self.decoder = SimpleNeuralModule(hidden_size, output_size, "decoder")

        # 생존 상태
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True

        # 학습 파라미터
        self.hebbian_lr = hebbian_lr
        self.exploration_threshold = exploration_threshold

        # 생존 기록
        self.energy_history: deque = deque(maxlen=100)
        self.viability_history: deque = deque(maxlen=100)
        self.action_history: deque = deque(maxlen=10)

        # 통계
        self.age = 0
        self.total_energy_gained = 0.0
        self.total_energy_lost = 0.0
        self.survival_episodes = 0

        print(f"PureViabilityEntity created: id={self.id}")
        print(f"  Architecture: {input_size} → {hidden_size} → {output_size}")
        print(f"  Initial energy: {initial_energy}")
        print(f"  Learning: Pure Hebbian (NO gradients!)")

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        행동 생성

        핵심: Ground truth 없음, 스스로 행동 생성

        Args:
            state: 환경 관찰

        Returns:
            action: 생성된 행동
        """
        # Encoder
        hidden = self.encoder.forward(state)

        # Decoder
        action = self.decoder.forward(hidden)

        return action

    def live_one_step(self, environment) -> Dict:
        """
        한 스텝 생존

        핵심 흐름:
            1. 환경 관찰
            2. 행동 생성 (스스로!)
            3. 환경과 상호작용
            4. 에너지 변화 관찰
            5. 생존력 평가
            6. Hebbian 학습 (성공시 강화)
            7. 실패시 구조 탐색

        Args:
            environment: Pure viability environment

        Returns:
            result: {
                'action': 행동,
                'energy_change': 에너지 변화,
                'viability': 생존력,
                'is_alive': 생존 여부,
                'learned': 학습 여부
            }
        """
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1

        # ============================================
        # 1. 환경 관찰
        # ============================================
        state = environment.get_state()

        # ============================================
        # 2. 행동 생성 (스스로!)
        # ============================================
        action = self.forward(state)
        action_scalar = float(action.flatten()[0])

        # ============================================
        # 3. 환경과 상호작용
        # ============================================
        next_state, energy_change, done, info = environment.step(action_scalar)

        # ============================================
        # 4. 에너지 업데이트
        # ============================================
        self.energy += energy_change

        if energy_change > 0:
            self.total_energy_gained += energy_change
        else:
            self.total_energy_lost += abs(energy_change)

        self.energy_history.append(self.energy)

        # ============================================
        # 5. 생존력 평가 (NO loss function!)
        # ============================================
        viability = self._assess_viability()
        self.viability_history.append(viability)

        # 성공 여부 (에너지 증가 = 성공)
        was_successful = (energy_change > 0)

        # ============================================
        # 6. Hebbian 학습 (성공시 강화!)
        # ============================================
        self.encoder.hebbian_update(was_successful, self.hebbian_lr)
        self.decoder.hebbian_update(was_successful, self.hebbian_lr)

        learned = was_successful

        # ============================================
        # 7. 구조적 적응 (실패시 탐색!)
        # ============================================
        if viability < self.exploration_threshold:
            # 생존력 낮음 → 구조 탐색
            exploration_rate = 0.1 * (self.exploration_threshold - viability)
            self.encoder.random_exploration(exploration_rate)
            self.decoder.random_exploration(exploration_rate)
            learned = True  # 탐색도 학습의 일종

        # ============================================
        # 8. 죽음 체크
        # ============================================
        if self.energy <= 0:
            self.is_alive = False
            print(f"Entity {self.id} died at age {self.age} (energy depleted)")

        # ============================================
        # 9. 생존 에피소드 카운트
        # ============================================
        if self.energy > self.initial_energy:
            self.survival_episodes += 1

        # 결과 반환
        result = {
            'action': action_scalar,
            'energy_change': energy_change,
            'energy': self.energy,
            'viability': viability,
            'is_alive': self.is_alive,
            'learned': learned,
            'age': self.age,
            'debug_info': info  # 디버깅용 (entity는 사용 안 함)
        }

        return result

    def _assess_viability(self) -> float:
        """
        생존력 평가 (내부 상태 기반)

        Components:
            1. 현재 에너지 수준 (40%)
            2. 최근 에너지 트렌드 (30%)
            3. 에너지 안정성 (20%)
            4. 생존 지속성 (10%)

        Returns:
            viability: 0~1 사이 값
        """
        # 1. 현재 에너지 수준
        energy_level = np.clip(self.energy / self.initial_energy, 0, 1)

        # 2. 최근 에너지 트렌드
        if len(self.energy_history) >= 10:
            recent = list(self.energy_history)[-10:]
            older = list(self.energy_history)[-20:-10] if len(self.energy_history) >= 20 else recent
            trend = (np.mean(recent) - np.mean(older)) / (np.mean(older) + 1e-8)
            trend_score = np.clip(trend + 0.5, 0, 1)  # normalize to 0-1
        else:
            trend_score = 0.5

        # 3. 에너지 안정성 (변동성 낮을수록 좋음)
        if len(self.energy_history) >= 10:
            recent_energies = list(self.energy_history)[-10:]
            stability = 1.0 / (1.0 + np.std(recent_energies))
        else:
            stability = 0.5

        # 4. 생존 지속성
        longevity = min(1.0, self.age / 100.0)

        # 가중 평균
        viability = (
            0.4 * energy_level +
            0.3 * trend_score +
            0.2 * stability +
            0.1 * longevity
        )

        return float(np.clip(viability, 0, 1))

    def get_summary(self) -> Dict:
        """Entity 상태 요약"""
        return {
            'id': self.id,
            'age': self.age,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'viability': self._assess_viability() if self.is_alive else 0.0,
            'total_energy_gained': self.total_energy_gained,
            'total_energy_lost': self.total_energy_lost,
            'net_energy': self.total_energy_gained - self.total_energy_lost,
            'survival_episodes': self.survival_episodes,
            'avg_energy': np.mean(list(self.energy_history)) if len(self.energy_history) > 0 else 0.0
        }


# =======================
# Testing
# =======================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from pure_viability_environment import ResourceEnvironment

    print("=" * 70)
    print("Pure Viability Entity Test")
    print("=" * 70)

    # 환경 생성
    env = ResourceEnvironment(input_dim=5, function_type='linear', energy_cost_per_step=0.05)

    # Entity 생성
    entity = PureViabilityEntity(
        input_size=5,
        hidden_size=16,
        output_size=1,
        initial_energy=5.0,
        hebbian_lr=0.01
    )

    # 생존 시뮬레이션
    print(f"\n{'='*70}")
    print("Survival Simulation (NO gradients, NO ground truth!)")
    print(f"{'='*70}\n")

    env.reset()
    results = []

    for step in range(100):
        result = entity.live_one_step(env)

        if not result['is_alive']:
            print(f"\n💀 Entity died at step {step}")
            break

        results.append(result)

        if step % 10 == 0:
            print(f"Step {step:3d} | Energy: {result['energy']:6.2f} | "
                  f"Viability: {result['viability']:.3f} | "
                  f"Action: {result['action']:+6.3f}")

    # 최종 요약
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    summary = entity.get_summary()
    print(f"\n**Survival Metrics**:")
    print(f"  Lifespan: {summary['age']} steps")
    print(f"  Final energy: {summary['energy']:.2f}")
    print(f"  Net energy: {summary['net_energy']:+.2f}")
    print(f"  Survival episodes: {summary['survival_episodes']}")
    print(f"  Average energy: {summary['avg_energy']:.2f}")

    print(f"\n**Learning Method**:")
    print(f"  ✓ Pure Hebbian (correlation-based)")
    print(f"  ✓ Structural exploration (failure-driven)")
    print(f"  ✗ NO gradient descent")
    print(f"  ✗ NO loss function")
    print(f"  ✗ NO ground truth")

    print("\n" + "=" * 70)
    print("This is TRUE viability-driven learning!")
    print("=" * 70)
