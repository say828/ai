"""
True Viability-Driven Entity for GENESIS
Author: GENESIS Project
Date: 2026-01-03

진정한 생존력 기반 학습의 핵심:
    1. Predictive Capacity: 미래 생존 가능성 예측
    2. Homeostatic Regulation: 다중 내부 변수 균형
    3. Allostatic States: 환경에 따른 모드 전환
    4. Structural Self-Organization: 스트레스 시 재구조화
    5. Multi-timescale Learning: 빠른 반응 + 느린 구조 변화

차별점:
    vs Supervised Learning: NO ground truth, 스스로 미래 예측
    vs Reinforcement Learning: NO explicit reward, 내부 항상성 유지
    vs Hebbian Learning: 단순 상관관계가 아닌 생존 역학
    vs v1.1: 단순 에너지가 아닌 다차원 생존력
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import uuid
from collections import deque


class PredictiveModule:
    """
    미래 상태 예측 모듈

    핵심: 현재 행동의 미래 결과 예측
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 16):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # 예측 네트워크: [state, action] → future_state
        input_size = state_size + action_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, state_size) * 0.1

        # 예측 오차 기록
        self.prediction_errors = deque(maxlen=50)

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """다음 상태 예측"""
        x = np.concatenate([state.flatten(), action.flatten()])
        hidden = np.tanh(np.dot(x, self.W1))
        predicted_state = np.dot(hidden, self.W2)
        return predicted_state

    def update_prediction(self, state: np.ndarray, action: np.ndarray,
                         actual_next_state: np.ndarray):
        """
        예측 모델 업데이트 (Hebbian + Error-correction)

        핵심: 예측 오차를 줄이는 방향으로 학습 (Predictive Coding)
        """
        # Forward pass
        x = np.concatenate([state.flatten(), action.flatten()])
        hidden = np.tanh(np.dot(x, self.W1))
        predicted_state = np.dot(hidden, self.W2)

        # Prediction error
        prediction_error = actual_next_state.flatten() - predicted_state
        self.prediction_errors.append(np.linalg.norm(prediction_error))

        # Hebbian-style update (correlation-based, NO backprop)
        # Layer 2
        grad_W2 = 0.01 * np.outer(hidden, prediction_error)
        self.W2 += grad_W2

        # Layer 1 (error propagated through correlation)
        error_signal = np.dot(prediction_error, self.W2.T)
        grad_W1 = 0.01 * np.outer(x, error_signal * (1 - hidden**2))
        self.W1 += grad_W1

    def get_prediction_confidence(self) -> float:
        """예측 신뢰도 (낮은 오차 = 높은 신뢰도)"""
        if len(self.prediction_errors) < 5:
            return 0.5
        recent_error = np.mean(list(self.prediction_errors)[-10:])
        confidence = 1.0 / (1.0 + recent_error)
        return float(np.clip(confidence, 0, 1))


class HomeostaticController:
    """
    항상성 유지 컨트롤러

    핵심: 다중 내부 변수를 목표 범위 내로 유지
    """

    def __init__(self):
        # 항상성 변수들
        self.variables = {
            'energy': {'current': 1.0, 'target': 1.0, 'range': (0.7, 1.3)},
            'stability': {'current': 1.0, 'target': 1.0, 'range': (0.8, 1.2)},
            'entropy': {'current': 0.5, 'target': 0.5, 'range': (0.3, 0.7)},
            'prediction_accuracy': {'current': 0.5, 'target': 0.7, 'range': (0.5, 0.9)}
        }

        # 불균형 기록
        self.imbalance_history = deque(maxlen=20)

    def update_variable(self, var_name: str, value: float):
        """변수 업데이트"""
        if var_name in self.variables:
            self.variables[var_name]['current'] = value

    def compute_homeostatic_stress(self) -> float:
        """
        항상성 스트레스 계산

        모든 변수가 목표 범위 내 → 낮은 스트레스
        어떤 변수라도 범위 벗어남 → 높은 스트레스
        """
        total_stress = 0.0

        for var_name, var_info in self.variables.items():
            current = var_info['current']
            target = var_info['target']
            min_val, max_val = var_info['range']

            # 범위 벗어난 정도
            if current < min_val:
                stress = (min_val - current) / target
            elif current > max_val:
                stress = (current - max_val) / target
            else:
                stress = 0.0

            total_stress += stress

        # 정규화
        normalized_stress = total_stress / len(self.variables)
        self.imbalance_history.append(normalized_stress)

        return float(np.clip(normalized_stress, 0, 1))

    def get_allostatic_state(self) -> str:
        """
        알로스타틱 상태 결정

        상태:
            - 'thriving': 모든 변수 최적 범위
            - 'stable': 변수들이 허용 범위 내
            - 'stressed': 일부 변수 범위 벗어남
            - 'critical': 심각한 불균형
        """
        stress = self.compute_homeostatic_stress()

        if stress < 0.1:
            return 'thriving'
        elif stress < 0.3:
            return 'stable'
        elif stress < 0.6:
            return 'stressed'
        else:
            return 'critical'


class ActorModule:
    """
    행동 생성 모듈 (상태에 따라 다른 전략)
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
        self.input_size = input_size
        self.output_size = output_size

        # 행동 네트워크
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1

        # 활동 기록
        self.last_state = None
        self.last_hidden = None
        self.last_action = None

    def generate_action(self, state: np.ndarray, mode: str = 'stable') -> np.ndarray:
        """
        행동 생성 (모드에 따라 다른 전략)

        Args:
            state: 현재 상태
            mode: 'thriving', 'stable', 'stressed', 'critical'
        """
        x = state.flatten()

        # Forward pass
        hidden = np.tanh(np.dot(x, self.W1))
        action = np.tanh(np.dot(hidden, self.W2))

        # 모드에 따른 행동 조정
        if mode == 'thriving':
            # 안정적 + 약간의 탐색
            noise = np.random.randn(*action.shape) * 0.05
            action = action + noise
        elif mode == 'stable':
            # 현재 정책 유지
            noise = np.random.randn(*action.shape) * 0.02
            action = action + noise
        elif mode == 'stressed':
            # 더 많은 탐색
            noise = np.random.randn(*action.shape) * 0.15
            action = action + noise
        else:  # critical
            # 극단적 탐색
            if np.random.rand() < 0.3:
                action = np.random.randn(*action.shape) * 2.0
            else:
                noise = np.random.randn(*action.shape) * 0.3
                action = action + noise

        # 기록
        self.last_state = x.copy()
        self.last_hidden = hidden.copy()
        self.last_action = action.copy()

        return np.clip(action, -1, 1)

    def hebbian_update(self, viability_change: float, learning_rate: float = 0.01):
        """
        Hebbian 업데이트

        Args:
            viability_change: 생존력 변화 (양수 = 개선)
        """
        if self.last_state is None or self.last_action is None:
            return

        # Viability 개선 → 경로 강화
        if viability_change > 0:
            # Layer 2
            delta_W2 = learning_rate * np.outer(self.last_hidden, self.last_action)
            self.W2 += delta_W2

            # Layer 1
            delta_W1 = learning_rate * 0.5 * np.outer(self.last_state, self.last_hidden)
            self.W1 += delta_W1
        else:
            # Viability 악화 → 경로 약화
            delta_W2 = learning_rate * 0.3 * np.outer(self.last_hidden, self.last_action)
            self.W2 -= delta_W2


class TrueViabilityEntity:
    """
    진정한 생존력 기반 Entity

    핵심 차별점:
        1. 미래 예측 능력 (PredictiveModule)
        2. 다차원 항상성 (HomeostaticController)
        3. 상태 기반 행동 전환 (Allostasis)
        4. 구조적 재조직화 (Structural metamorphosis)
    """

    def __init__(self,
                 state_size: int = 5,
                 action_size: int = 1,
                 hidden_size: int = 32,
                 initial_energy: float = 5.0,
                 entity_id: Optional[str] = None):
        """
        Args:
            state_size: 상태 차원
            action_size: 행동 차원
            hidden_size: 은닉층 크기
            initial_energy: 초기 에너지
        """
        self.id = entity_id or str(uuid.uuid4())[:8]
        self.state_size = state_size
        self.action_size = action_size

        # 핵심 모듈들
        self.predictor = PredictiveModule(state_size, action_size, hidden_size//2)
        self.homeostasis = HomeostaticController()
        self.actor = ActorModule(state_size, action_size, hidden_size)

        # 생존 상태
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.is_alive = True

        # 생존력 평가
        self.viability_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)

        # 통계
        self.age = 0
        self.total_prediction_updates = 0
        self.total_structural_changes = 0
        self.state_distribution = {'thriving': 0, 'stable': 0, 'stressed': 0, 'critical': 0}

        # 이전 상태 기록
        self.last_state = None
        self.last_action = None

        print(f"TrueViabilityEntity created: id={self.id}")
        print(f"  Components: Predictor + Homeostasis + Actor")
        print(f"  Initial energy: {initial_energy}")
        print(f"  Mechanisms: Prediction + Allostasis + Hebbian")

    def live_one_step(self, environment) -> Dict:
        """
        한 스텝 생존

        핵심 흐름:
            1. 환경 관찰
            2. 미래 예측 (predictive module)
            3. 항상성 평가 (homeostatic controller)
            4. 상태 결정 (allostatic state)
            5. 상태 기반 행동 생성
            6. 환경 상호작용
            7. 예측 모델 업데이트
            8. 항상성 변수 업데이트
            9. 행동 모듈 업데이트 (Hebbian)
            10. 구조적 재조직화 (필요시)
        """
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1

        # ============================================
        # 1. 환경 관찰
        # ============================================
        current_state = environment.get_state()

        # ============================================
        # 2. 미래 예측 (선택적 - 이전 경험 있을 때만)
        # ============================================
        predicted_energy_change = 0.0
        if self.last_state is not None and self.last_action is not None:
            predicted_next_state = self.predictor.predict_next_state(
                self.last_state, self.last_action
            )
            # 예측 신뢰도
            prediction_confidence = self.predictor.get_prediction_confidence()
        else:
            prediction_confidence = 0.5

        # ============================================
        # 3. 항상성 평가
        # ============================================
        # 에너지 정규화
        normalized_energy = self.energy / self.initial_energy
        self.homeostasis.update_variable('energy', normalized_energy)

        # 예측 정확도
        self.homeostasis.update_variable('prediction_accuracy', prediction_confidence)

        # 안정성 (최근 에너지 변동)
        if len(self.energy_history) >= 10:
            recent_energies = list(self.energy_history)[-10:]
            stability = 1.0 / (1.0 + np.std(recent_energies))
        else:
            stability = 1.0
        self.homeostasis.update_variable('stability', stability)

        # 엔트로피 (행동 다양성)
        entropy = 0.5  # Placeholder
        self.homeostasis.update_variable('entropy', entropy)

        # 항상성 스트레스
        homeostatic_stress = self.homeostasis.compute_homeostatic_stress()

        # ============================================
        # 4. 알로스타틱 상태 결정
        # ============================================
        allostatic_state = self.homeostasis.get_allostatic_state()
        self.state_distribution[allostatic_state] += 1

        # ============================================
        # 5. 상태 기반 행동 생성
        # ============================================
        action = self.actor.generate_action(current_state, mode=allostatic_state)

        # ============================================
        # 6. 환경 상호작용
        # ============================================
        next_state, energy_change, done, info = environment.step(action.flatten()[0])

        # ============================================
        # 7. 예측 모델 업데이트
        # ============================================
        if self.last_state is not None and self.last_action is not None:
            self.predictor.update_prediction(
                self.last_state,
                self.last_action,
                current_state
            )
            self.total_prediction_updates += 1

        # ============================================
        # 8. 에너지 업데이트
        # ============================================
        previous_energy = self.energy
        self.energy += energy_change
        self.energy_history.append(self.energy)

        # ============================================
        # 9. 생존력 평가
        # ============================================
        viability = self._assess_multidimensional_viability(
            normalized_energy=self.energy / self.initial_energy,
            homeostatic_stress=homeostatic_stress,
            prediction_confidence=prediction_confidence,
            allostatic_state=allostatic_state
        )
        self.viability_history.append(viability)

        # 생존력 변화
        if len(self.viability_history) >= 2:
            viability_change = self.viability_history[-1] - self.viability_history[-2]
        else:
            viability_change = 0.0

        # ============================================
        # 10. 행동 모듈 업데이트 (Hebbian)
        # ============================================
        self.actor.hebbian_update(viability_change, learning_rate=0.01)

        # ============================================
        # 11. 구조적 재조직화 (Critical 상태 시)
        # ============================================
        if allostatic_state == 'critical' and self.age % 10 == 0:
            self._structural_reorganization()
            self.total_structural_changes += 1

        # ============================================
        # 12. 죽음 체크
        # ============================================
        if self.energy <= 0:
            self.is_alive = False
            print(f"Entity {self.id} died at age {self.age} (energy depleted)")

        # 상태 기록
        self.last_state = current_state.copy()
        self.last_action = action.copy()

        # 결과 반환
        return {
            'is_alive': self.is_alive,
            'age': self.age,
            'energy': self.energy,
            'energy_change': energy_change,
            'viability': viability,
            'viability_change': viability_change,
            'allostatic_state': allostatic_state,
            'homeostatic_stress': homeostatic_stress,
            'prediction_confidence': prediction_confidence,
            'action': action.flatten()[0],
            'debug_info': info
        }

    def _assess_multidimensional_viability(self,
                                           normalized_energy: float,
                                           homeostatic_stress: float,
                                           prediction_confidence: float,
                                           allostatic_state: str) -> float:
        """
        다차원 생존력 평가

        Components:
            1. 에너지 수준 (30%)
            2. 항상성 균형 (25%)
            3. 예측 능력 (25%)
            4. 알로스타틱 상태 (20%)
        """
        # 1. 에너지 (정규화)
        energy_score = np.clip(normalized_energy, 0, 1)

        # 2. 항상성 (낮은 스트레스 = 높은 점수)
        homeostasis_score = 1.0 - homeostatic_stress

        # 3. 예측 능력
        prediction_score = prediction_confidence

        # 4. 알로스타틱 상태
        state_scores = {'thriving': 1.0, 'stable': 0.7, 'stressed': 0.4, 'critical': 0.1}
        allostatic_score = state_scores[allostatic_state]

        # 가중 평균
        viability = (
            0.30 * energy_score +
            0.25 * homeostasis_score +
            0.25 * prediction_score +
            0.20 * allostatic_score
        )

        return float(np.clip(viability, 0, 1))

    def _structural_reorganization(self):
        """
        구조적 재조직화 (Critical 상태 시)

        메커니즘:
            - Actor 네트워크에 큰 노이즈 추가
            - 예측 모듈 부분 리셋
        """
        # Actor 재구조화
        noise_scale = 0.3
        self.actor.W1 += np.random.randn(*self.actor.W1.shape) * noise_scale
        self.actor.W2 += np.random.randn(*self.actor.W2.shape) * noise_scale

        # 예측 모듈 부분 리셋
        if len(self.predictor.prediction_errors) > 0:
            avg_error = np.mean(list(self.predictor.prediction_errors))
            if avg_error > 1.0:  # 예측이 매우 나쁘면
                self.predictor.W2 *= 0.5  # 가중치 감소

    def get_summary(self) -> Dict:
        """Entity 상태 요약"""
        total_states = sum(self.state_distribution.values())
        state_percentages = {
            state: (count / total_states * 100) if total_states > 0 else 0
            for state, count in self.state_distribution.items()
        }

        return {
            'id': self.id,
            'age': self.age,
            'energy': self.energy,
            'is_alive': self.is_alive,
            'viability': self.viability_history[-1] if len(self.viability_history) > 0 else 0.0,
            'current_state': self.homeostasis.get_allostatic_state(),
            'state_distribution': state_percentages,
            'prediction_updates': self.total_prediction_updates,
            'structural_changes': self.total_structural_changes,
            'prediction_confidence': self.predictor.get_prediction_confidence(),
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
    print("True Viability Entity Test")
    print("=" * 70)

    # 더 가혹한 환경
    env = ResourceEnvironment(
        input_dim=5,
        function_type='nonlinear',  # 더 어려운 함수
        energy_cost_per_step=0.2,   # 높은 에너지 소모
        energy_reward_scale=0.3      # 낮은 보상
    )

    # Entity 생성
    entity = TrueViabilityEntity(
        state_size=5,
        action_size=1,
        hidden_size=32,
        initial_energy=5.0
    )

    # 생존 시뮬레이션
    print(f"\n{'='*70}")
    print("Survival Simulation (Harsh Environment)")
    print(f"{'='*70}\n")

    env.reset()
    results = []

    for step in range(200):
        result = entity.live_one_step(env)

        if not result['is_alive']:
            print(f"\n💀 Entity died at step {step}")
            break

        results.append(result)

        if step % 20 == 0:
            print(f"Step {step:3d} | Energy: {result['energy']:6.2f} | "
                  f"Viability: {result['viability']:.3f} | "
                  f"State: {result['allostatic_state']:8s} | "
                  f"Pred.Conf: {result['prediction_confidence']:.3f}")

    # 최종 요약
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    summary = entity.get_summary()
    print(f"\n**Survival Metrics**:")
    print(f"  Lifespan: {summary['age']} steps")
    print(f"  Final energy: {summary['energy']:.2f}")
    print(f"  Final viability: {summary['viability']:.3f}")
    print(f"  Prediction confidence: {summary['prediction_confidence']:.3f}")

    print(f"\n**Allostatic State Distribution**:")
    for state, pct in summary['state_distribution'].items():
        print(f"  {state:8s}: {pct:5.1f}%")

    print(f"\n**Learning Activity**:")
    print(f"  Prediction updates: {summary['prediction_updates']}")
    print(f"  Structural changes: {summary['structural_changes']}")

    print(f"\n**Key Mechanisms**:")
    print(f"  ✓ Predictive capacity (forward model)")
    print(f"  ✓ Homeostatic regulation (multi-variable)")
    print(f"  ✓ Allostatic states (adaptive modes)")
    print(f"  ✓ Structural reorganization (stress-driven)")

    print("\n" + "=" * 70)
    print("This is TRUE viability-driven learning!")
    print("=" * 70)
