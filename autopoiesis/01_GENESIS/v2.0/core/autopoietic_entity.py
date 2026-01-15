"""
GENESIS: Autopoietic Entity
Author: GENESIS Project
Date: 2026-01-04

Fundamental Paradigm Shift:
    FROM: Optimization of external objectives
    TO:   Maintenance of organizational identity

핵심 원칙:
    1. Organizational Closure: 시스템이 자기 자신을 생산
    2. Structural Coupling: 환경과 상호 교란
    3. Autonomy: 자체 규범 생성
    4. Circular Causality: 구조 → 기능 → 구조
    5. No External Goals: 오직 조직 유지

Maturana & Varela (1980): "A living system is organized as a network of
processes of production of components that produces the network of processes
that produced them."

Francisco Varela (1979): "Autonomy is the self-assertion of a system that
brings forth its own norms and is operationally closed."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import uuid


class InternalDynamics:
    """
    내부 역학 시스템

    핵심: 순환적 (recurrent), 자기생성적 (self-producing)
    NOT: 순방향 (feedforward), 입출력 매핑
    """

    def __init__(self, n_units: int = 20, connectivity: float = 0.3):
        """
        Args:
            n_units: 내부 단위 수
            connectivity: 연결 밀도 (0~1)
        """
        self.n_units = n_units

        # 순환 연결 구조 (recurrent)
        mask = np.random.rand(n_units, n_units) < connectivity
        self.W = np.random.randn(n_units, n_units) * 0.2 * mask

        # 자기 연결 제거 (no self-loops initially)
        np.fill_diagonal(self.W, 0)

        # 내부 상태
        self.state = np.random.randn(n_units) * 0.1

        # 역학 기록
        self.state_history = deque(maxlen=50)

        print(f"InternalDynamics created: {n_units} units, {connectivity:.1%} connectivity")

    def step(self, external_perturbation: np.ndarray) -> np.ndarray:
        """
        한 스텝 역학 실행

        Args:
            external_perturbation: 외부 교란

        Returns:
            new_state: 새로운 내부 상태
        """
        # 내부 역학: 순환적 활성화
        internal_influence = np.tanh(np.dot(self.W, self.state))

        # 외부 교란 통합
        perturbation_effect = external_perturbation[:self.n_units] if len(external_perturbation) >= self.n_units else np.zeros(self.n_units)

        # 새 상태 = 내부 역학 + 외부 교란
        # Leaky integration (시간 상수)
        tau = 0.7  # 0에 가까울수록 빠른 역학
        new_state = tau * self.state + (1 - tau) * (internal_influence + 0.1 * perturbation_effect)

        # 상태 업데이트
        self.state = np.clip(new_state, -2, 2)
        self.state_history.append(self.state.copy())

        return self.state

    def get_output(self) -> np.ndarray:
        """
        행동 생성 (내부 상태의 일부를 출력)

        NOT: 목표 지향적 행동
        BUT: 내부 역학의 자연스러운 표현
        """
        # 첫 몇 개 단위를 행동으로
        return np.tanh(self.state[:3])


class CoherenceAssessor:
    """
    조직적 일관성 평가

    핵심: 외부 기준(에너지, 보상) 없이 내부 역학의 일관성 측정
    """

    def __init__(self):
        self.coherence_history = deque(maxlen=100)

    def assess(self, dynamics: InternalDynamics) -> Dict[str, float]:
        """
        다차원 일관성 평가

        차원:
            1. Predictability: 내부 상태의 예측가능성
            2. Stability: 역학의 안정성
            3. Complexity: 적절한 복잡도 (너무 단순/복잡 모두 나쁨)
            4. Circularity: 순환 인과성

        Returns:
            coherence_scores: 각 차원 점수
        """
        if len(dynamics.state_history) < 10:
            return {
                'predictability': 0.5,
                'stability': 0.5,
                'complexity': 0.5,
                'circularity': 0.5,
                'composite': 0.5
            }

        states = np.array(list(dynamics.state_history))

        # 1. Predictability: 낮은 엔트로피 = 높은 예측가능성
        # 상태 변화의 분산 (낮을수록 예측가능)
        state_changes = np.diff(states, axis=0)
        predictability_score = 1.0 / (1.0 + np.mean(np.var(state_changes, axis=0)))

        # 2. Stability: 상태의 안정성
        # 최근 상태들의 표준편차 (낮을수록 안정)
        recent_states = states[-20:]
        stability_score = 1.0 / (1.0 + np.std(recent_states))

        # 3. Complexity: 적절한 복잡도
        # 상태의 엔트로피 (너무 높지도 낮지도 않아야 함)
        state_variance = np.var(states)
        # Optimal complexity around 0.5 variance
        complexity_score = 1.0 - abs(state_variance - 0.5)
        complexity_score = np.clip(complexity_score, 0, 1)

        # 4. Circularity: 자기상관 (순환 인과성)
        # 시간 지연 자기상관
        if len(states) >= 20:
            autocorr = np.corrcoef(states[:-10].flatten(), states[10:].flatten())[0, 1]
            circularity_score = abs(autocorr)  # 높은 자기상관 = 순환성
        else:
            circularity_score = 0.5

        # 종합 일관성
        composite = (
            0.3 * predictability_score +
            0.3 * stability_score +
            0.2 * complexity_score +
            0.2 * circularity_score
        )

        coherence = {
            'predictability': float(np.clip(predictability_score, 0, 1)),
            'stability': float(np.clip(stability_score, 0, 1)),
            'complexity': float(np.clip(complexity_score, 0, 1)),
            'circularity': float(np.clip(circularity_score, 0, 1)),
            'composite': float(np.clip(composite, 0, 1))
        }

        self.coherence_history.append(coherence['composite'])

        return coherence


class StructuralPlasticity:
    """
    구조적 가소성

    핵심: Gradient descent 없이 구조적 드리프트
    """

    def __init__(self, plasticity_rate: float = 0.01):
        self.plasticity_rate = plasticity_rate
        self.structural_changes = 0

    def perturb_structure(self, dynamics: InternalDynamics,
                          coherence_before: float) -> Tuple[bool, float]:
        """
        구조 교란 및 수용/거부

        Args:
            dynamics: 내부 역학
            coherence_before: 교란 전 일관성

        Returns:
            accepted: 변화 수용 여부
            coherence_after: 교란 후 일관성
        """
        # 작은 랜덤 구조 변화
        n = dynamics.W.shape[0]
        perturbation = np.random.randn(n, n) * self.plasticity_rate

        # 연결 마스크 유지 (기존 0인 곳은 0 유지)
        mask = (dynamics.W != 0)
        perturbation = perturbation * mask

        # 임시 적용
        W_original = dynamics.W.copy()
        dynamics.W += perturbation

        # 몇 스텝 실행하여 일관성 평가
        assessor = CoherenceAssessor()

        # 임시 시뮬레이션
        temp_state = dynamics.state.copy()
        for _ in range(5):
            dynamics.step(np.zeros(n))

        coherence_after = assessor.assess(dynamics)['composite']

        # 복원
        dynamics.state = temp_state

        # 일관성 유지/개선되면 수용
        if coherence_after >= coherence_before * 0.95:  # 5% 허용
            # 변화 수용
            self.structural_changes += 1
            accepted = True
        else:
            # 변화 거부, 복원
            dynamics.W = W_original
            accepted = False

        return accepted, coherence_after


class AutopoeticEntity:
    """
    자기생성 Entity

    핵심 차이:
        - NO loss function
        - NO optimization objective
        - ONLY organizational maintenance

    생존 = 조직적 일관성 유지
    학습 = 일관성을 유지하는 구조적 드리프트
    """

    def __init__(self,
                 n_internal_units: int = 20,
                 connectivity: float = 0.3,
                 plasticity_rate: float = 0.01,
                 coherence_threshold: float = 0.3,
                 entity_id: Optional[str] = None):
        """
        Args:
            n_internal_units: 내부 단위 수
            connectivity: 연결 밀도
            plasticity_rate: 구조 변화율
            coherence_threshold: 생존 임계값
            entity_id: Entity ID
        """
        self.id = entity_id or str(uuid.uuid4())[:8]

        # 내부 역학
        self.dynamics = InternalDynamics(n_internal_units, connectivity)

        # 일관성 평가
        self.assessor = CoherenceAssessor()

        # 구조 가소성
        self.plasticity = StructuralPlasticity(plasticity_rate)

        # 생존 상태
        self.is_alive = True
        self.age = 0
        self.coherence_threshold = coherence_threshold

        # 통계
        self.coherence_history = deque(maxlen=200)
        self.structural_change_history = []

        print(f"AutopoeticEntity created: id={self.id}")
        print(f"  Internal units: {n_internal_units}")
        print(f"  Connectivity: {connectivity:.1%}")
        print(f"  Coherence threshold: {coherence_threshold}")
        print(f"  Paradigm: Autopoiesis (NO external objectives)")

    def live_one_step(self, perturbation: np.ndarray) -> Dict:
        """
        한 스텝 생존

        핵심 흐름:
            1. 외부 교란 받음
            2. 내부 역학 실행
            3. 조직적 일관성 평가
            4. 일관성 낮으면 구조 변화 시도
            5. 일관성 붕괴시 죽음

        Args:
            perturbation: 환경으로부터의 교란

        Returns:
            result: 상태 정보
        """
        if not self.is_alive:
            return {'is_alive': False}

        self.age += 1

        # ============================================
        # 1. 내부 역학 실행 (교란 통합)
        # ============================================
        self.dynamics.step(perturbation)

        # ============================================
        # 2. 행동 생성 (내부 역학의 표현)
        # ============================================
        action = self.dynamics.get_output()

        # ============================================
        # 3. 조직적 일관성 평가
        # ============================================
        coherence = self.assessor.assess(self.dynamics)
        self.coherence_history.append(coherence['composite'])

        # ============================================
        # 4. 구조적 가소성 (일관성 낮으면)
        # ============================================
        structural_changed = False
        if coherence['composite'] < 0.6 and self.age % 5 == 0:
            # 구조 변화 시도
            accepted, new_coherence = self.plasticity.perturb_structure(
                self.dynamics,
                coherence['composite']
            )

            if accepted:
                structural_changed = True
                coherence['composite'] = new_coherence

        # ============================================
        # 5. 죽음 (조직 붕괴)
        # ============================================
        if coherence['composite'] < self.coherence_threshold:
            self.is_alive = False
            print(f"Entity {self.id} died at age {self.age} "
                  f"(coherence={coherence['composite']:.3f} < {self.coherence_threshold})")

        # ============================================
        # 6. 결과 반환
        # ============================================
        return {
            'is_alive': self.is_alive,
            'age': self.age,
            'action': action,
            'coherence': coherence,
            'structural_changed': structural_changed,
            'internal_state': self.dynamics.state.copy()
        }

    def get_fitness(self) -> float:
        """
        적합도 (번식 기준)

        NOT: 외부 성능
        BUT: 조직적 일관성의 안정성
        """
        if len(self.coherence_history) < 20:
            return 0.5

        recent = list(self.coherence_history)[-20:]

        # 평균 일관성 + 안정성
        avg_coherence = np.mean(recent)
        stability = 1.0 / (1.0 + np.std(recent))

        fitness = 0.7 * avg_coherence + 0.3 * stability

        return float(fitness)

    def reproduce(self, mutation_rate: float = 0.1) -> 'AutopoeticEntity':
        """
        번식 (구조 변이)

        Args:
            mutation_rate: 변이율

        Returns:
            offspring: 자손 entity
        """
        offspring = AutopoeticEntity(
            n_internal_units=self.dynamics.n_units,
            connectivity=0.3,
            plasticity_rate=self.plasticity.plasticity_rate,
            coherence_threshold=self.coherence_threshold
        )

        # 구조 복사 + 변이
        offspring.dynamics.W = self.dynamics.W.copy()
        mutation = np.random.randn(*offspring.dynamics.W.shape) * mutation_rate
        mask = (offspring.dynamics.W != 0)
        offspring.dynamics.W += mutation * mask

        return offspring

    def get_summary(self) -> Dict:
        """상태 요약"""
        return {
            'id': self.id,
            'age': self.age,
            'is_alive': self.is_alive,
            'current_coherence': self.coherence_history[-1] if len(self.coherence_history) > 0 else 0.0,
            'avg_coherence': np.mean(list(self.coherence_history)) if len(self.coherence_history) > 0 else 0.0,
            'structural_changes': self.plasticity.structural_changes,
            'fitness': self.get_fitness()
        }


# =======================
# Testing
# =======================

if __name__ == "__main__":
    print("=" * 70)
    print("Autopoietic Entity Test: True Viability Paradigm")
    print("=" * 70)

    # Entity 생성
    entity = AutopoeticEntity(
        n_internal_units=20,
        connectivity=0.3,
        plasticity_rate=0.02,
        coherence_threshold=0.25
    )

    print(f"\n{'='*70}")
    print("Living without External Objectives")
    print(f"{'='*70}\n")

    # 시뮬레이션
    n_steps = 200

    for step in range(n_steps):
        # 환경 교란 (랜덤)
        perturbation = np.random.randn(20) * 0.3

        # 생존 스텝
        result = entity.live_one_step(perturbation)

        if not result['is_alive']:
            print(f"\n💀 Entity died at step {step}")
            break

        if step % 20 == 0:
            coherence = result['coherence']
            print(f"Step {step:3d} | Coherence: {coherence['composite']:.3f} "
                  f"(P:{coherence['predictability']:.2f} "
                  f"S:{coherence['stability']:.2f} "
                  f"Cx:{coherence['complexity']:.2f} "
                  f"Cr:{coherence['circularity']:.2f})")

    # 최종 요약
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")

    summary = entity.get_summary()
    print(f"\n**Survival**:")
    print(f"  Lifespan: {summary['age']} steps")
    print(f"  Status: {'Alive' if summary['is_alive'] else 'Dead'}")
    print(f"  Fitness: {summary['fitness']:.3f}")

    print(f"\n**Coherence**:")
    print(f"  Current: {summary['current_coherence']:.3f}")
    print(f"  Average: {summary['avg_coherence']:.3f}")

    print(f"\n**Adaptation**:")
    print(f"  Structural changes: {summary['structural_changes']}")

    print(f"\n**Paradigm**:")
    print(f"  ✓ NO loss function")
    print(f"  ✓ NO optimization objective")
    print(f"  ✓ NO external goals")
    print(f"  ✓ ONLY organizational coherence")

    print("\n" + "=" * 70)
    print("This is TRUE autopoietic learning!")
    print("=" * 70)
