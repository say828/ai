# QED-LAML Hybrid: 설계 문서

## 실험 날짜
2026-01-03

---

## 1. 현재 상황 분석

### QED 성능
- Linear: 0.00989 (SGD 대비 +26%)
- Nonlinear: 0.10541 (SGD 대비 +32%)
- XOR: 0.27960 (SGD 대비 +65%)
- **강점**: Nonlinear와 XOR에서 압도적

### LAML-Q 성능
- Linear: 0.009491 (SGD 대비 +29.14%)
- Nonlinear: 0.114231 (SGD 대비 +26.68%)
- XOR: 0.451173 (SGD 대비 +43.90%)
- **강점**: Linear에서 QED보다 근소하게 우세

### 직접 비교
| 데이터셋 | 승자 | 차이 |
|---------|------|------|
| Linear | LAML-Q | -4.0% 더 좋음 |
| Nonlinear | QED | +8.4% 더 좋음 |
| XOR | QED | +61.3% 더 좋음 |

**결론**: QED가 전체적으로 우세하지만, 두 알고리즘이 상호 보완적

---

## 2. 하이브리드 전략: "Force + Action Fusion"

### 핵심 아이디어
QED의 강력한 탐색 능력과 LAML의 물리적 효율성 검증을 **깊은 수준에서 융합**

### 설계 원칙

```
┌─────────────────────────────────────────────┐
│         QED-LAML Hybrid Architecture        │
├─────────────────────────────────────────────┤
│                                             │
│  QED Layer (Exploration)                    │
│  ├─ 6 Forces (gradient, momentum, etc.)     │
│  ├─ Particle Swarm Intelligence             │
│  └─ Evolution (crossover, mutation)         │
│                 ↓↑                          │
│  LAML Layer (Verification)                  │
│  ├─ Multi-scale Endpoint Prediction         │
│  ├─ Action Computation (efficiency)         │
│  ├─ Least Action Verification              │
│  └─ Adaptive Learning Rate                  │
│                 ↓                           │
│  Hybrid Decision:                           │
│  - QED force 제안                           │
│  - LAML action 검증                         │
│  - 둘 다 만족하면 강하게 업데이트           │
│  - 하나만 만족하면 약하게 업데이트          │
│  - 둘 다 불만족하면 탐색                    │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 3. 구체적 구현 계획

### 3.1 HybridParticle 클래스

QED의 Particle을 확장하여 LAML 기능 추가:

```python
class HybridParticle(Particle):
    def __init__(self, network, particle_id):
        super().__init__(network, particle_id)

        # LAML 추가 속성
        self.predicted_endpoint = None
        self.trajectory = None
        self.action = float('inf')
        self.learning_rate = 0.1  # Adaptive LR
        self.success_count = 0
        self.fail_count = 0

    def predict_endpoint(self, X, y, temperature):
        """LAML의 multi-scale endpoint prediction"""
        # 1, 5, 10 steps lookahead
        # Weighted averaging by quality
        pass

    def compute_action(self, start, end, X, y):
        """LAML의 action calculation"""
        # S = ∫[½||θ̇||² + λL(θ)] dt
        pass

    def hybrid_update(self, X, y, lr, temp, global_best, swarm_center,
                     global_best_action):
        """
        QED + LAML 융합 업데이트

        1. QED force 계산
        2. LAML endpoint 예측
        3. 두 방향의 action 비교
        4. 최선의 방향 선택
        5. Adaptive step size
        """
        # QED의 6 forces
        qed_direction = self.compute_qed_forces(...)

        # LAML의 endpoint prediction
        laml_endpoint = self.predict_endpoint(X, y, temp)
        laml_direction = laml_endpoint - current_pos

        # 각 방향의 Action 계산
        qed_action = self.compute_action(current_pos,
                                         current_pos + qed_direction, X, y)
        laml_action = self.compute_action(current_pos,
                                          laml_endpoint, X, y)

        # 최선의 방향 선택 (action 기준)
        if laml_action < qed_action:
            direction = laml_direction
            action = laml_action
            method = "LAML"
        else:
            direction = qed_direction
            action = qed_action
            method = "QED"

        # Action 검증 (LAML 철학)
        if action < global_best_action * 1.2:  # 충분히 효율적
            step_size = self.learning_rate
            self.success_count += 1
            if self.success_count >= 3:
                self.learning_rate = min(0.5, self.learning_rate * 1.2)
        else:  # 비효율적 → 보수적
            step_size = self.learning_rate * 0.5
            self.fail_count += 1
            if self.fail_count >= 3:
                self.learning_rate = max(0.01, self.learning_rate * 0.8)

        # 업데이트
        new_pos = current_pos + step_size * direction
        return new_pos, action, method
```

### 3.2 QED_LAML_Optimizer

```python
class QED_LAML_Optimizer:
    """
    QED와 LAML의 완전한 융합

    특징:
    - QED의 particle swarm 구조
    - LAML의 action verification
    - 둘의 adaptive mechanism 결합
    - 진화 + 물리적 검증
    """

    def __init__(self, network_template, n_particles=10, ...):
        # HybridParticle들로 초기화
        self.particles = [HybridParticle(net, i) for i in range(n_particles)]

        # Global tracking
        self.global_best = ...
        self.global_best_action = float('inf')
        self.global_best_loss = float('inf')

        # Statistics
        self.qed_wins = 0
        self.laml_wins = 0

    def train(self, X, y, max_iters=100):
        for iteration in range(max_iters):
            for particle in self.particles:
                # Hybrid update
                new_pos, action, method = particle.hybrid_update(
                    X, y, ..., self.global_best_action
                )

                particle.network.set_weights(new_pos)
                loss = particle.network.loss(X, y)

                # Statistics
                if method == "QED":
                    self.qed_wins += 1
                else:
                    self.laml_wins += 1

                # Global best update
                if action < self.global_best_action:
                    self.global_best_action = action
                    self.global_best = new_pos
                    self.global_best_loss = loss

            # Evolution (from QED)
            if iteration % 5 == 0:
                diversity = self.get_diversity()
                self.evolve_particles(X, y, diversity)

            # Temperature decay
            self.temperature *= self.temp_decay
```

### 3.3 진화 메커니즘

QED의 진화에 LAML의 action 기준 추가:

```python
def evolve_particles(self, X, y, diversity):
    """
    QED evolution + LAML action selection
    """
    # Action 기준으로 정렬 (LAML)
    particles_ranked = sorted(self.particles, key=lambda p: p.action)

    # Diversity 기반 교체율 (LAML-Q Step 4)
    n_replace = ...

    # Tournament selection (LAML-Q Step 4)
    # But now using action as fitness metric
    ...
```

---

## 4. 예상 성능

### 가설

**Linear**: LAML-Q가 이미 이겼으므로, Hybrid는 동등 이상
- 예상: 0.009 이하

**Nonlinear**: QED가 강했으므로, Hybrid는 QED 수준 유지
- 예상: 0.10 이하

**XOR**: QED가 압도적이었으므로, Hybrid는 QED 기반으로 개선
- 예상: 0.25 이하

### 목표

모든 데이터셋에서:
1. SGD 대비 +30% 이상
2. QED와 LAML-Q의 최선 이상
3. 더 빠른 수렴 (fewer iterations)

---

## 5. 구현 순서

1. **HybridParticle 클래스 작성**
   - QED Particle 상속
   - LAML 기능 추가
   - hybrid_update 구현

2. **QED_LAML_Optimizer 작성**
   - 기본 구조 (QED 기반)
   - LAML verification 통합
   - Evolution 개선

3. **테스트 및 비교**
   - 3개 데이터셋
   - QED, LAML-Q, Hybrid, SGD 4-way 비교
   - 시각화 및 분석

4. **하이퍼파라미터 튜닝**
   - QED/LAML 밸런스 조정
   - Action threshold 최적화
   - Learning rate schedule

---

## 6. 핵심 혁신 포인트

1. **깊은 융합**: 단순 결합이 아닌, update 단계에서 통합
2. **상호 검증**: QED가 제안 → LAML이 검증 → 최선 선택
3. **Adaptive mechanism**: 두 방법의 성공/실패를 모두 추적
4. **통계 기록**: QED vs LAML 승률 추적으로 어느 방법이 언제 좋은지 분석

---

## 7. 예상 결과 시나리오

**Best Case**: Hybrid가 모든 데이터셋에서 QED와 LAML-Q 둘 다 뛰어넘음
- Linear: 0.008 대
- Nonlinear: 0.09 대
- XOR: 0.23 대

**Likely Case**: Hybrid가 QED 수준이거나 약간 개선
- QED의 강력한 탐색 유지
- LAML의 효율성 검증으로 약간 개선
- 더 안정적인 수렴

**Worst Case**: Overhead로 인해 QED보다 느림
- 하지만 최종 성능은 비슷할 것
- 이 경우에도 "두 철학의 융합"이라는 학술적 가치는 있음

---

## 8. 논문 기여도

이 Hybrid는 다음을 보여줍니다:

1. **물리학 기반 최적화 (LAML)** 가 실용 가능
2. **집단 지성 (QED)** 와 **물리적 원리 (LAML)** 의 시너지
3. **Meta-cognition**: 시스템이 자기 행동의 효율성을 평가
4. **새로운 패러다임**: "탐색 + 검증" 이중 구조

---

**작성자**: Claude Sonnet 4.5
**다음 단계**: 구현 시작
