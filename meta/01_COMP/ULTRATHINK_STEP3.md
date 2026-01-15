# ULTRATHINK Step 3: Compositional Optimization 구체화

## 핵심 아이디어

```
복잡한 최적화 = 단순한 Primitives의 지능적 조합

Update(θ, context) = Σ w_i(context) · Primitive_i(θ)
```

**철학**:
- 각 Primitive는 하나의 "관점"
- Context에 따라 어떤 관점이 중요한지 달라짐
- 시스템이 자동으로 가중치 결정

---

## Part 1: Primitives 설계

### 1.1 탐색 Primitives (Exploration)

**P1: Gradient Descent**
```python
def gradient_descent(theta, X, y, lr=0.01):
    grad = compute_gradient(theta, X, y)
    return -lr * grad
```
- 용도: 지역적 최적화
- 강점: 안정적, 이론적 보장
- 약점: 느림, local minima

**P2: Stochastic Jump**
```python
def stochastic_jump(theta, temperature=0.1):
    direction = random_direction()
    return temperature * direction
```
- 용도: Local minima 탈출
- 강점: 탐색력
- 약점: 불안정

**P3: Momentum**
```python
def momentum(theta, velocity, decay=0.9):
    return decay * velocity
```
- 용도: 과거 방향 유지
- 강점: 수렴 가속
- 약점: Overshoot 가능

**P4: Historical Best Direction**
```python
def best_direction(theta, history):
    best_theta = history.get_best()
    return (best_theta - theta) / norm(best_theta - theta)
```
- 용도: 검증된 방향
- 강점: 안전
- 약점: 보수적

### 1.2 적응 Primitives (Adaptation)

**P5: Adaptive Step Size**
```python
def adaptive_step(theta, success_rate):
    if success_rate > 0.8:
        return scale * 1.2  # 성공 많으면 증가
    elif success_rate < 0.3:
        return scale * 0.5  # 실패 많으면 감소
    return scale
```

**P6: Curvature-Aware**
```python
def curvature_step(theta, X, y):
    hessian_diag = compute_hessian_diagonal(theta, X, y)
    preconditioner = 1.0 / (hessian_diag + epsilon)
    grad = compute_gradient(theta, X, y)
    return -preconditioner * grad
```
- 용도: 2차 정보 활용
- 강점: 더 정확한 방향

### 1.3 정제 Primitives (Refinement)

**P7: Coordinate-Wise Update**
```python
def coordinate_update(theta, X, y, k=10):
    # 가장 중요한 k개 파라미터만 업데이트
    grad = compute_gradient(theta, X, y)
    top_k = argsort(abs(grad))[-k:]
    delta = zeros_like(theta)
    delta[top_k] = -grad[top_k]
    return delta
```

**P8: Layer-Wise Strategy**
```python
def layerwise_update(theta, X, y):
    # 레이어마다 다른 learning rate
    grad = compute_gradient(theta, X, y)
    layer_lrs = compute_layer_importance(theta)
    return -layer_lrs * grad
```

### 1.4 정규화 Primitives (Regularization)

**P9: Weight Decay**
```python
def weight_decay(theta, lambda_reg=0.01):
    return -lambda_reg * theta
```

**P10: Gradient Clipping Direction**
```python
def safe_direction(grad, max_norm=1.0):
    norm_g = norm(grad)
    if norm_g > max_norm:
        return grad * (max_norm / norm_g)
    return grad
```

---

## Part 2: Context 정의

**Context**: 현재 최적화 상황을 나타내는 정보

```python
class OptimizationContext:
    # 상태 정보
    current_loss: float
    current_grad_norm: float
    current_theta: np.ndarray

    # 역사 정보
    loss_history: List[float]
    grad_history: List[np.ndarray]
    success_rate: float  # 최근 개선 비율

    # 진행 정보
    iteration: int
    phase: str  # 'exploration', 'exploitation', 'refinement'

    # 통계 정보
    loss_variance: float  # 최근 손실의 분산
    grad_variance: float
    improvement_rate: float  # 개선 속도

    def get_phase(self):
        """학습 단계 자동 판단"""
        if self.iteration < 20:
            return 'exploration'  # 초기: 큰 탐색
        elif self.improvement_rate > 0.05:
            return 'exploitation'  # 중기: 빠른 수렴
        else:
            return 'refinement'  # 후기: 미세 조정
```

---

## Part 3: Weight Function 설계

**핵심**: Context → Primitive Weights

### 3.1 Rule-Based Weights

```python
def compute_weights(context: OptimizationContext):
    weights = np.zeros(10)  # 10 primitives

    # Phase 기반
    if context.phase == 'exploration':
        weights[1] = 0.4  # Stochastic Jump
        weights[0] = 0.3  # Gradient
        weights[2] = 0.2  # Momentum
        weights[3] = 0.1  # Best Direction

    elif context.phase == 'exploitation':
        weights[0] = 0.4  # Gradient (강화)
        weights[2] = 0.3  # Momentum
        weights[5] = 0.2  # Curvature
        weights[3] = 0.1  # Best Direction

    elif context.phase == 'refinement':
        weights[0] = 0.3  # Gradient
        weights[5] = 0.3  # Curvature
        weights[6] = 0.2  # Coordinate-wise
        weights[7] = 0.2  # Layer-wise

    # Success rate 기반 조정
    if context.success_rate < 0.3:
        # 실패 많으면 탐색 강화
        weights[1] *= 1.5  # Stochastic Jump
        weights[0] *= 0.7  # Gradient 약화

    # Gradient norm 기반
    if context.current_grad_norm < 0.01:
        # Gradient 작으면 탐색 강화
        weights[1] *= 2.0

    # Normalize
    weights = weights / weights.sum()
    return weights
```

### 3.2 Learned Weights (Advanced)

```python
class WeightPredictor(nn.Module):
    """Context → Weights 학습"""

    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_primitives),
            nn.Softmax()
        )

    def forward(self, context_vector):
        return self.encoder(context_vector)
```

**학습 방법**:
- 많은 최적화 과정 기록
- 좋은 업데이트와 나쁜 업데이트 레이블링
- Supervised learning

---

## Part 4: 전체 알고리즘

### 4.1 Compositional Optimizer

```python
class CompositionalOptimizer:
    def __init__(self, primitives, weight_function):
        self.primitives = primitives
        self.weight_fn = weight_function
        self.context = OptimizationContext()

    def step(self, theta, X, y):
        # 1. Context 업데이트
        self.context.update(theta, X, y)

        # 2. Weights 계산
        weights = self.weight_fn(self.context)

        # 3. Primitives 실행
        deltas = []
        for i, primitive in enumerate(self.primitives):
            delta = primitive(theta, X, y, self.context)
            deltas.append(delta)

        # 4. 가중 합산
        final_delta = sum(w * d for w, d in zip(weights, deltas))

        # 5. 업데이트
        theta_new = theta + final_delta

        # 6. 평가 및 역사 기록
        loss_new = compute_loss(theta_new, X, y)
        self.context.record(loss_new, final_delta)

        return theta_new, weights  # weights도 반환 (분석용)
```

### 4.2 실행 흐름

```
Initialization
│
├─→ Iteration Start
│   │
│   ├─→ Update Context (상황 파악)
│   │   ├─ 현재 loss, grad
│   │   ├─ 역사 통계
│   │   └─ Phase 판단
│   │
│   ├─→ Compute Weights (전략 결정)
│   │   └─ Context → w₁, w₂, ..., w₁₀
│   │
│   ├─→ Execute Primitives (각 관점의 제안)
│   │   ├─ P1: Gradient 방향
│   │   ├─ P2: Random 방향
│   │   ├─ ...
│   │   └─ P10: Safe 방향
│   │
│   ├─→ Compose Update (종합)
│   │   └─ Δθ = Σ wᵢ · Pᵢ
│   │
│   ├─→ Apply Update
│   │   └─ θ' = θ + Δθ
│   │
│   └─→ Record History
│       └─ 성공/실패, weights 기록
│
└─→ Next Iteration or Converge
```

---

## Part 5: 왜 이게 혁신적인가?

### 5.1 기존 방법과의 차이

| 특성 | SGD | Adam | QED | LAML-Q | **Compositional** |
|------|-----|------|-----|--------|-------------------|
| 단일 전략 | ✓ | ✓ | ✗ | ✗ | ✗ |
| 고정 규칙 | ✓ | ✓ | ✓ | ✓ | ✗ |
| 상황 인식 | ✗ | 부분적 | 부분적 | ✓ | ✓✓ |
| 확장 가능 | ✗ | ✗ | ✗ | ✗ | ✓✓ |
| 해석 가능 | ✓ | ✗ | ✗ | 부분적 | ✓✓ |

### 5.2 고유한 장점

**1. 모듈성**
- 새 Primitive 추가 쉬움
- 기존 것 제거/수정 쉬움

**2. 투명성**
- 각 Primitive의 기여도 추적 가능
- "왜 이렇게 업데이트했는가?" 설명 가능

**3. 적응성**
- 문제마다 Primitives 선택 가능
- Context definition 커스터마이징

**4. 학습 가능**
- Weight function을 학습하여 개선
- Meta-learning 자연스럽게 통합

**5. 이론과 실용의 균형**
- 각 Primitive는 이론적 근거 있음
- 조합은 실용적으로 작동

---

## Part 6: 예상 성능

### Scenario 1: Linear Problem
- 초기: Gradient + Momentum 강함 → 빠른 수렴
- 후기: Curvature-aware → 정확한 수렴
- **예상**: QED 수준 (0.01 이하)

### Scenario 2: Nonlinear Problem
- 초기: Stochastic + Gradient → 좋은 영역 찾기
- 중기: Momentum + Best Direction → 빠른 수렴
- 후기: Coordinate-wise → 미세 조정
- **예상**: QED 초과 (0.09 이하)

### Scenario 3: XOR
- 초기: 강한 탐색 (Stochastic) → Saddle point 탈출
- 중기: Curvature + Momentum → 비선형 구조 학습
- 후기: Layer-wise → 레이어별 최적화
- **예상**: QED 초과 (0.25 이하)

---

## Step 3 결론

### 설계 완료

**알고리즘 이름**: **"COMP"** (Compositional Optimizer)

**핵심 구성**:
1. ✅ 10개 Primitives
2. ✅ Context 정의
3. ✅ Weight function (rule-based)
4. ✅ 전체 알고리즘

### 다음 스텝

Step 4에서:
1. 구현 계획 수립
2. 파일 구조 설계
3. 테스트 전략

---

**작성**: 2026-01-03 Step 3/6
**다음**: Step 4 - 구현 계획
