# ULTRATHINK Step 4: 구현 계획

## 알고리즘 정식 명칭

**COMP**: **C**ompositional **O**ptimizer with **M**ulti-**P**rimitives

**발음**: "컴프"
**철학**: Compose simple strategies into intelligent optimization

---

## 파일 구조

```
05_COMP/
├── comp_optimizer.py      # 메인 구현
├── primitives.py          # 10개 primitive 함수들
├── context.py             # OptimizationContext 클래스
├── weight_functions.py    # Weight 계산 로직
├── COMP_THEORY.md         # 이론 문서
└── results/               # 실험 결과
```

---

## 구현 우선순위

### Phase A: 핵심 구조 (필수)
1. `context.py` - Context 클래스
2. `primitives.py` - 기본 5개 primitives
3. `weight_functions.py` - Rule-based weights
4. `comp_optimizer.py` - 메인 optimizer

### Phase B: 확장 (선택)
5. 추가 primitives (P6-P10)
6. Learned weight function
7. 시각화 및 분석 도구

---

## 상세 구현 계획

### 1. context.py

```python
"""
Optimization context tracking

역할:
- 현재 상태 추적
- 역사 기록
- Phase 자동 판단
- 통계 계산
"""

import numpy as np
from collections import deque

class OptimizationContext:
    def __init__(self, history_size=20):
        # 현재 상태
        self.iteration = 0
        self.current_loss = float('inf')
        self.current_grad_norm = 0.0
        self.current_theta = None

        # 역사 (최근 N개만)
        self.loss_history = deque(maxlen=history_size)
        self.grad_norm_history = deque(maxlen=history_size)
        self.update_history = deque(maxlen=history_size)

        # 통계
        self.success_count = 0
        self.total_count = 0

    def update(self, theta, loss, grad_norm):
        """매 iteration마다 호출"""
        # 개선 여부 판단
        if loss < self.current_loss:
            self.success_count += 1
        self.total_count += 1

        # 상태 업데이트
        self.current_theta = theta
        self.current_loss = loss
        self.current_grad_norm = grad_norm

        # 역사 기록
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

        self.iteration += 1

    @property
    def success_rate(self):
        """최근 성공률"""
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count

    @property
    def loss_variance(self):
        """최근 손실의 분산"""
        if len(self.loss_history) < 2:
            return 1.0
        return np.var(list(self.loss_history))

    @property
    def improvement_rate(self):
        """개선 속도"""
        if len(self.loss_history) < 2:
            return 0.0
        recent = list(self.loss_history)[-5:]
        if len(recent) < 2:
            return 0.0
        return (recent[0] - recent[-1]) / recent[0]

    @property
    def phase(self):
        """학습 단계 자동 판단"""
        if self.iteration < 15:
            return 'exploration'
        elif self.improvement_rate > 0.05:
            return 'exploitation'
        else:
            return 'refinement'

    def to_vector(self):
        """Context를 벡터로 (learned weight function용)"""
        return np.array([
            self.current_loss,
            self.current_grad_norm,
            self.success_rate,
            self.loss_variance,
            self.improvement_rate,
            1.0 if self.phase == 'exploration' else 0.0,
            1.0 if self.phase == 'exploitation' else 0.0,
            1.0 if self.phase == 'refinement' else 0.0,
        ])
```

### 2. primitives.py

```python
"""
Optimization primitives

각 primitive는 독립적인 "관점"을 제공
"""

import numpy as np

class Primitive:
    """Base class for all primitives"""
    def __init__(self, name):
        self.name = name

    def __call__(self, theta, grad, context):
        raise NotImplementedError

# === 탐색 Primitives ===

class GradientDescent(Primitive):
    def __init__(self, lr=0.05):
        super().__init__("GradientDescent")
        self.lr = lr

    def __call__(self, theta, grad, context):
        return -self.lr * grad

class StochasticJump(Primitive):
    def __init__(self, temperature=0.1):
        super().__init__("StochasticJump")
        self.temperature = temperature

    def __call__(self, theta, grad, context):
        # Context의 온도 활용
        effective_temp = self.temperature * (1.0 if context.phase == 'exploration' else 0.3)
        return np.random.randn(len(theta)) * effective_temp

class Momentum(Primitive):
    def __init__(self, decay=0.9):
        super().__init__("Momentum")
        self.decay = decay
        self.velocity = None

    def __call__(self, theta, grad, context):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)

        # Velocity 업데이트
        self.velocity = self.decay * self.velocity - 0.05 * grad
        return self.velocity

class BestDirection(Primitive):
    def __init__(self):
        super().__init__("BestDirection")
        self.best_theta = None
        self.best_loss = float('inf')

    def __call__(self, theta, grad, context):
        # Best 업데이트
        if context.current_loss < self.best_loss:
            self.best_loss = context.current_loss
            self.best_theta = theta.copy()

        if self.best_theta is None:
            return np.zeros_like(theta)

        # Best로 향하는 방향
        direction = self.best_theta - theta
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm * 0.1
        return np.zeros_like(theta)

class AdaptiveStep(Primitive):
    def __init__(self):
        super().__init__("AdaptiveStep")
        self.base_lr = 0.05

    def __call__(self, theta, grad, context):
        # Success rate 기반 adaptive LR
        if context.success_rate > 0.7:
            lr = self.base_lr * 1.5
        elif context.success_rate < 0.3:
            lr = self.base_lr * 0.5
        else:
            lr = self.base_lr

        return -lr * grad

# === Primitive Registry ===

def get_default_primitives():
    """기본 5개 primitives"""
    return [
        GradientDescent(lr=0.05),
        StochasticJump(temperature=0.1),
        Momentum(decay=0.9),
        BestDirection(),
        AdaptiveStep(),
    ]
```

### 3. weight_functions.py

```python
"""
Weight functions: Context → Primitive Weights
"""

import numpy as np

def rule_based_weights(context, n_primitives=5):
    """
    Rule-based weight assignment

    Primitives order:
    0: GradientDescent
    1: StochasticJump
    2: Momentum
    3: BestDirection
    4: AdaptiveStep
    """
    weights = np.zeros(n_primitives)

    # Phase-based base weights
    if context.phase == 'exploration':
        weights[0] = 0.25  # Gradient
        weights[1] = 0.35  # Stochastic (강화)
        weights[2] = 0.20  # Momentum
        weights[3] = 0.10  # Best
        weights[4] = 0.10  # Adaptive

    elif context.phase == 'exploitation':
        weights[0] = 0.30  # Gradient
        weights[1] = 0.15  # Stochastic (약화)
        weights[2] = 0.30  # Momentum (강화)
        weights[3] = 0.15  # Best
        weights[4] = 0.10  # Adaptive

    else:  # refinement
        weights[0] = 0.25  # Gradient
        weights[1] = 0.05  # Stochastic (최소)
        weights[2] = 0.25  # Momentum
        weights[3] = 0.20  # Best
        weights[4] = 0.25  # Adaptive (강화)

    # Success rate adjustment
    if context.success_rate < 0.3:
        # 실패 많으면 탐색 강화
        weights[1] *= 2.0  # Stochastic
        weights[0] *= 0.7  # Gradient 약화

    # Gradient norm adjustment
    if context.current_grad_norm < 0.01:
        # Gradient 작으면 탐색 강화
        weights[1] *= 1.5
        weights[3] *= 1.3  # Best direction도 활용

    # Normalize
    return weights / weights.sum()
```

### 4. comp_optimizer.py

```python
"""
COMP: Compositional Optimizer with Multi-Primitives

메인 optimizer 구현
"""

import numpy as np
import matplotlib.pyplot as plt
from context import OptimizationContext
from primitives import get_default_primitives
from weight_functions import rule_based_weights

class COMP_Optimizer:
    def __init__(self, network, primitives=None, weight_fn=None):
        self.network = network
        self.primitives = primitives or get_default_primitives()
        self.weight_fn = weight_fn or rule_based_weights
        self.context = OptimizationContext()

        # 통계 추적
        self.weight_history = []

    def step(self, X, y):
        """Single optimization step"""
        theta = self.network.get_weights()
        grad = self.network.gradient(X, y)
        loss = self.network.loss(X, y)
        grad_norm = np.linalg.norm(grad)

        # Context 업데이트
        self.context.update(theta, loss, grad_norm)

        # Weights 계산
        weights = self.weight_fn(self.context, len(self.primitives))
        self.weight_history.append(weights.copy())

        # Primitives 실행 및 조합
        deltas = []
        for primitive in self.primitives:
            delta = primitive(theta, grad, self.context)
            deltas.append(delta)

        # 가중 합산
        final_delta = sum(w * d for w, d in zip(weights, deltas))

        # 업데이트
        new_theta = theta + final_delta
        self.network.set_weights(new_theta)

        return loss, weights

    def train(self, X, y, max_iters=100, tolerance=1e-3, verbose=True):
        """Full training loop"""
        losses = []

        for i in range(max_iters):
            loss, weights = self.step(X, y)
            losses.append(loss)

            if verbose and i % 10 == 0:
                phase = self.context.phase
                success_rate = self.context.success_rate
                dominant = self.primitives[np.argmax(weights)].name

                print(f"[{i:3d}] Loss: {loss:.5f} | "
                      f"Phase: {phase:12s} | "
                      f"Success: {success_rate:.1%} | "
                      f"Dominant: {dominant}")

            # 수렴 체크
            if i > 10 and abs(losses[-1] - losses[-2]) < tolerance:
                if verbose:
                    print(f"\n✓ Converged at iteration {i}")
                break

        return {
            'final_loss': losses[-1],
            'losses': losses,
            'iterations': len(losses),
            'weight_history': self.weight_history
        }
```

---

## 테스트 전략

### Test 1: 기본 동작 확인
```python
# Linear, Nonlinear, XOR 각각 실행
# Loss 감소 확인
# 수렴 확인
```

### Test 2: Phase 전환 확인
```python
# Iteration에 따라 phase 변화 확인
# exploration → exploitation → refinement
```

### Test 3: Primitive 기여도 분석
```python
# Weight history 시각화
# 어느 phase에서 어떤 primitive 중요한지
```

### Test 4: SGD/QED/LAML-Q 비교
```python
# 4-way comparison
# 성능 + 해석가능성
```

---

## 시각화 계획

### Plot 1: Loss Curve
- 기본 학습 곡선
- SGD와 비교

### Plot 2: Weight Evolution
```
Stacked area chart:
  시간 → X축
  Weight → Y축
  각 primitive를 색깔로 구분

어느 시점에 어떤 전략이 지배적인지 한눈에
```

### Plot 3: Phase Transitions
```
Timeline:
  Exploration | Exploitation | Refinement
  각 phase의 시작/끝 표시
```

### Plot 4: Primitive Contribution
```
Pie chart or Bar chart:
  전체 학습 과정에서 각 primitive의 평균 기여도
```

---

## 예상 실험 결과

### Linear
- Exploration: Gradient + Momentum 지배
- Exploitation: Adaptive step 증가
- Refinement: Best direction 활용
- **예상 Loss**: 0.009 대

### Nonlinear
- Exploration: Stochastic jump 많이 활용
- Exploitation: Momentum 강화
- Refinement: Adaptive + Best
- **예상 Loss**: 0.09 대

### XOR
- Exploration: Stochastic 지배 (saddle 탈출)
- Exploitation: Gradient + Momentum
- Refinement: 모든 primitive 균형적 사용
- **예상 Loss**: 0.23 대

---

## 확장 가능성

### 단기 (이번 실험)
- 5개 primitives로 증명

### 중기 (추후)
- P6-P10 추가
- Learned weight function
- 더 복잡한 데이터셋

### 장기 (논문)
- User-defined primitives
- Transfer learning (다른 문제의 weight 재사용)
- Meta-COMP (primitives도 학습)

---

## Step 4 결론

### 구현 계획 완료
- ✅ 4개 파일 구조
- ✅ 각 파일의 상세 코드
- ✅ 테스트 전략
- ✅ 시각화 계획

### 다음 스텝
Step 5: 실제 구현
- 파일 생성
- 코드 작성
- 실행

---

**작성**: 2026-01-03 Step 4/6
**다음**: Step 5 - 구현
