# ULTRATHINK Step 2: 기존 가정 파괴하기

## 우리가 무의식적으로 받아들인 가정들

### 가정 1: "가중치 공간에서 움직여야 한다"

**기존**: θ → θ' (가중치 직접 업데이트)

**왜 이게 유일한 방법인가?**

대안:
- Gradient 공간에서 작업 후 적분
- Loss landscape 공간에서 작업
- 활성화 공간에서 작업
- **메타 공간**: 업데이트 규칙 자체를 업데이트

### 가정 2: "모든 스텝이 동등하다"

**기존**: 매 iteration마다 동일한 로직

**문제**: 학습 초기와 후기는 완전히 다른 상황
- 초기: 큰 탐색 필요
- 중기: 탐색 + 수렴
- 후기: 미세 조정

**대안**: **Phase-Aware Optimization**
각 단계마다 완전히 다른 전략 사용

### 가정 3: "한 번에 한 스텝만 생각한다"

**기존**: θ_t → θ_{t+1}

**문제**: 근시안적

**대안**: **Multi-Horizon Planning**
- 1 step: 즉각적 개선
- 5 steps: 중기 전략
- 20 steps: 장기 수렴

동시에 모두 고려!

### 가정 4: "손실 함수만 최적화하면 된다"

**기존**: min L(θ)

**문제**: 손실이 낮다고 학습이 좋은 건 아님
- Overfitting 가능
- Generalization 고려 안됨
- 학습 속도 고려 안됨

**대안**: **Multi-Objective Meta-Loss**
```
M(θ) = α·L(θ) + β·complexity(θ) + γ·stability(θ) + δ·speed(learning)
```

### 가정 5: "gradient를 따라가야 한다"

**기존**: 방향 = -∇L

**왜?** 그게 지역적 최선이니까

**하지만**: 지역적 최선 ≠ 전역적 최선

**대안**: **Gradient를 정보로만 사용**
- Gradient는 "하나의 의견"일 뿐
- 다른 의견들과 종합하여 결정

---

## 신경망 학습의 고유한 특성들

### 특성 1: 고차원성
- 파라미터가 수천~수백만 개
- 시각화 불가능
- 직관이 작동 안함

**활용법**: 차원 축소 or 부분 공간 활용

### 특성 2: 비볼록성
- 수많은 local minima
- Saddle points
- Plateau

**활용법**: 탈출 메커니즘 내장

### 특성 3: 계층 구조
- Input → Hidden → Output
- 각 레이어의 역할이 다름

**활용법**: 레이어별 다른 전략

### 특성 4: 과적합 가능성
- Train loss ↓, Test loss ↑

**활용법**: Regularization 내장

### 특성 5: 학습 궤적의 패턴
- 대부분의 학습은 비슷한 패턴
- 빠른 개선 → 느린 수렴

**활용법**: 패턴 학습하여 예측

---

## 완전히 새로운 관점들

### 관점 1: "Learning to Optimize"

**아이디어**: 최적화기 자체를 학습

```
Meta-Optimizer:
  Input: 현재 상태 (θ, ∇L, L, history)
  Output: 최선의 업데이트 Δθ

  이 Meta-Optimizer를 어떻게 학습?
  → 많은 학습 과정을 관찰하여 패턴 추출
```

**장점**: 데이터에서 직접 학습 전략 발견

**단점**: Meta-training 필요 (느림)

### 관점 2: "Optimization as Compression"

**아이디어**: 학습 = 데이터를 가중치로 압축하는 과정

```
Raw Data (X, y) → Compressed Representation (θ)

최적 압축 = 최소 정보 손실
```

**핵심**: 정보이론적 관점
- Entropy 최소화
- Mutual Information 최대화

### 관점 3: "Multi-Resolution Optimization"

**아이디어**: 여러 해상도에서 동시에 작업

```
Coarse (큰 스케일):
  - 전역 구조 파악
  - 큰 움직임

Fine (작은 스케일):
  - 지역 최적화
  - 미세 조정

Middle (중간):
  - 둘 사이 조율
```

웨이블릿 변환처럼!

### 관점 4: "Optimization as Inference"

**아이디어**: 최적 가중치를 "추론"

```
P(θ* | X, y, θ_current) = ?

Bayesian 추론으로 다음 상태 결정
```

**장점**: 불확실성 자연스럽게 포함

### 관점 5: "Compositional Optimization"

**아이디어**: 복잡한 업데이트를 단순한 요소들의 조합으로

```
Update = Σ w_i · Primitive_i

Primitives:
  - GradientDescent
  - Momentum
  - AdaptiveLR
  - NoiseInjection
  - ...

Weights w_i를 상황에 맞게 동적 조정
```

---

## 가장 유망한 방향

위 관점들을 평가:

| 관점 | 혁신성 | 구현 가능성 | 이론적 근거 | 종합 |
|------|--------|-------------|-------------|------|
| Learning to Optimize | ★★★★★ | ★★☆☆☆ | ★★★★☆ | 14/15 |
| Optimization as Compression | ★★★★☆ | ★★★☆☆ | ★★★★★ | 14/15 |
| Multi-Resolution | ★★★★☆ | ★★★★☆ | ★★★★☆ | 14/15 |
| Optimization as Inference | ★★★★☆ | ★★★☆☆ | ★★★★★ | 14/15 |
| Compositional | ★★★★★ | ★★★★★ | ★★★★☆ | 14/15 |

**선택**: **"Compositional Optimization"**

**이유**:
1. ✅ 구현 가능 (가장 높음)
2. ✅ 혁신적
3. ✅ 기존 방법들을 통합 가능
4. ✅ 확장 가능
5. ✅ 해석 가능

---

## Step 2 결론

### 파괴한 가정들
1. 가중치 공간만 사용
2. 모든 스텝 동등
3. 한 스텝만 생각
4. 손실만 최적화
5. Gradient를 따라감

### 발견한 새 관점
**Compositional Optimization**: 기본 업데이트 요소들을 상황에 맞게 동적으로 조합

### 다음 스텝
Step 3에서 구체화:
1. 어떤 Primitives?
2. 가중치를 어떻게 결정?
3. 상황을 어떻게 판단?
4. 학습을 어떻게?

---

**작성**: 2026-01-03 Step 2/6
**다음**: Step 3 - Compositional Optimization 구체화
