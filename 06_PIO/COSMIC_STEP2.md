# COSMIC Optimization: Step 2 - 경로 적분 최적화 설계

## Feynman 경로 적분 복습

### 양자역학에서

**입자가 A에서 B로 이동하는 확률 진폭**:
```
K(B,A) = ∫ e^(iS[x(t)]/ℏ) Dx(t)

여기서:
- x(t): 한 경로
- S[x(t)]: 그 경로의 작용
- Dx(t): 모든 경로에 대한 적분
- ℏ: 플랑크 상수
```

**핵심 아이디어**:
1. 입자는 A→B의 **모든 가능한 경로를 동시에** 간다
2. 각 경로는 위상 e^(iS/ℏ)을 기여
3. 작용 S가 작은 경로들이 보강간섭
4. 작용 S가 큰 경로들은 상쇄간섭
5. 결과: 고전적 경로(최소 작용)가 지배적

### AI에 적용하려면?

**문제**:
- i(허수)와 ℏ를 어떻게 해석?
- 확률 진폭이 아니라 실제 업데이트 필요

**해결책**: Euclidean 경로 적분
```
Wick rotation: it → τ (허수 시간을 실수로)
e^(iS/ℏ) → e^(-S_E/T)

S_E: Euclidean action
T: 온도 (ℏ의 역할)
```

---

## AI 경로 적분 정식화

### 설정

**상태 공간**:
- 현재 가중치: θ_t
- 목표: 다음 가중치 θ_{t+1}

**경로**:
```
Δθ: θ_t에서 θ_{t+1}로의 "경로"
실제로는 벡터이지만, "경로"로 해석
```

**작용 (Action)**:
```
S[Δθ] = 운동 에너지 + 포텐셜 에너지

S[Δθ] = (1/2)||Δθ||² + λ·L(θ_t + Δθ)

여기서:
- ||Δθ||²: 너무 크게 움직이는 것 방지
- L(θ_t + Δθ): 새 위치의 손실
- λ: 균형 파라미터
```

### 경로 적분 업데이트

**아이디어**: 모든 가능한 업데이트 Δθ에 대해 적분

```
θ_{t+1} = θ_t + ⟨Δθ⟩

where:
⟨Δθ⟩ = (1/Z) ∫ Δθ · e^(-S[Δθ]/T) D(Δθ)

Z = ∫ e^(-S[Δθ]/T) D(Δθ)  (partition function)
```

**해석**:
- 모든 업데이트 Δθ를 고려
- 작용 S가 작을수록 큰 가중치
- 온도 T로 탐색/수렴 조절
- 가중 평균으로 최종 업데이트

---

## 실용적 구현: Monte Carlo

### 문제

무한 차원 적분을 어떻게 계산?

### 해결: Importance Sampling

**아이디어**:
- 모든 Δθ를 시도할 수 없음
- 중요한 Δθ들만 샘플링
- Monte Carlo로 근사

**방법 1: Metropolis-Hastings**
```python
# 현재 상태
current_update = Δθ_0
current_action = S[Δθ_0]

for i in range(N_samples):
    # 제안 (proposal)
    proposed = current_update + noise
    proposed_action = S[proposed]

    # 수락/거절 (Metropolis criterion)
    acceptance_prob = exp(-(proposed_action - current_action) / T)

    if random() < acceptance_prob:
        current_update = proposed
        current_action = proposed_action

    samples.append(current_update)

# 평균
final_update = mean(samples)
```

**방법 2: Langevin Dynamics**
```python
# 더 효율적: gradient를 활용

Δθ_i = Δθ_{i-1} - (dt/2)·∇S + sqrt(dt·T)·noise

여기서:
- ∇S: 작용의 기울기
- dt: 시간 스텝
- T: 온도
```

---

## 구체적 설계

### 알고리즘: PIO (Path Integral Optimizer)

```python
class PathIntegralOptimizer:
    """
    경로 적분 최적화

    철학:
    - 모든 업데이트를 중첩으로 고려
    - 작용이 작은 것이 자연스럽게 선택
    - 온도로 탐색/수렴 조절
    """

    def __init__(self, network, n_samples=20, temperature=0.1, lambda_loss=1.0):
        """
        Args:
            n_samples: Monte Carlo 샘플 수
            temperature: 온도 (탐색 강도)
            lambda_loss: 손실의 가중치
        """
        self.network = network
        self.n_samples = n_samples
        self.temperature = temperature
        self.lambda_loss = lambda_loss

    def compute_action(self, delta_theta, theta_current, X, y):
        """
        작용 계산

        S[Δθ] = (1/2)||Δθ||² + λ·L(θ + Δθ)
        """
        # 운동 항 (kinetic)
        kinetic = 0.5 * np.linalg.norm(delta_theta)**2

        # 포텐셜 항 (potential = loss)
        theta_new = theta_current + delta_theta
        self.network.set_weights(theta_new)
        loss = self.network.loss(X, y)
        potential = self.lambda_loss * loss

        action = kinetic + potential
        return action

    def sample_updates(self, theta_current, X, y):
        """
        Monte Carlo로 업데이트 샘플링

        Langevin dynamics 사용:
        더 효율적 (gradient 활용)
        """
        # 초기화
        delta = np.zeros_like(theta_current)
        samples = []
        weights = []

        # Gradient 계산 (한 번만)
        grad = self.network.gradient(X, y)

        # Langevin dynamics
        dt = 0.01
        n_steps = self.n_samples

        for step in range(n_steps):
            # 작용의 gradient
            grad_action = delta + self.lambda_loss * grad

            # Langevin update
            noise = np.random.randn(len(delta)) * np.sqrt(2 * dt * self.temperature)
            delta = delta - dt * grad_action + noise

            # 작용 계산
            action = self.compute_action(delta, theta_current, X, y)

            # Boltzmann weight
            weight = np.exp(-action / self.temperature)

            samples.append(delta.copy())
            weights.append(weight)

        return samples, weights

    def step(self, X, y):
        """
        한 번의 최적화 스텝
        """
        theta = self.network.get_weights()

        # Monte Carlo 샘플링
        samples, weights = self.sample_updates(theta, X, y)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 가중 평균
        final_update = sum(w * s for w, s in zip(weights, samples))

        # 적용
        theta_new = theta + final_update
        self.network.set_weights(theta_new)

        return self.network.loss(X, y), final_update
```

---

## 기존 방법들과의 연결

### SGD와의 관계

**SGD**:
```
θ_{t+1} = θ_t - lr·∇L
```

**PIO (low temperature, 1 sample)**:
```
θ_{t+1} = θ_t - ⟨Δθ⟩

T→0 극한에서:
⟨Δθ⟩ ≈ -∇S ≈ -∇L

→ SGD가 특수 케이스!
```

### LAML과의 관계

**LAML**: 끝점 예측 → 경로 계산 → 검증
**PIO**: 모든 경로 중첩 → 자동 선택

**차이**:
- LAML: "계획" (실패)
- PIO: "중첩" (자연스러움)

### QED와의 관계

**QED**: N개 입자 탐색
**PIO**: 무한개 "경로" 탐색 (샘플링)

**유사점**:
- 둘 다 "여러 가능성" 동시 고려
- QED는 discrete, PIO는 continuous

### COMP와의 관계

**COMP**: 여러 primitive 조합
**PIO**: 여러 업데이트 조합

**차이**:
- COMP: 사전 정의된 primitives
- PIO: 자연스럽게 생성

---

## 이론적 장점

### 1. 우주의 실제 작동 방식

- 자연이 실제로 사용하는 방법
- 이론적으로 완벽
- 수학적으로 아름다움

### 2. 자동 탐색/수렴 균형

```
T 크면:
- 넓은 탐색
- 많은 경로 고려
- 초기에 유용

T 작으면:
- 좁은 탐색
- 최소 작용 경로만
- 후기에 유용

T를 감소시키면 자동으로 탐색→수렴!
```

### 3. Local Minima 자연스럽게 탈출

- 온도가 높으면 언덕 넘기 가능
- Simulated annealing과 유사
- 하지만 더 원리적

### 4. 불확실성 정량화

```
분산(Δθ) = 불확실성
온도 높으면 분산 큼 = 불확실
온도 낮으면 분산 작음 = 확실
```

---

## 실용적 고려사항

### 장점

1. ✅ **원리적**: 우주의 법칙
2. ✅ **자동 조절**: 온도만 조절
3. ✅ **이론 보장**: 수렴 증명 가능
4. ✅ **일반적**: 모든 문제 적용

### 단점

1. ⚠️ **계산 비용**: N_samples × forward pass
2. ⚠️ **샘플링 품질**: MCMC 수렴 시간
3. ⚠️ **하이퍼파라미터**: T, λ, n_samples

### 최적화 방안

**방안 1: 적응적 샘플 수**
```python
if loss > 0.5:  # 초기
    n_samples = 5  # 적게
else:  # 후기
    n_samples = 20  # 많이
```

**방안 2: Gradient 재사용**
```python
# Gradient는 한 번만 계산
# 모든 샘플에 재사용
```

**방안 3: 병렬화**
```python
# N개 샘플 동시에
# GPU 활용
```

---

## 온도 스케줄

### Simulated Annealing

```python
T(t) = T_0 * decay^t

초기: T_0 = 1.0 (큰 탐색)
후기: T → 0 (수렴)
```

### Adaptive Temperature

```python
if loss improving:
    T *= 0.95  # 감소
else:
    T *= 1.05  # 증가 (갇혔을 수 있음)
```

### Phase-Based

```python
if iteration < 20:
    T = 0.5  # exploration
elif improvement_rate > 0.05:
    T = 0.2  # exploitation
else:
    T = 0.05  # refinement
```

---

## Step 2 결론

### 설계 완료

**알고리즘**: PIO (Path Integral Optimizer)

**핵심 요소**:
1. ✅ 작용 함수 S[Δθ]
2. ✅ Monte Carlo 샘플링
3. ✅ Langevin dynamics
4. ✅ 온도 스케줄

### 예상 성능

**이론적**:
- 최적 (T→0 극한에서 최소 작용)
- SGD를 일반화

**실용적**:
- 계산 비용 vs 성능 trade-off
- 샘플 수가 관건
- 온도 조절이 핵심

### 다음 스텝

Step 3에서:
1. 완전한 구현
2. 실험 및 테스트
3. 기존 방법들과 비교

---

**작성**: 2026-01-03 COSMIC Step 2/6
**다음**: Step 3 - 구현 및 실험
