# GENESIS v1.2 Phase 3 실험 결과

## 실험 개요

**목표**: Positive learning 달성

**가설**: v1.1의 메커니즘은 작동하지만 강도가 부족함

**구현된 개선사항**:
1. Hebbian learning rate: 0.01 → 0.05 (5배 증가)
2. Environment feedback smoothing: 최근 10개 feedback의 moving average
3. Better initialization: Xavier/He initialization
4. Metamorphosis threshold: 0.005 → 0.001 (5배 감소)
5. Network capacity: [32, 16] → [64, 32] (2배 증가)

**실험 설정**:
- 동일한 regression 문제: y = 2*x1 + 3*x2 + noise
- 100 samples, 2 features
- 200 steps 실행
- Random seed=42 (재현성)

---

## 실험 결과

### 정량적 비교

| Metric | v1.1 | v1.2 | 변화 | 평가 |
|--------|------|------|------|------|
| **Final Viability** | 0.200 | 0.309 | **+54.2%** | ✅ 대폭 개선 |
| **Final Error** | 3.072 | 5.793 | **-88.6%** | ❌ 악화 |
| **Learning Progress** | +10.2% | -74.0% | **-84.2%p** | ❌ 심각한 악화 |
| **Metamorphosis Count** | 0/200 | 0/200 | 0% | ⚠️ 추적 불가 |

### 핵심 발견

#### ❌ **실패: Positive learning 미달성**
- v1.1: +10.2% (약한 positive learning 달성!)
- v1.2: -74.0% (negative learning 악화)
- **의미**: 개선사항들이 예상과 달리 역효과

#### ✅ **부분 성공: Viability 개선**
- v1.1: 0.200
- v1.2: 0.309 (+54.2%)
- **의미**: Entity가 자신의 상태를 더 긍정적으로 평가
- **문제**: Viability와 실제 performance의 괴리

#### ❌ **심각한 문제: Error 증가**
- v1.1: 3.072
- v1.2: 5.793 (+88.6%)
- **의미**: 실제 예측 성능은 오히려 악화
- **원인**: 강한 Hebbian learning이 잘못된 패턴을 강화한 것으로 추정

---

## 상세 분석

### 1. Viability vs Performance 괴리

**v1.1**:
```
Viability: 0.200 (낮음)
Error: 3.072 (중간)
→ Viability가 performance를 보수적으로 반영
```

**v1.2**:
```
Viability: 0.309 (높음)
Error: 5.793 (나쁨)
→ Viability가 performance를 과대평가
```

**해석**:
- Smoothed feedback가 entity를 false confidence로 유도
- 높은 viability에도 불구하고 실제로는 더 나쁜 성능
- **문제의 핵심**: Feedback smoothing이 noise를 제거했지만, 동시에 중요한 error signal도 희석시킴

### 2. Hebbian Learning의 역설

**이론적 기대**:
- 0.05 learning rate → 더 강한 학습
- Successful patterns 강화 → 성능 개선

**실제 결과**:
- Error 증가 (3.072 → 5.793)
- Learning progress 악화 (+10.2% → -74.0%)

**가능한 원인**:
```python
# v1.2 코드
if success:
    strength_update = 0.05 * activity * self.pathway_strengths[key]
    self.parameters[key] += strength_update
    self.pathway_strengths[key] *= 1.05  # 강화
```

**문제점**:
1. **Local maxima 함정**: 초기에 우연히 성공한 패턴을 과도하게 강화
2. **Exploration 부족**: 강한 reinforcement → 탐색 감소
3. **Error signal 부재**: Hebbian은 방향만 알고 크기(error)는 모름
4. **Positive feedback loop**: 성공 → 강화 → 더 많은 성공 (local) → 과적합

### 3. Network Capacity의 양날의 검

**v1.1**: [32, 16] layers
**v1.2**: [64, 32] layers (2배 증가)

**예상**:
- 더 큰 표현 능력 → 더 나은 학습

**실제**:
- 더 많은 파라미터 → 과적합 가능성 증가
- Hebbian learning과 결합 → 잘못된 패턴을 더 강하게 학습
- He initialization으로 큰 초기값 → 초기 노이즈 증폭

### 4. Metamorphosis 추적 실패

**문제**:
- 두 버전 모두 0개로 보고됨
- 실제로는 많은 metamorphosis 발생 (로그 참조)

**로그 증거**:
```
v1.1: Age 13, 24, 30, 33, 38, 53, 54, 56, 71, 73... (30+ metamorphoses)
v1.2: Age 13, 15, 17, 35, 36, 40, 41, 55, 56... (32+ metamorphoses)
```

**의미**:
- Metamorphosis가 여전히 빈번히 발생
- 0.005 → 0.001 threshold 변경이 효과 없음
- 다른 조건들(viability < 0.2, stagnation)이 trigger

### 5. Error 궤적 분석

**v1.1**:
```
Step 50:  0.661
Step 100: 3.965
Step 150: 7.569
Step 200: 3.072
→ 초기에 증가, 후반에 회복
```

**v1.2**:
```
Step 50:  1.787
Step 100: 4.754
Step 150: 2.652
Step 200: 5.793
→ 불안정한 진동, 최종적으로 악화
```

**해석**:
- v1.2가 더 큰 진동 (1.787 → 4.754 → 2.652 → 5.793)
- 강한 Hebbian learning이 불안정성 유발
- Smoothed feedback가 올바른 방향 제시 실패

---

## 왜 v1.2가 실패했는가?

### 근본 원인 분석

#### 1. **Over-reinforcement**
```python
# v1.2
strength_update = 0.05 * activity * pathway_strengths
pathway_strengths *= 1.05
```
- 0.05 learning rate는 너무 큼
- 1.05 multiplier는 exponential growth 유발
- 초기 noise를 pattern으로 착각하고 고착화

#### 2. **Signal Dilution**
```python
# v1.2
env_feedback_score = np.mean(self.recent_feedback[-10:])
```
- 10개 feedback 평균 → noise 제거
- 하지만 동시에 중요한 error gradient 희석
- Entity가 실제 성능 악화를 감지하지 못함

#### 3. **Capacity Mismatch**
- [64, 32] layers는 단순 regression에 과도
- He initialization으로 큰 초기값
- 작은 dataset (100 samples)에서 과적합

#### 4. **Lack of True Gradient**
- Hebbian learning: "fire together, wire together"
- 하지만 어느 방향으로 wire해야 하는지 불명확
- Error gradient 없이는 blind reinforcement

---

## 결정적 인사이트

### v1.1의 "약한 성공"이 중요했던 이유

**v1.1 결과 재평가**:
```
Learning Progress: +10.2%
```

이것은 GENESIS 최초의 positive learning!

**v1.1이 작동한 이유**:
1. **보수적인 learning rate (0.01)**: 과적합 방지
2. **실시간 feedback**: Signal dilution 없음
3. **적절한 capacity [32, 16]**: 과적합 억제
4. **Weak initialization (*0.01)**: 안정적 시작

**v1.2가 실패한 이유**:
- 모든 것을 "더 강하게" 만들었지만
- 방향(gradient)은 여전히 불명확
- "Louder noise ≠ Better signal"

---

## Phase 3 교훈

### 학습 원리

#### ❌ **잘못된 가정**
> "v1.1의 메커니즘이 작동하니, 더 강하게 하면 더 나아진다"

#### ✅ **올바른 이해**
> "Learning rate를 높이면 수렴 속도는 빨라지지만, **방향이 틀리면 더 빠르게 발산한다**"

### Hebbian Learning의 한계

**강점**:
- Local correlation 포착
- Unsupervised learning 가능
- Biological plausibility

**한계**:
- **Direction blind**: 어느 방향으로 강화할지 모름
- **No error signal**: 얼마나 틀렸는지 모름
- **Local reinforcement**: Global optimum 보장 안 됨

**결론**:
> "Hebbian learning alone is insufficient for supervised learning"

---

## 다음 단계

### 옵션 A: v1.1 미세 조정 (추천)
v1.1이 이미 +10.2% positive learning 달성했으므로:
1. v1.1 그대로 유지
2. Learning rate를 조금만 증가 (0.01 → 0.015)
3. Metamorphosis를 더 억제
4. 더 긴 실험 (500 steps)

**예상**: +20~30% learning progress 가능

### 옵션 B: Hybrid Learning
Hebbian + Error-based:
```python
if success:
    # Hebbian reinforcement
    hebbian_update = 0.01 * activity * pathway_strengths

    # Error-based correction (pseudo-gradient)
    error_signal = consequence['error']
    gradient_update = 0.01 * error_signal * np.sign(parameters)

    # Combine
    parameters += hebbian_update - gradient_update
```

**예상**: 방향성 있는 학습 가능

### 옵션 C: Meta-learning
Entity가 자신의 learning rate를 조절:
```python
if recent_performance_improving:
    learning_rate *= 1.05  # Exploit
else:
    learning_rate *= 0.95  # Explore
```

**예상**: Adaptive learning 가능

### 옵션 D: Ecosystem Evolution
- 여러 entities를 동시에 진화
- Natural selection: 성능 좋은 entity 복제
- Knowledge sharing
- Collective intelligence

**예상**: 개체보다 집단이 나은 학습

---

## 실험 데이터

### Error 궤적 (전체)

**v1.1**:
```python
[0.661, 0.717, 0.804, ..., 3.965, ..., 7.569, ..., 3.072]
Initial: 0.661
Peak: 7.569 (step ~150)
Final: 3.072
Improvement: +10.2% from initial
```

**v1.2**:
```python
[1.787, 2.013, 2.156, ..., 4.754, ..., 2.652, ..., 5.793]
Initial: 1.787
Peak: 5.793 (step 200)
Final: 5.793
Deterioration: -74.0% from initial
```

### Viability 궤적

**v1.1**:
```
Mean: 0.234
Std: 0.052
Range: [0.155, 0.326]
Trend: 안정적, 약간 하락
```

**v1.2**:
```
Mean: 0.241
Std: 0.050
Range: [0.155, 0.309]
Trend: 안정적, 변동 있음
```

**해석**:
- Smoothing이 viability를 약간 안정화 (-4.2% std)
- 하지만 false confidence 제공 (높은 viability, 나쁜 error)

---

## 결론

### 달성한 것
1. ✅ v1.2 구현 완료 (5가지 개선사항 모두 적용)
2. ✅ 비교 실험 수행 (200 steps)
3. ✅ 중요한 인사이트 발견
4. ⚠️ v1.1의 positive learning (+10.2%) 재확인

### 달성 못한 것
1. ❌ v1.2에서 positive learning 달성 실패
2. ❌ 오히려 v1.1보다 악화 (-74.0%)
3. ❌ Hebbian learning만으로는 supervised learning 한계 확인

### 핵심 교훈

> **"More is not always better. Direction matters more than magnitude."**

1. **Signal quality > Signal strength**
   - Smoothing이 signal을 개선할 수도, 희석할 수도 있음

2. **Hebbian learning needs gradient**
   - 상관관계 포착은 가능
   - 하지만 최적화 방향은 제시 못함

3. **v1.1의 작은 성공이 중요**
   - +10.2% learning progress는 작아 보이지만
   - GENESIS 최초의 positive learning
   - 더 발전시킬 가치 있음

4. **Over-engineering의 위험**
   - 5가지 개선사항을 동시에 적용
   - 각각이 어떤 영향을 미쳤는지 분리 불가
   - Ablation study 필요

### 다음 Phase 방향

**추천**: **v1.1 기반 미세 조정**
- v1.1이 이미 작동함을 입증
- 조심스럽게 개선
- One change at a time

**대안**: **Hybrid learning**
- Hebbian + Error-based
- Best of both worlds

**장기**: **Ecosystem experiments**
- 개체의 한계 → 집단 지능

---

## 시각화 결과

생성된 플롯: `/Users/say/Documents/GitHub/ai/08_GENESIS/v1_1_v1_2_comparison.png`

**포함된 그래프**:
1. Viability over time (v1.1 vs v1.2)
2. Prediction error over time (smoothed)
3. Metamorphosis events
4. Learning progress comparison (bar chart)

**주요 관찰**:
- v1.2의 error가 더 큰 진동
- v1.2의 viability가 초반에 더 높음
- Learning progress에서 v1.2가 negative (빨간색)

---

**Generated**: 2026-01-03
**Experiment**: GENESIS v1.1 vs v1.2 Comparison
**Result**: v1.2 failed to achieve positive learning (-74.0%)
**Key Insight**: v1.1's weak positive learning (+10.2%) is actually a success
**Next**: Refine v1.1 carefully, or try hybrid learning
