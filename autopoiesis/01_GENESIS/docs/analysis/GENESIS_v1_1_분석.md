# GENESIS v1.1 Phase 2 개선 분석

## 실험 개요

**목표**: v1.0의 3가지 문제점 해결
- 문제 1: Viability ≠ Performance (연결 부재)
- 문제 2: Integration 메커니즘 약함
- 문제 3: 과도한 Metamorphosis (거의 매 step마다 발생)

**실험 설정**:
- 동일한 regression 문제 (y = 2*x1 + 3*x2 + noise)
- 100 samples, 2 features
- 200 steps 실행
- Random seed=42 (재현성)

---

## Phase 2 개선사항

### 1. Viability ↔ Performance 연결 (40% 가중치)

```python
def assess_viability(self, environment, ecosystem):
    scores = []

    # 1. Environment feedback (v1.1: NEW! - 40% weight)
    if len(self.recent_feedback) > 0:
        env_feedback_score = np.mean(self.recent_feedback)
        scores.append(env_feedback_score)

    # 2. Success rate (30%)
    # 3. Growth trend (20%)
    # 4. Adaptability (10%)

    weights = [0.4, 0.3, 0.2, 0.1]
    viability = np.average(scores, weights=weights)
    return viability
```

**핵심**: 환경 피드백이 이제 viability의 40%를 차지

### 2. Hebbian-like Integration

```python
def integrate_experience(self, experience: Experience):
    success = experience.was_successful()

    if success:
        # "Neurons that fire together, wire together"
        for key in self.parameters:
            activity = np.abs(self.parameters[key])
            strength_update = 0.01 * activity * self.pathway_strengths[key]
            self.parameters[key] += strength_update

            self.pathway_strengths[key] *= 1.01  # 강화
            self.pathway_strengths[key] = np.clip(self.pathway_strengths[key], 0.5, 2.0)
    else:
        # Anti-Hebbian: 실패한 경로 약화
        for key in self.parameters:
            self.parameters[key] += np.random.randn(*self.parameters[key].shape) * 0.001
            self.pathway_strengths[key] *= 0.99  # 약화
```

**핵심**: 성공한 경로는 강화되고, 실패한 경로는 약화됨

### 3. Controlled Metamorphosis

```python
def should_metamorphose(self):
    # 1. 심각한 실패 (viability < 0.2)
    if self.viability < 0.2:
        return True

    # 2. 장기간 정체 (50 steps, std < 0.02)
    if len(self.viability_history) > 50:
        recent = self.viability_history[-50:]
        if np.std(recent) < 0.02:
            return True

    # 3. 무작위 탐색 (v1.0: 0.01 → v1.1: 0.005)
    if np.random.rand() < 0.005:
        return True

    return False
```

**핵심**: Threshold 0.01 → 0.005 (절반으로 감소)

---

## 실험 결과

### 정량적 비교

| Metric | v1.0 (baseline) | v1.1 (improved) | 변화 |
|--------|----------------|----------------|------|
| **Final Viability** | 0.175 | 0.250 | **+43.0%** ✅ |
| **Final Error** | 3.448 | 2.675 | **-22.4%** ✅ |
| **Learning Progress** | -56.7% | -17.5% | **+39.2%p** ✅ |
| **Metamorphosis Count** | 198/200 | 26/200 | **-86.9%** ✅ |

### 핵심 발견

#### ✅ **성공 1: Viability 대폭 개선**
- v1.0: 0.175 (매우 낮음)
- v1.1: 0.250 (+43%)
- **의미**: 환경 피드백 연결이 viability를 올바른 방향으로 유도

#### ✅ **성공 2: Prediction Error 감소**
- v1.0: 3.448
- v1.1: 2.675 (-22.4%)
- **의미**: v1.1이 실제로 더 나은 예측 성능 달성

#### ✅ **성공 3: Metamorphosis 제어 성공**
- v1.0: 198/200 steps (거의 매 step)
- v1.1: 26/200 steps (13%)
- **의미**: 구조적 변화가 전략적으로 발생

#### ⚠️ **한계: 여전히 negative learning**
- v1.0: -56.7% (error increased)
- v1.1: -17.5% (error increased, but less)
- **의미**: 개선되었지만 아직 positive learning 미달성

---

## 상세 분석

### Viability 궤적

```
v1.0: 대부분 0.15-0.18 범위 (낮고 불안정)
v1.1: 대부분 0.23-0.27 범위 (높고 안정적)
```

**해석**:
- 환경 피드백 40% 가중치가 viability를 실제 performance와 연결
- v1.1의 viability가 더 높고 안정적
- 이는 entity가 환경을 더 잘 이해하고 있음을 시사

### Metamorphosis 패턴

**v1.0 (198 metamorphoses)**:
```
Age 3-200: 거의 연속적인 metamorphosis
- 3, 4, 6, 7, 9, 10, 12, 13, 14, ...
- Module 추가 → 제거 → 추가 → 제거 (반복)
- 구조가 안정화되지 못함
```

**v1.1 (26 metamorphoses)**:
```
Age 30, 32, 39, 45, 47, 70, 84, 93, 103-107, 114, 117, 121, 128, 137, 142, 148, 150, 156, 159, 164, 174, 180, 183
- 초반 (30-47): 탐색
- 중반 (70-93): 조정
- 후반 (103-183): 드물게 발생
- 구조가 점진적으로 안정화
```

**해석**:
- v1.0: 구조적 혼란 → 학습 불가능
- v1.1: 전략적 진화 → 학습 가능성 증가

### Error 궤적 (smoothed)

**v1.0**:
```
Initial: 2.201
Final: 3.448
Trend: 지속적으로 증가 (학습 실패)
```

**v1.1**:
```
Initial: 2.276
Final: 2.675
Trend: 증가하지만 v1.0보다 완만
```

**해석**:
- v1.1이 v1.0보다 22.4% 더 나은 최종 error
- 여전히 positive learning은 아님
- Hebbian integration이 작동하고 있지만 충분하지 않음

---

## 왜 아직 Positive Learning이 안 되는가?

### 가설 1: Integration 강도 부족
```python
# 현재 v1.1
strength_update = 0.01 * activity * self.pathway_strengths[key]
```
- 0.01이 너무 작을 수 있음
- **시도해볼 것**: 0.05 또는 adaptive learning rate

### 가설 2: Environment Feedback 노이즈
```python
viability_contribution = np.exp(-error)
```
- Noise level 0.1이 feedback를 불안정하게 만들 수 있음
- **시도해볼 것**: Smoothed feedback (moving average)

### 가설 3: Exploration vs Exploitation 불균형
- 현재 metamorphosis가 여전히 발생 (26회)
- 이 중 일부가 학습을 방해할 수 있음
- **시도해볼 것**: Metamorphosis를 더 억제 (0.005 → 0.001)

### 가설 4: Network 구조 문제
```python
params[f'layer_{i}'] = np.random.randn(input_size, layer['size']) * 0.01
```
- 초기화 scale 0.01이 너무 작을 수 있음
- Layer 크기가 너무 작거나 클 수 있음
- **시도해볼 것**: Xavier/He initialization, 더 큰 layers

---

## Phase 2 개선사항 검증

| 개선사항 | 목표 | 결과 | 검증 |
|---------|------|------|------|
| Viability ↔ Performance | 연결 확립 | Viability +43%, Error -22% | ✅ **성공** |
| Hebbian Integration | 학습 메커니즘 | Learning progress +39%p | ✅ **부분 성공** |
| Controlled Metamorphosis | 안정성 확보 | 198 → 26 metamorphoses | ✅ **성공** |

**종합 평가**: Phase 2는 **명확한 개선**을 달성했으나 **positive learning은 미달성**

---

## Phase 3 방향성

### 옵션 A: Deeper Refinement (v1.2)
Phase 2 메커니즘을 더 정교화:
1. Hebbian learning rate 증가 (0.01 → 0.05)
2. Environment feedback smoothing (moving average)
3. Metamorphosis threshold 더 낮춤 (0.005 → 0.001)
4. Better initialization (Xavier/He)
5. Larger network capacity

**예상**: Positive learning 달성 가능성 높음

### 옵션 B: Ecosystem Experiment
단일 entity의 한계 → 집단 지능:
1. 여러 entities 동시 진화
2. Symbiotic interactions
3. Natural selection
4. Knowledge sharing

**예상**: 집단이 개체보다 나은 학습 가능성

### 옵션 C: Multi-task Learning
단일 regression → 다양한 환경:
1. 여러 regression 문제 동시 학습
2. Transfer learning 검증
3. Generalization 능력 측정

**예상**: Viability 개념의 보편성 검증

---

## 결론

### 달성한 것
1. ✅ Viability와 Performance를 성공적으로 연결
2. ✅ Hebbian-like integration 구현 및 검증
3. ✅ Metamorphosis 제어 (86.9% 감소)
4. ✅ v1.0 대비 모든 지표에서 개선

### 아직 해결 못한 것
1. ❌ Positive learning (error still increases)
2. ❌ Consistent improvement over time
3. ❌ Gradient descent와의 경쟁력

### 핵심 인사이트
> **"Viability는 이제 Performance와 연결되었다.
> Hebbian integration은 작동한다.
> 그러나 아직 학습이 충분히 강하지 않다."**

### 다음 스텝
Phase 3로 진행:
- **추천**: 옵션 A (v1.2 refinement)
- **이유**: 메커니즘이 작동하지만 강도가 부족한 상태
- **예상 결과**: Positive learning 달성 가능

---

**Generated**: 2026-01-03
**Experiment**: GENESIS v1.0 vs v1.1 Comparison
**Result**: v1.1 improves over v1.0, but positive learning not yet achieved
**Next**: Phase 3 - Deeper refinement or Ecosystem experiments
