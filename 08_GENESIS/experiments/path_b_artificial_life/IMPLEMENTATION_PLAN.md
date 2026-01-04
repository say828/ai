# Path B: 구현 계획 및 복잡도 분석

**Project**: GENESIS - Artificial Life Implementation Plan
**Date**: 2026-01-04
**Status**: Planning Phase

---

## 1. 컴포넌트 리스트

### 1.1 핵심 컴포넌트 (Core Components)

| ID | Component | Description | Dependencies |
|----|-----------|-------------|--------------|
| C1 | GridWorld | 64x64 토로이달 그리드 환경 | NumPy |
| C2 | ResourceSystem | 자원 재생/분포 시스템 | C1 |
| C3 | AutopoeticAgentV2 | 확장된 autopoietic agent | 기존 코드 |
| C4 | VisionSystem | Egocentric 시각 시스템 | C1 |
| C5 | ActionSystem | 연속 행동 공간 처리 | C1 |
| C6 | CoherenceModule | 조직적 일관성 평가 확장 | 기존 코드 |
| C7 | ReproductionSystem | 번식/변이 시스템 | C3, C6 |
| C8 | PopulationManager | 개체군 관리 | C3, C7 |

### 1.2 Baseline 컴포넌트

| ID | Component | Description | Dependencies |
|----|-----------|-------------|--------------|
| B1 | ICM_Agent | Intrinsic Curiosity Module RL | PyTorch, Stable-Baselines3 |
| B2 | ES_Agent | Evolution Strategies agent | NumPy |
| B3 | RandomAgent | 무작위 baseline | NumPy |
| B4 | BaselineWrapper | 환경 호환 래퍼 | C1, Gym |

### 1.3 측정/분석 컴포넌트

| ID | Component | Description | Dependencies |
|----|-----------|-------------|--------------|
| M1 | MetricsCollector | 지표 수집기 | Pandas |
| M2 | DiversityAnalyzer | 행동 다양성 분석 | SciPy, Scikit-learn |
| M3 | EmergenceDetector | 창발 탐지 | M2 |
| M4 | Visualizer | 시각화 도구 | Matplotlib, Plotly |
| M5 | ExperimentRunner | 실험 자동화 | Ray (optional) |

### 1.4 유틸리티 컴포넌트

| ID | Component | Description | Dependencies |
|----|-----------|-------------|--------------|
| U1 | ConfigManager | 설정 관리 | YAML |
| U2 | Logger | 로깅 시스템 | logging |
| U3 | Checkpointer | 체크포인트 저장/복원 | pickle |
| U4 | VideoRecorder | 시뮬레이션 비디오 녹화 | OpenCV (optional) |

---

## 2. 구현 시간 예상

### 2.1 개발자 프로필 가정

- **경험**: AI/ML 3년+, Python 숙련
- **하루 작업 시간**: 6시간 (실효 코딩)
- **테스트 포함**: 구현 시간의 30% 추가

### 2.2 컴포넌트별 시간 예상

#### Phase 1: 환경 구축 (1.5주)

| Component | Estimated Hours | Difficulty | Notes |
|-----------|-----------------|------------|-------|
| C1 GridWorld | 8h | Medium | 토로이달 경계, 다중 레이어 |
| C2 ResourceSystem | 6h | Medium | Logistic growth, diffusion |
| C4 VisionSystem | 4h | Low | NumPy 슬라이싱 |
| C5 ActionSystem | 3h | Low | 연속 행동 처리 |
| 테스트 | 6h | - | Unit tests |
| **소계** | **27h** | - | **~4.5일** |

#### Phase 2: Agent 확장 (2주)

| Component | Estimated Hours | Difficulty | Notes |
|-----------|-----------------|------------|-------|
| C3 AutopoeticAgentV2 | 16h | High | 기존 코드 확장 |
| C6 CoherenceModule | 8h | High | 새 coherence 지표 |
| C7 ReproductionSystem | 6h | Medium | 변이, 유전 |
| C8 PopulationManager | 8h | Medium | 죽음/탄생 관리 |
| 통합 테스트 | 12h | - | Integration tests |
| **소계** | **50h** | - | **~8.3일** |

#### Phase 3: Baselines (1.5주)

| Component | Estimated Hours | Difficulty | Notes |
|-----------|-----------------|------------|-------|
| B1 ICM_Agent | 16h | High | PyTorch, PPO+ICM |
| B2 ES_Agent | 8h | Medium | OpenAI ES 구현 |
| B3 RandomAgent | 2h | Low | 간단 |
| B4 BaselineWrapper | 6h | Medium | Gym 호환 |
| 테스트 | 10h | - | 각 baseline 검증 |
| **소계** | **42h** | - | **~7일** |

#### Phase 4: 측정/분석 (1주)

| Component | Estimated Hours | Difficulty | Notes |
|-----------|-----------------|------------|-------|
| M1 MetricsCollector | 6h | Low | Pandas 기반 |
| M2 DiversityAnalyzer | 8h | Medium | Clustering, distances |
| M3 EmergenceDetector | 12h | High | 창발 탐지 알고리즘 |
| M4 Visualizer | 8h | Medium | 다양한 플롯 |
| **소계** | **34h** | - | **~5.7일** |

#### Phase 5: 실험 인프라 (0.5주)

| Component | Estimated Hours | Difficulty | Notes |
|-----------|-----------------|------------|-------|
| M5 ExperimentRunner | 8h | Medium | 병렬화 (optional) |
| U1-U4 유틸리티 | 8h | Low | 설정, 로깅 등 |
| **소계** | **16h** | - | **~2.7일** |

#### Phase 6: 통합 및 디버깅 (1주)

| Task | Estimated Hours | Notes |
|------|-----------------|-------|
| 전체 통합 | 12h | 모든 컴포넌트 연결 |
| 버그 수정 | 18h | 예상치 못한 문제 |
| 성능 최적화 | 10h | 병목 해결 |
| **소계** | **40h** | **~6.7일** |

### 2.3 총 예상 시간

```
┌────────────────────────────────────────────┐
│           총 구현 시간 예상                  │
├────────────────────────────────────────────┤
│ Phase 1: 환경 구축         27h  (4.5일)    │
│ Phase 2: Agent 확장        50h  (8.3일)    │
│ Phase 3: Baselines         42h  (7일)      │
│ Phase 4: 측정/분석         34h  (5.7일)    │
│ Phase 5: 실험 인프라       16h  (2.7일)    │
│ Phase 6: 통합/디버깅       40h  (6.7일)    │
├────────────────────────────────────────────┤
│ 총계                       209h (34.8일)   │
│                                            │
│ 버퍼 (+30%)               62h  (10.3일)    │
├────────────────────────────────────────────┤
│ 최종 예상                  271h (45일)     │
│                           ≈ 9주 (풀타임)    │
└────────────────────────────────────────────┘
```

### 2.4 실험 실행 시간 (구현 후)

```
실험 실행 예상:
- 1 condition x 1 seed: 3 hours (CPU)
- 4 conditions x 10 seeds: 120 hours
- GPU 가속 시: ~12 hours
- 분석/시각화: ~8 hours

총 실험 시간: ~1주 (GPU) 또는 ~5주 (CPU only)
```

---

## 3. 기술적 도전 과제

### 3.1 높은 난이도 (High Difficulty)

#### 3.1.1 Coherence-Survival Mapping 튜닝

**문제**: Coherence가 실제로 생존에 의미있게 기여하도록 매핑하는 것

**도전**:
- 너무 약하면: coherence가 무관해짐
- 너무 강하면: coherence만으로 결정됨, 환경 영향 감소
- 적절한 비선형 매핑 필요

**예상 해결 시간**: 추가 20h (실험적 튜닝)

**접근법**:
```python
# 여러 매핑 함수 비교 실험
mappings = {
    'linear': lambda c: c,
    'sigmoid': lambda c: 1/(1+exp(-k*(c-threshold))),
    'polynomial': lambda c: c**2,
    'threshold': lambda c: 1 if c > threshold else 0
}
```

#### 3.1.2 Open-Ended Evolution 달성

**문제**: 실제로 open-ended evolution을 달성하는 것

**도전**:
- 대부분의 ALife 시스템은 수렴하거나 멸종
- 지속적 혁신(novelty)을 유지하기 어려움
- 복잡도가 증가하지 않을 수 있음

**예상 해결 시간**: 미지수 (연구 문제)

**접근법**:
- Quality-Diversity 압력 추가
- Novelty search 요소 도입
- 환경 복잡도 동적 증가

#### 3.1.3 창발 현상 탐지

**문제**: "창발"을 정의하고 자동으로 탐지하는 것

**도전**:
- 창발의 정의가 모호
- False positive/negative 가능
- 계산 비용이 높을 수 있음

**예상 해결 시간**: 추가 15h

**접근법**:
```python
# 창발 탐지 휴리스틱
def is_emergent(population_behavior, individual_behaviors):
    """
    전체 행동이 개체 행동의 합 이상인가?
    """
    predicted_collective = aggregate(individual_behaviors)
    actual_collective = population_behavior
    
    emergence_score = distance(actual_collective, predicted_collective)
    return emergence_score > THRESHOLD
```

### 3.2 중간 난이도 (Medium Difficulty)

#### 3.2.1 공정한 Baseline 비교

**문제**: Autopoietic agent와 RL/ES를 공정하게 비교하는 것

**도전**:
- 파라미터 수 맞추기
- 계산 예산 맞추기
- 학습 시간 vs 진화 시간

**접근법**:
- 동일한 네트워크 크기 강제
- Wall-clock time 대신 environment steps 사용
- 여러 비교 축 제시

#### 3.2.2 Scalability

**문제**: 큰 population에서 효율적 시뮬레이션

**도전**:
- O(n^2) 상호작용 비용
- 메모리 제한
- 시각화 병목

**접근법**:
- Spatial hashing으로 O(n) 근사
- 배치 처리
- 선택적 시각화

### 3.3 낮은 난이도 (Low Difficulty)

- GridWorld 구현: 잘 정립된 패턴
- Random baseline: 간단
- 기본 metrics: 표준 공식

---

## 4. 리스크 분석

### 4.1 기술적 리스크

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Population 멸종 | High | High | 환경 난이도 조절, 초기 개체 수 증가 |
| 수렴/정체 | High | Medium | Diversity pressure, novelty search |
| 계산 병목 | Medium | Medium | GPU 가속, 근사 알고리즘 |
| Coherence 무의미 | Medium | High | 신중한 매핑, ablation study |
| Baseline 승리 | Medium | Low | 이것도 유효한 결과 |

### 4.2 연구 리스크

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 의미있는 창발 없음 | High | High | 현실적 기대치, 정량적 분석 |
| 재현 불가 | Low | High | 철저한 seed 관리, 코드 공개 |
| 결과 해석 어려움 | Medium | Medium | 다양한 시각화, 질적 분석 |

### 4.3 프로젝트 리스크

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 일정 초과 | High | Medium | 30% 버퍼, 우선순위 조정 |
| 범위 확장 | Medium | High | 명확한 MVP 정의 |
| 동기 저하 | Low | High | 작은 마일스톤 설정 |

---

## 5. MVP 정의 (Minimum Viable Product)

### 5.1 필수 (Must Have)

- [ ] 64x64 GridWorld with resources
- [ ] AutopoeticAgentV2 (기존 코드 기반)
- [ ] 기본 coherence-survival mapping
- [ ] Reproduction system
- [ ] Random baseline
- [ ] 기본 metrics (survival time, population size)
- [ ] 기본 시각화

**MVP 예상 시간**: 120h (~20일)

### 5.2 권장 (Should Have)

- [ ] ICM-RL baseline
- [ ] ES baseline
- [ ] Behavioral diversity metrics
- [ ] 4가지 환경 조건
- [ ] 10 seeds 실험

**권장 예상 시간**: +80h (~13일)

### 5.3 선택 (Nice to Have)

- [ ] Emergence detector
- [ ] GPU 가속
- [ ] 비디오 녹화
- [ ] 병렬 실험 실행
- [ ] Interactive 시각화

---

## 6. 의존성 관리

### 6.1 필수 라이브러리

```python
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
pyyaml>=5.4.0
```

### 6.2 선택적 라이브러리

```python
# requirements-optional.txt
torch>=1.9.0           # RL baseline용
stable-baselines3>=1.3  # PPO 구현
ray>=1.9.0             # 병렬화
plotly>=5.3.0          # Interactive viz
opencv-python>=4.5.0   # 비디오
scikit-learn>=1.0.0    # 클러스터링
```

### 6.3 기존 코드 의존성

```
현재 GENESIS v2.0 코드:
├── autopoietic_entity.py     # 핵심 재사용
├── autopoietic_population.py # 부분 재사용
├── pure_viability_environment.py  # 참조
└── true_viability_entity.py  # 참조

재사용률: ~40%
수정 필요: ~60%
```

---

## 7. 마일스톤

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | 환경 구축 | Working GridWorld + Resources |
| 3-4 | Agent 확장 | AutopoeticAgentV2 동작 |
| 5-6 | Baselines | 3 baselines 동작 |
| 7 | Metrics | 수집/분석 파이프라인 |
| 8 | 통합 | 전체 시스템 동작 |
| 9 | 실험 실행 | 결과 데이터 |
| 10+ | 분석/논문 | 최종 보고서 |

---

## 8. 결론

### 8.1 총 예상 비용

```
구현: 9주 (1명 풀타임)
실험: 1-5주 (계산 자원에 따라)
분석: 2주

총: 12-16주 (3-4개월)
```

### 8.2 권장 사항

1. **MVP 우선**: 전체 시스템보다 MVP 먼저 완성
2. **점진적 확장**: 성공 확인 후 확장
3. **조기 실패 감지**: 2주차에 feasibility 체크
4. **유연한 계획**: 결과에 따라 방향 조정

### 8.3 다음 단계

이 구현 계획을 진행하기 전에:
1. `RELATED_WORK.md` 검토 (선행 연구)
2. `RECOMMENDATION.md` 검토 (최종 권장)
3. 팀/리소스 확인
4. Go/No-Go 결정

---

*This document is a planning artifact. Actual implementation may vary.*
