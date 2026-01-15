# Path B: Artificial Life 환경 설계 문서

**Project**: GENESIS - Autopoietic Agents in Open-Ended Evolution
**Date**: 2026-01-04
**Status**: Design Phase (No Implementation)

---

## 1. Executive Summary

이 문서는 autopoietic agents가 open-ended evolution 환경에서 보이는 행동을 분석하기 위한 실험 환경의 설계 명세를 정의합니다. 핵심 목표는 기존 GENESIS 코드(`autopoietic_entity.py`, `autopoietic_population.py`)의 철학을 확장하여 더 풍부한 ALife 환경에서 검증하는 것입니다.

---

## 2. 환경 규격 (Environment Specification)

### 2.1 Grid World 정의

```
┌─────────────────────────────────────────┐
│  Grid Parameters                        │
├─────────────────────────────────────────┤
│  Size: 64 x 64 (토로이달 경계)           │
│  Cell Type: 연속 값 (0.0 ~ 1.0)          │
│  Layers: 3 (resource, agent, pheromone) │
│  Update: 동기적 (모든 agent 동시 행동)    │
└─────────────────────────────────────────┘
```

**설계 근거**:
- 64x64: 계산 가능하면서도 충분한 복잡도 제공
- 토로이달: 경계 효과 제거, 공정한 공간
- 연속 값: 이산적 격자보다 풍부한 역학 허용

### 2.2 Resource Dynamics

```python
# 자원 재생 모델 (Logistic Growth)
resource[t+1] = resource[t] + r * resource[t] * (1 - resource[t]/K) - consumption

# Parameters
r: float = 0.1      # 재생률 (intrinsic growth rate)
K: float = 1.0      # 수용력 (carrying capacity)
diffusion: float = 0.05  # 확산 계수 (spatial spreading)
```

**자원 분포 타입**:

| Type | Description | Complexity |
|------|-------------|------------|
| Uniform | 균일 분포 | Low |
| Patchy | 패치 형태 클러스터 | Medium |
| Gradient | 공간적 기울기 | Medium |
| Dynamic | 시간에 따라 이동 | High |
| Adversarial | Agent 위치 피함 | Very High |

### 2.3 Predator Behavior (Optional)

```python
class PredatorSpec:
    """선택적 포식자 시스템"""
    
    n_predators: int = 0  # 기본: 없음 (확장 가능)
    
    # 포식자 행동 규칙
    movement: str = 'random'  # 'random', 'chase', 'patrol'
    vision_range: int = 5
    kill_range: int = 1
    
    # 난이도 조절
    spawn_rate: float = 0.0  # 스텝당 생성 확률
    despawn_after: int = 100  # 스텝 후 소멸
```

**설계 결정**: 초기 실험에서는 포식자 없이 진행. 자원 경쟁만으로도 충분한 선택압 제공.

### 2.4 Physics System

**Movement Model**:
```python
# 에너지 소모 기반 이동
movement_cost = base_cost + velocity * friction_coefficient

# Parameters
base_cost: float = 0.01      # 존재 비용 (매 스텝)
max_velocity: float = 1.0     # 최대 이동 속도 (cells/step)
friction: float = 0.1         # 이동 마찰 계수
```

**Collision Handling**:
- 같은 셀에 여러 agent 허용 (자원 경쟁)
- 물리적 충돌 없음 (계산 단순화)

**Vision Model**:
```python
class VisionSpec:
    """시각 시스템 정의"""
    
    type: str = 'egocentric'  # agent 중심 좌표계
    range: int = 5            # 시야 반경 (cells)
    channels: int = 3         # [resource, agent_density, pheromone]
    resolution: str = 'full'  # 'full' or 'coarse'
```

---

## 3. Agent 규격 (Agent Specification)

### 3.1 Sensory Input Format

```python
# Observation Space
observation = {
    'visual': np.ndarray,      # shape: (2*range+1, 2*range+1, channels)
    'internal': np.ndarray,    # shape: (n_internal_vars,)
    'proprioception': np.ndarray  # shape: (n_proprio,)
}

# Visual channels
# [0] resource_map: 자원 분포 (0~1)
# [1] agent_density: 다른 agent 밀도 (0~1)  
# [2] pheromone: 페로몬 농도 (0~1)

# Internal state variables
internal = [
    energy,          # 0~1, 정규화된 에너지
    coherence,       # 0~1, 조직적 일관성
    age_normalized,  # 0~1, 정규화된 나이
    reproduction_readiness  # 0~1, 번식 준비도
]

# Proprioception
proprio = [
    velocity_x,      # -1~1, 현재 속도
    velocity_y,      # -1~1
    heading          # -pi~pi, 방향
]
```

**Total observation size**: 
- Visual: 11 x 11 x 3 = 363 (range=5)
- Internal: 4
- Proprio: 3
- **Total: 370 dimensions**

### 3.2 Motor Output Format

```python
# Action Space (Continuous)
action = {
    'movement': np.ndarray,    # shape: (2,), [-1, 1] each
    'pheromone': float,        # [0, 1], 분비량
    'eat': float               # [0, 1], 섭취 강도
}

# Movement interpretation
delta_x = movement[0] * max_velocity
delta_y = movement[1] * max_velocity

# Eat interpretation  
resource_consumed = eat * eat_efficiency * available_resource
energy_gained = resource_consumed * conversion_efficiency
```

### 3.3 Coherence-to-Survival Mapping

기존 `autopoietic_entity.py`의 coherence 메커니즘을 확장:

```python
class CoherenceSurvivalMapping:
    """
    조직적 일관성 → 생존 능력 매핑
    
    핵심 원리 (Maturana & Varela):
    "A living system is organized as a network of processes
    of production of components that produces the network."
    """
    
    def compute_survival_probability(self, coherence: Dict) -> float:
        """
        일관성 점수를 생존 확률로 변환
        
        Args:
            coherence: {
                'predictability': float,  # 내부 역학 예측가능성
                'stability': float,       # 상태 안정성
                'complexity': float,      # 적절한 복잡도
                'circularity': float      # 순환 인과성
            }
        
        Returns:
            survival_prob: 이 스텝에서 생존할 확률
        """
        composite = (
            0.30 * coherence['predictability'] +
            0.30 * coherence['stability'] +
            0.20 * coherence['complexity'] +
            0.20 * coherence['circularity']
        )
        
        # Sigmoid mapping: coherence → survival probability
        # threshold=0.3에서 50% 생존
        survival_prob = 1.0 / (1.0 + np.exp(-10 * (composite - 0.3)))
        
        return survival_prob
    
    def should_die(self, coherence: Dict, energy: float) -> bool:
        """
        사망 조건 판정
        
        두 가지 사망 경로:
        1. 에너지 고갈: energy <= 0
        2. 조직 붕괴: coherence < threshold (확률적)
        """
        # 에너지 고갈
        if energy <= 0:
            return True
        
        # 조직 붕괴 (확률적)
        survival_prob = self.compute_survival_probability(coherence)
        return np.random.rand() > survival_prob
```

### 3.4 Reproduction Mechanism

```python
class ReproductionSpec:
    """번식 규격"""
    
    # 번식 조건
    min_energy: float = 0.7    # 최소 에너지 (정규화)
    min_age: int = 50          # 최소 나이 (steps)
    min_coherence: float = 0.6 # 최소 일관성
    
    # 번식 비용
    energy_cost: float = 0.3   # 에너지 비용
    
    # 변이
    mutation_rate: float = 0.1
    mutation_scale: float = 0.05  # 가중치 변이 크기
    
    # 유전 vs 학습
    inherited: List[str] = [
        'internal_dynamics.W',     # 내부 역학 구조
        'coherence_threshold',     # 생존 임계값
        'plasticity_rate'          # 가소성률
    ]
    
    not_inherited: List[str] = [
        'internal_dynamics.state', # 내부 상태
        'age',                     # 나이
        'energy'                   # 에너지 (초기값으로 리셋)
    ]
```

---

## 4. Baseline 설계 (Comparison Methods)

### 4.1 Baseline 1: RL Agent with Intrinsic Motivation

```python
class IntrinsicMotivationRL:
    """
    호기심 기반 RL (ICM-style)
    
    References:
    - Pathak et al. (2017) "Curiosity-driven Exploration"
    - Burda et al. (2018) "Large-Scale Study of Curiosity-Driven Learning"
    """
    
    # Architecture
    feature_encoder: CNN  # Visual features
    inverse_model: MLP    # Predicts action from s, s'
    forward_model: MLP    # Predicts s' from s, a
    policy: Actor-Critic  # PPO or A2C
    
    # Intrinsic reward
    intrinsic_reward = eta * ||f(s') - f_pred(s, a)||^2
    
    # Hyperparameters
    eta: float = 0.01       # Intrinsic reward scale
    lr: float = 0.0003      # Learning rate
    gamma: float = 0.99     # Discount factor
    
    # Extrinsic reward (survival)
    extrinsic_reward = {
        'energy_gain': +1.0 * delta_energy,
        'survival_bonus': +0.01 per step,
        'death_penalty': -10.0 on death
    }
```

**비교 포인트**:
- Autopoietic: coherence 유지가 목표, 외부 보상 없음
- ICM-RL: 호기심 + 생존 보상 최대화

### 4.2 Baseline 2: Evolution Strategies (ES)

```python
class EvolutionStrategies:
    """
    진화 전략 (OpenAI ES style)
    
    Reference:
    - Salimans et al. (2017) "Evolution Strategies as Scalable Alternative to RL"
    """
    
    # Population
    population_size: int = 50
    sigma: float = 0.02  # Noise std
    
    # Fitness function
    fitness = total_energy_collected + survival_time_bonus
    
    # Update
    theta = theta + alpha * (1/n*sigma) * sum(F_i * epsilon_i)
```

**비교 포인트**:
- Autopoietic: 개체 내 학습 (Hebbian/structural) + 진화
- ES: 진화만 (개체는 고정)

### 4.3 Baseline 3: Random Agent

```python
class RandomAgent:
    """제어 조건: 무작위 행동"""
    
    def act(self, observation):
        return {
            'movement': np.random.uniform(-1, 1, size=2),
            'pheromone': np.random.uniform(0, 1),
            'eat': np.random.uniform(0, 1)
        }
```

### 4.4 공정한 비교 방법

```python
class FairComparisonProtocol:
    """공정한 비교를 위한 프로토콜"""
    
    # 1. 동일한 환경
    environment_seed: int = 42  # 모든 방법에 동일
    n_episodes: int = 100       # 충분한 통계적 유의성
    
    # 2. 동일한 계산 예산
    total_environment_steps: int = 1_000_000
    # Autopoietic: 1M steps (population * steps/agent)
    # RL: 1M steps (training)
    # ES: 1M evaluations
    
    # 3. 동일한 아키텍처 용량
    parameter_budget: int = 50_000  # 모든 방법 동일 파라미터 수
    
    # 4. 여러 시드로 반복
    n_seeds: int = 10
    
    # 5. 통계 검정
    test_method: str = 'Mann-Whitney U'  # 비모수 검정
    significance_level: float = 0.05
```

---

## 5. Metrics 정의

### 5.1 정량적 지표 (Quantitative)

#### 5.1.1 생존 관련

| Metric | Formula | Description |
|--------|---------|-------------|
| **Survival Time** | `mean(lifespan)` | 평균 생존 시간 |
| **Population Size** | `len(alive_agents)` | 현재 생존 개체 수 |
| **Extinction Risk** | `P(population=0)` | 멸종 확률 |
| **Generation Count** | `max(generation)` | 도달한 최대 세대 |

#### 5.1.2 에너지/자원 관련

| Metric | Formula | Description |
|--------|---------|-------------|
| **Energy Efficiency** | `energy_gained / steps_alive` | 스텝당 에너지 획득 |
| **Foraging Success** | `eating_attempts > 0 / total_attempts` | 섭식 성공률 |
| **Energy Variance** | `std(energy_history)` | 에너지 안정성 |

#### 5.1.3 Coherence 관련 (Autopoietic 전용)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Mean Coherence** | `mean(coherence)` | 평균 일관성 |
| **Coherence Stability** | `1 / (1 + std(coherence))` | 일관성 안정성 |
| **Structural Changes** | `count(metamorphosis)` | 구조 변화 횟수 |

### 5.2 질적 지표 (Qualitative)

#### 5.2.1 Behavioral Diversity

```python
def compute_behavioral_diversity(population: List[Agent]) -> float:
    """
    행동 다양성 측정 (MAP-Elites style)
    
    Behavior Space:
    - movement_pattern: [mean_velocity, turn_frequency]
    - foraging_style: [exploration_ratio, exploitation_ratio]
    - social_behavior: [clustering_coefficient, pheromone_usage]
    """
    behaviors = [extract_behavior_descriptor(a) for a in population]
    
    # Pairwise distances
    distances = pdist(behaviors, metric='euclidean')
    
    # Diversity = mean pairwise distance
    return np.mean(distances)
```

#### 5.2.2 Emergence Detection

```python
class EmergenceDetector:
    """
    창발 현상 탐지
    
    탐지 대상:
    1. Division of labor (역할 분화)
    2. Spatial clustering (공간 군집)
    3. Temporal coordination (시간 조율)
    4. Communication patterns (의사소통)
    """
    
    def detect_division_of_labor(self, population):
        """역할 분화 탐지"""
        behaviors = [self.classify_behavior(a) for a in population]
        # Shannon entropy of behavior distribution
        return entropy(behavior_distribution)
    
    def detect_spatial_clustering(self, positions):
        """공간 군집 탐지"""
        # DBSCAN clustering
        clusters = DBSCAN(eps=3, min_samples=2).fit(positions)
        return len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
    
    def detect_temporal_coordination(self, action_histories):
        """시간 조율 탐지"""
        # Cross-correlation of actions
        correlations = []
        for i, j in combinations(range(len(action_histories)), 2):
            corr = np.correlate(action_histories[i], action_histories[j])
            correlations.append(np.max(corr))
        return np.mean(correlations)
```

#### 5.2.3 Open-Endedness Metrics

```python
class OpenEndednessMetrics:
    """
    Open-ended evolution 지표
    
    References:
    - Taylor et al. (2016) "Open-ended evolution: perspectives"
    - Packard et al. (2019) "Overview of metrics for OEE"
    """
    
    def novelty_rate(self, history: List[State]) -> float:
        """
        시간당 새로운 상태 발생률
        
        MODES metric의 일부
        """
        novel_states = self.count_novel_states(history)
        return novel_states / len(history)
    
    def complexity_growth(self, population_history) -> float:
        """
        복잡도 성장률
        
        Measure: Mean phenotype complexity over time
        """
        complexities = [self.measure_complexity(p) for p in population_history]
        # Linear regression slope
        slope, _, _, _, _ = linregress(range(len(complexities)), complexities)
        return slope
    
    def activity_rate(self, history) -> float:
        """
        진화적 활동률
        
        Evolutionary Activity Statistics의 일부
        """
        mutations = self.count_mutations(history)
        selections = self.count_selections(history)
        return (mutations + selections) / len(history)
```

### 5.3 측정 방법

```python
class MeasurementProtocol:
    """측정 프로토콜"""
    
    # 샘플링 빈도
    survival_metrics: every_step  # 매 스텝
    population_metrics: every_10_steps  # 10스텝마다
    behavioral_metrics: every_100_steps  # 100스텝마다
    emergence_metrics: every_1000_steps  # 1000스텝마다
    
    # 저장 형식
    format: str = 'parquet'  # 효율적 저장
    
    # 시각화
    real_time: bool = False  # 실시간 시각화 (선택적)
    save_video: bool = True  # 비디오 저장 (100 프레임마다)
```

---

## 6. 실험 설계

### 6.1 실험 조건

| Condition | Description | Difficulty |
|-----------|-------------|------------|
| **Easy** | Uniform resource, no predators | Low |
| **Medium** | Patchy resource, mild depletion | Medium |
| **Hard** | Dynamic resource, competition | High |
| **Adversarial** | Resource avoids agents | Very High |

### 6.2 주요 실험 질문

1. **Q1**: Autopoietic agent가 RL baseline보다 오래 생존하는가?
2. **Q2**: Coherence 유지가 실제로 생존에 기여하는가?
3. **Q3**: Open-ended evolution의 징후가 관찰되는가?
4. **Q4**: 예상치 못한 창발 현상이 있는가?

### 6.3 Ablation Studies

| Ablation | What's Removed | Purpose |
|----------|----------------|---------|
| No Coherence | Random coherence | Coherence 필요성 검증 |
| No Plasticity | Fixed structure | 구조 적응 필요성 검증 |
| No Reproduction | Immortal agents | 진화 필요성 검증 |
| No Pheromone | No communication | 의사소통 효과 검증 |

---

## 7. 기술적 고려사항

### 7.1 계산 요구사항

```
예상 계산 비용:
- 1 simulation step: ~10ms (CPU)
- 1M steps: ~3 hours
- 10 seeds x 4 conditions: ~120 hours (5일)
- GPU 가속 시: ~12 hours
```

### 7.2 병렬화 전략

```python
# Ray를 이용한 병렬화
@ray.remote
def run_experiment(seed, condition):
    env = create_environment(condition, seed)
    population = create_population()
    return simulate(env, population, n_steps=1_000_000)

# 병렬 실행
futures = [run_experiment.remote(s, c) for s in seeds for c in conditions]
results = ray.get(futures)
```

### 7.3 재현성

```python
# 재현성 보장
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

---

## 8. 예상 결과 및 가설

### 8.1 가설

**H1**: Autopoietic agents는 단기적으로 RL보다 낮은 성능을 보이지만, 장기적으로 더 안정적인 생존을 보일 것이다.

**H2**: 높은 coherence를 유지하는 agent가 더 오래 생존할 것이다.

**H3**: Population level에서 behavioral diversity가 창발할 것이다.

**H4**: Adversarial 조건에서 autopoietic 접근이 더 robust할 것이다.

### 8.2 예상 실패 모드

| Failure Mode | Cause | Mitigation |
|--------------|-------|------------|
| Population Extinction | 너무 가혹한 환경 | 난이도 조절 |
| Degenerate Behavior | 단순 행동 수렴 | Diversity pressure |
| Computational Bottleneck | 느린 시뮬레이션 | GPU 가속/근사 |
| Unfair Comparison | 다른 계산 예산 | 엄격한 예산 통제 |

---

## 9. 버전 관리 및 변경 이력

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-04 | Initial design document |

---

## 10. 참고 문헌

1. Maturana, H. R., & Varela, F. J. (1980). Autopoiesis and cognition.
2. Varela, F. J., Maturana, H. R., & Uribe, R. (1974). Autopoiesis.
3. Pathak, D., et al. (2017). Curiosity-driven exploration.
4. Taylor, T., et al. (2016). Open-ended evolution: perspectives.
5. Pugh, J. K., et al. (2016). Quality diversity.

---

*This document is a design specification only. No implementation has been performed.*
