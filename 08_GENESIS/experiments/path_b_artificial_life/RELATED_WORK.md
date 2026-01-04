# Path B: 선행 연구 조사

**Project**: GENESIS - Related Work Analysis
**Date**: 2026-01-04
**Status**: Literature Review

---

## 1. Artificial Life 분야 표준 벤치마크

### 1.1 Classical ALife Environments

#### 1.1.1 Tierra (Ray, 1991)

**개요**: 
- 자기 복제 프로그램들이 경쟁하는 가상 컴퓨터
- CPU 시간과 메모리를 놓고 경쟁
- 기생자, 면역, 공생 등 창발

**벤치마크 측면**:
- Diversity: 프로그램 종류의 수
- Novelty: 새로운 유전자 서열 출현
- Complexity: 프로그램 길이/기능

**한계**:
- 수렴 경향 (최적 복제자로 수렴)
- 제한된 "열린" 진화

**GENESIS 관련성**: 
- 자기 복제 vs 자기 생산(autopoiesis)
- Tierra는 복제, GENESIS는 조직 유지

#### 1.1.2 Avida (Ofria & Wilke, 2004)

**개요**:
- Tierra의 확장
- 디지털 유기체가 논리 연산을 수행하면 보상
- 복잡한 기능 진화 가능

**벤치마크 지표**:
- Task completion: 수행 가능한 논리 연산 수
- Fitness landscape analysis
- Evolutionary dynamics

**한계**:
- 여전히 외부 피트니스 함수
- 환경이 고정적

**GENESIS 관련성**:
- Avida는 외부 보상, GENESIS는 내재적 생존력

#### 1.1.3 Polyworld (Yaeger, 1994)

**개요**:
- 3D 환경의 신경망 기반 에이전트
- 먹이, 교배, 전투
- 최초의 "생태계" ALife

**벤치마크**:
- Population dynamics
- Neural network complexity
- Behavioral diversity

**한계**:
- 계산 비용 높음
- 재현성 이슈

**GENESIS 관련성**:
- 가장 유사한 선행 연구
- 차이점: Polyworld는 신경망 최적화, GENESIS는 autopoiesis

### 1.2 Modern ALife Platforms

#### 1.2.1 NEAT (Stanley & Miikkulainen, 2002)

**개요**:
- Neuroevolution of Augmenting Topologies
- 구조와 가중치 동시 진화

**벤치마크 태스크**:
- XOR, Double pole balancing
- Maze navigation
- Game playing (NERO)

**GENESIS 관련성**:
- NEAT: 진화로 구조 변경
- GENESIS: 개체 내 구조 변경(metamorphosis) + 진화

#### 1.2.2 Lenia (Chan, 2019)

**개요**:
- 연속 Game of Life
- 아름다운 자기 조직화 패턴
- "디지털 생명체"

**참조**: [Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity](https://www.researchgate.net/publication/381227259)

**벤치마크**:
- Pattern novelty
- Self-organization metrics
- Complexity measures

**GENESIS 관련성**:
- Lenia는 패턴, GENESIS는 행동하는 에이전트
- 둘 다 연속 공간 사용

#### 1.2.3 Chromaria / Flow Lenia

**개요**:
- Lenia 확장
- 다중 채널, 화학적 상호작용
- 더 복잡한 창발

**GENESIS 관련성**:
- 화학적 상호작용 vs 내부 역학

---

## 2. Open-Ended Evolution 지표

### 2.1 표준 지표 프레임워크

#### 2.1.1 Evolutionary Activity Statistics (Bedau & Packard, 1992)

**세 가지 핵심 통계**:

```python
# 1. Activity: 진화적 변화 빈도
activity = sum(new_genotypes_per_timestep)

# 2. Diversity: 현재 존재하는 유형의 수
diversity = len(unique_genotypes)

# 3. Novelty: 새로운 유형 출현률
novelty = new_genotypes / total_genotypes
```

**해석**:
- 높은 Activity + 높은 Diversity = Open-ended 가능성
- 낮은 Activity + 낮은 Diversity = 수렴/정체

#### 2.1.2 MODES Metrics (Dolson et al., 2019)

**확장된 프레임워크**:

| Metric | Description | Formula |
|--------|-------------|---------|
| **M**easure | 측정 대상 정의 | phenotype/genotype/behavior |
| **O**rdering | 시간 순서 | temporal ordering |
| **D**ifference | 차이 측정 | distance function |
| **E**valuation | 평가 방법 | cumulative/rate/final |
| **S**election | 선택 대상 | what counts as "alive" |

**구현**:
```python
def modes_complexity(history):
    """MODES 복잡도 측정"""
    phenotypes = [extract_phenotype(h) for h in history]
    
    # 누적 복잡도
    cumulative = sum(complexity(p) for p in phenotypes)
    
    # 복잡도 성장률
    growth_rate = np.polyfit(range(len(phenotypes)), 
                             [complexity(p) for p in phenotypes], 1)[0]
    
    return cumulative, growth_rate
```

#### 2.1.3 ONEBench (2025)

**최신 접근**:
- 참조: [ONEBench to Test Them All](https://www.researchgate.net/publication/394271216)
- Sample-level benchmarking
- Open-ended capabilities 측정
- Foundation model 기반 novelty 측정

### 2.2 Quality-Diversity 지표

#### 2.2.1 Coverage

```python
def coverage(archive, behavior_space):
    """
    행동 공간에서 커버된 영역 비율
    
    Reference: MAP-Elites (Mouret & Clune, 2015)
    """
    occupied_cells = len([c for c in archive if c.is_occupied])
    total_cells = behavior_space.total_cells
    return occupied_cells / total_cells
```

#### 2.2.2 QD-Score

```python
def qd_score(archive):
    """
    Quality-Diversity 점수
    
    = Sum of fitness in all occupied cells
    """
    return sum(cell.fitness for cell in archive if cell.is_occupied)
```

### 2.3 최신 연구 방향

**Foundation Model 기반 측정** (2024-2025):
- 참조: [Automating the Search for Artificial Life with Foundation Models](https://arxiv.org/html/2412.17799v2)
- LLM/VLM으로 novelty 자동 평가
- 인간 판단과의 상관관계 측정

**주요 이점**:
- 사전 정의된 metrics 없이 novelty 평가
- 의미적(semantic) novelty 포착

**한계**:
- 계산 비용
- Foundation model의 편향

---

## 3. 기존 Autopoietic ALife 연구

### 3.1 초기 연구 (1970s-1990s)

#### 3.1.1 Varela, Maturana & Uribe (1974)

**최초의 계산적 autopoiesis**:
- 참조: [30 Years of Computational Autopoiesis](https://www.researchgate.net/publication/242657349)
- 2D 셀룰러 오토마타
- 자기 생산하는 "막(membrane)"

**핵심 통찰**:
```
Autopoietic system은:
1. 자신의 경계를 생산
2. 자신의 구성요소를 생산
3. 순환적 인과성
```

**한계**:
- 매우 단순한 시스템
- 행동 없음 (정적)

#### 3.1.2 McMullin (2004) Review

**30년 리뷰**:
- 참조: [Thirty years of computational autopoiesis: a review](https://dl.acm.org/doi/10.1162/1064546041255548)
- 주요 발전 정리
- 미해결 문제 식별

**미해결 문제 (2004 기준)**:
1. Autopoiesis와 cognition의 연결
2. 더 복잡한 autopoietic 시스템
3. 다중 autopoietic 개체 상호작용

**GENESIS 관련성**: 이 문제들을 직접 다룸

### 3.2 현대 연구 (2010s-2020s)

#### 3.2.1 Game of Life에서의 Autopoiesis

**Beer (2015)**:
- 참조: [Characterizing autopoiesis in the game of life](https://dl.acm.org/doi/abs/10.1162/ARTL_a_00143)
- GoL에서 autopoietic 패턴 식별
- Glider, oscillator 등 분석

**핵심 발견**:
- 일부 GoL 패턴은 autopoietic 특성 가짐
- 경계 유지 + 자기 생산

**한계**:
- GoL은 매우 제한적
- 학습/적응 없음

#### 3.2.2 Froese & Ziemke (2009)

**Enactive AI 관점**:
- Autopoiesis + Embodied cognition
- AI는 "살아있어야" 한다

**핵심 주장**:
> "진정한 인지를 위해서는 진정한 생명이 필요하다"

**GENESIS 관련성**: 철학적 토대 공유

#### 3.2.3 Virgo et al. (2013)

**Autopoiesis와 Agency**:
- 자율성(autonomy)의 조작적 정의
- 계산적 측정 가능성

**주요 기여**:
```python
# Autonomy 측정 (근사)
autonomy = mutual_information(internal_states, environment) / 
           entropy(internal_states)
```

### 3.3 2023-2025 최신 연구

#### 3.3.1 Evolution in Lenia (2024)

**참조**: [Evolution of Autopoiesis and Multicellularity in the Game of Life](https://direct.mit.edu/artl/article/27/1/26/101060)

**발견**:
- GoL 확장에서 autopoietic 패턴 진화
- 다세포 구조 창발

#### 3.3.2 Foundation Models for ALife (2024-2025)

**참조**: [Automating the Search for Artificial Life with Foundation Models](https://arxiv.org/html/2412.17799v2)

**접근**:
- GPT/Claude로 ALife 시스템 분석
- 창발 자동 탐지

**GENESIS 관련성**: 분석 도구로 활용 가능

---

## 4. 우리 접근의 Novelty

### 4.1 기존 연구와의 비교

| Aspect | 기존 연구 | GENESIS |
|--------|----------|---------|
| **주체** | 패턴/프로그램 | 행동하는 Agent |
| **학습** | 없음 또는 진화만 | 개체 내 학습 + 진화 |
| **Coherence** | 암시적/없음 | 명시적 측정 |
| **환경** | 단순/고정 | 복잡/동적 |
| **Baseline** | 없음 | RL/ES 비교 |
| **목표** | 이론적 탐구 | 실용적 AI |

### 4.2 구체적 Novelty

#### 4.2.1 Coherence-Driven Survival

**기존**: 에너지 또는 fitness로 생존 결정
**GENESIS**: 조직적 일관성(coherence)으로 생존 결정

```python
# 기존 접근
if energy <= 0:
    die()

# GENESIS 접근
if coherence < threshold:  # 조직 붕괴
    die()
elif energy <= 0:          # 자원 고갈
    die()
```

**의의**: "삶"의 정의를 재고

#### 4.2.2 내부 역학 기반 행동

**기존**: 센서 → 액션 직접 매핑
**GENESIS**: 센서 → 내부 역학 교란 → 액션

```python
# 기존
action = network(observation)

# GENESIS
internal_state = internal_dynamics.step(observation_as_perturbation)
action = internal_state[:action_dim]  # 자연스러운 표현
```

**의의**: 행동이 "계산된" 것이 아니라 "표현된" 것

#### 4.2.3 다중 스케일 학습

**기존**: 진화만 또는 개체 학습만
**GENESIS**: 
- 개체 내 가소성 (Hebbian-like)
- 구조적 변화 (Metamorphosis)
- 개체군 진화 (Selection + Mutation)

**의의**: 생물학적 학습의 다중 스케일 반영

### 4.3 이론적 기여

1. **Autopoiesis의 조작화**: 
   - 추상적 개념을 측정 가능한 지표로
   - Coherence = predictability + stability + complexity + circularity

2. **Viability vs Optimality**: 
   - 최적화가 아닌 생존 가능성
   - "Good enough" vs "Best"

3. **Open-ended Evolution + Learning**:
   - 진화와 학습의 상호작용
   - 기존 연구에서 드문 조합

### 4.4 실험적 기여

1. **체계적 비교**: RL, ES, Random과 공정 비교
2. **다양한 환경 조건**: Easy → Adversarial
3. **다차원 지표**: 생존, 다양성, 창발

---

## 5. 관련 분야 연결

### 5.1 Reinforcement Learning

**관련 연구**:
- Intrinsic Motivation (Schmidhuber, 1991; Oudeyer et al., 2007)
- Curiosity-Driven Exploration (Pathak et al., 2017)
- 참조: [Curiosity-driven exploration based on hierarchical vision transformer](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009245)

**GENESIS 차이**:
- RL: 보상 최대화 (외부 또는 내재적)
- GENESIS: 조직 유지 (생존력)

### 5.2 Evolutionary Computation

**관련 연구**:
- Novelty Search (Lehman & Stanley, 2011)
- 참조: [Evolution through the Search for Novelty Alone](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf)
- Quality-Diversity (Pugh et al., 2016)
- 참조: [Quality Diversity: A New Frontier](https://www.frontiersin.org/articles/10.3389/frobt.2016.00040/full)

**GENESIS 차이**:
- QD: 다양성을 명시적 목표로
- GENESIS: 다양성이 자연스럽게 창발

### 5.3 Developmental Systems

**관련 연구**:
- Developmental Encoding (Stanley, 2007)
- Morphogenesis 기반 AI (Etcheverry et al., 2020)

**GENESIS 차이**:
- 발달: 유전자 → 표현형 (한 방향)
- GENESIS: 지속적 자기 생산 (순환)

---

## 6. 연구 격차 (Research Gaps)

### 6.1 식별된 격차

1. **Autopoiesis + Learning + Evolution의 통합**
   - 기존 연구: 개별적으로 다룸
   - 격차: 세 가지 통합한 시스템 부재

2. **정량적 Coherence 측정**
   - 기존 연구: 질적 분석
   - 격차: 조작화된 측정 방법 부재

3. **Open-ended Evolution 달성 조건**
   - 기존 연구: 경험적/산발적
   - 격차: 체계적 이해 부족

4. **Autopoietic AI와 RL/ES 비교**
   - 기존 연구: 없음
   - 격차: 공정한 비교 연구 부재

### 6.2 GENESIS가 채우는 격차

| Gap | GENESIS Contribution |
|-----|---------------------|
| 통합 | 세 가지 통합 시스템 구현 |
| 측정 | Coherence metrics 정의 |
| OEE | 다양한 조건에서 실험 |
| 비교 | 체계적 baseline 비교 |

---

## 7. 참고 문헌

### 7.1 Autopoiesis 핵심

1. Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and cognition*.
2. Varela, F. J., et al. (1974). *Autopoiesis: The organization of living systems*.
3. McMullin, B. (2004). *Thirty years of computational autopoiesis: a review*.
4. Beer, R. D. (2015). *Characterizing autopoiesis in the game of life*.

### 7.2 Artificial Life

5. Ray, T. S. (1991). *An approach to the synthesis of life*.
6. Yaeger, L. (1994). *Computational genetics, physiology, metabolism, neural systems*.
7. Ofria, C., & Wilke, C. O. (2004). *Avida: A software platform for research*.
8. Chan, B. (2019). *Lenia: Biology of Artificial Life*.

### 7.3 Open-Ended Evolution

9. Bedau, M. A., & Packard, N. H. (1992). *Measurement of evolutionary activity*.
10. Taylor, T., et al. (2016). *Open-ended evolution: Perspectives*.
11. Dolson, E., et al. (2019). *MODES: A unified framework*.
12. Stanley, K. O. (2019). *Why open-endedness matters*.

### 7.4 Quality-Diversity

13. Mouret, J. B., & Clune, J. (2015). *Illuminating search spaces by mapping elites*.
14. Pugh, J. K., et al. (2016). *Quality diversity: A new frontier*.
15. Lehman, J., & Stanley, K. O. (2011). *Abandoning objectives*.

### 7.5 Intrinsic Motivation

16. Schmidhuber, J. (1991). *A possibility for implementing curiosity*.
17. Oudeyer, P. Y., et al. (2007). *Intrinsic motivation systems*.
18. Pathak, D., et al. (2017). *Curiosity-driven exploration*.

---

## 8. 소스 링크

- [Open-Ended Evolution - Artificial Life Encyclopedia](https://alife.org/encyclopedia/introduction/open-ended-evolution/)
- [Quality-Diversity Papers List](https://quality-diversity.github.io/papers.html)
- [The Future of AI is Open-Ended (2025 Blog)](https://richardcsuwandi.github.io/blog/2025/open-endedness/)
- [Darwin Godel Machine (2025)](https://arxiv.org/abs/2505.22954)
- [LLM-Driven Intrinsic Motivation (2025)](https://arxiv.org/html/2508.18420)

---

*This document is a literature review. Citations should be verified before publication.*
