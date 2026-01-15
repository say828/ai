# GENESIS Path B - Phase 0: Minimal Artificial Life Prototype

## Overview

Phase 0는 Path B의 핵심 메커니즘을 빠르게 검증하기 위한 최소 프로토타입입니다.

**목표 기간**: 2일  
**목표**: 핵심 메커니즘이 작동하는지 확인

## Core Questions

1. **일관성-생존 연결**: 일관성이 높은 에이전트가 실제로 더 오래 생존하는가?
2. **번식/선택**: 번식과 자연선택이 작동하는가?
3. **진화**: 개체군이 시간에 따라 진화하는가?

## Simplified Design

### Environment
- **Grid**: 16x16 토러스 (경계 연결)
- **Resources**: 각 셀 [0, 1] 범위
- **Growth**: 로지스틱 성장 `r_t+1 = r_t + growth_rate * r_t * (1 - r_t)`

### Agent
- **Position**: (x, y) on grid
- **Internal State**: 5 dimensions (RNN)
- **Sensor**: 8방향 + 현재 위치 = 9 positions, 2 features each = 18 dimensions
- **Action**: [dx, dy, consume] = 3 dimensions
- **Based on**: `v2.0/core/autopoietic_entity.py` (CoherenceAssessor)

### Population Dynamics
- **Initial**: 20 agents (random initialization)
- **Reproduction**: coherence > 0.6 AND age > 50 AND energy > 0.5
- **Death**: coherence < 0.3 OR energy < 0
- **Mutation**: Gaussian noise on child weights (std=0.1)

### Energy Model
```
energy_t+1 = energy_t + consumed_resources - base_cost - action_cost
base_cost = 0.01 (existence cost per step)
action_cost = 0.02 * |action| (movement cost)
```

## Files

| File | Description |
|------|-------------|
| `minimal_environment.py` | 16x16 토러스 그리드 환경 |
| `minimal_agent.py` | Autopoietic 에이전트 |
| `minimal_population.py` | 개체군 관리 |
| `phase0_experiment.py` | 메인 실험 스크립트 |
| `visualize_phase0.py` | 결과 시각화 |

## Running the Experiment

```bash
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_phase0
python phase0_experiment.py
```

## Success Criteria

1. [x] 코드가 오류 없이 실행됨
2. [x] 일관성과 생존 간 상관관계 관찰됨 (coherence-age correlation = 0.24)
3. [x] 개체군이 진화함 (avg 526 births, active selection)
4. [x] 실행 시간 < 2분 (1,000 steps in ~5 seconds)

## Key Design Decisions

### Why Coherence as Fitness?
- 외부 보상 없이 에이전트의 "품질" 측정
- 내부 역학의 예측가능성, 안정성, 복잡성 기반
- Autopoiesis 원칙: 자기 유지 = 생존

### Why Torus Grid?
- 경계 효과 제거
- 모든 위치가 동등한 기회
- 단순한 구현

### Why Simple RNN?
- Phase 0는 메커니즘 검증이 목표
- 복잡한 아키텍처는 Phase 1+에서

## Results Location

결과는 다음 위치에 저장됩니다:
```
/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_b_phase0/
├── phase0_results_TIMESTAMP.json
├── phase0_plots_TIMESTAMP.png
└── phase0_summary.md
```

## Next Steps (Phase 1)

Phase 0 성공 시:
- 더 복잡한 환경 (다중 자원, 장애물)
- 더 복잡한 에이전트 (더 큰 RNN, attention)
- 에이전트 간 상호작용
- 진화 분석 (계통수, 형질 변화)

---
Created: 2026-01-04
Project: GENESIS Path B
