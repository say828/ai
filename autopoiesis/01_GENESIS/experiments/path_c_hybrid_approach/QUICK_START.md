# Path C: Hybrid Approach - Quick Start Guide

## Overview

이 실험은 Autopoietic 원리(Coherence Regularization)를 표준 ML에 추가하여 성능 향상을 검증합니다.

## 파일 구조

```
path_c_hybrid_approach/
├── baseline_model.py        # 비교용 표준 PyTorch 모델
├── coherence_regularizer.py # 4D Coherence Metric 구현
├── hybrid_model.py          # Coherence Regularization 적용 모델
├── quick_experiment.py      # 빠른 검증 실험 스크립트
└── results/                 # 상세 실험 결과
```

## Quick Start

### 1. 환경 설정
```bash
source /Users/say/Documents/GitHub/ai/08_GENESIS/venv/bin/activate
```

### 2. 빠른 실험 실행 (약 20분)
```bash
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_c_hybrid_approach
python quick_experiment.py
```

### 3. 결과 확인
- JSON: `/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_c/initial_results.json`
- 시각화: `/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_c/initial_results.png`

## 실험 구성

| Condition | Coherence Reg | 설명 |
|-----------|---------------|------|
| Baseline | No | 표준 MLP (512-256-128) |
| +Coherence | Yes (lambda=0.01) | Coherence loss 추가 |

## 초기 결과 (N=3 trials, 10 epochs)

| Condition | Test Accuracy | Loss Variance |
|-----------|--------------|---------------|
| Baseline | 98.58% +/- 0.10% | 0.000679 |
| +Coherence | 98.54% +/- 0.05% | 0.000681 |

### 분석

1. **정확도**: Baseline과 +Coherence가 거의 동등한 성능 (차이 < 0.1%)
2. **안정성**: +Coherence가 더 낮은 std (0.05% vs 0.10%), 학습이 더 일관됨
3. **Coherence 값**: ~0.86으로 안정적으로 유지됨

### 해석

현재 설정에서 Coherence Regularization은:
- MNIST에서는 눈에 띄는 정확도 향상을 보이지 않음
- 그러나 학습 분산이 줄어드는 경향 (더 안정적인 학습)
- 더 어려운 태스크(CIFAR-10, noisy data)에서 효과가 클 수 있음

## 다음 단계

1. **더 긴 학습**: 20-30 epochs로 수렴 후 성능 비교
2. **Hyperparameter 튜닝**: coherence_weight 조정 (0.001, 0.1 등)
3. **어려운 데이터셋**: CIFAR-10, noisy MNIST에서 테스트
4. **Robustness 테스트**: adversarial examples, domain shift 등

## 4D Coherence Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| Predictability | 0.3 | 이전 활성화로부터 현재 예측 가능성 |
| Stability | 0.3 | 활성화의 시간적 일관성 |
| Complexity | 0.2 | 정보 복잡도 (엔트로피 기반) |
| Circularity | 0.2 | 레이어 간 자기참조 일관성 |

## 참고

- 전체 ablation study: `python experiment.py --dataset mnist --n_trials 5 --n_epochs 20`
- Robustness 평가: `python robustness.py --dataset mnist`
