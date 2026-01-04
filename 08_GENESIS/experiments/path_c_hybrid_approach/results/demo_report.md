# Hybrid Autopoietic-ML Experiment Results
**Dataset:** DEMO
**Trials:** 3
**Epochs:** 20
**Timestamp:** 2026-01-04T02:30:57.606552

## Ablation Study Results
| Condition | Accuracy | Std | Improvement | Time (s) |
|-----------|----------|-----|-------------|----------|
| Baseline | 95.02% | 0.11% | +0.00% | 60.0 |
| +Coherence | 95.58% | 0.25% | +0.56% | 65.0 |
| +Plasticity | 95.20% | 0.24% | +0.19% | 63.0 |
| +SelfOrg | 95.39% | 0.20% | +0.37% | 64.0 |
| +Coh+Plas | 95.94% | 0.27% | +0.93% | 68.0 |
| +Coh+Self | 95.51% | 0.14% | +0.50% | 67.0 |
| +All | 96.15% | 0.24% | +1.14% | 70.0 |

## Key Findings
1. **Best performing condition:** +All (96.15%, +1.14% over baseline)
2. **Coherence regularization impact:** +0.56%
3. **Self-organizing layers impact:** +0.37%
