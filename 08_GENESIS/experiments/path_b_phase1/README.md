# GENESIS Path B Phase 1: Full Artificial Life System

## Overview

Phase 1 is a complete implementation of the autopoietic artificial life system, building on the validated mechanisms from Phase 0. This represents a publication-ready system with comprehensive metrics and baseline comparisons.

## Key Differences from Phase 0

| Component | Phase 0 | Phase 1 |
|-----------|---------|---------|
| Grid Size | 16x16 | 64x64 |
| Population | 20 | 100 |
| Simulation Steps | 1,000 | 10,000 |
| Sensor Dimension | 18 | 370 |
| Internal State | 5 | 128 |
| Resource Types | 1 (energy) | 2 (energy + material) |
| Interactions | None | Spatial awareness |
| Metrics | Basic stats | QD + Phylogeny |
| Baselines | None | Random, Fixed, RL |

## Architecture

### Environment (`full_environment.py`)
- 64x64 toroidal grid
- Two resource types:
  - **Energy**: Fast regeneration (growth rate 0.15)
  - **Material**: Slow regeneration (growth rate 0.05)
- Spatial heterogeneity via resource patches
- Visual field sensing (7x7 grid)

### Agent (`full_agent.py`)
- **Sensor**: 370 dimensions
  - Visual field: 147 (7x7x3)
  - Proprioception: 32
  - Memory: 128
  - Gradient: 4
  - Temporal: 59
- **Internal State**: 128-dim RNN
- **Action**: 5-dim continuous
  - dx, dy (movement)
  - consume_energy, consume_material
  - reproduction signal
- **Coherence**: 4D metric
  - Predictability
  - Stability
  - Complexity
  - Circularity

### Population (`full_population.py`)
- Spatial hashing for efficient neighbor queries
- Phylogenetic tree tracking
- Quality-Diversity archive
- Comprehensive death/birth logging

### Baselines (`baselines.py`)
1. **RandomAgent**: Uniform random actions
2. **FixedPolicyAgent**: Greedy resource-seeking
3. **RLAgent**: Policy gradient with energy reward

## Usage

### Quick Test (5 minutes)
```bash
python phase1_experiment.py --quick
```

### Full Experiment (30-60 minutes)
```bash
python phase1_experiment.py --steps 10000 --trials 3
```

### Autopoietic Only (no baselines)
```bash
python phase1_experiment.py --steps 10000 --trials 3 --no-baselines
```

### Visualization
```bash
python visualize_phase1.py --latest
```

## Expected Results

### Success Criteria
- Population survives 10,000 steps (no extinction)
- QD Coverage > 50 cells
- Coherence-Age Correlation > 0.2
- Autopoietic outperforms baselines

### Key Metrics
1. **Population Dynamics**: Stable population with active reproduction
2. **Coherence Evolution**: Mean coherence stabilizes above death threshold
3. **QD Coverage**: Diverse behavioral strategies emerge
4. **Phylogenetic Depth**: Multi-generation lineages

## File Structure

```
path_b_phase1/
├── __init__.py           # Package exports
├── README.md             # This file
├── full_environment.py   # 64x64 environment
├── full_agent.py         # 370-dim sensor agent
├── full_population.py    # Population manager
├── baselines.py          # Comparison agents
├── phase1_experiment.py  # Main experiment
└── visualize_phase1.py   # Visualization

results/path_b_phase1/
├── phase1_results_*.json # Raw results
├── phase1_*_plots.png    # Main visualizations
└── phase1_*_comparison.png # Baseline comparison
```

## Core Mechanisms

### Coherence-Based Survival
The key innovation is that agent survival depends on maintaining internal coherence:

```python
# High coherence = low metabolic cost = survival
coherence_multiplier = 2.0 - 1.5 * coherence
metabolic_cost = base_cost * coherence_multiplier
```

### Reproduction Requirements
```python
can_reproduce = (
    coherence > 0.55 and
    age > 100 and
    energy > 0.7 and
    material > 0.4
)
```

### Death Conditions
```python
should_die = (
    energy <= 0 or
    recent_coherence < 0.25
)
```

## Theory

This system demonstrates that:

1. **Self-organization emerges** from coherence-based selection without external fitness functions

2. **Open-ended evolution** is possible when survival equals maintaining organizational identity

3. **Behavioral diversity** naturally arises through the Quality-Diversity mechanism

4. **Autopoietic agents** outperform reward-based baselines in long-term survival

## References

- Varela, F., Maturana, H., & Uribe, R. (1974). Autopoiesis: The organization of living systems
- Lehman, J., & Stanley, K. O. (2011). Evolving a diversity of virtual creatures through novelty search and local competition
- Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation

## Next Steps (Phase 2)

- Neural network controllers (PyTorch)
- Communication mechanisms
- Tool use emergence
- Multi-objective coherence
