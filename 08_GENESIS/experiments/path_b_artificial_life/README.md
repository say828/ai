# Path B: Artificial Life Validation

## Overview

This experiment validates the autopoietic paradigm by comparing agent behaviors in an open-ended artificial life environment. The key question: **Do autopoietic agents exhibit qualitatively different (and more "natural") behaviors than traditional RL agents?**

## Core Hypothesis

Autopoietic agents, which maintain organizational coherence rather than optimizing external rewards, should exhibit:
1. More diverse and exploratory behaviors
2. Natural niche specialization
3. Emergent collective patterns
4. Longer survival through self-organization

## Components

### 1. Grid World Environment (`grid_world.py`)
- 100x100 2D grid
- Resources: Spawn randomly, clustered, deplete on consumption
- Predators: Rule-based chase behavior
- Physics: Movement, collision, 9x9 vision cone

### 2. Autopoietic Agent (`autopoietic_grid_agent.py`)
Extension of v2.0 AutopoieticEntity:
- Sensory: 9x9x5 local perception (405 dimensions)
- Internal: 50-unit recurrent dynamics
- Motor: 6 actions (stay, 4 directions, consume)
- Survival = coherence maintenance
- Reproduction when coherence > 0.7

### 3. Baseline Agents (`baseline_agents.py`)
1. **RL Agent (PPO + ICM)**: Intrinsic curiosity only, NO explicit rewards
2. **NEAT Agent**: Evolved feedforward network
3. **Random Agent**: Uniform random actions (control)

### 4. Analysis Tools (`analysis.py`)
- Behavioral clustering (t-SNE)
- Diversity metrics (entropy, pairwise distance)
- Emergence detection (complexity, niche specialization)
- Survival statistics

### 5. Visualization (`visualize.py`)
- Grid world animation
- Population dynamics plots
- Behavioral embedding visualization
- Niche heatmaps

## Running the Experiment

### Quick Start
```bash
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_artificial_life
source ../../venv/bin/activate
python experiment.py
```

### Full Options
```bash
python experiment.py \
    --grid-size 100 \
    --resources 200 \
    --predators 5 \
    --agents-per-type 10 \
    --max-population 50 \
    --max-steps 5000 \
    --save-interval 100 \
    --seed 42 \
    --output results
```

### Individual Tests
```bash
# Test grid world
python grid_world.py

# Test autopoietic agent
python autopoietic_grid_agent.py

# Test baseline agents
python baseline_agents.py

# Test analysis tools
python analysis.py

# Test visualization
python visualize.py
```

## Success Criteria

1. **Survival**: Autopoietic population survives 5000+ steps
2. **Diversity**: Behavioral entropy > baseline agents
3. **Emergence**: Observable unique behaviors (niche specialization, collective patterns)

## Key Design Decisions

### Fair Comparison
- RL agent uses ONLY intrinsic motivation (curiosity), NO explicit rewards
- All agents have identical sensory/motor interfaces
- Same energy dynamics and survival rules

### "Success" Definition
We use both quantitative and qualitative metrics:
- **Quantitative**: Survival time, diversity scores, complexity measures
- **Qualitative**: Visual inspection of behavioral patterns

### Population Scaling
- Maximum population limit prevents explosion
- Reproduction threshold ensures quality
- Energy cost for reproduction

## Output Structure

```
results/
  run_YYYYMMDD_HHMMSS/
    experiment_population.png    # Population over time
    experiment_survival.png      # Survival curves
    experiment_diversity.png     # Diversity metrics
    experiment_embedding.png     # Behavioral t-SNE
    experiment_actions.png       # Action distributions
    experiment_niches.png        # Spatial heatmaps
    animation.gif                # Grid world animation
    summary.json                 # Statistics summary
```

## Theoretical Background

### Autopoiesis vs Reinforcement Learning

| Aspect | Autopoiesis | RL |
|--------|-------------|-----|
| Objective | Coherence maintenance | Reward maximization |
| Learning | Structural drift | Gradient descent |
| Reward | None (intrinsic coherence) | External signal |
| Behavior | Self-organized | Optimized |

### Expected Differences

1. **Action Patterns**: Autopoietic agents should show more variable, context-sensitive actions
2. **Exploration**: RL explores to maximize future reward; autopoietic explores to maintain coherence
3. **Niche Formation**: Autopoietic agents may naturally specialize without explicit incentive
4. **Robustness**: Autopoietic agents should be more robust to environmental changes

## References

- Maturana & Varela (1980): Autopoiesis and Cognition
- Varela (1979): Principles of Biological Autonomy
- Di Paolo (2005): Autopoiesis, Adaptivity, Teleology, Agency
- Pathak et al. (2017): Curiosity-driven Exploration (ICM)

## Author

GENESIS Project - 2026-01-04
