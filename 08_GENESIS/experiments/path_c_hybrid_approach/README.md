# Path C: Hybrid Autopoietic-ML Approach

## Overview

This experiment validates whether adding autopoietic principles to standard machine learning improves performance. We implement and test three mechanisms:

1. **Coherence Regularization**: Adds a penalty to encourage stable, predictable internal representations
2. **Structural Plasticity**: Dynamic neuron pruning and growing during training
3. **Self-Organizing Layers**: Hybrid gradient + Hebbian learning

## Installation

```bash
# Activate virtual environment
source /Users/say/Documents/GitHub/ai/08_GENESIS/venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib scipy tqdm
```

## Quick Start

### Run Quick Test
```bash
python experiment.py --dataset mnist --quick --device cpu
```

### Run Full Ablation Study
```bash
# MNIST (approximately 1 hour)
python experiment.py --dataset mnist --n_trials 5 --n_epochs 20 --device cpu

# CIFAR-10 (approximately 3-4 hours)
python experiment.py --dataset cifar10 --n_trials 5 --n_epochs 30 --device cpu
```

### Run Robustness Evaluation
```bash
python robustness.py --dataset mnist --n_epochs 20 --device cpu
```

### Generate Visualizations
```bash
# From existing results
python visualize.py --ablation results/ablation_mnist_*.json --output results/

# Demo with synthetic data
python visualize.py --demo --output results/
```

## Project Structure

```
path_c_hybrid_approach/
├── coherence_regularizer.py   # 4D coherence metrics
├── structural_plasticity.py   # Neuron pruning/growing
├── self_organizing_layer.py   # Hebbian learning layers
├── hybrid_model.py            # Integrated model
├── experiment.py              # Ablation study
├── robustness.py              # Robustness evaluation
├── visualize.py               # Result visualization
├── results/                   # Output directory
│   ├── *.json                 # Raw results
│   ├── *.png                  # Plots
│   └── summary_report.md      # Summary
└── README.md                  # This file
```

## Ablation Conditions

| Condition | Coherence | Plasticity | Self-Org |
|-----------|-----------|------------|----------|
| Baseline  | -         | -          | -        |
| +Coherence| Yes       | -          | -        |
| +Plasticity| -        | Yes        | -        |
| +SelfOrg  | -         | -          | Yes      |
| +Coh+Plas | Yes       | Yes        | -        |
| +Coh+Self | Yes       | -          | Yes      |
| +All      | Yes       | Yes        | Yes      |

## Key Components

### 1. Coherence Regularizer

Adds a coherence penalty to the loss function:

```python
loss = cross_entropy_loss + lambda * coherence_penalty
```

Where coherence is computed from 4 dimensions:
- **Predictability** (0.3): How well current activations can be predicted from history
- **Stability** (0.3): Temporal consistency of activation norms
- **Complexity** (0.2): Information richness (entropy-based)
- **Circularity** (0.2): Self-referential consistency between layers

### 2. Structural Plasticity

Dynamic network modification during training:
- **Pruning**: Remove neurons with consistently low activation
- **Growing**: Add neurons to bottleneck layers (high variance)
- Controlled by adaptive scheduler with warmup and cooldown

### 3. Self-Organizing Layers

Hybrid learning combining:
- Standard gradient descent (backpropagation)
- Hebbian learning ("fire together, wire together")
- Homeostatic scaling (maintain stable activity levels)

```python
# Forward pass uses combined weights
y = linear(x, weight + hebbian_weight) * homeostatic_scale

# Hebbian update after successful predictions
delta_hebbian = lr * success_signal * outer(y, x)
```

## Expected Results

### Success Criteria
- Accuracy: >= +0.5% improvement (statistically significant)
- Robustness: >= +2% improvement on noisy inputs
- Ablation: Clear contribution from each mechanism

### Typical Results (MNIST)
| Condition | Expected Accuracy | Expected Improvement |
|-----------|------------------|---------------------|
| Baseline  | ~98.0%           | -                   |
| +Coherence| ~98.3%           | +0.3%               |
| +SelfOrg  | ~98.4%           | +0.4%               |
| +All      | ~98.6%           | +0.6%               |

## Hyperparameters

### Coherence Regularization
- `coherence_weight`: 0.01 (default), grid search [0.001, 0.01, 0.1]
- Weight distribution: pred=0.3, stab=0.3, comp=0.2, circ=0.2

### Structural Plasticity
- `prune_threshold`: 0.01 (remove neurons with <1% max activity)
- `warmup_steps`: 500 (wait before applying)
- `apply_every`: 200 (steps between modifications)

### Self-Organizing Layers
- `hebbian_lr`: 0.001
- `hebbian_decay`: 0.999 (prevent weight explosion)
- `homeostatic_target`: 0.1 (target mean activation)

## Output Files

After running experiments:

### JSON Results
- `ablation_mnist_YYYYMMDD_HHMMSS.json`: Full ablation results
- `robustness_mnist_YYYYMMDD_HHMMSS.json`: Robustness metrics

### Visualizations
- `ablation_results_mnist.png`: Bar chart of accuracy by condition
- `improvement_mnist.png`: Improvement over baseline
- `robustness_curves_mnist.png`: Noise and adversarial robustness
- `coherence_evolution_mnist.png`: Coherence metrics over training
- `all_training_mnist.png`: Training curves comparison
- `timing_mnist.png`: Training time comparison

### Report
- `summary_report.md`: Markdown summary with key findings

## Running on Different Hardware

### CPU
```bash
python experiment.py --device cpu
```

### GPU (CUDA)
```bash
python experiment.py --device cuda
```

### Apple Silicon (MPS)
```bash
python experiment.py --device mps
```

### Auto-detect
```bash
python experiment.py --device auto
```

## Computational Notes

- MNIST full experiment: ~1 hour on CPU, ~15 min on GPU
- CIFAR-10 full experiment: ~4 hours on CPU, ~45 min on GPU
- Memory usage: ~2GB for MNIST, ~4GB for CIFAR-10
- Autopoietic overhead: +10-20% training time

## References

- Maturana & Varela (1980): Autopoiesis and Cognition
- Hebb (1949): The Organization of Behavior
- Turrigiano (2008): Homeostatic plasticity in neuronal networks
