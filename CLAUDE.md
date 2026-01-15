# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ultrathink**: A research progression through 8 paradigms exploring physics-based, bio-inspired, and autopoietic approaches to neural network optimization. The project spans from classical Lagrangian mechanics (LAML) through quantum-inspired ensembles (QED) to self-organizing autopoietic systems (GENESIS).

**Research Period**: 2026-01-03 ~
**Language**: Mixed Korean/English documentation, Python code with English comments

## Quick Start Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install numpy matplotlib scipy
```

### Running Experiments

```bash
# Physics-based optimizers
python physics_based/01_LAML/laml_experiment.py     # Failed baseline
python physics_based/02_LAML_Q/laml_q.py            # LAML redeemed: +26.68-43.90%
python physics_based/03_PIO/pio_optimizer.py        # Path integral: XOR specialist

# Ensemble optimizers
python ensemble/01_QED/qed_optimizer.py             # First success: +26-65%

# Meta optimizers
python meta/01_COMP/comp_optimizer.py               # Compositional: 2/3 wins
python meta/02_ULTIMATE/test_ultimate.py            # Meta-learner: Concept proven

# Autopoiesis
python autopoiesis/01_GENESIS/v2.0/experiments/*/main.py  # Autopoietic system
```

### Test Problems
All optimizers evaluated on 3 standard tasks:
- **Linear Regression**: Simple baseline
- **Nonlinear Regression**: Moderate complexity
- **XOR Classification**: Hardest nonlinear problem

Results saved to each directory's `results/` folder as PNG images.

## Architecture Overview

### Research Progression

The codebase represents an evolutionary journey through 8 distinct optimization paradigms:

```
physics_based/          ensemble/           meta/               autopoiesis/
├── 01_LAML (Failed)    ├── 01_QED          ├── 01_COMP         └── 01_GENESIS
├── 02_LAML_Q           └── 02_HYBRID       └── 02_ULTIMATE
└── 03_PIO
```

**Performance Summary**:
| Paradigm | Wins | Key Innovation | Status |
|----------|------|----------------|--------|
| QED | 3/3 | Quantum-inspired ensemble | ⭐⭐⭐⭐⭐ Production |
| LAML-Q | 3/3 | Ensemble endpoint prediction | ⭐⭐⭐⭐⭐ Production |
| COMP | 2/3 | Compositional primitives | ⭐⭐⭐⭐ Interpretable |
| PIO | 1/3 | Feynman path integrals | ⭐⭐⭐ XOR best |
| ULTIMATE | 1/3 | Meta-learning policy | ⭐⭐⭐ Concept proven |
| LAML | 0/3 | Least action principle | ⭐ Learning experience |

**Best-in-class**: PIO achieves 0.18683 on XOR (best across all paradigms)

### Core Architectural Patterns

#### Pattern 1: Primitive-Based Composition (COMP, ULTIMATE, GENESIS)

The most successful architectures compose simple primitives dynamically:

```python
# Base abstraction
class Primitive:
    def compute_update(self, network, X, y, context):
        """Return parameter updates based on strategy"""
        pass

# COMP uses 5 primitives:
- GradientDescent: Stable local optimization
- StochasticJump: Escape local minima
- Momentum: Directional acceleration
- BestDirection: Proven path following
- AdaptiveStep: Learning rate adaptation

# ULTIMATE extends to 10 universal primitives
# GENESIS uses modular components with self-organization
```

**Key Files**:
- `meta/01_COMP/primitives.py` - Template for implementing primitives
- `meta/02_ULTIMATE/primitives.py` - Extended primitive set
- `autopoiesis/01_GENESIS/v2.0/core/autopoietic_entity.py` - Beyond optimization

#### Pattern 2: Context-Aware Strategy Selection (COMP, ULTIMATE)

Optimizers track problem state and adapt strategy:

```python
class OptimizationContext:
    """Track optimization state for intelligent decisions"""

    # Current metrics
    current_loss, current_grad_norm, current_theta

    # Historical tracking
    loss_history, grad_norm_history, success_history

    # Derived features
    success_rate      # Fraction of improving steps
    improvement_rate  # Convergence speed
    loss_variance     # Stability
    phase             # 'exploration' | 'exploitation' | 'refinement'
```

**Implementation**:
- `meta/01_COMP/context.py` - State tracking
- `meta/01_COMP/weight_functions.py` - Context → primitive weights
- `meta/02_ULTIMATE/policy_network.py` - Neural policy for weight selection

#### Pattern 3: Ensemble/Multi-Agent (QED, LAML-Q, PIO)

Multiple particles/candidates explore simultaneously:

```python
class EnsembleOptimizer:
    """N entities exploring weight space"""

    def step(self):
        # Independent exploration
        for particle in particles:
            particle.update_with_gradient()
            particle.apply_stochastic_effect()

        # Collective intelligence
        ensemble_gradient = aggregate(particles)
        best_particle = select_best(particles)

        # Evolutionary selection
        if particle.loss > threshold:
            particle.mutate_or_replace()
```

**Key Files**:
- `ensemble/01_QED/qed_optimizer.py` - Quantum-inspired ensemble (cleanest implementation)
- `physics_based/02_LAML_Q/laml_q.py` - Endpoint prediction ensemble
- `physics_based/03_PIO/pio_optimizer.py` - Path integral sampling

#### Pattern 4: Autopoietic Self-Organization (GENESIS v2.0)

Radical departure from optimization - systems maintain coherence through structural drift:

```python
class AutopoieticEntity:
    """Self-maintaining organization via internal coherence"""

    def assess_coherence(self):
        # Internal metrics (NOT external loss)
        predictability = 1 / (1 + variance(state_changes))
        stability = 1 / (1 + std(states))
        complexity = 1 - abs(variance - 0.5)
        circularity = abs(autocorr(states))

        return weighted_combination(predictability, stability, complexity, circularity)

    def structural_drift(self):
        # Random perturbation (NOT gradient)
        W_new = W + random_noise * plasticity

        # Accept if coherence maintained (95% threshold)
        if assess_coherence(W_new) >= coherence * 0.95:
            W = W_new  # Evolution through drift, not optimization
```

**Architecture**: GENESIS v2.0 uses modular phenotype with:
- Shared encoder (task-agnostic features)
- Functional modules (reusable components)
- Task-specific heads
- Task router for dynamic module selection
- Meta-controller for architecture evolution

**Key Files**:
- `autopoiesis/01_GENESIS/v2.0/design/architecture_spec.md` - Complete v2.0 design
- `autopoiesis/01_GENESIS/v2.0/core/autopoietic_entity.py` - Core implementation
- `autopoiesis/01_GENESIS/experiments/*/README.md` - Experimental results

### Hybrid Learning System (GENESIS v2.0)

Three-stage integration:
1. **Gradient-based learning** (primary) - Direct error minimization
2. **Hebbian consolidation** (secondary) - Memory strengthening, catastrophic forgetting prevention
3. **Viability assessment** (tertiary) - Survival threshold based on performance + growth + adaptability

## Key Concepts

### Action Functional (LAML, LAML-Q, PIO)
```
S[θ] = ∫[½||θ̇||² + λL(θ)] dt

where:
  θ̇ = velocity of weight changes (kinetic energy)
  L(θ) = loss function (potential energy)
  S = path efficiency measure
```

Physics principle: Nature selects paths minimizing action (Least Action Principle)

### Quantum-Inspired Ensemble (QED)
- Multiple particles in superposition (explore simultaneously)
- 6 forces: gradient, momentum, personal/global/center best, quantum tunneling
- Evolutionary selection (good particles survive)
- Temperature annealing (exploration → exploitation)

### Compositional Optimization (COMP)
Complex optimization = intelligent composition of simple primitives
- Context-aware weighting
- Full interpretability (track which primitive contributes)
- Easy extensibility (add/remove primitives)

### Path Integral Optimization (PIO)
Sample all possible update paths weighted by Euclidean action (Feynman formulation)
- Best for hardest problems (XOR)
- Theory vs practice gap demonstrated

### Meta-Learning (ULTIMATE)
3-layer architecture:
```
Layer 3: Meta-Learner (experience → knowledge)
    ↕
Layer 2: Policy Network (context → primitive weights)
    ↕
Layer 1: Primitive Pool (10 universal strategies)
```

Learns to select strategies based on problem characteristics.

### Autopoiesis (GENESIS)
Self-organizing systems maintaining internal coherence:
- Organization → Self-Production → Organization (circular)
- Internal coherence > external objectives
- Structural drift > gradient descent
- 72x sample efficiency vs ML baseline

## Critical Insights

1. **No Free Lunch Validated**: Each paradigm excels on different problems
   - QED: Balanced performance (all tasks)
   - LAML-Q: Linear regression best
   - PIO: XOR classification best

2. **Theory ≠ Practice**: Beautiful mathematics (LAML, PIO) doesn't guarantee success
   - LAML failed despite perfect theory
   - PIO only succeeds on hardest problem

3. **Ensemble Power**: Multiple agents > single-path optimization
   - QED first success through parallelism
   - LAML-Q redeems LAML through ensemble

4. **Adaptivity Matters**: Context-aware selection > fixed strategy
   - COMP: Rule-based weights (2/3 wins)
   - ULTIMATE: Learned weights (1/3 wins but demonstrates adaptation)

5. **Paradigm Shift**: GENESIS goes beyond optimization entirely
   - Not gradient descent, not evolutionary - self-organization
   - Coherence-based survival, not loss minimization
   - 82.2% accuracy vs 58-62% ML baseline

## Development Patterns

When extending this codebase:

### Adding New Primitives
```python
# 1. Inherit from Primitive base class (see meta/01_COMP/primitives.py)
class MyPrimitive(Primitive):
    def compute_update(self, network, X, y, context):
        # Implement your strategy
        return update_vector

# 2. Register in optimizer
optimizer.primitives.append(MyPrimitive())

# 3. Track contribution
optimizer.track_primitive_weights()
```

### Implementing New Optimizer
```python
# 1. Define update mechanism
class MyOptimizer:
    def step(self, X, y):
        # Update parameters
        pass

    def train(self, X_train, y_train, iterations=100):
        for i in range(iterations):
            loss = self.step(X_train, y_train)
            self.loss_history.append(loss)

# 2. Run on standard test problems
from test_utils import run_standard_tests
results = run_standard_tests(MyOptimizer)

# 3. Compare against baselines
plot_comparison(results, baseline='SGD')
```

### Context-Based Decision Making
```python
# Update context after each step
context.update(theta, loss, grad_norm)

# Use context for decisions
if context.phase == 'exploration':
    weights = favor_stochastic_primitives()
elif context.phase == 'exploitation':
    weights = favor_gradient_primitives()
else:  # refinement
    weights = favor_adaptive_primitives()
```

## Important Notes

### Language and Documentation
- **Korean README.md**: Main project overview in Korean
- **English code comments**: All code uses English
- **Mixed documentation**: Some analysis files in Korean, some in English
- When editing, maintain existing language conventions

### No External ML Frameworks
- Pure NumPy implementation throughout (except GENESIS uses scipy)
- Intentional choice for clarity and interpretability
- Do NOT introduce PyTorch/TensorFlow dependencies

### Result Tracking
All experiments save results to `results/` subdirectories:
- Learning curves (PNG)
- Performance comparisons
- Primitive contribution plots (COMP, ULTIMATE)

### Phase vs Path Naming
- `01_LAML` through `07_ULTIMATE` called "Phases"
- `08_GENESIS` uses "Path" terminology (Path A, Path B)
- Maintain naming conventions when documenting

### GENESIS v2.0 Status
Currently in design/early implementation phase:
- v1.1 completed (catastrophic forgetting resistance proven)
- v2.0 architecture designed (modular phenotype, direct feedback, knowledge sharing)
- Implementation phases defined in `08_GENESIS/v2.0/design/architecture_spec.md`

## Navigation Guide

**For understanding research progression**:
1. `README.md` - Overview and results
2. `docs/THE_JOURNEY.md` - Evolution of ideas (if exists)
3. `ensemble/01_QED/qed_optimizer.py` - First success (easiest to understand)
4. `meta/01_COMP/` - Best code architecture
5. `autopoiesis/01_GENESIS/v2.0/design/architecture_spec.md` - Cutting edge

**For implementing new algorithms**:
1. Study `meta/01_COMP/primitives.py` - Clean abstraction pattern
2. Review `meta/01_COMP/context.py` - State tracking
3. Check `ensemble/01_QED/qed_optimizer.py` - Ensemble pattern
4. Reference `meta/02_ULTIMATE/ultimate_optimizer.py` - Meta-learning pattern

**For GENESIS development**:
1. Read `autopoiesis/01_GENESIS/v2.0/design/architecture_spec.md` completely
2. Check experimental results in `autopoiesis/01_GENESIS/experiments/*/README.md`
3. Study core entity in `autopoiesis/01_GENESIS/v2.0/core/autopoietic_entity.py`

## Research Philosophy

This project embodies:
1. **Complete objectivity**: Record success and failure equally
2. **Scientific rigor**: Validate all claims experimentally
3. **Innovative thinking**: Don't settle for existing methods
4. **Interdisciplinary fusion**: Physics + AI + Evolution + Quantum mechanics

> "True innovation comes from trying bold ideas without fearing failure"

The progression from LAML's failure → QED's success → LAML-Q's redemption exemplifies learning from mistakes to achieve breakthroughs.
