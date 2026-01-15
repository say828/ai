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

# Install core dependencies
pip install numpy matplotlib scipy

# Install additional dependencies for Phase 4C (emergent communication)
pip install torch  # For neural message encoding/decoding

# Optional: Neo4j for scalable knowledge graphs (Phase 4A)
# See autopoiesis/01_GENESIS/experiments/path_b_phase1/neo4j_backend.py
```

### Running Experiments

```bash
# Physics-based optimizers
python physics_based/01_LAML/laml_experiment.py     # Failed baseline (learning experience)
python physics_based/02_LAML_Q/laml_q.py            # LAML redeemed: +26.68-43.90%
python physics_based/03_PIO/pio_optimizer.py        # Path integral: XOR specialist

# Ensemble optimizers
python ensemble/01_QED/qed_optimizer.py             # First success: +26-65%

# Meta optimizers
python meta/01_COMP/comp_optimizer.py               # Compositional: 2/3 wins
python meta/02_ULTIMATE/test_ultimate.py            # Meta-learner: Concept proven

# Autopoiesis - Phase 4 experiments (path_b_phase1)
cd autopoiesis/01_GENESIS/experiments/path_b_phase1
python test_phase4_minimal.py                       # Phase 4A: Advanced intelligence
python test_phase4b.py                              # Phase 4B: Open-ended learning
python test_phase4c.py                              # Phase 4C: Emergent communication
python benchmark_comparison.py                      # Compare all paradigms
python long_term_experiment.py                      # Extended experiments with checkpointing

# Optimization testing
python test_optimizations.py                        # Test optimized implementations
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
| QED | 3/3 | Quantum-inspired ensemble | ⭐⭐⭐⭐⭐ Production (+41% avg) |
| LAML-Q | 3/3 | Ensemble endpoint prediction | ⭐⭐⭐⭐⭐ Production (+33% avg) |
| COMP | 2/3 | Compositional primitives | ⭐⭐⭐⭐ Interpretable (+16% avg) |
| PIO | 1/3 | Feynman path integrals | ⭐⭐⭐ XOR specialist |
| ULTIMATE | 1/3 | Meta-learning policy | ⭐⭐⭐ Concept proven |
| LAML | 0/3 | Least action principle | ⭐ Learning experience |
| GENESIS | N/A | Autopoietic intelligence | ⭐⭐⭐⭐⭐ **Paradigm shift** |

**Best-in-class**:
- Overall: QED (3/3 wins, +41%)
- XOR: PIO (0.18683, best on hardest problem)
- Paradigm shift: GENESIS (+37% performance, +7264% sample efficiency vs ML)

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
- `autopoiesis/01_GENESIS/experiments/path_b_phase1/` - Phase 4 implementations
- `autopoiesis/01_GENESIS/experiments/path_b_phase1/OPTIMIZATION_GUIDE.md` - Performance optimization analysis

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
   - 82.2% performance vs 61.8% ML baseline (+37%)
   - 72x better sample efficiency (7264% improvement)

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

### Dependencies Policy
- **Physics-based optimizers** (LAML, QED, PIO, COMP, ULTIMATE): Pure NumPy only
- **GENESIS Core**: NumPy + scipy for statistical functions
- **GENESIS Phase 4C** (Emergent Communication): PyTorch for neural message encoding
- Intentional choice for clarity and interpretability in core algorithms
- PyTorch only used where neural networks are conceptually central (communication protocols)

### Result Tracking
All experiments save results to `results/` subdirectories:
- Learning curves (PNG)
- Performance comparisons
- Primitive contribution plots (COMP, ULTIMATE)

### Phase vs Path Naming
- `01_LAML` through `07_ULTIMATE` called "Phases" or "Paradigms"
- `GENESIS` uses "Path" terminology for different research directions:
  - **Path A**: Continual learning approach
  - **Path B**: Artificial life approach (main direction)
  - **Path C**: Hybrid approach
- Within Path B:
  - **Phase 0**: Initial autopoietic baseline
  - **Phase 1**: Full system with Phase 4A+4B+4C
- Maintain naming conventions when documenting

### GENESIS Experiments Structure
```
autopoiesis/01_GENESIS/experiments/
├── path_b_phase0/          # Baseline autopoietic system
├── path_b_phase1/          # Main implementation (Phase 4A+4B+4C)
│   ├── test_phase4_minimal.py      # Phase 4A test
│   ├── test_phase4b.py             # Phase 4B test
│   ├── test_phase4c.py             # Phase 4C test
│   ├── benchmark_comparison.py     # Compare all paradigms
│   ├── long_term_experiment.py     # Extended experiments
│   ├── test_optimizations.py       # Optimized implementations
│   └── OPTIMIZATION_GUIDE.md       # Performance analysis
├── path_a_continual_learning/
└── path_c_hybrid_approach/
```

**Current Focus**: `path_b_phase1/` contains all production code and experiments

### GENESIS Status
**v1.1**: ✅ Complete (catastrophic forgetting resistance proven)
**v2.0**: ✅ Architecture designed (modular phenotype)
**Phase 4A**: ✅ Complete (advanced intelligence - multi-teacher, learned memory)
**Phase 4B**: ✅ Complete (open-ended learning - novelty search, MAP-Elites, POET)
**Phase 4C**: ✅ Complete (emergent communication - neural protocols)
**Optimizations**: ✅ Ready (9-18x speedup expected from batch processing, caching, sparse updates)

All experiments fully operational with comprehensive testing and validation.

## Navigation Guide

**For understanding research progression**:
1. `RESEARCH_ACHIEVEMENTS.md` - Complete summary of all achievements
2. `FINAL_EVALUATION.md` - Comprehensive analysis and assessment
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
2. Check Phase 4 implementations in `autopoiesis/01_GENESIS/experiments/path_b_phase1/`
3. Study core entity in `autopoiesis/01_GENESIS/v2.0/core/autopoietic_entity.py`
4. Review optimization strategies in `autopoiesis/01_GENESIS/experiments/path_b_phase1/OPTIMIZATION_GUIDE.md`

**For performance optimization**:
1. Read `autopoiesis/01_GENESIS/experiments/path_b_phase1/OPTIMIZATION_GUIDE.md` - Detailed analysis
2. Study `autopoiesis/01_GENESIS/experiments/path_b_phase1/test_optimizations.py` - Optimized implementations
3. Review `autopoiesis/01_GENESIS/experiments/path_b_phase1/benchmark_comparison.py` - Benchmarking framework

## Performance & Optimization

### GENESIS Phase 4 Optimizations

The Phase 4 system initially had 25x performance overhead. Three key optimizations were implemented:

#### 1. Batch Neural Network Processing (2-3x speedup)
```python
# Instead of processing messages one-by-one
for agent in agents:
    message = encoder(agent.state)  # Slow!

# Process all at once
all_states = np.stack([a.state for a in agents])
all_messages = encoder(all_states)  # Fast!
```

**Implementation**: `test_optimizations.py` - `BatchedMessageProcessor` class

#### 2. Cached Coherence Computation (1.5-2x speedup)
```python
# Cache coherence values, only recompute when state changes
if state_hash == last_hash:
    return cached_coherence  # Skip expensive computation
```

**Implementation**: `test_optimizations.py` - `CachedCoherenceAgent` wrapper

#### 3. Sparse MAP-Elites Updates (2-3x speedup)
```python
# Only update archive when agent improves or changes niche
if not (fitness_improved or niche_changed):
    skip_update()  # Avoid unnecessary archive operations
```

**Implementation**: `test_optimizations.py` - `SparseMapElites` class

### Expected Performance Gains

| Configuration | Speedup | Time per Step |
|--------------|---------|---------------|
| Baseline | 1x | 0.500s |
| + Batch Processing | 2.5x | 0.200s |
| + Cached Coherence | 3.75x | 0.133s |
| + Sparse MAP-Elites | **9-18x** | **0.028-0.056s** |

### Running Optimizations

```bash
cd autopoiesis/01_GENESIS/experiments/path_b_phase1

# Test optimized implementations
python test_optimizations.py

# Compare baseline vs optimized
python benchmark_comparison.py

# Long-term experiment with checkpointing
python long_term_experiment.py --steps 10000 --checkpoint_interval 1000
```

### Future Optimization Opportunities

From `OPTIMIZATION_GUIDE.md`:
- **Spatial Indexing** (KD-Tree): 5-10x additional speedup
- **GPU Acceleration**: 20-50x speedup (requires PyTorch GPU)
- **Parallel Processing**: 2-4x speedup (complex synchronization)

**Potential Total**: 50x (CPU) to 1500x (GPU) faster than baseline

## Research Philosophy

This project embodies:
1. **Complete objectivity**: Record success and failure equally
2. **Scientific rigor**: Validate all claims experimentally
3. **Innovative thinking**: Don't settle for existing methods
4. **Interdisciplinary fusion**: Physics + AI + Evolution + Quantum mechanics

> "True innovation comes from trying bold ideas without fearing failure"

The progression from LAML's failure → QED's success → LAML-Q's redemption exemplifies learning from mistakes to achieve breakthroughs.
