# GENESIS v2.0 Architecture Specification

**Version**: 2.0.0-alpha
**Date**: 2026-01-03
**Status**: Design Phase

---

## 1. Design Goals

### Primary Objectives
1. **Enable Positive Transfer Learning**: Multi-task learning should improve over single-task
2. **Fix Feedback Loop**: Direct Entity-Environment coupling
3. **Achieve Collective Intelligence**: Ecosystem > Individual
4. **Maintain Strengths**: Keep catastrophic forgetting resistance

### Performance Targets
| Metric | v1.1 | v2.0 Target | Improvement |
|--------|------|-------------|-------------|
| Single-task learning | +10% | +50% | 5x |
| Multi-task transfer | -11% | +20% | Positive! |
| Ecosystem advantage | 0% | +30% | Significant |
| Convergence speed | Baseline | 3x faster | Efficiency |

---

## 2. Core Architecture

### 2.1 Hierarchical Modular Design

```
GENESIS_Entity_v2_0
├── Perception Layer
│   └── Sensory encoding (unchanged from v1.1)
│
├── Modular Phenotype (NEW!)
│   ├── Shared Encoder
│   │   └── Low-level feature extraction (task-agnostic)
│   ├── Functional Modules
│   │   ├── LinearModule (for linear relationships)
│   │   ├── NonlinearModule (for complex patterns)
│   │   ├── InteractionModule (for multiplicative effects)
│   │   └── [Dynamically added modules]
│   ├── Task-Specific Heads
│   │   └── Per-task output layers
│   └── Task Router (NEW!)
│       ├── Task Detector (identify current task)
│       └── Module Selector (activate relevant modules)
│
├── Learning System
│   ├── Gradient-Based Learning (NEW!)
│   │   └── Direct error minimization
│   ├── Hebbian Consolidation
│   │   └── Memory strengthening
│   └── Viability Assessment
│       └── Survival threshold
│
├── Meta-Controller (NEW!)
│   ├── Task Memory (track per-task performance)
│   ├── Sharing Policy (decide when to share/specialize)
│   └── Architecture Manager (add/remove modules)
│
└── Ecosystem Interface
    ├── Knowledge Sharing Protocol (NEW!)
    └── Social Learning (NEW!)
```

---

## 3. Component Specifications

### 3.1 ModularPhenotype_v2_0

**Purpose**: Hierarchical architecture enabling task abstraction

**Structure**:
```python
class ModularPhenotype_v2_0:
    def __init__(self, input_size=10):
        # Layer 1: Shared encoder (task-agnostic features)
        self.shared_encoder = SharedEncoder(input_size, hidden_size=32)

        # Layer 2: Functional modules (reusable components)
        self.modules = {
            'linear': LinearModule(32, 16),
            'nonlinear': NonlinearModule(32, 16),
            'interaction': InteractionModule(32, 16)
        }

        # Layer 3: Task-specific heads
        self.task_heads = {}  # Dynamically created

        # Meta: Task router
        self.task_router = TaskRouter(shared_size=32)

        # Hebbian: Pathway strengths
        self.pathway_strengths = {}
```

**Key Methods**:
- `forward(x, task_context)`: Hierarchical forward pass
- `add_task(task_id, task_type)`: Create new task head
- `add_module(module_type)`: Dynamically add functional module
- `get_active_modules(task_id)`: Retrieve task-specific module set

**Advantages**:
- ✅ Shared low-level features (positive transfer)
- ✅ Task-specific high-level outputs (no interference)
- ✅ Compositional (modules can be reused)
- ✅ Scalable (add modules/tasks incrementally)

---

### 3.2 TaskRouter

**Purpose**: Identify task and route to appropriate modules

**Components**:

#### 3.2.1 TaskDetector
```python
class TaskDetector:
    def __init__(self, feature_size=32):
        self.task_embeddings = {}  # task_id → embedding vector
        self.threshold = 0.8  # Similarity threshold for known tasks

    def identify_task(self, features):
        """
        Returns: task_id (str) or 'new_task'

        Method:
        1. Compute similarity to all known task embeddings
        2. If max_similarity > threshold, return that task_id
        3. Else, return 'new_task'
        """
```

#### 3.2.2 ModuleSelector
```python
class ModuleSelector:
    def __init__(self):
        self.task_module_map = {}  # task_id → [module_ids]

    def select_modules(self, task_id, all_modules):
        """
        Returns: List of active module names

        Method:
        1. If task_id known, return cached module set
        2. Else, return default modules (all active)
        """
```

**Routing Process**:
```
Input features (from shared encoder)
    ↓
TaskDetector
    ├─ Known task → task_id
    └─ Unknown → 'new_task' → create new task_id
    ↓
ModuleSelector
    ├─ Get active modules for task_id
    └─ Process through selected modules
    ↓
Task-specific head
    ↓
Output
```

---

### 3.3 MetaController

**Purpose**: High-level decision making for architecture evolution

**State**:
```python
class MetaController:
    def __init__(self):
        # Task tracking
        self.task_memory = {}  # task_id → PerformanceHistory
        self.task_similarities = {}  # (task_i, task_j) → similarity

        # Module tracking
        self.module_usage = {}  # module_id → usage_count
        self.module_performance = {}  # module_id → avg_performance

        # Policies
        self.sharing_policy = SharingPolicy()
        self.specialization_threshold = 0.7
```

**Key Decisions**:

#### Decision 1: New Task Handling
```python
def handle_new_task(self, task_id, initial_features):
    # Find similar existing tasks
    similar_tasks = self.find_similar_tasks(task_id, threshold=0.7)

    if len(similar_tasks) > 0:
        # Share modules from similar task
        return {'action': 'share', 'from': similar_tasks[0]}
    else:
        # Create task-specific head with default modules
        return {'action': 'create_new', 'modules': ['linear', 'nonlinear']}
```

#### Decision 2: Module Addition
```python
def should_add_module(self, task_id, performance_history):
    # Add module if:
    # 1. Performance plateaued
    # 2. Task is complex
    # 3. No existing module covers pattern

    if self.is_plateaued(performance_history, window=20):
        missing_capability = self.diagnose_missing_capability(task_id)
        return {'add': True, 'type': missing_capability}
    return {'add': False}
```

#### Decision 3: Sharing vs Specialization
```python
def decide_sharing_policy(self, task_i, task_j):
    similarity = self.compute_task_similarity(task_i, task_j)

    if similarity > self.specialization_threshold:
        # Tasks are similar → share modules
        return 'share'
    else:
        # Tasks are different → specialize
        return 'specialize'
```

---

### 3.4 Hybrid Learning System

**Purpose**: Combine gradient-based + Hebbian learning

**Three-Stage Integration**:

#### Stage 1: Direct Error Gradient (Primary)
```python
def gradient_update(self, prediction, target, input_data):
    # Compute prediction error
    error = target - prediction

    # Backpropagate through active modules
    gradients = self.compute_gradients(error, input_data)

    # Update parameters
    for param_name, gradient in gradients.items():
        self.parameters[param_name] -= self.learning_rate * gradient

    return error
```

**Purpose**: Rapid learning, global optimization

#### Stage 2: Hebbian Consolidation (Secondary)
```python
def hebbian_update(self, error, was_successful):
    if was_successful:
        # Strengthen successful pathways
        for module_name in self.active_modules:
            activity = self.get_module_activity(module_name)
            strength_update = 0.01 * activity * self.pathway_strengths[module_name]
            self.parameters[module_name] += strength_update
            self.pathway_strengths[module_name] *= 1.01
    else:
        # Weaken failed pathways (mild)
        for module_name in self.active_modules:
            self.pathway_strengths[module_name] *= 0.99
```

**Purpose**: Memory consolidation, catastrophic forgetting prevention

#### Stage 3: Viability Assessment (Tertiary)
```python
def assess_viability(self, error, history):
    # Direct from error (50% weight - increased!)
    performance_score = np.exp(-np.abs(error))

    # Success rate (20%)
    success_rate = self.compute_success_rate(history)

    # Growth trend (20%)
    growth = self.compute_growth_trend(history)

    # Adaptability (10%)
    adaptability = len(self.active_modules) / 10.0

    # Weighted average
    viability = 0.5*performance_score + 0.2*success_rate + 0.2*growth + 0.1*adaptability

    return viability
```

**Purpose**: Survival threshold, long-term sustainability

**Learning Rate Schedule**:
```python
learning_rates = {
    'gradient': 0.01,      # Fast updates
    'hebbian': 0.01,       # Consolidation
    'viability': None      # Aggregated metric
}
```

---

## 4. Direct Feedback Loop Design

### 4.1 Problem in v1.1

```
Entity → vague action → Environment
   ↑                        ↓
   └────── no direct error ──┘
           (only viability)
```

**Issue**: Entity never receives direct error signal

### 4.2 Solution in v2.0

```
Entity ← Environment
   ↓         ↓
Predict    Ground Truth
   ↓         ↓
   └─→ Error ←┘
       ↓
   Integration
   (Gradient + Hebbian + Viability)
```

**Implementation**:
```python
def live_one_step_v2_0(self, environment):
    # 1. PERCEIVE
    input_data = environment.get_input()
    task_context = environment.get_task_context()

    # 2. PREDICT (NEW!)
    prediction = self.phenotype.forward(input_data, task_context)

    # 3. GET GROUND TRUTH (NEW!)
    target = environment.get_target(input_data)

    # 4. COMPUTE ERROR (NEW!)
    error = self.compute_error(prediction, target)

    # 5. THREE-STAGE INTEGRATION (NEW!)
    # 5a. Gradient update (primary)
    self.gradient_update(error, input_data, task_context)

    # 5b. Hebbian consolidation (secondary)
    was_successful = (error < self.success_threshold)
    self.hebbian_update(error, was_successful)

    # 5c. Viability assessment (tertiary)
    self.viability = self.assess_viability(error, self.error_history)

    # 6. META-LEARNING (NEW!)
    self.meta_controller.update(task_context, error, self.viability)

    # 7. EVOLVE (if needed)
    if self.should_metamorphose():
        self.metamorphose()

    return self.viability
```

**Key Changes**:
- ✅ Direct prediction → error → learning pipeline
- ✅ No ambiguous action format
- ✅ Environment provides clear targets
- ✅ Three-stage learning (gradient + hebbian + viability)

---

## 5. Knowledge Sharing Protocol

### 5.1 Purpose
Enable entities in ecosystem to learn from each other

### 5.2 Protocol

#### Step 1: Identify Compatible Neighbors
```python
def find_neighbors(self, entity, similarity_threshold=0.7):
    neighbors = []
    for other in self.entities:
        if other.id == entity.id:
            continue

        # Compute compatibility
        compatibility = self.compute_compatibility(entity, other)

        if compatibility > similarity_threshold:
            neighbors.append(other)

    return neighbors
```

**Compatibility Metrics**:
- Task overlap (are they working on similar tasks?)
- Viability gap (is other significantly better?)
- Genetic similarity (similar genome?)

#### Step 2: Extract Successful Pathways
```python
def get_strong_pathways(self, task_id):
    # Get modules with high pathway strengths
    strong_pathways = {}

    for module_name, strength in self.pathway_strengths.items():
        if task_id in self.module_usage.get(module_name, []):
            if np.mean(strength) > 1.2:  # Above baseline
                strong_pathways[module_name] = self.parameters[module_name]

    return strong_pathways
```

#### Step 3: Adaptive Transfer
```python
def incorporate_pathways(self, pathways, adaptation_rate=0.1):
    for module_name, external_params in pathways.items():
        if module_name in self.parameters:
            # Blend external knowledge with local knowledge
            self.parameters[module_name] = (
                (1 - adaptation_rate) * self.parameters[module_name] +
                adaptation_rate * external_params
            )
        else:
            # Adopt new module
            self.add_module(module_name, initial_params=external_params)
```

### 5.3 Ecosystem Evolution with Knowledge Sharing

```python
def evolve_one_generation_v2_0(self):
    # 1. Individual learning (FIRST!)
    for entity in self.entities:
        for _ in range(10):
            entity.live_one_step(self.environment, self)

    # 2. Knowledge sharing (NEW!)
    self.enable_knowledge_sharing()

    # 3. Natural selection
    self.selection()

    # 4. Reproduction
    self.reproduction()

    # 5. Measure emergence
    stats = self.measure_emergence()

    return stats
```

**Critical**: Individual learning BEFORE selection ensures entities have knowledge to share

---

## 6. Implementation Plan

### Phase 4A: Core Architecture (Week 1-2)
- [ ] Implement ModularPhenotype_v2_0
  - SharedEncoder
  - Functional modules (Linear, Nonlinear, Interaction)
  - Task-specific heads
- [ ] Implement TaskRouter
  - TaskDetector
  - ModuleSelector
- [ ] Implement MetaController
  - Task memory
  - Sharing policy
  - Architecture manager
- [ ] Direct feedback loop in live_one_step
- [ ] Single-task sanity check

### Phase 4B: Learning Mechanism (Week 3)
- [ ] Gradient-based learning
- [ ] Hybrid integration (Gradient + Hebbian)
- [ ] Calibrated viability
- [ ] Test on regression tasks

### Phase 4C: Multi-Task Testing (Week 4)
- [ ] Re-run 4-task experiment
- [ ] Measure transfer learning
- [ ] Compare v1.1 vs v2.0
- [ ] Validate positive transfer

### Phase 4D: Ecosystem Enhancement (Week 5)
- [ ] Knowledge sharing protocol
- [ ] Compatibility metrics
- [ ] Re-run ecosystem experiment
- [ ] Measure collective intelligence

### Phase 4E: Documentation (Week 6)
- [ ] Technical paper
- [ ] API documentation
- [ ] Benchmark results
- [ ] Public release

---

## 7. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Single-task learning: +50% improvement
- [ ] No regression from v1.1 catastrophic forgetting resistance
- [ ] Clean, documented code

### Target Performance
- [ ] Multi-task transfer: Positive (>+10%)
- [ ] Ecosystem advantage: +30%
- [ ] Convergence speed: 3x faster

### Stretch Goals
- [ ] Transfer learning across 10+ tasks
- [ ] Emergent specialization in ecosystem
- [ ] Meta-learning: Learning to learn

---

## 8. Risk Mitigation

### Risk 1: Gradient + Hebbian Conflict
**Risk**: Two learning signals might interfere
**Mitigation**:
- Sequential application (gradient first, then hebbian)
- Different learning rates (gradient 0.01, hebbian 0.01)
- Hebbian only for consolidation, not primary learning

### Risk 2: Task Router Failure
**Risk**: Misidentification of tasks
**Mitigation**:
- Conservative similarity threshold (0.8)
- Fallback to "new task" on uncertainty
- Manual task labels for debugging

### Risk 3: Meta-Controller Overhead
**Risk**: Too much computational cost
**Mitigation**:
- Lazy evaluation (only when needed)
- Caching of similarities
- Pruning of unused modules

### Risk 4: Knowledge Sharing Interference
**Risk**: Bad knowledge transfer hurts performance
**Mitigation**:
- Low adaptation rate (0.1)
- Compatibility check before transfer
- Option to disable sharing

---

## 9. Testing Strategy

### Unit Tests
- [ ] Each module forward pass
- [ ] TaskRouter identification
- [ ] MetaController decisions
- [ ] Gradient computation
- [ ] Hebbian updates

### Integration Tests
- [ ] End-to-end single-task learning
- [ ] Multi-task sequential learning
- [ ] Multi-task interleaved learning
- [ ] Ecosystem evolution

### Performance Tests
- [ ] Convergence speed vs v1.1
- [ ] Memory usage
- [ ] Scaling with task count
- [ ] Scaling with entity count

---

## 10. Appendix

### A. Mathematical Formulations

#### A.1 Gradient Update
```
L = ||y_pred - y_true||^2
∇L/∇θ = 2(y_pred - y_true) * ∂y_pred/∂θ
θ ← θ - α * ∇L/∂θ
```

#### A.2 Hebbian Update
```
Δw_ij = η * x_i * x_j * s_ij
s_ij ← s_ij * (1 + β) if success else s_ij * (1 - β)
```

#### A.3 Viability
```
V = 0.5*exp(-|error|) + 0.2*success_rate + 0.2*growth + 0.1*adaptability
```

#### A.4 Task Similarity
```
sim(task_i, task_j) = cosine(embedding_i, embedding_j)
```

### B. Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| gradient_lr | 0.01 | Standard for SGD |
| hebbian_lr | 0.01 | Conservative consolidation |
| hebbian_strengthen | 1.01 | Mild reinforcement |
| hebbian_weaken | 0.99 | Mild forgetting |
| task_similarity_threshold | 0.8 | High confidence for known tasks |
| sharing_compatibility | 0.7 | Moderate for knowledge transfer |
| adaptation_rate | 0.1 | Conservative blending |
| success_threshold | 2.0 | Task-dependent (regression) |

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Status**: Ready for Implementation
**Next**: Begin Phase 4A implementation
