# GENESIS Phase 4B: Open-Ended Learning Design

**Date:** 2026-01-04
**Status:** 🚧 In Design

## Overview

Phase 4B implements **open-ended learning** - the ability to continuously generate new challenges and discover new behaviors without predefined objectives.

### Key Innovation

Instead of optimizing for a fixed objective, Phase 4B:
1. **Generates** new environments/challenges automatically
2. **Discovers** unexpected solutions through novelty search
3. **Maintains** a diversity of high-quality solutions (Quality-Diversity)
4. **Coevolves** agents and environments together (POET)

## Components

### 1. POET (Paired Open-Ended Trailblazer)

**What:** Coevolve agents and environments together

**How:**
- Start with minimal environment
- Agents evolve to solve current environment
- When agents master environment → generate harder variant
- Transfer agents between environments (horizontal transfer)
- Maintain population of agent-environment pairs

**Benefits:**
- Automatic curriculum generation
- Prevents stagnation (always new challenges)
- Transfer learning across domains
- Discovers increasingly complex behaviors

**Implementation:**
```python
class POETSystem:
    def __init__(self):
        self.pairs = []  # List of (agent, environment) pairs
        self.archive = []  # Historical pairs

    def step(self):
        # 1. Evolve agents in their environments
        for pair in self.pairs:
            pair.agent.evolve_in(pair.environment)

        # 2. Generate new environment variants
        new_envs = self.generate_environment_variants()

        # 3. Transfer agents to new environments
        successful_transfers = self.attempt_transfers(new_envs)

        # 4. Prune unsuccessful pairs
        self.pairs = self.select_active_pairs()
```

### 2. Novelty Search

**What:** Reward behavioral novelty instead of objective performance

**How:**
- Define behavior characterization (e.g., trajectory, resource usage patterns)
- Measure distance to k-nearest neighbors in behavior space
- Reward = Distance to nearest neighbors (high = novel)
- Maintain archive of all discovered behaviors

**Benefits:**
- Discovers unexpected solutions
- Avoids deceptive local optima
- Explores behavior space systematically
- Can be combined with objective (novelty + performance)

**Behavior Characterization Examples:**
1. **Trajectory:** Final (x, y) position after T steps
2. **Resource pattern:** (avg_energy, avg_material, reproduction_count)
3. **Movement style:** (speed, turn_rate, exploration_radius)
4. **Social:** (interaction_count, cooperation_index)

**Implementation:**
```python
class NoveltySearch:
    def __init__(self, k_nearest=15):
        self.archive = []  # All behaviors seen
        self.k_nearest = k_nearest

    def compute_novelty(self, behavior):
        if len(self.archive) < self.k_nearest:
            return 1.0  # Automatically novel if archive small

        # Find k-nearest neighbors
        distances = [self.behavior_distance(behavior, b) for b in self.archive]
        nearest_k = sorted(distances)[:self.k_nearest]

        # Novelty = average distance to k-nearest
        novelty = sum(nearest_k) / self.k_nearest

        return novelty

    def add_to_archive(self, behavior, novelty_threshold=0.5):
        novelty = self.compute_novelty(behavior)
        if novelty > novelty_threshold:
            self.archive.append(behavior)
        return novelty
```

### 3. Quality-Diversity (MAP-Elites)

**What:** Fill behavior space with diverse high-quality solutions

**How:**
- Discretize behavior space into grid (e.g., 10x10 bins)
- Each bin stores the BEST solution with that behavior
- New solutions compete within their behavioral bin
- Result: Archive of high-quality diverse solutions

**Benefits:**
- Combines novelty (diversity) and quality (performance)
- Provides "stepping stones" for further evolution
- Reveals structure of fitness landscape
- Useful for transfer learning (pick solution matching new task)

**Implementation:**
```python
class MAPElites:
    def __init__(self, behavior_dims=2, bins_per_dim=10):
        self.behavior_dims = behavior_dims
        self.bins_per_dim = bins_per_dim

        # Archive: dict mapping bin_index → (agent, fitness, behavior)
        self.archive = {}

    def add(self, agent, fitness, behavior):
        # Discretize behavior to bin
        bin_index = self.behavior_to_bin(behavior)

        # Check if bin empty or new agent better
        if bin_index not in self.archive or fitness > self.archive[bin_index][1]:
            self.archive[bin_index] = (agent, fitness, behavior)
            return True
        return False

    def behavior_to_bin(self, behavior):
        # Map continuous behavior to discrete bin
        normalized = (behavior - self.min_behavior) / (self.max_behavior - self.min_behavior)
        bin_coords = (normalized * self.bins_per_dim).astype(int)
        bin_coords = np.clip(bin_coords, 0, self.bins_per_dim - 1)
        return tuple(bin_coords)
```

## Integration with GENESIS

### Architecture

```
Phase4B_OpenEndedManager (extends Phase4PopulationManager)
├── POETSystem
│   ├── EnvironmentGenerator (creates variants)
│   ├── TransferEvaluator (tests agents in new envs)
│   └── PairSelector (manages active pairs)
├── NoveltySearch
│   ├── BehaviorCharacterizer
│   └── NoveltyArchive
└── MAPElites
    ├── BehaviorSpace (discretization)
    └── EliteArchive (best per bin)
```

### Usage Modes

**Mode 1: POET-only**
```python
manager = Phase4B_OpenEndedManager(
    mode='poet',
    env_complexity_start='minimal',
    env_complexity_max='extreme'
)
```

**Mode 2: Novelty Search**
```python
manager = Phase4B_OpenEndedManager(
    mode='novelty',
    behavior_characterization='trajectory',
    novelty_weight=0.8,  # 80% novelty, 20% fitness
    fitness_weight=0.2
)
```

**Mode 3: Quality-Diversity (MAP-Elites)**
```python
manager = Phase4B_OpenEndedManager(
    mode='map_elites',
    behavior_dims=['energy_usage', 'exploration_range'],
    bins_per_dim=20  # 20x20 = 400 behavioral niches
)
```

**Mode 4: All combined (recommended)**
```python
manager = Phase4B_OpenEndedManager(
    mode='full',
    use_poet=True,
    use_novelty=True,
    use_map_elites=True
)
```

## Expected Outcomes

### Performance Improvements

1. **Continuous Discovery:** Never stops finding new behaviors
2. **Robustness:** Solutions work across environment variants
3. **Transfer Learning:** Solutions from one env help in others
4. **Behavior Diversity:** Rich repertoire of strategies

### Metrics to Track

```python
stats = {
    'poet': {
        'active_pairs': 10,
        'total_environments_generated': 157,
        'successful_transfers': 23,
        'max_environment_difficulty': 0.87
    },
    'novelty': {
        'archive_size': 1543,
        'avg_novelty': 0.45,
        'unique_behaviors_discovered': 89
    },
    'map_elites': {
        'bins_filled': 234,  # out of 400
        'coverage': 0.585,  # 58.5% of behavior space
        'avg_quality': 0.73,
        'best_per_bin_count': 234
    }
}
```

## Implementation Phases

### Phase 1: Core Algorithms ✅
- [x] Design document (this file)
- [ ] Implement NoveltySearch
- [ ] Implement MAPElites
- [ ] Implement POETSystem
- [ ] Implement EnvironmentGenerator

### Phase 2: Integration
- [ ] Create Phase4B_OpenEndedManager
- [ ] Integrate with Phase4PopulationManager
- [ ] Add behavior characterization options
- [ ] Implement statistics tracking

### Phase 3: Testing
- [ ] Test NoveltySearch on simple task
- [ ] Test MAP-Elites coverage
- [ ] Test POET environment generation
- [ ] Test full integration

### Phase 4: Validation
- [ ] Run 10K+ step experiments
- [ ] Measure diversity improvements
- [ ] Measure behavior discovery rate
- [ ] Compare vs baseline (Phase 4A)

## Success Criteria

| Metric | Target | Why |
|--------|--------|-----|
| Behavior diversity | 10x more unique behaviors than Phase 4A | Novelty search working |
| Environment difficulty | Grows continuously for 10K steps | POET generating challenges |
| MAP-Elites coverage | >60% of behavior space | Quality-diversity working |
| Transfer success rate | >30% of transfers successful | Cross-domain learning |
| No stagnation | New behaviors every 100 steps | Open-ended property |

## References

1. **POET:** Wang et al. (2019) "Paired Open-Ended Trailblazer (POET)"
2. **Novelty Search:** Lehman & Stanley (2011) "Abandoning Objectives"
3. **MAP-Elites:** Mouret & Clune (2015) "Illuminating the Search Space"
4. **Quality-Diversity:** Pugh et al. (2016) "Quality Diversity"

---

**Next:** Implement novelty_search.py
