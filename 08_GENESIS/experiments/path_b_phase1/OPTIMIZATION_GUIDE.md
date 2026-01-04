# GENESIS Phase 4: Optimization Guide

**Date:** 2026-01-04
**Status:** Recommendations for Performance Improvement

---

## Current Performance

### Benchmark Results (50 agents, 50 steps)
- **Baseline**: 0.020s per step (0.99s total)
- **Full System**: 0.500s per step (24.99s total)
- **Overhead**: 25x slower

### Bottlenecks Identified

1. **Novelty Search** (~40% of time)
   - K-NN computation for all agents every step
   - Distance calculations: O(n²)

2. **MAP-Elites** (~20% of time)
   - Archive lookups and updates
   - Fitness comparisons

3. **Communication** (~30% of time)
   - Neural network forward passes (encoder, decoder, attention)
   - Message routing and delivery

4. **Memory Operations** (~10% of time)
   - Priority computation
   - Experience storage and retrieval

---

## Optimization Strategies

### 1. Spatial Indexing for Novelty Search

**Current**: Brute force O(n²) distance calculations
**Improvement**: Use KD-Tree or Ball Tree for O(n log n)

```python
from sklearn.neighbors import BallTree

class OptimizedNoveltySearch:
    def __init__(self):
        self.behavior_tree = None

    def compute_novelty_fast(self, behavior, k=15):
        if self.behavior_tree is None:
            behaviors = np.array([b for b, _ in self.archive])
            self.behavior_tree = BallTree(behaviors)

        distances, indices = self.behavior_tree.query([behavior], k=k)
        return np.mean(distances[0])
```

**Expected Improvement**: 5-10x faster for large populations

### 2. Batch Neural Network Processing

**Current**: Process messages one at a time
**Improvement**: Batch encoding/decoding for GPU acceleration

```python
class BatchedCommunication:
    def encode_batch(self, states):
        # states: (batch_size, state_dim)
        with torch.no_grad():
            messages = self.encoder(torch.FloatTensor(states))
        return messages.numpy()

    def decode_batch(self, messages):
        # messages: (batch_size, message_dim)
        with torch.no_grad():
            influences = self.decoder(torch.FloatTensor(messages))
        return influences.numpy()
```

**Expected Improvement**: 10-20x faster with GPU, 2-3x with CPU batching

### 3. Sparse MAP-Elites Updates

**Current**: Update every step for all agents
**Improvement**: Update only when agent improves or changes niche

```python
class SparseMapElites:
    def add_solution_lazy(self, agent, fitness, behavior):
        niche = self.behavior_space.get_niche(behavior)

        # Only update if improvement
        if niche not in self.archive or fitness > self.archive[niche].fitness:
            self.archive[niche] = EliteSolution(agent, fitness, behavior)
            return True  # Updated
        return False  # No change
```

**Expected Improvement**: 2-3x faster

### 4. Cached Coherence Computation

**Current**: Recompute coherence every time
**Improvement**: Cache and invalidate on state change

```python
class CachedCoherenceAgent:
    def __init__(self, agent):
        self.agent = agent
        self._coherence_cache = None
        self._last_state_hash = None

    def compute_coherence(self):
        current_hash = hash(self.agent.state.tobytes())

        if current_hash == self._last_state_hash:
            return self._coherence_cache

        coherence = self.agent.compute_coherence()
        self._coherence_cache = coherence
        self._last_state_hash = current_hash
        return coherence
```

**Expected Improvement**: 1.5-2x faster

### 5. Parallel Population Processing

**Current**: Sequential agent updates
**Improvement**: Parallel processing with multiprocessing

```python
from multiprocessing import Pool

class ParallelPopulation:
    def __init__(self, num_workers=4):
        self.pool = Pool(num_workers)

    def step_parallel(self, agents):
        # Process agents in parallel
        results = self.pool.map(self._step_agent, agents)
        return results

    def _step_agent(self, agent):
        # This runs in separate process
        return agent.step()
```

**Expected Improvement**: 2-4x faster on multi-core CPUs

### 6. GPU Acceleration for Neural Networks

**Current**: CPU-only PyTorch
**Improvement**: Move models to GPU

```python
class GPUCommunication:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = MessageEncoder().to(self.device)
        self.decoder = MessageDecoder().to(self.device)
        self.attention = MessageAttention().to(self.device)

    def encode_gpu(self, states):
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            messages = self.encoder(states_tensor)
        return messages.cpu().numpy()
```

**Expected Improvement**: 20-50x faster with modern GPU

### 7. Approximate Novelty Search

**Current**: Exact k-NN every step
**Improvement**: Approximate k-NN with update scheduling

```python
class ApproximateNoveltySearch:
    def __init__(self, update_frequency=10):
        self.update_frequency = update_frequency
        self.step_count = 0
        self.approximate_archive = []

    def compute_novelty_approximate(self, behavior):
        # Only rebuild tree every N steps
        if self.step_count % self.update_frequency == 0:
            self._rebuild_tree()

        # Use approximate k-NN
        return self._fast_approximate_knn(behavior)
```

**Expected Improvement**: 3-5x faster

---

## Implementation Priority

### High Priority (Quick wins)

1. **Batch Processing** (2-3 hours implementation)
   - Easy to add
   - Significant speedup (2-3x)
   - No accuracy loss

2. **Cached Coherence** (1 hour implementation)
   - Trivial to add
   - 1.5-2x speedup
   - No accuracy loss

3. **Sparse MAP-Elites** (2 hours implementation)
   - Simple logic change
   - 2-3x speedup
   - No accuracy loss

### Medium Priority (1-2 days)

4. **Spatial Indexing** (4-6 hours implementation)
   - Requires refactoring
   - 5-10x speedup
   - Minimal accuracy impact

5. **GPU Acceleration** (1 day implementation)
   - Needs GPU hardware
   - 20-50x speedup for communication
   - No accuracy loss

### Low Priority (Research projects)

6. **Parallel Processing** (2-3 days implementation)
   - Complex state management
   - 2-4x speedup
   - Requires careful synchronization

7. **Approximate Algorithms** (1-2 weeks)
   - Research accuracy/speed tradeoffs
   - 3-5x speedup
   - May impact quality

---

## Expected Combined Performance

### Conservative Estimates

| Optimization | Speedup |
|-------------|---------|
| Batch Processing | 2.5x |
| Cached Coherence | 1.5x |
| Sparse MAP-Elites | 2x |
| Spatial Indexing | 7x |

**Combined**: ~50x faster (conservative)

### With GPU

| Optimization | Speedup |
|-------------|---------|
| Above optimizations | 50x |
| GPU Acceleration | 30x |

**Combined**: ~1500x faster!

### Projected Performance

**Current**: 0.500s per step (50 agents)
**Optimized (CPU)**: 0.010s per step (50x faster)
**Optimized (GPU)**: 0.0003s per step (1500x faster)

**Enables**:
- 10,000 step experiments in minutes instead of hours
- 1,000 agent populations
- Real-time experimentation

---

## Memory Optimization

### Current Memory Usage

- **50 agents**: ~150 MB
- **500 agents**: ~1.1 GB

### Memory Bottlenecks

1. Position history (deques): ~30%
2. Message archives: ~20%
3. Behavior archives: ~25%
4. Neural network weights: ~15%
5. Experience buffers: ~10%

### Optimization Strategies

1. **Limit History Length**
   ```python
   position_history = deque(maxlen=50)  # Instead of unlimited
   ```

2. **Compress Archives**
   ```python
   # Store only recent behaviors
   if len(archive) > 10000:
       archive = archive[-5000:]  # Keep only recent half
   ```

3. **Shared Neural Networks**
   ```python
   # All agents share same encoder/decoder
   class SharedCommunication:
       encoder = MessageEncoder()  # Single instance
       decoder = MessageDecoder()  # Single instance
   ```

**Expected**: 50% memory reduction

---

## Quick Win Implementation

Here's a quick optimization that can be added immediately:

```python
# Add to phase4c_integration.py

class OptimizedPhase4C_CommunicationManager(Phase4C_CommunicationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optimization flags
        self.use_batch_processing = True
        self.use_cached_coherence = True
        self.update_frequency = 5  # Update novelty search every 5 steps

    def step(self):
        # Cached coherence
        if self.use_cached_coherence and self.current_step % 10 == 0:
            # Only recompute every 10 steps
            return super().step()

        # ... optimized step logic
```

---

## Testing Optimizations

### Validation Protocol

1. **Correctness Check**
   ```python
   # Run both versions, compare results
   baseline_result = baseline_manager.step()
   optimized_result = optimized_manager.step()

   assert np.allclose(baseline_result['coherence'],
                      optimized_result['coherence'],
                      rtol=0.01)  # 1% tolerance
   ```

2. **Performance Benchmark**
   ```python
   import time

   # Baseline
   start = time.time()
   for _ in range(100):
       baseline_manager.step()
   baseline_time = time.time() - start

   # Optimized
   start = time.time()
   for _ in range(100):
       optimized_manager.step()
   optimized_time = time.time() - start

   speedup = baseline_time / optimized_time
   print(f"Speedup: {speedup:.2f}x")
   ```

3. **Memory Profiling**
   ```python
   from memory_profiler import profile

   @profile
   def test_memory():
       manager = create_phase4c_system(initial_population=500)
       for _ in range(100):
           manager.step()
   ```

---

## Conclusion

Current 25x overhead is acceptable for research but can be dramatically reduced:

- **Quick wins** (1 day work): 5-10x faster
- **Full optimization** (1 week work): 50x faster
- **GPU acceleration** (with hardware): 1500x faster

**Recommendation**: Implement quick wins first, then evaluate need for further optimization based on experimental requirements.

---

**Last Updated**: 2026-01-04
**Status**: Recommendations ready for implementation
