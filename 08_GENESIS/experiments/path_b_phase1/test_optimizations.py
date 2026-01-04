"""
Unit Tests for GENESIS Phase 4C Optimizations

Tests all optimization components:
- BatchedMessageProcessor
- CachedCoherenceAgent
- SparseMapElites
- OptimizedPhase4C_Manager

Run with: python test_optimizations.py
"""

import numpy as np
import torch
import unittest
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_phase4c import (
    BatchedMessageProcessor,
    CachedCoherenceAgent,
    SparseMapElites,
    create_optimized_phase4c_system
)


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, agent_id=0):
        self.id = agent_id
        self.state = np.random.randn(128)
        self._coherence = 0.5
        self.coherence_computations = 0

    def compute_coherence(self):
        self.coherence_computations += 1
        time.sleep(0.001)  # Simulate computation
        return self._coherence


class MockMapElites:
    """Mock MAP-Elites for testing"""
    def __init__(self):
        self.archive = {}
        self.add_solution_calls = 0
        self.behavior_space = MockBehaviorSpace()

    def add_solution(self, agent, fitness, behavior):
        self.add_solution_calls += 1
        niche = self.behavior_space.get_niche(behavior)
        if niche not in self.archive or self.archive[niche].fitness < fitness:
            self.archive[niche] = MockSolution(agent, fitness, behavior)

class MockBehaviorSpace:
    """Mock behavior space"""
    def get_niche(self, behavior):
        return tuple(np.round(behavior * 10).astype(int))

class MockSolution:
    """Mock solution"""
    def __init__(self, agent, fitness, behavior):
        self.agent = agent
        self.fitness = fitness
        self.behavior = behavior


class TestBatchedMessageProcessor(unittest.TestCase):
    """Test batch neural network processing"""

    def setUp(self):
        self.processor = BatchedMessageProcessor(
            message_dim=8,
            influence_dim=32,
            state_dim=128,
            device='cpu'
        )

    def test_encode_single(self):
        """Test encoding single state"""
        state = np.random.randn(128)
        states = np.array([state])

        messages = self.processor.encode_batch(states)

        self.assertEqual(messages.shape, (1, 8))
        self.assertTrue(np.all(np.abs(messages) <= 1.0))  # Tanh bounded

    def test_encode_batch(self):
        """Test encoding multiple states"""
        batch_size = 50
        states = np.random.randn(batch_size, 128)

        start_time = time.time()
        messages = self.processor.encode_batch(states)
        batch_time = time.time() - start_time

        self.assertEqual(messages.shape, (batch_size, 8))
        self.assertTrue(np.all(np.abs(messages) <= 1.0))

        # Test statistics tracking
        stats = self.processor.get_statistics()
        self.assertEqual(stats['batches_processed'], 1)
        self.assertEqual(stats['total_items'], batch_size)

    def test_decode_batch(self):
        """Test decoding multiple messages"""
        batch_size = 50
        messages = np.random.randn(batch_size, 8)

        influences = self.processor.decode_batch(messages)

        self.assertEqual(influences.shape, (batch_size, 32))
        self.assertTrue(np.all(np.abs(influences) <= 1.0))

    def test_attention_batch(self):
        """Test batch attention computation"""
        batch_size = 50
        states = np.random.randn(batch_size, 128)
        messages = np.random.randn(batch_size, 8)

        attention = self.processor.compute_attention_batch(states, messages)

        self.assertEqual(attention.shape, (batch_size,))
        self.assertTrue(np.all((attention >= 0) & (attention <= 1)))

    def test_batch_performance(self):
        """Test that batching is faster than sequential"""
        batch_size = 100
        states = np.random.randn(batch_size, 128)

        # Batch processing
        start_time = time.time()
        batch_messages = self.processor.encode_batch(states)
        batch_time = time.time() - start_time

        # Sequential processing (simulated)
        start_time = time.time()
        seq_messages = []
        for state in states:
            msg = self.processor.encode_batch(np.array([state]))
            seq_messages.append(msg[0])
        seq_time = time.time() - start_time

        # Batch should be faster
        self.assertLess(batch_time, seq_time)
        print(f"\nBatch speedup: {seq_time/batch_time:.2f}x")


class TestCachedCoherenceAgent(unittest.TestCase):
    """Test coherence caching"""

    def setUp(self):
        self.mock_agent = MockAgent()
        self.cached_agent = CachedCoherenceAgent(self.mock_agent)

    def test_first_computation(self):
        """Test first coherence computation"""
        coherence = self.cached_agent.compute_coherence()

        self.assertEqual(coherence, 0.5)
        self.assertEqual(self.mock_agent.coherence_computations, 1)

    def test_cache_hit(self):
        """Test that cache is used when state unchanged"""
        # First call
        coherence1 = self.cached_agent.compute_coherence()
        computations_after_first = self.mock_agent.coherence_computations

        # Second call (should use cache)
        coherence2 = self.cached_agent.compute_coherence()
        computations_after_second = self.mock_agent.coherence_computations

        self.assertEqual(coherence1, coherence2)
        self.assertEqual(computations_after_first, computations_after_second)  # No new computation

        # Check statistics
        stats = self.cached_agent.get_cache_statistics()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)

    def test_cache_miss_on_state_change(self):
        """Test that cache misses when state changes"""
        # First call
        coherence1 = self.cached_agent.compute_coherence()

        # Change state
        self.cached_agent.agent.state = np.random.randn(128)

        # Second call (should recompute)
        coherence2 = self.cached_agent.compute_coherence()

        self.assertEqual(self.mock_agent.coherence_computations, 2)

    def test_manual_invalidation(self):
        """Test manual cache invalidation"""
        coherence1 = self.cached_agent.compute_coherence()

        # Invalidate cache
        self.cached_agent.invalidate_cache()

        # Should recompute
        coherence2 = self.cached_agent.compute_coherence()

        self.assertEqual(self.mock_agent.coherence_computations, 2)

    def test_cache_performance(self):
        """Test that caching improves performance"""
        n_calls = 100

        # With cache
        start_time = time.time()
        for _ in range(n_calls):
            self.cached_agent.compute_coherence()
        cached_time = time.time() - start_time

        # Without cache (always invalidate)
        start_time = time.time()
        for _ in range(n_calls):
            self.cached_agent.invalidate_cache()
            self.cached_agent.compute_coherence()
        uncached_time = time.time() - start_time

        # Cached should be much faster
        self.assertLess(cached_time, uncached_time)
        speedup = uncached_time / cached_time
        print(f"\nCache speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 10)  # Should be at least 10x faster


class TestSparseMapElites(unittest.TestCase):
    """Test sparse MAP-Elites updates"""

    def setUp(self):
        self.map_elites = MockMapElites()
        self.sparse_map_elites = SparseMapElites(self.map_elites)

    def test_first_addition(self):
        """Test first solution addition"""
        agent = MockAgent(0)
        fitness = 0.8
        behavior = np.array([0.5, 0.3])

        updated = self.sparse_map_elites.add_solution_lazy(agent, fitness, behavior)

        self.assertTrue(updated)
        self.assertEqual(self.map_elites.add_solution_calls, 1)

    def test_skip_worse_solution(self):
        """Test that worse solutions are skipped"""
        agent1 = MockAgent(0)
        agent2 = MockAgent(1)
        behavior = np.array([0.5, 0.3])

        # Add good solution
        self.sparse_map_elites.add_solution_lazy(agent1, 0.8, behavior)

        # Try to add worse solution (same niche)
        updated = self.sparse_map_elites.add_solution_lazy(agent2, 0.6, behavior)

        self.assertFalse(updated)
        self.assertEqual(self.sparse_map_elites.skipped_updates, 1)

    def test_accept_better_solution(self):
        """Test that better solutions are accepted"""
        agent1 = MockAgent(0)
        agent2 = MockAgent(1)
        behavior = np.array([0.5, 0.3])

        # Add initial solution
        self.sparse_map_elites.add_solution_lazy(agent1, 0.6, behavior)

        # Add better solution
        updated = self.sparse_map_elites.add_solution_lazy(agent2, 0.8, behavior)

        self.assertTrue(updated)
        self.assertEqual(self.map_elites.add_solution_calls, 2)

    def test_skip_statistics(self):
        """Test skip rate statistics"""
        agents = [MockAgent(i) for i in range(100)]
        behavior = np.array([0.5, 0.3])

        # Add many solutions with same behavior
        for i, agent in enumerate(agents):
            fitness = 0.5 + np.random.randn() * 0.1
            self.sparse_map_elites.add_solution_lazy(agent, fitness, behavior)

        stats = self.sparse_map_elites.get_statistics()

        self.assertEqual(stats['update_attempts'], 100)
        self.assertGreater(stats['skipped_updates'], 0)
        self.assertGreater(stats['skip_rate'], 0)

        print(f"\nSparse MAP-Elites skip rate: {stats['skip_rate']:.1%}")
        print(f"Speedup estimate: {stats['speedup_estimate']:.2f}x")


class TestIntegration(unittest.TestCase):
    """Integration tests for optimized system"""

    def test_create_system(self):
        """Test creating optimized system"""
        manager = create_optimized_phase4c_system(
            env_size=20,
            initial_population=10,
            use_batch_processing=True,
            use_cached_coherence=True,
            use_sparse_map_elites=True,
            device='cpu'
        )

        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, 'batch_processor'))
        self.assertTrue(hasattr(manager, 'cached_agents'))
        self.assertTrue(hasattr(manager, 'sparse_map_elites'))

    def test_system_step(self):
        """Test that optimized system can step"""
        manager = create_optimized_phase4c_system(
            env_size=20,
            initial_population=10,
            use_batch_processing=True,
            use_cached_coherence=True,
            use_sparse_map_elites=True,
            device='cpu'
        )

        # Run a few steps
        for _ in range(5):
            stats = manager.step()

            self.assertIn('population_size', stats)
            self.assertIn('avg_coherence', stats)

    def test_optimization_statistics(self):
        """Test that optimization statistics are collected"""
        manager = create_optimized_phase4c_system(
            env_size=20,
            initial_population=10,
            use_batch_processing=True,
            use_cached_coherence=True,
            use_sparse_map_elites=True,
            device='cpu'
        )

        # Run steps
        for _ in range(10):
            stats = manager.step()

        # Check optimization stats
        self.assertIn('optimization', stats)
        opt_stats = stats['optimization']

        if manager.batch_processor:
            self.assertIn('batch_processing', opt_stats)

        if manager.cached_agents:
            self.assertIn('cached_coherence', opt_stats)

        if manager.sparse_map_elites:
            self.assertIn('sparse_map_elites', opt_stats)


def run_performance_tests():
    """Run performance comparison tests"""
    print("\n" + "="*70)
    print("PERFORMANCE TESTS")
    print("="*70)

    # Test 1: Batch Processing Performance
    print("\n1. Batch Processing Performance")
    print("-" * 40)

    processor = BatchedMessageProcessor(device='cpu')
    batch_sizes = [10, 50, 100, 200]

    for batch_size in batch_sizes:
        states = np.random.randn(batch_size, 128)

        start = time.time()
        messages = processor.encode_batch(states)
        elapsed = time.time() - start

        throughput = batch_size / elapsed
        print(f"Batch size {batch_size:3d}: {elapsed:.4f}s ({throughput:.0f} items/sec)")

    # Test 2: Cache Hit Rate
    print("\n2. Cache Performance")
    print("-" * 40)

    mock_agent = MockAgent()
    cached = CachedCoherenceAgent(mock_agent)

    # Simulate realistic access pattern
    for _ in range(100):
        cached.compute_coherence()  # Same state

    for _ in range(10):
        cached.agent.state = np.random.randn(128)
        cached.compute_coherence()  # Different state

    stats = cached.get_cache_statistics()
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")

    # Test 3: Sparse MAP-Elites Skip Rate
    print("\n3. Sparse MAP-Elites Performance")
    print("-" * 40)

    map_elites = MockMapElites()
    sparse = SparseMapElites(map_elites)

    # Add many solutions
    for i in range(200):
        agent = MockAgent(i)
        behavior = np.random.rand(2)
        fitness = np.random.rand()
        sparse.add_solution_lazy(agent, fitness, behavior)

    stats = sparse.get_statistics()
    print(f"Update attempts: {stats['update_attempts']}")
    print(f"Actual updates: {stats['actual_updates']}")
    print(f"Skipped: {stats['skipped_updates']}")
    print(f"Skip rate: {stats['skip_rate']:.1%}")
    print(f"Speedup estimate: {stats['speedup_estimate']:.2f}x")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENESIS Phase 4C: Optimization Tests")
    print("="*70)

    # Run unit tests
    print("\nRunning unit tests...")
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run performance tests
    if result.wasSuccessful():
        run_performance_tests()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("="*70)
        sys.exit(1)
