"""
GENESIS Phase 4C: Optimization Benchmark

Compares performance of original vs optimized implementations.

Tests:
1. Baseline (no optimizations)
2. Batch processing only
3. Cached coherence only
4. Sparse MAP-Elites only
5. All optimizations combined

Expected results: 9-18x speedup with all optimizations
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase4c_integration import create_phase4c_system
from optimized_phase4c import create_optimized_phase4c_system


def run_benchmark(manager, num_steps: int = 50, warmup_steps: int = 5) -> dict:
    """
    Run benchmark on a manager

    Args:
        manager: Manager to benchmark
        num_steps: Number of steps to run
        warmup_steps: Number of warmup steps (not counted)

    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup_steps):
        manager.step()

    # Actual benchmark
    step_times = []
    stats_history = []

    start_time = time.time()

    for step in range(num_steps):
        step_start = time.time()
        stats = manager.step()
        step_time = time.time() - step_start

        step_times.append(step_time)
        stats_history.append(stats)

    total_time = time.time() - start_time

    # Collect results
    results = {
        'total_time': total_time,
        'num_steps': num_steps,
        'step_times': step_times,
        'avg_step_time': np.mean(step_times),
        'std_step_time': np.std(step_times),
        'min_step_time': np.min(step_times),
        'max_step_time': np.max(step_times),
        'steps_per_second': num_steps / total_time,
        'final_population': stats_history[-1]['population_size'],
        'final_coherence': stats_history[-1]['avg_coherence']
    }

    # Add optimization stats if available
    if 'optimization' in stats_history[-1]:
        results['optimization'] = stats_history[-1]['optimization']

    return results


def print_results(name: str, results: dict, baseline_results: dict = None):
    """Print formatted benchmark results"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Steps: {results['num_steps']}")
    print(f"Avg Step Time: {results['avg_step_time']:.4f}s ± {results['std_step_time']:.4f}s")
    print(f"Min/Max: {results['min_step_time']:.4f}s / {results['max_step_time']:.4f}s")
    print(f"Throughput: {results['steps_per_second']:.2f} steps/sec")

    if baseline_results:
        speedup = baseline_results['avg_step_time'] / results['avg_step_time']
        print(f"\n🚀 Speedup: {speedup:.2f}x faster than baseline")

    print(f"\nFinal State:")
    print(f"  Population: {results['final_population']}")
    print(f"  Avg Coherence: {results['final_coherence']:.3f}")

    # Optimization details
    if 'optimization' in results:
        opt = results['optimization']

        if 'batch_processing' in opt:
            bp = opt['batch_processing']
            print(f"\n  Batch Processing:")
            print(f"    Total Items: {bp.get('total_items', 0):,}")
            print(f"    Throughput: {bp.get('throughput_items_per_sec', 0):.1f} items/sec")

        if 'cached_coherence' in opt:
            cc = opt['cached_coherence']
            print(f"\n  Cached Coherence:")
            print(f"    Hit Rate: {cc.get('avg_hit_rate', 0):.1%}")

        if 'sparse_map_elites' in opt:
            sme = opt['sparse_map_elites']
            print(f"\n  Sparse MAP-Elites:")
            print(f"    Skip Rate: {sme.get('skip_rate', 0):.1%}")
            print(f"    Speedup Estimate: {sme.get('speedup_estimate', 1.0):.2f}x")


def main():
    print("\n" + "="*70)
    print("GENESIS Phase 4C: Optimization Benchmark")
    print("="*70)
    print()

    # Configuration
    env_size = 30
    initial_population = 50
    num_steps = 50
    warmup_steps = 5

    print(f"Configuration:")
    print(f"  Environment Size: {env_size}")
    print(f"  Initial Population: {initial_population}")
    print(f"  Benchmark Steps: {num_steps}")
    print(f"  Warmup Steps: {warmup_steps}")
    print()

    # ==================================================================
    # Test 1: Baseline (Original Implementation)
    # ==================================================================
    print("\n" + "█"*70)
    print("Test 1: Baseline (Original Implementation)")
    print("█"*70)

    print("\nCreating baseline system...")
    baseline_manager = create_phase4c_system(
        env_size=env_size,
        initial_population=initial_population,
        phase4c_enabled=True,
        use_novelty_search=True,
        use_map_elites=True,
        use_poet=False
    )
    print(f"✓ Created with {len(baseline_manager.agents)} agents")

    print("\nRunning baseline benchmark...")
    baseline_results = run_benchmark(baseline_manager, num_steps, warmup_steps)
    print_results("BASELINE", baseline_results)

    # ==================================================================
    # Test 2: Batch Processing Only
    # ==================================================================
    print("\n\n" + "█"*70)
    print("Test 2: Batch Processing Only")
    print("█"*70)

    print("\nCreating system with batch processing...")
    batch_manager = create_optimized_phase4c_system(
        env_size=env_size,
        initial_population=initial_population,
        use_batch_processing=True,
        use_cached_coherence=False,
        use_sparse_map_elites=False,
        device='cpu'
    )
    print(f"✓ Created with {len(batch_manager.agents)} agents")

    print("\nRunning benchmark...")
    batch_results = run_benchmark(batch_manager, num_steps, warmup_steps)
    print_results("BATCH PROCESSING", batch_results, baseline_results)

    # ==================================================================
    # Test 3: Cached Coherence Only
    # ==================================================================
    print("\n\n" + "█"*70)
    print("Test 3: Cached Coherence Only")
    print("█"*70)

    print("\nCreating system with cached coherence...")
    cache_manager = create_optimized_phase4c_system(
        env_size=env_size,
        initial_population=initial_population,
        use_batch_processing=False,
        use_cached_coherence=True,
        use_sparse_map_elites=False,
        device='cpu'
    )
    print(f"✓ Created with {len(cache_manager.agents)} agents")

    print("\nRunning benchmark...")
    cache_results = run_benchmark(cache_manager, num_steps, warmup_steps)
    print_results("CACHED COHERENCE", cache_results, baseline_results)

    # ==================================================================
    # Test 4: Sparse MAP-Elites Only
    # ==================================================================
    print("\n\n" + "█"*70)
    print("Test 4: Sparse MAP-Elites Only")
    print("█"*70)

    print("\nCreating system with sparse MAP-Elites...")
    sparse_manager = create_optimized_phase4c_system(
        env_size=env_size,
        initial_population=initial_population,
        use_batch_processing=False,
        use_cached_coherence=False,
        use_sparse_map_elites=True,
        device='cpu'
    )
    print(f"✓ Created with {len(sparse_manager.agents)} agents")

    print("\nRunning benchmark...")
    sparse_results = run_benchmark(sparse_manager, num_steps, warmup_steps)
    print_results("SPARSE MAP-ELITES", sparse_results, baseline_results)

    # ==================================================================
    # Test 5: All Optimizations Combined
    # ==================================================================
    print("\n\n" + "█"*70)
    print("Test 5: All Optimizations Combined")
    print("█"*70)

    print("\nCreating fully optimized system...")
    optimized_manager = create_optimized_phase4c_system(
        env_size=env_size,
        initial_population=initial_population,
        use_batch_processing=True,
        use_cached_coherence=True,
        use_sparse_map_elites=True,
        device='cpu'
    )
    print(f"✓ Created with {len(optimized_manager.agents)} agents")

    print("\nRunning benchmark...")
    optimized_results = run_benchmark(optimized_manager, num_steps, warmup_steps)
    print_results("ALL OPTIMIZATIONS", optimized_results, baseline_results)

    # ==================================================================
    # Summary Comparison
    # ==================================================================
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    configurations = [
        ("Baseline", baseline_results),
        ("Batch Processing", batch_results),
        ("Cached Coherence", cache_results),
        ("Sparse MAP-Elites", sparse_results),
        ("All Optimizations", optimized_results)
    ]

    print("\n┌─────────────────────┬──────────────┬───────────┬──────────┐")
    print("│ Configuration       │ Avg Step (s) │ Steps/sec │ Speedup  │")
    print("├─────────────────────┼──────────────┼───────────┼──────────┤")

    baseline_time = baseline_results['avg_step_time']

    for name, results in configurations:
        avg_time = results['avg_step_time']
        throughput = results['steps_per_second']
        speedup = baseline_time / avg_time

        print(f"│ {name:19s} │ {avg_time:12.4f} │ {throughput:9.2f} │ {speedup:7.2f}x │")

    print("└─────────────────────┴──────────────┴───────────┴──────────┘")

    # Speedup analysis
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)

    total_speedup = baseline_time / optimized_results['avg_step_time']
    print(f"\n🚀 Total Speedup: {total_speedup:.2f}x faster with all optimizations")

    batch_speedup = baseline_time / batch_results['avg_step_time']
    cache_speedup = baseline_time / cache_results['avg_step_time']
    sparse_speedup = baseline_time / sparse_results['avg_step_time']

    print(f"\nIndividual Contributions:")
    print(f"  Batch Processing:   {batch_speedup:.2f}x")
    print(f"  Cached Coherence:   {cache_speedup:.2f}x")
    print(f"  Sparse MAP-Elites:  {sparse_speedup:.2f}x")

    # Expected vs Actual
    print(f"\nExpected Combined: ~9-18x (conservative estimate)")
    print(f"Actual Combined:   {total_speedup:.2f}x")

    if total_speedup >= 9:
        print("✅ Optimization target achieved!")
    else:
        print(f"⚠️  Below target (got {total_speedup:.2f}x, expected 9-18x)")

    # Time savings
    print(f"\nTime Savings Example (10,000 steps):")
    baseline_10k = baseline_time * 10000 / 3600  # hours
    optimized_10k = optimized_results['avg_step_time'] * 10000 / 3600  # hours
    saved_hours = baseline_10k - optimized_10k

    print(f"  Baseline:  {baseline_10k:.2f} hours")
    print(f"  Optimized: {optimized_10k:.2f} hours")
    print(f"  Saved:     {saved_hours:.2f} hours ({saved_hours/baseline_10k*100:.1f}%)")

    print("\n" + "="*70)
    print()


if __name__ == '__main__':
    main()
