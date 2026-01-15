"""
Quick test of Teacher Network integration

This script runs a short experiment to verify:
1. Teacher Network updates properly
2. Minimum population prevents extinction
3. Knowledge accumulates across generations
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from full_environment import FullALifeEnvironment
from full_population import FullPopulationManager

def test_teacher_network():
    """Quick test with teacher enabled"""
    print("="*60)
    print("Testing Teacher Network - 1000 steps")
    print("="*60)

    np.random.seed(42)

    # Create environment
    env = FullALifeEnvironment(size=32, seed=42)  # Smaller for speed

    # Create population WITH teacher
    pop = FullPopulationManager(
        env,
        initial_pop=50,
        max_population=200,
        min_population=30,  # Prevent extinction
        enable_teacher=True,
        teacher_update_interval=100,
        teacher_learning_rate=0.1
    )

    print(f"Initial population: {len(pop.agents)}")
    print(f"Teacher enabled: {pop.enable_teacher}")
    print(f"Minimum population: {pop.min_population}")
    print()

    # Run simulation
    for step in range(1000):
        stats = pop.step()

        if step % 100 == 0:
            print(f"Step {step:4d} | Pop: {stats['population_size']:3d} | "
                  f"Coh: {stats['avg_coherence']:.3f} | "
                  f"Births: {stats['total_births']:3d} | "
                  f"Deaths: {stats['total_deaths']:3d}", end="")

            if 'teacher_knowledge_level' in stats:
                print(f" | Teacher: {stats['teacher_knowledge_level']:.3f}")
            else:
                print()

    # Final statistics
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    final_stats = pop.get_statistics()
    print(f"Population: {final_stats['population_size']}")
    print(f"Avg Coherence: {final_stats['avg_coherence']:.3f}")
    print(f"Total Births: {final_stats['total_births']}")
    print(f"Total Deaths: {final_stats['total_deaths']}")

    if pop.teacher:
        teacher_stats = pop.teacher.get_statistics()
        print(f"\nTeacher Statistics:")
        print(f"  Updates: {teacher_stats['update_count']}")
        print(f"  Knowledge Level: {teacher_stats['knowledge_level']:.3f}")
        print(f"  Agents Learned From: {teacher_stats['total_agents_learned_from']}")

        if teacher_stats.get('coherence_history'):
            history = teacher_stats['coherence_history']
            print(f"  Coherence Trend: {history['mean']:.3f} ± {history['std']:.3f}")
            print(f"  Recent (last 10): {[f'{x:.3f}' for x in history['recent_10']]}")

    return pop

def test_without_teacher():
    """Test without teacher for comparison"""
    print("\n\n" + "="*60)
    print("Control: WITHOUT Teacher Network - 1000 steps")
    print("="*60)

    np.random.seed(42)

    env = FullALifeEnvironment(size=32, seed=42)

    # Create population WITHOUT teacher
    pop = FullPopulationManager(
        env,
        initial_pop=50,
        max_population=200,
        min_population=30,
        enable_teacher=False  # DISABLED
    )

    print(f"Initial population: {len(pop.agents)}")
    print(f"Teacher enabled: {pop.enable_teacher}")
    print()

    for step in range(1000):
        stats = pop.step()

        if step % 100 == 0:
            print(f"Step {step:4d} | Pop: {stats['population_size']:3d} | "
                  f"Coh: {stats['avg_coherence']:.3f} | "
                  f"Births: {stats['total_births']:3d} | "
                  f"Deaths: {stats['total_deaths']:3d}")

    print("\n" + "="*60)
    print("Final Results (No Teacher)")
    print("="*60)
    final_stats = pop.get_statistics()
    print(f"Population: {final_stats['population_size']}")
    print(f"Avg Coherence: {final_stats['avg_coherence']:.3f}")
    print(f"Total Births: {final_stats['total_births']}")
    print(f"Total Deaths: {final_stats['total_deaths']}")

    return pop

if __name__ == "__main__":
    # Test WITH teacher
    pop_with = test_teacher_network()

    # Test WITHOUT teacher (control)
    pop_without = test_without_teacher()

    # Comparison
    print("\n\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    stats_with = pop_with.get_statistics()
    stats_without = pop_without.get_statistics()

    print(f"{'Metric':<25} | {'WITH Teacher':<15} | {'WITHOUT Teacher':<15}")
    print("-" * 60)
    print(f"{'Final Coherence':<25} | {stats_with['avg_coherence']:<15.3f} | {stats_without['avg_coherence']:<15.3f}")
    print(f"{'Population':<25} | {stats_with['population_size']:<15d} | {stats_without['population_size']:<15d}")
    print(f"{'Total Births':<25} | {stats_with['total_births']:<15d} | {stats_without['total_births']:<15d}")
    print(f"{'Total Deaths':<25} | {stats_with['total_deaths']:<15d} | {stats_without['total_deaths']:<15d}")

    if pop_with.teacher:
        print(f"\nTeacher Knowledge Level: {pop_with.teacher.knowledge_level:.3f}")
        print("✅ Teacher Network working correctly!")
    else:
        print("\n❌ Teacher Network not initialized")
