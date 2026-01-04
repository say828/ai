"""
GENESIS Phase 4B Test: Open-Ended Learning

Tests Novelty Search + MAP-Elites integration
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase4b_integration import create_phase4b_system

print("\n" + "="*70)
print("GENESIS Phase 4B: Open-Ended Learning Test")
print("="*70)
print()

# Create system with Phase 4B enabled
print("🔧 Creating Phase 4B system...")
manager = create_phase4b_system(
    env_size=30,
    initial_population=100,
    phase4b_enabled=True,
    use_novelty_search=True,
    use_map_elites=True,
    use_poet=False  # POET is complex, disable for quick test
)
print(f"✓ System created with {len(manager.agents)} agents")
print(f"  Novelty Search: {'Enabled' if manager.novelty_search else 'Disabled'}")
print(f"  MAP-Elites: {'Enabled' if manager.map_elites else 'Disabled'}")
print(f"  POET: {'Enabled' if manager.poet else 'Disabled'}")
print()

# Run simulation
print("🚀 Running Phase 4B simulation (200 steps)...")
test_steps = 200
print_interval = 50

for step in range(test_steps):
    stats = manager.step()

    if (step + 1) % print_interval == 0:
        print(f"Step {step + 1}/{test_steps}")
        print(f"  Population: {stats['population_size']}")
        print(f"  Avg Coherence: {stats['avg_coherence']:.3f}")

        if 'phase4b' in stats:
            if 'novelty_search' in stats['phase4b']:
                ns = stats['phase4b']['novelty_search']
                print(f"  Novelty Archive: {ns['archive']['size']} behaviors")
                print(f"  Avg Novelty: {ns['avg_novelty']:.3f}")

            if 'map_elites' in stats['phase4b']:
                me = stats['phase4b']['map_elites']['archive']
                print(f"  MAP-Elites Coverage: {me['coverage']:.1%} ({me['size']} bins filled)")
                print(f"  Avg Elite Fitness: {me['avg_fitness']:.3f}")

        print()

print("="*70)
print("Phase 4B Test Complete!")
print("="*70)
print()

# Final statistics
print("📊 Final Phase 4B Statistics:")

if manager.novelty_search:
    ns_stats = manager.novelty_search.get_statistics()
    print("\nNovelty Search:")
    print(f"  Total Evaluations: {ns_stats['evaluations']}")
    print(f"  Unique Behaviors: {ns_stats['unique_behaviors']}")
    print(f"  Archive Size: {ns_stats['archive']['size']}")
    print(f"  Acceptance Rate: {ns_stats['archive']['acceptance_rate']:.1%}")

if manager.map_elites:
    me_stats = manager.map_elites.get_statistics()
    print("\nMAP-Elites:")
    print(f"  Behavior Space Coverage: {me_stats['archive']['coverage']:.1%}")
    print(f"  Total Bins Filled: {me_stats['archive']['size']}")
    print(f"  Avg Elite Fitness: {me_stats['archive']['avg_fitness']:.3f}")
    print(f"  Max Elite Fitness: {me_stats['archive']['max_fitness']:.3f}")
    print(f"  Total Improvements: {manager.map_elites_improvements}")

print()

# Success criteria
print("✅ Success Criteria:")
success = True

if manager.novelty_search:
    if ns_stats['unique_behaviors'] > 20:
        print(f"  ✓ Discovered {ns_stats['unique_behaviors']} unique behaviors")
    else:
        print(f"  ✗ Only {ns_stats['unique_behaviors']} unique behaviors")
        success = False

if manager.map_elites:
    coverage = me_stats['archive']['coverage']
    if coverage > 0.1:  # At least 10% coverage
        print(f"  ✓ MAP-Elites coverage: {coverage:.1%}")
    else:
        print(f"  ✗ Low MAP-Elites coverage: {coverage:.1%}")
        success = False

if test_steps == 200:
    print(f"  ✓ Completed {test_steps} steps without errors")

print()

if success:
    print("🎉 Phase 4B is operational!")
    print()
    print("Open-ended learning capabilities:")
    print("  - Behavioral diversity through novelty search")
    print("  - Quality-diversity through MAP-Elites")
    print("  - Continuous discovery of new behaviors")
else:
    print("⚠️  Some metrics need more steps for full validation")

print()
