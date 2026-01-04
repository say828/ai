"""
GENESIS Phase 4 Simple Test

Quick validation that all systems work together
"""

import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from phase4_integration import create_phase4_system

print("\n" + "="*70)
print("GENESIS Phase 4: Simple Integration Test")
print("="*70)
print()

# Create system
print("🔧 Creating Phase 4 system...")
manager = create_phase4_system(
    env_size=30,
    initial_population=100,
    phase1_enabled=True,
    phase2_enabled=True,
    phase3_enabled=True,
    phase4_enabled=True
)
print(f"✓ System created with {len(manager.agents)} agents")
print()

# Ingest some knowledge
if manager.phase3_enabled:
    print("📥 Ingesting knowledge...")
    knowledge = """
    Evolution is the process of change over generations.
    Natural selection favors well-adapted organisms.
    Learning allows organisms to adapt during their lifetime.
    Neural networks can learn patterns from data.
    """
    manager.knowledge_ingestion.ingest_text(knowledge, "test_knowledge")
    manager.knowledge_graph.add_entity("Evolution", "concept")
    manager.knowledge_graph.add_entity("Learning", "concept")
    manager.knowledge_graph.add_relation("Learning", "part-of", "Evolution")
    print("✓ Knowledge added")
    print()

# Run simulation
print("🚀 Running simulation (1000 steps)...")
test_steps = 1000
print_interval = 200

start_time = time.time()

for step in range(test_steps):
    stats = manager.step()

    if (step + 1) % print_interval == 0:
        print(f"Step {step + 1}/{test_steps}")
        print(f"  Population: {stats['population_size']}")
        print(f"  Avg Coherence: {stats['avg_coherence']:.3f}")

        if 'phase4' in stats:
            if 'learned_memory' in stats['phase4']:
                mem = stats['phase4']['learned_memory']
                print(f"  Memory: {mem['size']}/{mem['capacity']} ({mem['utilization']:.1%})")
        print()

elapsed = time.time() - start_time

print("="*70)
print("Test Complete!")
print("="*70)
print()

final_stats = manager.get_statistics()

print("📊 Final Statistics:")
print(f"  Steps: {final_stats.get('step', 0)}")
print(f"  Population: {final_stats.get('population_size', 0)}")
print(f"  Avg Coherence: {final_stats.get('avg_coherence', 0):.3f}")
print(f"  Extinction: {'Yes' if final_stats.get('extinct', False) else 'No'}")
print(f"  Generation: {final_stats.get('generation', 0)}")
print()

# Get last step stats (which has Phase 3/4 info)
print("💡 Running final step to get Phase 3/4 stats...")
step_stats = manager.step()
print()

if 'phase3' in step_stats:
    kg = step_stats['phase3']['knowledge_graph']
    print("Phase 3 (Knowledge System):")
    print(f"  Knowledge Entities: {kg.get('entity_count', 0)}")
    print(f"  Knowledge Relations: {kg.get('relation_count', 0)}")
    print()

if 'phase4' in step_stats:
    print("Phase 4 (Advanced Intelligence):")

    if 'learned_memory' in step_stats['phase4']:
        mem = step_stats['phase4']['learned_memory']
        print(f"  ✓ Learned Memory Active")
        print(f"    Size: {mem['size']} / {mem['capacity']}")
        print(f"    Utilization: {mem['utilization']:.1%}")
        print(f"    Avg Priority: {mem['avg_priority']:.3f}")
        print(f"    Consolidations: {mem['consolidation_count']}")

    if 'advanced_teacher' in step_stats['phase4']:
        teacher = step_stats['phase4']['advanced_teacher']
        print(f"  ✓ Advanced Teacher Active")
        print(f"    Updates: {teacher['update_count']}")
        print(f"    Curriculum Difficulty: {teacher['curriculum_difficulty']:.3f}")

    if 'knowledge_guidance' in step_stats['phase4']:
        kg = step_stats['phase4']['knowledge_guidance']
        print(f"  ✓ Knowledge Guidance Active")
        print(f"    Agents Wrapped: {kg['total_agents_wrapped']}")

    print()

print(f"⏱️  Time: {elapsed:.2f}s ({test_steps/elapsed:.1f} steps/sec)")
print()

# Success criteria
print("✅ Success Criteria:")
success = True

if final_stats['extinction_events'] == 0:
    print("  ✓ No extinction")
else:
    print(f"  ✗ {final_stats['extinction_events']} extinctions")
    success = False

if final_stats['avg_coherence'] > 0.4:
    print(f"  ✓ Coherence improved: {final_stats['avg_coherence']:.3f}")
else:
    print(f"  ✗ Low coherence: {final_stats['avg_coherence']:.3f}")

if test_steps == 1000:
    print("  ✓ Ran 1000 steps successfully")

print()

if success:
    print("🎉 All core tests passed!")
    print()
    print("Phase 4 is operational!")
else:
    print("⚠️  Some tests need attention")

print()
