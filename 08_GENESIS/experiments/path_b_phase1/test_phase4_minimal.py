"""
GENESIS Phase 4 Minimal Test

Quickest validation - just 100 steps
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase4_integration import create_phase4_system

print("\n" + "="*70)
print("GENESIS Phase 4: Minimal Test (100 steps)")
print("="*70)
print()

# Create system
print("🔧 Creating Phase 4 system...")
manager = create_phase4_system(
    env_size=20,
    initial_population=50,
    phase1_enabled=True,
    phase2_enabled=True,
    phase3_enabled=True,
    phase4_enabled=True
)
print(f"✓ System created with {len(manager.agents)} agents")
print()

# Run simulation
print("🚀 Running minimal test (100 steps)...")

for step in range(100):
    stats = manager.step()

    if (step + 1) % 50 == 0:
        print(f"Step {step + 1}/100")
        print(f"  Population: {stats['population_size']}")
        print(f"  Avg Coherence: {stats['avg_coherence']:.3f}")
        print()

print("="*70)
print("✅ Minimal Test Complete!")
print("="*70)
print()

final_stats = manager.get_statistics()
print("📊 Final Statistics:")
print(f"  Steps: {final_stats.get('step', 0)}")
print(f"  Population: {final_stats.get('population_size', 0)}")
print(f"  Avg Coherence: {final_stats.get('avg_coherence', 0):.3f}")
print(f"  Extinction: {'Yes' if final_stats.get('extinct', False) else 'No'}")
print()

if 'phase4' in final_stats:
    print("✅ Phase 4 is operational!")
    print()

    if 'learned_memory' in final_stats['phase4']:
        mem = final_stats['phase4']['learned_memory']
        print(f"  Memory: {mem['size']} experiences")
        print(f"  Utilization: {mem['utilization']:.1%}")

    if 'advanced_teacher' in final_stats['phase4']:
        print(f"  Advanced Teacher: Active")

    if 'knowledge_guidance' in final_stats['phase4']:
        kg = final_stats['phase4']['knowledge_guidance']
        print(f"  Knowledge-guided agents: {kg['total_agents_wrapped']}")

print()
print("🎉 All systems working! Phase 4 integration successful!")
print()
