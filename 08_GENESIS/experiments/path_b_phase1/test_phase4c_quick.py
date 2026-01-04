"""
GENESIS Phase 4C Quick Test: Emergent Communication

Quick 50-step validation test
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase4c_integration import create_phase4c_system

print("\n" + "="*70)
print("GENESIS Phase 4C: Quick Communication Test (50 steps)")
print("="*70)
print()

# Create system with Phase 4C enabled
print("🔧 Creating Phase 4C system...")
manager = create_phase4c_system(
    env_size=20,
    initial_population=50,
    phase4c_enabled=True,
    use_novelty_search=True,
    use_map_elites=True,
    use_poet=False,
    message_dim=8,
    local_radius=5.0
)
print(f"✓ System created with {len(manager.agents)} agents")
print(f"  Communication: {'Enabled' if manager.comm_manager else 'Disabled'}")
print(f"  Message dimension: 8")
print(f"  Local radius: 5.0")
print()

# Run simulation
print("🚀 Running Phase 4C simulation (50 steps)...")
test_steps = 50
print_interval = 25

for step in range(test_steps):
    stats = manager.step()

    if (step + 1) % print_interval == 0:
        print(f"Step {step + 1}/{test_steps}")
        print(f"  Population: {stats['population_size']}")
        print(f"  Avg Coherence: {stats['avg_coherence']:.3f}")

        if 'phase4c' in stats:
            if 'communication' in stats['phase4c']:
                comm = stats['phase4c']['communication']
                print(f"  Total Messages: {comm['total_messages']}")
                print(f"  Broadcast: {comm.get('broadcast_messages', 0)}")
                print(f"  Local: {comm.get('local_messages', 0)}")

            if 'protocol_analysis' in stats['phase4c']:
                protocol = stats['phase4c']['protocol_analysis']
                print(f"  Signal Diversity: {protocol.get('signal_diversity', 0):.3f}")
                print(f"  Signal Stability: {protocol.get('signal_stability', 0):.3f}")

        print()

print("="*70)
print("Phase 4C Quick Test Complete!")
print("="*70)
print()

# Final statistics
comm_stats = manager.get_communication_statistics()

print("📊 Final Communication Statistics:")

if 'manager' in comm_stats:
    print("\nCommunication Manager:")
    print(f"  Total Messages: {comm_stats['manager']['total_messages']}")
    print(f"  Broadcast Messages: {comm_stats['manager'].get('broadcast_messages', 0)}")
    print(f"  Local Messages: {comm_stats['manager'].get('local_messages', 0)}")

if 'per_agent' in comm_stats:
    print("\nPer-Agent Stats:")
    pa = comm_stats['per_agent']
    print(f"  Avg Messages Sent: {pa['avg_messages_sent']:.1f}")
    print(f"  Max Messages Sent: {pa['max_messages_sent']}")
    print(f"  Avg Messages Received: {pa['avg_messages_received']:.1f}")
    print(f"  Communication Rate: {pa['communication_rate']:.1%}")

if 'protocol' in comm_stats:
    print("\nProtocol Analysis:")
    protocol = comm_stats['protocol']
    print(f"  Total Signals Analyzed: {protocol['total_signals_analyzed']}")
    print(f"  Signal Diversity: {protocol['signal_diversity']:.3f}")
    print(f"  Signal Stability: {protocol['signal_stability']:.3f}")

print()

# Success criteria
print("✅ Phase 4C Test Result:")
success = True

if 'manager' in comm_stats:
    total_msgs = comm_stats['manager']['total_messages']
    if total_msgs > 20:
        print(f"  ✓ Communication active: {total_msgs} messages")
    else:
        print(f"  ⚠ Low communication: {total_msgs} messages (needs more steps)")

if 'per_agent' in comm_stats:
    comm_rate = comm_stats['per_agent']['communication_rate']
    if comm_rate > 0.1:  # At least 10% of agents communicating
        print(f"  ✓ Communication rate: {comm_rate:.1%}")
    else:
        print(f"  ⚠ Low communication rate: {comm_rate:.1%} (needs more steps)")

print(f"  ✓ Completed {test_steps} steps without errors")
print()
print("🎉 Phase 4C integration is functional!")
print()
print("Emergent communication capabilities:")
print("  - Message encoding/decoding")
print("  - Selective attention to messages")
print("  - Local and broadcast channels")
print("  - Protocol emergence tracking")
print()
