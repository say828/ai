"""
GENESIS Phase 4: Long-Term Experiment

Runs extended experiments (10K-100K steps) with comprehensive logging,
checkpointing, and analysis.

Usage:
    python long_term_experiment.py --steps 10000 --checkpoint-interval 1000
"""

import numpy as np
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from phase4c_integration import create_phase4c_system

def parse_args():
    parser = argparse.ArgumentParser(description='GENESIS Phase 4 Long-Term Experiment')

    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of simulation steps (default: 10000)')
    parser.add_argument('--env-size', type=int, default=30,
                       help='Environment size (default: 30)')
    parser.add_argument('--population', type=int, default=100,
                       help='Initial population (default: 100)')

    parser.add_argument('--phase4a', action='store_true', default=True,
                       help='Enable Phase 4A (Advanced Intelligence)')
    parser.add_argument('--phase4b', action='store_true', default=True,
                       help='Enable Phase 4B (Open-Ended Learning)')
    parser.add_argument('--phase4c', action='store_true', default=True,
                       help='Enable Phase 4C (Emergent Communication)')

    parser.add_argument('--novelty-search', action='store_true', default=True,
                       help='Enable Novelty Search')
    parser.add_argument('--map-elites', action='store_true', default=True,
                       help='Enable MAP-Elites')
    parser.add_argument('--poet', action='store_true', default=False,
                       help='Enable POET (warning: expensive)')

    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Save checkpoint every N steps (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Print log every N steps (default: 100)')
    parser.add_argument('--output-dir', type=str, default='results/long_term',
                       help='Output directory (default: results/long_term)')

    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')

    return parser.parse_args()


def create_output_directory(output_dir):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = Path(output_dir) / timestamp
    full_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (full_path / 'checkpoints').mkdir(exist_ok=True)
    (full_path / 'logs').mkdir(exist_ok=True)
    (full_path / 'analysis').mkdir(exist_ok=True)

    return full_path


def save_checkpoint(manager, step, output_dir):
    """Save system checkpoint"""
    checkpoint_path = output_dir / 'checkpoints' / f'checkpoint_{step:06d}.pkl'

    checkpoint = {
        'step': step,
        'current_step': manager.current_step,
        'agents': manager.agents,
        'best_fitness': manager.best_fitness if hasattr(manager, 'best_fitness') else None,

        # Phase 4B
        'novelty_archive': manager.novelty_search.archive if manager.novelty_search else None,
        'map_elites_archive': manager.map_elites.archive if manager.map_elites else None,

        # Phase 4C
        'communication_stats': manager.get_communication_statistics() if manager.comm_manager else None,
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"  💾 Checkpoint saved: {checkpoint_path.name}")


def load_checkpoint(checkpoint_path):
    """Load system checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"📂 Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint


def save_statistics(stats_history, output_dir):
    """Save statistics history"""
    stats_path = output_dir / 'logs' / 'statistics.json'

    # Convert numpy arrays to lists for JSON serialization
    serializable_stats = []
    for stats in stats_history:
        serializable = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                serializable[key] = {k: float(v) if isinstance(v, (np.number, np.ndarray)) else v
                                    for k, v in value.items()}
            elif isinstance(value, (np.number, np.ndarray)):
                serializable[key] = float(value)
            else:
                serializable[key] = value
        serializable_stats.append(serializable)

    with open(stats_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2)


def print_statistics(step, total_steps, stats, elapsed_time):
    """Print formatted statistics"""
    steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
    eta_sec = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
    eta_hours = eta_sec / 3600

    print(f"\n{'='*70}")
    print(f"Step {step:,}/{total_steps:,} ({step/total_steps*100:.1f}%)")
    print(f"Elapsed: {elapsed_time/3600:.2f}h | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.2f}h")
    print(f"{'='*70}")

    print(f"  Population: {stats['population_size']}")
    print(f"  Avg Coherence: {stats['avg_coherence']:.3f}")

    if 'phase4b' in stats:
        if 'novelty_search' in stats['phase4b']:
            ns = stats['phase4b']['novelty_search']
            print(f"\n  📊 Novelty Search:")
            print(f"     Archive Size: {ns['archive']['size']:,} behaviors")
            print(f"     Avg Novelty: {ns['avg_novelty']:.3f}")
            print(f"     Unique Behaviors: {ns['unique_behaviors']:,}")

        if 'map_elites' in stats['phase4b']:
            me = stats['phase4b']['map_elites']['archive']
            print(f"\n  🗺️  MAP-Elites:")
            print(f"     Coverage: {me['coverage']:.2%} ({me['size']:,} bins)")
            print(f"     Avg Elite Fitness: {me['avg_fitness']:.3f}")
            print(f"     Max Elite Fitness: {me['max_fitness']:.3f}")

    if 'phase4c' in stats and 'communication' in stats['phase4c']:
        comm = stats['phase4c']['communication']
        print(f"\n  💬 Communication:")
        print(f"     Total Messages: {comm['total_messages']:,}")
        print(f"     Broadcast: {comm.get('broadcast_messages', 0):,}")
        print(f"     Local: {comm.get('local_messages', 0):,}")

        if 'protocol_analysis' in stats['phase4c']:
            protocol = stats['phase4c']['protocol_analysis']
            print(f"     Signal Diversity: {protocol.get('signal_diversity', 0):.3f}")
            print(f"     Signal Stability: {protocol.get('signal_stability', 0):.3f}")


def analyze_results(stats_history, output_dir):
    """Analyze and save final results"""
    analysis_path = output_dir / 'analysis' / 'final_analysis.txt'

    with open(analysis_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GENESIS Phase 4: Long-Term Experiment Analysis\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total Steps: {len(stats_history):,}\n")
        f.write(f"Final Population: {stats_history[-1]['population_size']}\n")
        f.write(f"Initial Coherence: {stats_history[0]['avg_coherence']:.3f}\n")
        f.write(f"Final Coherence: {stats_history[-1]['avg_coherence']:.3f}\n")
        f.write(f"Coherence Improvement: {stats_history[-1]['avg_coherence'] - stats_history[0]['avg_coherence']:+.3f}\n")
        f.write("\n")

        # Analyze Phase 4B
        if 'phase4b' in stats_history[-1]:
            final_4b = stats_history[-1]['phase4b']

            if 'novelty_search' in final_4b:
                ns = final_4b['novelty_search']
                f.write("Phase 4B - Novelty Search:\n")
                f.write(f"  Total Unique Behaviors: {ns['unique_behaviors']:,}\n")
                f.write(f"  Final Archive Size: {ns['archive']['size']:,}\n")
                f.write(f"  Average Novelty: {ns['avg_novelty']:.3f}\n")
                f.write("\n")

            if 'map_elites' in final_4b:
                me = final_4b['map_elites']['archive']
                f.write("Phase 4B - MAP-Elites:\n")
                f.write(f"  Behavior Space Coverage: {me['coverage']:.2%}\n")
                f.write(f"  Total Bins Filled: {me['size']:,}\n")
                f.write(f"  Average Elite Fitness: {me['avg_fitness']:.3f}\n")
                f.write(f"  Max Elite Fitness: {me['max_fitness']:.3f}\n")
                f.write("\n")

        # Analyze Phase 4C
        if 'phase4c' in stats_history[-1]:
            final_4c = stats_history[-1]['phase4c']

            if 'communication' in final_4c:
                comm = final_4c['communication']
                f.write("Phase 4C - Communication:\n")
                f.write(f"  Total Messages Sent: {comm['total_messages']:,}\n")
                f.write(f"  Broadcast Messages: {comm.get('broadcast_messages', 0):,}\n")
                f.write(f"  Local Messages: {comm.get('local_messages', 0):,}\n")

                if 'protocol_analysis' in final_4c:
                    protocol = final_4c['protocol_analysis']
                    f.write(f"  Signal Diversity: {protocol.get('signal_diversity', 0):.3f}\n")
                    f.write(f"  Signal Stability: {protocol.get('signal_stability', 0):.3f}\n")
                f.write("\n")

        f.write("="*70 + "\n")
        f.write("Analysis complete. Check logs/ for detailed statistics.\n")

    print(f"\n📊 Analysis saved: {analysis_path}")


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("GENESIS Phase 4: Long-Term Experiment")
    print("="*70)
    print()

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    print(f"📁 Output directory: {output_dir}")
    print()

    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"💾 Configuration saved: {config_path}")
    print()

    # Create or load system
    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # TODO: Reconstruct manager from checkpoint
        print("⚠️  Resume not yet implemented. Starting fresh.")

    print("🔧 Creating GENESIS Phase 4 system...")
    manager = create_phase4c_system(
        env_size=args.env_size,
        initial_population=args.population,
        phase4c_enabled=args.phase4c,
        use_novelty_search=args.novelty_search,
        use_map_elites=args.map_elites,
        use_poet=args.poet,
        message_dim=8,
        local_radius=5.0
    )
    print(f"✓ System created with {len(manager.agents)} agents")
    print(f"  Phase 4A: Advanced Intelligence ✓")
    print(f"  Phase 4B: Open-Ended Learning ✓" if args.phase4b else "  Phase 4B: Disabled")
    print(f"  Phase 4C: Emergent Communication ✓" if args.phase4c else "  Phase 4C: Disabled")
    print()

    # Main experiment loop
    print("🚀 Starting long-term experiment...")
    print(f"   Total steps: {args.steps:,}")
    print(f"   Checkpoint every: {args.checkpoint_interval:,} steps")
    print(f"   Log every: {args.log_interval:,} steps")
    print()

    stats_history = []
    start_time = time.time()

    for step in range(start_step, args.steps):
        # Run simulation step
        stats = manager.step()
        stats_history.append(stats)

        # Print progress
        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed_time = time.time() - start_time
            print_statistics(step + 1, args.steps, stats, elapsed_time)

        # Save checkpoint
        if (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(manager, step + 1, output_dir)
            save_statistics(stats_history, output_dir)

    # Final analysis
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("🎉 Long-Term Experiment Complete!")
    print("="*70)
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Average Speed: {args.steps/total_time:.2f} steps/second")
    print()

    # Save final results
    save_checkpoint(manager, args.steps, output_dir)
    save_statistics(stats_history, output_dir)
    analyze_results(stats_history, output_dir)

    print(f"\n📁 All results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
