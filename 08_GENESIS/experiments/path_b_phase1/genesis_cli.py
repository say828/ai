#!/usr/bin/env python3
"""
GENESIS Phase 4: Unified Command-Line Interface

Comprehensive CLI for all GENESIS operations:
- Run experiments
- Analyze results
- Visualize data
- Benchmark performance
- Manage configurations

Usage:
    python genesis_cli.py run --config configs/quick_test.json
    python genesis_cli.py analyze results/long_term/20260104_120000
    python genesis_cli.py visualize results/long_term/20260104_120000
    python genesis_cli.py benchmark
    python genesis_cli.py compare exp1 exp2

Author: GENESIS Research Team
Date: 2026-01-04
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import subprocess

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


class GenesisCLI:
    """Main CLI application"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description='GENESIS Phase 4: Unified Command-Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run quick test
  %(prog)s run --preset quick_test

  # Run custom experiment
  %(prog)s run --config my_config.json

  # Analyze results
  %(prog)s analyze results/long_term/20260104_120000

  # Generate visualizations
  %(prog)s visualize results/long_term/20260104_120000

  # Benchmark optimizations
  %(prog)s benchmark

  # Compare experiments
  %(prog)s compare exp1 exp2 exp3

  # List all experiments
  %(prog)s list

  # Run tests
  %(prog)s test
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Command to run')

        # Run command
        run_parser = subparsers.add_parser('run', help='Run experiment')
        run_parser.add_argument('--preset', choices=['quick_test', 'medium', 'long_term', 'production'],
                               help='Use preset configuration')
        run_parser.add_argument('--config', type=str, help='Path to config JSON file')
        run_parser.add_argument('--steps', type=int, help='Override number of steps')
        run_parser.add_argument('--population', type=int, help='Override population size')
        run_parser.add_argument('--optimized', action='store_true', help='Use optimized implementation')
        run_parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
        analyze_parser.add_argument('path', type=str, help='Path to experiment results')
        analyze_parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')

        # Visualize command
        viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
        viz_parser.add_argument('path', type=str, help='Path to experiment results')
        viz_parser.add_argument('--output', type=str, help='Output directory for figures')
        viz_parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                               help='Output format')

        # Benchmark command
        bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
        bench_parser.add_argument('--quick', action='store_true', help='Quick benchmark (less thorough)')
        bench_parser.add_argument('--output', type=str, help='Output file for results')

        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare multiple experiments')
        compare_parser.add_argument('paths', nargs='+', help='Paths to experiments')
        compare_parser.add_argument('--metrics', nargs='+', help='Metrics to compare')
        compare_parser.add_argument('--output', type=str, help='Output file for comparison')

        # List command
        list_parser = subparsers.add_parser('list', help='List all experiments')
        list_parser.add_argument('--base-dir', default='results/long_term', help='Base directory')
        list_parser.add_argument('--sort-by', choices=['date', 'steps', 'coherence'],
                                default='date', help='Sort order')

        # Test command
        test_parser = subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument('--suite', choices=['unit', 'integration', 'performance', 'all'],
                                default='all', help='Test suite to run')

        # Info command
        info_parser = subparsers.add_parser('info', help='Show system information')

        return parser

    def run(self, args=None):
        """Run CLI"""
        args = self.parser.parse_args(args)

        if not args.command:
            self.parser.print_help()
            return 0

        # Dispatch to appropriate handler
        handler = getattr(self, f'_handle_{args.command}', None)
        if handler:
            return handler(args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            return 1

    def _handle_run(self, args):
        """Handle run command"""
        print("="*70)
        print("RUNNING EXPERIMENT")
        print("="*70)
        print()

        # Build command
        cmd = ['python', 'long_term_experiment.py']

        if args.preset:
            from experiment_utils import ExperimentConfig
            config = ExperimentConfig(args.preset)

            cmd.extend(['--steps', str(config.get('steps'))])
            cmd.extend(['--population', str(config.get('initial_population'))])
            cmd.extend(['--env-size', str(config.get('env_size', 50))])

            print(f"Using preset: {args.preset}")

        elif args.config:
            with open(args.config) as f:
                config_data = json.load(f)

            exp_config = config_data['experiment']
            env_config = config_data['environment']

            cmd.extend(['--steps', str(exp_config['steps'])])
            cmd.extend(['--population', str(env_config['initial_population'])])
            cmd.extend(['--env-size', str(env_config['size'])])

            print(f"Using config: {args.config}")

        # Overrides
        if args.steps:
            cmd.extend(['--steps', str(args.steps)])

        if args.population:
            cmd.extend(['--population', str(args.population)])

        print(f"\nCommand: {' '.join(cmd)}")
        print()

        # Run
        return subprocess.call(cmd)

    def _handle_analyze(self, args):
        """Handle analyze command"""
        print("="*70)
        print("ANALYZING RESULTS")
        print("="*70)
        print()

        from experiment_utils import load_experiment

        try:
            exp = load_experiment(args.path)
            summary = exp.get_summary()

            print(f"Experiment: {exp.timestamp}")
            print(f"Steps: {summary['n_steps']:,}")
            print()

            print("Population:")
            print(f"  Initial: {summary['initial_population']}")
            print(f"  Final: {summary['final_population']}")
            print(f"  Growth: {summary['population_growth']:+.1f}%")
            print()

            print("Coherence:")
            print(f"  Initial: {summary['initial_coherence']:.3f}")
            print(f"  Final: {summary['final_coherence']:.3f}")
            print(f"  Improvement: {summary['coherence_improvement']:+.3f}")
            print()

            if summary['novelty_search']:
                print("Novelty Search:")
                print(f"  Unique Behaviors: {summary['novelty_search']['unique_behaviors']:,}")
                print(f"  Archive Size: {summary['novelty_search']['archive_size']:,}")
                print()

            if summary['map_elites']:
                print("MAP-Elites:")
                print(f"  Coverage: {summary['map_elites']['coverage']:.1f}%")
                print(f"  Archive Size: {summary['map_elites']['size']:,}")
                print()

            if summary['communication']:
                print("Communication:")
                print(f"  Total Messages: {summary['communication']['total_messages']:,}")
                if 'signal_diversity' in summary['communication']:
                    print(f"  Signal Diversity: {summary['communication']['signal_diversity']:.3f}")
                print()

            if args.detailed:
                print("="*70)
                print("DETAILED ANALYSIS")
                print("="*70)
                print()

                from experiment_utils import compute_performance_metrics

                metrics = compute_performance_metrics(exp.statistics)
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")

            return 0

        except Exception as e:
            print(f"Error analyzing {args.path}: {e}")
            return 1

    def _handle_visualize(self, args):
        """Handle visualize command"""
        print("="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        print()

        from visualization_tools import ExperimentVisualizer, load_experiment_data

        try:
            # Load data
            data = load_experiment_data(args.path)
            stats = data.get('statistics', [])

            if not stats:
                print(f"No statistics found in {args.path}")
                return 1

            # Create visualizer
            viz = ExperimentVisualizer()

            # Output directory
            if args.output:
                output_dir = Path(args.output)
            else:
                output_dir = Path(args.path) / 'figures'

            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate plots
            print("Generating learning curves...")
            viz.plot_learning_curves(
                stats,
                save_path=str(output_dir / f'learning_curves.{args.format}')
            )

            print("Generating phase comparison...")
            viz.plot_phase_comparison(
                stats,
                save_path=str(output_dir / f'phase_comparison.{args.format}')
            )

            print()
            print(f"✓ Figures saved to: {output_dir}")

            return 0

        except Exception as e:
            print(f"Error visualizing {args.path}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _handle_benchmark(self, args):
        """Handle benchmark command"""
        print("="*70)
        print("RUNNING BENCHMARKS")
        print("="*70)
        print()

        cmd = ['python', 'benchmark_optimizations.py']

        if args.quick:
            print("Running quick benchmark...")
        else:
            print("Running full benchmark (this may take 30-60 minutes)...")

        print()

        return subprocess.call(cmd)

    def _handle_compare(self, args):
        """Handle compare command"""
        print("="*70)
        print("COMPARING EXPERIMENTS")
        print("="*70)
        print()

        from experiment_utils import load_experiment, compare_experiments, print_comparison

        try:
            # Load experiments
            experiments = []
            for path in args.paths:
                exp = load_experiment(path)
                experiments.append(exp)
                print(f"Loaded: {exp.timestamp} ({exp.n_steps} steps)")

            print()

            # Compare
            comparison = compare_experiments(experiments, metrics=args.metrics)
            print_comparison(comparison)

            # Save if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
                print(f"\n✓ Comparison saved to: {args.output}")

            return 0

        except Exception as e:
            print(f"Error comparing experiments: {e}")
            return 1

    def _handle_list(self, args):
        """Handle list command"""
        print("="*70)
        print("AVAILABLE EXPERIMENTS")
        print("="*70)
        print()

        from experiment_utils import find_experiments

        experiments = find_experiments(args.base_dir)

        if not experiments:
            print(f"No experiments found in {args.base_dir}")
            return 0

        # Sort
        if args.sort_by == 'date':
            experiments.sort(key=lambda x: x.timestamp, reverse=True)
        elif args.sort_by == 'steps':
            experiments.sort(key=lambda x: x.n_steps, reverse=True)
        elif args.sort_by == 'coherence':
            experiments.sort(key=lambda x: x.get_final_metrics().get('avg_coherence', 0), reverse=True)

        # Print
        print(f"{'Timestamp':<20} {'Steps':>10} {'Population':>12} {'Coherence':>12}")
        print("-" * 70)

        for exp in experiments:
            summary = exp.get_summary()
            print(f"{exp.timestamp:<20} {summary['n_steps']:>10,} "
                  f"{summary['final_population']:>12} {summary['final_coherence']:>12.3f}")

        print()
        print(f"Total: {len(experiments)} experiments")

        return 0

    def _handle_test(self, args):
        """Handle test command"""
        print("="*70)
        print("RUNNING TESTS")
        print("="*70)
        print()

        if args.suite == 'unit' or args.suite == 'all':
            print("Running unit tests...")
            result = subprocess.call(['python', 'test_optimizations.py'])
            if result != 0:
                return result

        if args.suite == 'integration' or args.suite == 'all':
            print("\nRunning integration tests...")
            result = subprocess.call(['python', 'test_phase4b_quick.py'])
            if result != 0:
                return result

            result = subprocess.call(['python', 'test_phase4c_quick.py'])
            if result != 0:
                return result

        if args.suite == 'performance' or args.suite == 'all':
            print("\nRunning performance tests...")
            result = subprocess.call(['python', 'spatial_indexing.py'])
            if result != 0:
                return result

        print("\n✓ All tests passed")
        return 0

    def _handle_info(self, args):
        """Handle info command"""
        print("="*70)
        print("GENESIS PHASE 4: SYSTEM INFORMATION")
        print("="*70)
        print()

        import torch

        print("Environment:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Working Directory: {Path.cwd()}")
        print()

        print("Dependencies:")
        print(f"  NumPy: {__import__('numpy').__version__}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print()

        print("Components:")
        components = [
            ('phase4c_integration.py', 'Phase 4C Integration'),
            ('optimized_phase4c.py', 'Optimized Implementation'),
            ('spatial_indexing.py', 'Spatial Indexing'),
            ('long_term_experiment.py', 'Long-term Experiments'),
            ('visualization_tools.py', 'Visualization Tools'),
            ('experiment_utils.py', 'Utility Functions'),
        ]

        for filename, description in components:
            path = Path(filename)
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {description:<30} ({filename})")

        print()

        print("Documentation:")
        docs = [
            ('INDEX.md', 'Master Index'),
            ('QUICKSTART_GUIDE.md', 'Quick-Start Guide'),
            ('OPTIMIZATION_GUIDE.md', 'Optimization Guide'),
            ('FINAL_SYSTEM_REPORT.md', 'System Report'),
        ]

        for filename, description in docs:
            path = Path(filename)
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {description:<30} ({filename})")

        print()

        return 0


def main():
    """Main entry point"""
    cli = GenesisCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
