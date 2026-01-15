"""
GENESIS Phase 4: Experiment Utilities

Helper functions for common operations:
- Configuration management
- Results loading and analysis
- Checkpoint handling
- Statistics extraction
- Comparison utilities
- Batch processing helpers

Usage:
    from experiment_utils import *

    # Load and compare experiments
    exp1 = load_experiment('results/exp1')
    exp2 = load_experiment('results/exp2')
    compare_experiments([exp1, exp2])
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict


# ============================================================================
# Configuration Management
# ============================================================================

class ExperimentConfig:
    """Experiment configuration builder with sensible defaults"""

    PRESETS = {
        'quick_test': {
            'steps': 100,
            'population': 50,
            'env_size': 30,
            'checkpoint_interval': 50,
            'log_interval': 20
        },
        'medium': {
            'steps': 1000,
            'population': 100,
            'env_size': 40,
            'checkpoint_interval': 100,
            'log_interval': 50
        },
        'long_term': {
            'steps': 10000,
            'population': 300,
            'env_size': 50,
            'checkpoint_interval': 1000,
            'log_interval': 100
        },
        'production': {
            'steps': 100000,
            'population': 500,
            'env_size': 60,
            'checkpoint_interval': 5000,
            'log_interval': 500
        }
    }

    def __init__(self, preset: str = 'medium'):
        """
        Initialize with preset

        Args:
            preset: 'quick_test', 'medium', 'long_term', or 'production'
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")

        self.config = self.PRESETS[preset].copy()
        self.config['preset'] = preset

    def set(self, **kwargs):
        """Set configuration parameters"""
        self.config.update(kwargs)
        return self

    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            loaded = json.load(f)

        preset = loaded.get('preset', 'medium')
        config = cls(preset)
        config.config = loaded
        return config

    def __repr__(self):
        return f"ExperimentConfig({self.config['preset']}, {len(self.config)} parameters)"


# ============================================================================
# Results Loading
# ============================================================================

class ExperimentResults:
    """Container for experiment results with analysis methods"""

    def __init__(self, results_dir: str):
        """
        Load experiment results

        Args:
            results_dir: Path to experiment results directory
        """
        self.results_dir = Path(results_dir)

        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")

        # Load components
        self.config = self._load_config()
        self.statistics = self._load_statistics()
        self.checkpoints = self._find_checkpoints()

        # Derived properties
        self.n_steps = len(self.statistics) if self.statistics else 0
        self.timestamp = self.results_dir.name

    def _load_config(self) -> Dict:
        """Load configuration"""
        config_path = self.results_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_statistics(self) -> List[Dict]:
        """Load statistics"""
        stats_path = self.results_dir / 'logs' / 'statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
        return []

    def _find_checkpoints(self) -> List[Path]:
        """Find checkpoint files"""
        checkpoint_dir = self.results_dir / 'checkpoints'
        if checkpoint_dir.exists():
            return sorted(checkpoint_dir.glob('checkpoint_*.pkl'))
        return []

    def get_metric_history(self, metric: str) -> np.ndarray:
        """
        Extract metric history

        Args:
            metric: Metric name (e.g., 'avg_coherence', 'population_size')

        Returns:
            Array of metric values over time
        """
        values = []
        for stats in self.statistics:
            if metric in stats:
                values.append(stats[metric])
            else:
                # Try nested metrics
                parts = metric.split('.')
                val = stats
                try:
                    for part in parts:
                        val = val[part]
                    values.append(val)
                except (KeyError, TypeError):
                    values.append(np.nan)

        return np.array(values)

    def get_final_metrics(self) -> Dict:
        """Get final statistics"""
        if not self.statistics:
            return {}
        return self.statistics[-1]

    def get_summary(self) -> Dict:
        """Get experiment summary"""
        if not self.statistics:
            return {'error': 'No statistics available'}

        initial = self.statistics[0]
        final = self.statistics[-1]

        summary = {
            'timestamp': self.timestamp,
            'n_steps': self.n_steps,
            'config': self.config,

            # Basic metrics
            'initial_population': initial['population_size'],
            'final_population': final['population_size'],
            'population_growth': (final['population_size'] / initial['population_size'] - 1) * 100,

            'initial_coherence': initial['avg_coherence'],
            'final_coherence': final['avg_coherence'],
            'coherence_improvement': final['avg_coherence'] - initial['avg_coherence'],

            # Phase 4B
            'novelty_search': {},
            'map_elites': {},

            # Phase 4C
            'communication': {}
        }

        # Extract Phase 4B metrics
        if 'phase4b' in final:
            if 'novelty_search' in final['phase4b']:
                ns = final['phase4b']['novelty_search']
                summary['novelty_search'] = {
                    'unique_behaviors': ns.get('unique_behaviors', 0),
                    'archive_size': ns.get('archive', {}).get('size', 0),
                    'avg_novelty': ns.get('avg_novelty', 0)
                }

            if 'map_elites' in final['phase4b']:
                me = final['phase4b']['map_elites']['archive']
                summary['map_elites'] = {
                    'coverage': me.get('coverage', 0) * 100,
                    'size': me.get('size', 0),
                    'avg_fitness': me.get('avg_fitness', 0),
                    'max_fitness': me.get('max_fitness', 0)
                }

        # Extract Phase 4C metrics
        if 'phase4c' in final:
            if 'communication' in final['phase4c']:
                comm = final['phase4c']['communication']
                summary['communication'] = {
                    'total_messages': comm.get('total_messages', 0),
                    'broadcast_messages': comm.get('broadcast_messages', 0),
                    'local_messages': comm.get('local_messages', 0)
                }

            if 'protocol_analysis' in final['phase4c']:
                protocol = final['phase4c']['protocol_analysis']
                summary['communication'].update({
                    'signal_diversity': protocol.get('signal_diversity', 0),
                    'signal_stability': protocol.get('signal_stability', 0)
                })

        return summary

    def __repr__(self):
        return f"ExperimentResults({self.results_dir.name}, {self.n_steps} steps)"


def load_experiment(results_dir: str) -> ExperimentResults:
    """
    Load experiment results (convenience function)

    Args:
        results_dir: Path to results directory

    Returns:
        ExperimentResults object
    """
    return ExperimentResults(results_dir)


def find_experiments(base_dir: str = 'results/long_term') -> List[ExperimentResults]:
    """
    Find all experiments in directory

    Args:
        base_dir: Base directory to search

    Returns:
        List of ExperimentResults
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    experiments = []
    for exp_dir in sorted(base_path.iterdir()):
        if exp_dir.is_dir() and (exp_dir / 'logs' / 'statistics.json').exists():
            try:
                experiments.append(ExperimentResults(str(exp_dir)))
            except Exception as e:
                print(f"Warning: Failed to load {exp_dir}: {e}")

    return experiments


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_experiments(experiments: List[ExperimentResults],
                       metrics: Optional[List[str]] = None) -> Dict:
    """
    Compare multiple experiments

    Args:
        experiments: List of ExperimentResults
        metrics: List of metrics to compare (default: common metrics)

    Returns:
        Comparison dictionary
    """
    if not experiments:
        return {}

    if metrics is None:
        metrics = [
            'final_coherence',
            'coherence_improvement',
            'final_population',
            'population_growth'
        ]

    comparison = {
        'n_experiments': len(experiments),
        'experiments': [],
        'metrics': {}
    }

    # Collect data
    for exp in experiments:
        summary = exp.get_summary()
        comparison['experiments'].append({
            'name': exp.timestamp,
            'n_steps': exp.n_steps,
            'summary': summary
        })

    # Compare metrics
    for metric in metrics:
        values = []
        for exp in experiments:
            summary = exp.get_summary()
            if metric in summary:
                values.append(summary[metric])

        if values:
            comparison['metrics'][metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'best_experiment': experiments[np.argmax(values)].timestamp
            }

    return comparison


def print_comparison(comparison: Dict):
    """Print comparison in readable format"""
    print("\n" + "="*70)
    print(f"EXPERIMENT COMPARISON ({comparison['n_experiments']} experiments)")
    print("="*70)

    for metric, stats in comparison['metrics'].items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Best: {stats['best_experiment']}")


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load checkpoint file

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint data
    """
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def extract_checkpoint_info(checkpoint_path: str) -> Dict:
    """
    Extract basic info from checkpoint without loading full data

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Info dictionary
    """
    checkpoint = load_checkpoint(checkpoint_path)

    info = {
        'path': checkpoint_path,
        'step': checkpoint.get('step', 0),
        'current_step': checkpoint.get('current_step', 0),
        'n_agents': len(checkpoint.get('agents', [])),
        'best_fitness': checkpoint.get('best_fitness', None),
        'has_novelty_archive': checkpoint.get('novelty_archive') is not None,
        'has_map_elites': checkpoint.get('map_elites_archive') is not None,
        'has_communication': checkpoint.get('communication_stats') is not None
    }

    return info


# ============================================================================
# Statistics Extraction
# ============================================================================

def extract_learning_curves(stats_history: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract all metrics as time series

    Args:
        stats_history: List of statistics

    Returns:
        Dictionary of metric arrays
    """
    curves = defaultdict(list)

    for stats in stats_history:
        curves['population_size'].append(stats['population_size'])
        curves['avg_coherence'].append(stats['avg_coherence'])

        # Phase 4B
        if 'phase4b' in stats:
            if 'novelty_search' in stats['phase4b']:
                ns = stats['phase4b']['novelty_search']
                curves['novelty_archive_size'].append(ns['archive']['size'])
                curves['unique_behaviors'].append(ns['unique_behaviors'])

            if 'map_elites' in stats['phase4b']:
                me = stats['phase4b']['map_elites']['archive']
                curves['map_elites_coverage'].append(me['coverage'])
                curves['map_elites_size'].append(me['size'])

        # Phase 4C
        if 'phase4c' in stats:
            if 'communication' in stats['phase4c']:
                comm = stats['phase4c']['communication']
                curves['total_messages'].append(comm.get('total_messages', 0))

    # Convert to arrays
    return {k: np.array(v) for k, v in curves.items()}


def compute_performance_metrics(stats_history: List[Dict]) -> Dict:
    """
    Compute aggregate performance metrics

    Args:
        stats_history: Statistics history

    Returns:
        Performance metrics
    """
    if not stats_history:
        return {}

    curves = extract_learning_curves(stats_history)

    metrics = {
        'n_steps': len(stats_history),

        # Population
        'initial_population': int(curves['population_size'][0]),
        'final_population': int(curves['population_size'][-1]),
        'avg_population': float(np.mean(curves['population_size'])),
        'population_growth_rate': float(
            (curves['population_size'][-1] / curves['population_size'][0] - 1) * 100
        ),

        # Coherence
        'initial_coherence': float(curves['avg_coherence'][0]),
        'final_coherence': float(curves['avg_coherence'][-1]),
        'avg_coherence': float(np.mean(curves['avg_coherence'])),
        'coherence_improvement': float(curves['avg_coherence'][-1] - curves['avg_coherence'][0]),
        'coherence_std': float(np.std(curves['avg_coherence'])),

        # Learning rate (coherence increase per step)
        'learning_rate': float(
            (curves['avg_coherence'][-1] - curves['avg_coherence'][0]) / len(stats_history)
        )
    }

    # Novelty Search
    if 'unique_behaviors' in curves and len(curves['unique_behaviors']) > 0:
        metrics['unique_behaviors'] = int(curves['unique_behaviors'][-1])
        metrics['behavioral_diversity_rate'] = float(
            curves['unique_behaviors'][-1] / len(stats_history)
        )

    # MAP-Elites
    if 'map_elites_coverage' in curves and len(curves['map_elites_coverage']) > 0:
        metrics['map_elites_coverage'] = float(curves['map_elites_coverage'][-1])
        metrics['map_elites_final_size'] = int(curves['map_elites_size'][-1])

    # Communication
    if 'total_messages' in curves and len(curves['total_messages']) > 0:
        total_messages = np.sum(curves['total_messages'])
        metrics['total_messages'] = int(total_messages)
        metrics['messages_per_step'] = float(total_messages / len(stats_history))

    return metrics


# ============================================================================
# Batch Processing Helpers
# ============================================================================

def batch_run_experiments(configs: List[ExperimentConfig],
                         output_base_dir: str = 'results/batch') -> List[str]:
    """
    Run multiple experiments with different configurations

    Args:
        configs: List of experiment configurations
        output_base_dir: Base directory for outputs

    Returns:
        List of output directories
    """
    from long_term_experiment import main as run_experiment
    import sys

    output_dirs = []

    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"Running Experiment {i+1}/{len(configs)}")
        print(f"Config: {config.config['preset']}")
        print(f"{'='*70}\n")

        # Set up arguments
        output_dir = f"{output_base_dir}/exp_{i+1:03d}"

        # Would need to modify long_term_experiment.py to accept config object
        # For now, this is a placeholder showing the intended usage

        print(f"✓ Would run experiment with config: {config}")
        print(f"  Output: {output_dir}")

        output_dirs.append(output_dir)

    return output_dirs


# ============================================================================
# Quick Analysis Functions
# ============================================================================

def quick_summary(results_dir: str):
    """Print quick summary of experiment"""
    exp = load_experiment(results_dir)
    summary = exp.get_summary()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY: {exp.timestamp}")
    print(f"{'='*70}\n")

    print(f"Steps: {summary['n_steps']:,}")
    print(f"Population: {summary['initial_population']} → {summary['final_population']} ({summary['population_growth']:+.1f}%)")
    print(f"Coherence: {summary['initial_coherence']:.3f} → {summary['final_coherence']:.3f} ({summary['coherence_improvement']:+.3f})")

    if summary['novelty_search']:
        print(f"\nNovelty Search:")
        print(f"  Unique Behaviors: {summary['novelty_search']['unique_behaviors']:,}")

    if summary['map_elites']:
        print(f"\nMAP-Elites:")
        print(f"  Coverage: {summary['map_elites']['coverage']:.1f}%")

    if summary['communication']:
        print(f"\nCommunication:")
        print(f"  Total Messages: {summary['communication']['total_messages']:,}")

    print()


def compare_two(dir1: str, dir2: str, name1: str = "Experiment 1", name2: str = "Experiment 2"):
    """Quick comparison of two experiments"""
    exp1 = load_experiment(dir1)
    exp2 = load_experiment(dir2)

    sum1 = exp1.get_summary()
    sum2 = exp2.get_summary()

    print(f"\n{'='*70}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*70}\n")

    metrics = [
        ('Final Coherence', 'final_coherence'),
        ('Coherence Improvement', 'coherence_improvement'),
        ('Population Growth', 'population_growth'),
    ]

    for label, key in metrics:
        val1 = sum1[key]
        val2 = sum2[key]
        diff = val2 - val1
        better = name2 if diff > 0 else name1

        print(f"{label}:")
        print(f"  {name1}: {val1:.3f}")
        print(f"  {name2}: {val2:.3f}")
        print(f"  Difference: {diff:+.3f} (better: {better})")
        print()


# ============================================================================
# Main (Example Usage)
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENESIS Phase 4: Experiment Utilities Demo")
    print("="*70)
    print()

    # 1. Configuration Management
    print("1. Configuration Management")
    print("-" * 40)

    config = ExperimentConfig('quick_test')
    config.set(population=100, steps=200)
    print(f"Created config: {config}")
    print(f"  Steps: {config.get('steps')}")
    print(f"  Population: {config.get('population')}")
    print()

    # 2. Load Experiments
    print("2. Finding Experiments")
    print("-" * 40)

    experiments = find_experiments('results/long_term')
    print(f"Found {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp.timestamp}: {exp.n_steps} steps")
    print()

    # 3. Quick Summary
    if experiments:
        print("3. Quick Summary")
        print("-" * 40)
        quick_summary(str(experiments[0].results_dir))

    print("✓ Utilities demo complete")
