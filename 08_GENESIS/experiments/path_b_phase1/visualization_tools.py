"""
GENESIS Phase 4: Visualization Tools

Comprehensive visualization suite for analyzing experiment results:
- Learning curves
- Population dynamics
- Behavioral diversity (MAP-Elites)
- Communication networks
- Optimization performance
- Long-term trends

Usage:
    from visualization_tools import ExperimentVisualizer

    viz = ExperimentVisualizer()
    viz.plot_learning_curves(stats_history)
    viz.plot_communication_network(manager)
    viz.save_all_figures('results/')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
from pathlib import Path
import json


class ExperimentVisualizer:
    """
    Comprehensive visualization suite for GENESIS experiments
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize=(12, 8)):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.figsize = figsize
        self.figures = {}

    def plot_learning_curves(self, stats_history: List[Dict],
                            title: str = "Learning Curves",
                            save_path: Optional[str] = None):
        """
        Plot comprehensive learning curves

        Args:
            stats_history: List of statistics dictionaries
            title: Plot title
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        steps = np.arange(len(stats_history))

        # Extract metrics
        population = [s['population_size'] for s in stats_history]
        coherence = [s['avg_coherence'] for s in stats_history]

        # 1. Population Size
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(steps, population, 'b-', linewidth=2)
        ax1.set_title('Population Size', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Agents')
        ax1.grid(True, alpha=0.3)

        # 2. Average Coherence
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(steps, coherence, 'g-', linewidth=2)
        ax2.set_title('Average Coherence', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Coherence')
        ax2.grid(True, alpha=0.3)

        # 3. Coherence Distribution (last 20%)
        ax3 = fig.add_subplot(gs[0, 2])
        recent_coherence = coherence[-len(coherence)//5:]
        ax3.hist(recent_coherence, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax3.set_title('Coherence Distribution (Recent)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Coherence')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Phase 4B: Novelty Search
        if 'phase4b' in stats_history[-1] and 'novelty_search' in stats_history[-1]['phase4b']:
            ax4 = fig.add_subplot(gs[1, 0])
            archive_sizes = []
            for s in stats_history:
                if 'phase4b' in s and 'novelty_search' in s['phase4b']:
                    archive_sizes.append(s['phase4b']['novelty_search']['archive']['size'])
                else:
                    archive_sizes.append(0)

            ax4.plot(steps, archive_sizes, 'purple', linewidth=2)
            ax4.set_title('Novelty Archive Growth', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Unique Behaviors')
            ax4.grid(True, alpha=0.3)

        # 5. Phase 4B: MAP-Elites Coverage
        if 'phase4b' in stats_history[-1] and 'map_elites' in stats_history[-1]['phase4b']:
            ax5 = fig.add_subplot(gs[1, 1])
            coverage = []
            for s in stats_history:
                if 'phase4b' in s and 'map_elites' in s['phase4b']:
                    coverage.append(s['phase4b']['map_elites']['archive']['coverage'] * 100)
                else:
                    coverage.append(0)

            ax5.plot(steps, coverage, 'orange', linewidth=2)
            ax5.set_title('MAP-Elites Coverage', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Coverage (%)')
            ax5.grid(True, alpha=0.3)

        # 6. Phase 4C: Communication Volume
        if 'phase4c' in stats_history[-1] and 'communication' in stats_history[-1]['phase4c']:
            ax6 = fig.add_subplot(gs[1, 2])
            messages = []
            for s in stats_history:
                if 'phase4c' in s and 'communication' in s['phase4c']:
                    messages.append(s['phase4c']['communication'].get('total_messages', 0))
                else:
                    messages.append(0)

            ax6.plot(steps, np.cumsum(messages), 'red', linewidth=2)
            ax6.set_title('Cumulative Messages', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Total Messages')
            ax6.grid(True, alpha=0.3)

        # 7. Phase 4C: Protocol Metrics
        if 'phase4c' in stats_history[-1] and 'protocol_analysis' in stats_history[-1]['phase4c']:
            ax7 = fig.add_subplot(gs[2, 0])
            diversity = []
            stability = []
            for s in stats_history:
                if 'phase4c' in s and 'protocol_analysis' in s['phase4c']:
                    diversity.append(s['phase4c']['protocol_analysis'].get('signal_diversity', 0))
                    stability.append(s['phase4c']['protocol_analysis'].get('signal_stability', 0))
                else:
                    diversity.append(0)
                    stability.append(0)

            ax7.plot(steps, diversity, 'blue', linewidth=2, label='Diversity')
            ax7.plot(steps, stability, 'green', linewidth=2, label='Stability')
            ax7.set_title('Communication Protocol', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Step')
            ax7.set_ylabel('Score')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Growth Rates
        ax8 = fig.add_subplot(gs[2, 1])
        if len(population) > 10:
            window = min(10, len(population) // 10)
            growth_rate = np.diff(population)
            growth_rate_smooth = np.convolve(growth_rate, np.ones(window)/window, mode='valid')

            ax8.plot(steps[1:len(growth_rate_smooth)+1], growth_rate_smooth, 'darkgreen', linewidth=2)
            ax8.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax8.set_title('Population Growth Rate', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Step')
            ax8.set_ylabel('Agents/Step')
            ax8.grid(True, alpha=0.3)

        # 9. Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
EXPERIMENT SUMMARY

Total Steps: {len(stats_history):,}
Final Population: {population[-1]}
Initial Coherence: {coherence[0]:.3f}
Final Coherence: {coherence[-1]:.3f}
Improvement: {coherence[-1] - coherence[0]:+.3f}

Population Change: {((population[-1] / population[0]) - 1) * 100:+.1f}%
"""

        if 'phase4b' in stats_history[-1]:
            if 'novelty_search' in stats_history[-1]['phase4b']:
                ns = stats_history[-1]['phase4b']['novelty_search']
                summary_text += f"\nNovelty Archive: {ns['unique_behaviors']:,} behaviors"

            if 'map_elites' in stats_history[-1]['phase4b']:
                me = stats_history[-1]['phase4b']['map_elites']['archive']
                summary_text += f"\nMAP-Elites Coverage: {me['coverage']:.1%}"

        if 'phase4c' in stats_history[-1]:
            if 'communication' in stats_history[-1]['phase4c']:
                comm = stats_history[-1]['phase4c']['communication']
                total_msgs = sum(messages)
                summary_text += f"\n\nTotal Messages: {total_msgs:,}"

        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        self.figures['learning_curves'] = fig
        return fig

    def plot_map_elites_heatmap(self, manager,
                                title: str = "MAP-Elites Behavior Space",
                                save_path: Optional[str] = None):
        """
        Plot MAP-Elites archive as heatmap

        Args:
            manager: Manager with MAP-Elites
            title: Plot title
            save_path: Path to save figure
        """
        if not hasattr(manager, 'map_elites') or manager.map_elites is None:
            print("No MAP-Elites archive available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Get archive
        archive = manager.map_elites.archive
        behavior_space = manager.map_elites.behavior_space

        # Create grid
        grid_size = behavior_space.grid_size
        fitness_grid = np.full((grid_size, grid_size), np.nan)
        count_grid = np.zeros((grid_size, grid_size))

        for niche, solution in archive.items():
            i, j = divmod(niche, grid_size)
            if i < grid_size and j < grid_size:
                fitness_grid[i, j] = solution.fitness
                count_grid[i, j] = 1

        # Plot fitness heatmap
        im1 = axes[0].imshow(fitness_grid, cmap='viridis', aspect='auto', origin='lower')
        axes[0].set_title('Fitness by Niche', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Behavior Dimension 2')
        axes[0].set_ylabel('Behavior Dimension 1')
        plt.colorbar(im1, ax=axes[0], label='Fitness')

        # Plot coverage heatmap
        im2 = axes[1].imshow(count_grid, cmap='Blues', aspect='auto', origin='lower')
        axes[1].set_title('Coverage Map', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Behavior Dimension 2')
        axes[1].set_ylabel('Behavior Dimension 1')
        plt.colorbar(im2, ax=axes[1], label='Occupied')

        # Statistics
        coverage = np.sum(count_grid > 0) / (grid_size * grid_size)
        avg_fitness = np.nanmean(fitness_grid)
        max_fitness = np.nanmax(fitness_grid)

        fig.text(0.5, 0.02,
                f"Coverage: {coverage:.1%} | Avg Fitness: {avg_fitness:.3f} | Max Fitness: {max_fitness:.3f}",
                ha='center', fontsize=12, fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        self.figures['map_elites'] = fig
        return fig

    def plot_optimization_comparison(self, baseline_results: Dict,
                                    optimized_results: Dict,
                                    title: str = "Optimization Performance",
                                    save_path: Optional[str] = None):
        """
        Compare baseline vs optimized performance

        Args:
            baseline_results: Baseline benchmark results
            optimized_results: Optimized benchmark results
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Step time comparison
        ax1 = axes[0, 0]
        configurations = ['Baseline', 'Optimized']
        step_times = [baseline_results['avg_step_time'], optimized_results['avg_step_time']]
        colors = ['red', 'green']

        bars = ax1.bar(configurations, step_times, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Avg Step Time (s)', fontweight='bold')
        ax1.set_title('Step Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, time in zip(bars, step_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.4f}s', ha='center', va='bottom', fontweight='bold')

        # 2. Speedup visualization
        ax2 = axes[0, 1]
        speedup = baseline_results['avg_step_time'] / optimized_results['avg_step_time']

        ax2.barh(['Speedup'], [speedup], color='gold', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Speedup Factor', fontweight='bold')
        ax2.set_title(f'Overall Speedup: {speedup:.2f}x', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.text(speedup/2, 0, f'{speedup:.2f}x', ha='center', va='center',
                fontsize=16, fontweight='bold')

        # 3. Step time distributions
        ax3 = axes[1, 0]

        baseline_times = baseline_results['step_times']
        optimized_times = optimized_results['step_times']

        ax3.hist(baseline_times, bins=20, alpha=0.5, label='Baseline', color='red', edgecolor='black')
        ax3.hist(optimized_times, bins=20, alpha=0.5, label='Optimized', color='green', edgecolor='black')
        ax3.set_xlabel('Step Time (s)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Step Time Distributions', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Optimization breakdown
        ax4 = axes[1, 1]

        if 'optimization' in optimized_results:
            opt = optimized_results['optimization']

            breakdown_text = "OPTIMIZATION BREAKDOWN\n\n"

            if 'cached_coherence' in opt:
                cc = opt['cached_coherence']
                breakdown_text += f"Cached Coherence:\n"
                breakdown_text += f"  Hit Rate: {cc.get('avg_hit_rate', 0):.1%}\n"
                breakdown_text += f"  Speedup: ~{1.0 + cc.get('avg_hit_rate', 0):.2f}x\n\n"

            if 'sparse_map_elites' in opt:
                sme = opt['sparse_map_elites']
                breakdown_text += f"Sparse MAP-Elites:\n"
                breakdown_text += f"  Skip Rate: {sme.get('skip_rate', 0):.1%}\n"
                breakdown_text += f"  Speedup: ~{sme.get('speedup_estimate', 1.0):.2f}x\n\n"

            if 'batch_processing' in opt:
                bp = opt['batch_processing']
                breakdown_text += f"Batch Processing:\n"
                breakdown_text += f"  Items: {bp.get('total_items', 0):,}\n"
                breakdown_text += f"  Throughput: {bp.get('throughput_items_per_sec', 0):.1f}/s\n"

            ax4.text(0.1, 0.9, breakdown_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax4.axis('off')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        self.figures['optimization'] = fig
        return fig

    def plot_phase_comparison(self, stats_history: List[Dict],
                             title: str = "Phase Comparison",
                             save_path: Optional[str] = None):
        """
        Compare contributions of different phases

        Args:
            stats_history: Statistics history
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = np.arange(len(stats_history))

        # Extract metrics
        base_coherence = [s['avg_coherence'] for s in stats_history]

        # 1. Overall coherence
        ax1 = axes[0, 0]
        ax1.plot(steps, base_coherence, 'blue', linewidth=2, label='System Coherence')
        ax1.set_title('Overall System Coherence', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Coherence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Phase 4A contribution (if available)
        ax2 = axes[0, 1]
        if 'phase4a' in stats_history[-1]:
            # Could track specific Phase 4A metrics
            ax2.text(0.5, 0.5, 'Phase 4A:\nAdvanced Intelligence\n\n✓ Multi-Teacher\n✓ Learned Memory\n✓ Knowledge Guidance',
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax2.set_title('Phase 4A Contribution', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Phase 4B contribution
        ax3 = axes[1, 0]
        if 'phase4b' in stats_history[-1]:
            unique_behaviors = []
            for s in stats_history:
                if 'phase4b' in s and 'novelty_search' in s['phase4b']:
                    unique_behaviors.append(s['phase4b']['novelty_search']['unique_behaviors'])
                else:
                    unique_behaviors.append(0)

            ax3.plot(steps, unique_behaviors, 'purple', linewidth=2)
            ax3.set_title('Phase 4B: Behavioral Diversity', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Unique Behaviors')
            ax3.grid(True, alpha=0.3)

        # 4. Phase 4C contribution
        ax4 = axes[1, 1]
        if 'phase4c' in stats_history[-1]:
            messages_per_step = []
            for s in stats_history:
                if 'phase4c' in s and 'communication' in s['phase4c']:
                    messages_per_step.append(s['phase4c']['communication'].get('total_messages', 0))
                else:
                    messages_per_step.append(0)

            # Running average
            if len(messages_per_step) > 10:
                window = 10
                messages_smooth = np.convolve(messages_per_step, np.ones(window)/window, mode='valid')
                ax4.plot(steps[:len(messages_smooth)], messages_smooth, 'red', linewidth=2)
            else:
                ax4.plot(steps, messages_per_step, 'red', linewidth=2)

            ax4.set_title('Phase 4C: Communication Activity', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Messages/Step')
            ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        self.figures['phase_comparison'] = fig
        return fig

    def save_all_figures(self, output_dir: str):
        """
        Save all created figures

        Args:
            output_dir: Directory to save figures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in self.figures.items():
            save_path = output_path / f"{name}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

    def close_all(self):
        """Close all figures"""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()


def load_experiment_data(results_dir: str) -> Dict:
    """
    Load experiment data from directory

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Dictionary with loaded data
    """
    results_path = Path(results_dir)

    data = {}

    # Load statistics
    stats_file = results_path / 'logs' / 'statistics.json'
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            data['statistics'] = json.load(f)

    # Load config
    config_file = results_path / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            data['config'] = json.load(f)

    return data


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENESIS Phase 4: Visualization Tools Test")
    print("="*70)
    print()

    # Example usage
    print("Creating example visualizations...")

    # Generate dummy data for demonstration
    np.random.seed(42)
    n_steps = 100

    stats_history = []
    for step in range(n_steps):
        stats = {
            'population_size': int(50 + step * 0.5 + np.random.randn() * 2),
            'avg_coherence': 0.3 + (step / n_steps) * 0.4 + np.random.randn() * 0.05,
            'phase4b': {
                'novelty_search': {
                    'archive': {'size': step * 10},
                    'unique_behaviors': step * 10,
                    'avg_novelty': 0.5 + np.random.randn() * 0.1
                },
                'map_elites': {
                    'archive': {
                        'coverage': min(1.0, step / n_steps * 0.8),
                        'size': step * 5,
                        'avg_fitness': 0.5 + (step / n_steps) * 0.3
                    }
                }
            },
            'phase4c': {
                'communication': {
                    'total_messages': max(0, int(step * 2 + np.random.randn() * 5))
                },
                'protocol_analysis': {
                    'signal_diversity': min(1.0, step / n_steps * 0.9),
                    'signal_stability': min(1.0, step / n_steps * 0.95)
                }
            }
        }
        stats_history.append(stats)

    # Create visualizer
    viz = ExperimentVisualizer()

    # Generate plots
    print("Generating learning curves...")
    viz.plot_learning_curves(stats_history, save_path='example_learning_curves.png')

    print("Generating phase comparison...")
    viz.plot_phase_comparison(stats_history, save_path='example_phase_comparison.png')

    print("\n✓ Example visualizations created")
    print("  - example_learning_curves.png")
    print("  - example_phase_comparison.png")
    print()
