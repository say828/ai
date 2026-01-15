"""
GENESIS Path B Phase 1: Visualization

Creates comprehensive visualizations for Phase 1 results:
1. Population dynamics over time
2. Coherence evolution
3. Quality-Diversity metrics
4. Phylogenetic analysis
5. Comparison with baselines
6. Resource and spatial dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys


def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_population_dynamics(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot population size over time"""
    trials = results.get(condition, [])
    if not trials:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    for i, trial in enumerate(trials):
        history = trial['history']
        steps = history['steps']
        pop = history['population_size']
        alpha = 0.7 if len(trials) > 1 else 1.0
        ax.plot(steps, pop, alpha=alpha, label=f'Trial {i+1}' if len(trials) > 1 else None)
    
    # Mean line if multiple trials
    if len(trials) > 1:
        min_len = min(len(t['history']['steps']) for t in trials)
        mean_pop = np.mean([t['history']['population_size'][:min_len] for t in trials], axis=0)
        steps = trials[0]['history']['steps'][:min_len]
        ax.plot(steps, mean_pop, 'k-', linewidth=2, label='Mean')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Population Size')
    ax.set_title(f'Population Dynamics ({condition.title()})')
    ax.grid(True, alpha=0.3)
    if len(trials) > 1:
        ax.legend()


def plot_coherence_evolution(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot coherence over time with std band"""
    trials = results.get(condition, [])
    if not trials or 'avg_coherence' not in trials[0]['history']:
        ax.text(0.5, 0.5, 'No coherence data', ha='center', va='center', transform=ax.transAxes)
        return
    
    for i, trial in enumerate(trials):
        history = trial['history']
        steps = history['steps']
        coh = history['avg_coherence']
        std = history.get('std_coherence', [0] * len(coh))
        
        ax.plot(steps, coh, alpha=0.7)
        ax.fill_between(steps, 
                       np.array(coh) - np.array(std), 
                       np.array(coh) + np.array(std), 
                       alpha=0.2)
    
    ax.axhline(y=0.55, color='g', linestyle='--', alpha=0.5, label='Reproduction threshold')
    ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='Death threshold')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Coherence')
    ax.set_title('Coherence Evolution')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')


def plot_births_deaths(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot cumulative births and deaths"""
    trials = results.get(condition, [])
    if not trials:
        return
    
    # Use first trial for clarity
    trial = trials[0]
    history = trial['history']
    steps = history['steps']
    births = history['total_births']
    deaths = history['total_deaths']
    
    ax.plot(steps, births, 'g-', linewidth=2, label='Births')
    ax.plot(steps, deaths, 'r-', linewidth=2, label='Deaths')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Count')
    ax.set_title('Births vs Deaths')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_qd_coverage(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot QD coverage over time"""
    trials = results.get(condition, [])
    if not trials or 'qd_coverage' not in trials[0]['history']:
        ax.text(0.5, 0.5, 'No QD data', ha='center', va='center', transform=ax.transAxes)
        return
    
    for i, trial in enumerate(trials):
        history = trial['history']
        steps = history['steps']
        qd = history['qd_coverage']
        ax.plot(steps, qd, alpha=0.7)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('QD Archive Size')
    ax.set_title('Quality-Diversity Coverage')
    ax.grid(True, alpha=0.3)


def plot_energy_material(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot average energy and material over time"""
    trials = results.get(condition, [])
    if not trials:
        return
    
    trial = trials[0]
    history = trial['history']
    steps = history['steps']
    energy = history['avg_energy']
    material = history.get('avg_material', [0] * len(energy))
    
    ax.plot(steps, energy, 'b-', linewidth=2, label='Energy')
    ax.plot(steps, material, 'orange', linewidth=2, label='Material')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Resource')
    ax.set_title('Resource Levels')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_age_distribution(results: Dict, ax: plt.Axes, condition: str = 'autopoietic'):
    """Plot max age over time"""
    trials = results.get(condition, [])
    if not trials:
        return
    
    trial = trials[0]
    history = trial['history']
    steps = history['steps']
    max_age = history['max_age']
    avg_age = history['avg_age']
    
    ax.plot(steps, max_age, 'b-', linewidth=2, label='Max Age')
    ax.plot(steps, avg_age, 'g-', linewidth=1, alpha=0.7, label='Avg Age')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Age (steps)')
    ax.set_title('Agent Longevity')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_condition_comparison(results: Dict, ax: plt.Axes, metric: str = 'population_size'):
    """Compare conditions on a specific metric"""
    conditions = ['autopoietic', 'random', 'fixed', 'rl']
    colors = ['blue', 'gray', 'orange', 'green']
    
    for condition, color in zip(conditions, colors):
        trials = results.get(condition, [])
        if not trials or metric not in trials[0]['history']:
            continue
        
        # Average across trials
        min_len = min(len(t['history']['steps']) for t in trials)
        mean_vals = np.mean([t['history'][metric][:min_len] for t in trials], axis=0)
        steps = trials[0]['history']['steps'][:min_len]
        
        ax.plot(steps, mean_vals, color=color, linewidth=2, label=condition.title())
    
    ax.set_xlabel('Step')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_summary_bars(results: Dict, ax: plt.Axes):
    """Bar chart summary of key metrics"""
    summary = results.get('summary', {})
    if not summary:
        ax.text(0.5, 0.5, 'No summary data', ha='center', va='center', transform=ax.transAxes)
        return
    
    conditions = ['autopoietic', 'random', 'fixed', 'rl']
    colors = ['blue', 'gray', 'orange', 'green']
    
    x = np.arange(4)
    width = 0.35
    
    # Metric 1: Survival time
    survival_times = []
    for cond in conditions:
        if cond in summary:
            survival_times.append(summary[cond]['survival_time_mean'])
        else:
            survival_times.append(0)
    
    # Metric 2: Total births
    births = []
    for cond in conditions:
        if cond in summary:
            births.append(summary[cond]['total_births_mean'])
        else:
            births.append(0)
    
    # Normalize for comparison
    max_survival = max(survival_times) if max(survival_times) > 0 else 1
    max_births = max(births) if max(births) > 0 else 1
    
    bars1 = ax.bar(x - width/2, np.array(survival_times)/max_survival, width, 
                   label='Survival (norm)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, np.array(births)/max_births, width,
                   label='Births (norm)', color='forestgreen', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions])
    ax.set_ylabel('Normalized Score')
    ax.set_title('Condition Comparison Summary')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def create_phase1_plots(results: Dict, output_path: Optional[str] = None):
    """Create comprehensive Phase 1 visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Main dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    plot_population_dynamics(results, ax1, 'autopoietic')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_coherence_evolution(results, ax2, 'autopoietic')
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_births_deaths(results, ax3, 'autopoietic')
    
    # Row 2: Additional metrics
    ax4 = fig.add_subplot(gs[1, 0])
    plot_qd_coverage(results, ax4, 'autopoietic')
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_energy_material(results, ax5, 'autopoietic')
    
    ax6 = fig.add_subplot(gs[1, 2])
    plot_age_distribution(results, ax6, 'autopoietic')
    
    # Row 3: Comparisons
    ax7 = fig.add_subplot(gs[2, 0])
    plot_condition_comparison(results, ax7, 'population_size')
    
    ax8 = fig.add_subplot(gs[2, 1])
    plot_condition_comparison(results, ax8, 'total_births')
    
    ax9 = fig.add_subplot(gs[2, 2])
    plot_summary_bars(results, ax9)
    
    # Title
    exp_info = results.get('experiment_info', {})
    fig.suptitle(f"GENESIS Phase 1: Full Artificial Life System\n"
                 f"Steps: {exp_info.get('n_steps', '?')}, "
                 f"Grid: {exp_info.get('grid_size', '?')}x{exp_info.get('grid_size', '?')}, "
                 f"Initial Pop: {exp_info.get('initial_pop', '?')}", 
                 fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    return fig


def create_comparison_plots(results: Dict, output_path: Optional[str] = None):
    """Create detailed comparison plots for autopoietic vs baselines"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = ['autopoietic', 'random', 'fixed', 'rl']
    colors = {'autopoietic': 'blue', 'random': 'gray', 'fixed': 'orange', 'rl': 'green'}
    
    # 1. Population size comparison
    ax = axes[0, 0]
    for cond in conditions:
        trials = results.get(cond, [])
        if trials:
            for trial in trials:
                steps = trial['history']['steps']
                pop = trial['history']['population_size']
                ax.plot(steps, pop, color=colors[cond], alpha=0.5)
    
    # Add legend
    handles = [Patch(facecolor=colors[c], label=c.title()) for c in conditions if results.get(c)]
    ax.legend(handles=handles)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('Population Size by Condition')
    ax.grid(True, alpha=0.3)
    
    # 2. Coherence comparison (where available)
    ax = axes[0, 1]
    for cond in conditions:
        trials = results.get(cond, [])
        if trials and 'avg_coherence' in trials[0]['history']:
            for trial in trials:
                steps = trial['history']['steps']
                coh = trial['history']['avg_coherence']
                ax.plot(steps, coh, color=colors[cond], alpha=0.5)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Coherence')
    ax.set_title('Coherence by Condition')
    ax.grid(True, alpha=0.3)
    
    # 3. Survival analysis
    ax = axes[1, 0]
    summary = results.get('summary', {})
    
    survival_data = []
    labels = []
    for cond in conditions:
        if cond in summary:
            survival_data.append(summary[cond]['survival_time_mean'])
            labels.append(cond.title())
    
    if survival_data:
        bars = ax.bar(labels, survival_data, color=[colors[c.lower()] for c in labels])
        ax.set_ylabel('Mean Survival Time (steps)')
        ax.set_title('Survival Time by Condition')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add error bars
        errors = [summary[c.lower()]['survival_time_std'] for c in labels]
        ax.errorbar(range(len(labels)), survival_data, yerr=errors, 
                   fmt='none', color='black', capsize=5)
    
    # 4. Births comparison
    ax = axes[1, 1]
    
    births_data = []
    labels = []
    for cond in conditions:
        if cond in summary:
            births_data.append(summary[cond]['total_births_mean'])
            labels.append(cond.title())
    
    if births_data:
        bars = ax.bar(labels, births_data, color=[colors[c.lower()] for c in labels])
        ax.set_ylabel('Mean Total Births')
        ax.set_title('Reproductive Success by Condition')
        ax.grid(True, alpha=0.3, axis='y')
        
        errors = [summary[c.lower()]['total_births_std'] for c in labels]
        ax.errorbar(range(len(labels)), births_data, yerr=errors,
                   fmt='none', color='black', capsize=5)
    
    fig.suptitle('GENESIS Phase 1: Autopoietic vs Baseline Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    
    return fig


def visualize_from_file(json_path: str):
    """Load results and create all visualizations"""
    results = load_results(json_path)
    
    # Determine output paths
    json_path = Path(json_path)
    base_name = json_path.stem.replace('_results', '')
    output_dir = json_path.parent
    
    # Create main plots
    main_plot_path = output_dir / f'{base_name}_plots.png'
    create_phase1_plots(results, str(main_plot_path))
    
    # Create comparison plots
    comparison_plot_path = output_dir / f'{base_name}_comparison.png'
    create_comparison_plots(results, str(comparison_plot_path))
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Phase 1 results')
    parser.add_argument('--results', type=str, help='Path to results JSON file')
    parser.add_argument('--latest', action='store_true', help='Use latest results file')
    
    args = parser.parse_args()
    
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'path_b_phase1'
    
    if args.results:
        json_path = args.results
    elif args.latest:
        # Find latest results file
        json_files = list(results_dir.glob('phase1_results_*.json'))
        if json_files:
            json_path = str(max(json_files, key=lambda p: p.stat().st_mtime))
            print(f"Using latest results: {json_path}")
        else:
            print("No results files found!")
            sys.exit(1)
    else:
        # Demo mode with sample data
        print("No results file specified. Creating demo visualization...")
        
        # Create sample results for testing
        demo_results = {
            'experiment_info': {
                'n_steps': 1000,
                'grid_size': 64,
                'initial_pop': 100,
                'n_trials': 1
            },
            'autopoietic': [{
                'history': {
                    'steps': list(range(0, 1001, 100)),
                    'population_size': [100, 95, 102, 98, 110, 105, 108, 112, 115, 110, 108],
                    'avg_coherence': [0.5, 0.52, 0.55, 0.58, 0.60, 0.62, 0.61, 0.63, 0.62, 0.64, 0.63],
                    'std_coherence': [0.1] * 11,
                    'avg_energy': [1.0, 0.95, 0.98, 1.02, 1.05, 1.03, 1.06, 1.04, 1.05, 1.06, 1.04],
                    'avg_material': [0.5, 0.48, 0.52, 0.55, 0.53, 0.56, 0.54, 0.55, 0.57, 0.56, 0.55],
                    'avg_age': [0, 50, 80, 100, 120, 130, 140, 145, 150, 155, 160],
                    'max_age': [0, 100, 200, 300, 400, 450, 500, 550, 600, 650, 700],
                    'total_births': [0, 10, 25, 45, 70, 95, 125, 160, 200, 240, 280],
                    'total_deaths': [0, 15, 23, 47, 60, 90, 117, 148, 185, 230, 272],
                    'qd_coverage': [0, 5, 12, 20, 28, 35, 42, 50, 58, 65, 72],
                    'generation': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                },
                'final_stats': {'population_size': 108, 'total_births': 280, 'total_deaths': 272},
                'qd_metrics': {'coverage': 72},
                'survival_analysis': {'coherence_age_correlation': 0.28}
            }],
            'random': [{
                'history': {
                    'steps': list(range(0, 1001, 100)),
                    'population_size': [100, 85, 70, 55, 40, 25, 15, 8, 3, 0, 0],
                    'avg_coherence': [0.3] * 11,
                    'avg_energy': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0, 0],
                    'avg_age': [0, 30, 50, 60, 65, 70, 72, 74, 76, 0, 0],
                    'total_births': [0, 5, 8, 10, 12, 13, 14, 14, 14, 14, 14],
                    'total_deaths': [0, 20, 38, 55, 72, 88, 99, 106, 111, 114, 114]
                },
                'final_stats': {'population_size': 0, 'total_births': 14, 'total_deaths': 114}
            }],
            'summary': {
                'autopoietic': {
                    'survival_time_mean': 1000,
                    'survival_time_std': 0,
                    'final_pop_mean': 108,
                    'final_pop_std': 0,
                    'total_births_mean': 280,
                    'total_births_std': 0
                },
                'random': {
                    'survival_time_mean': 800,
                    'survival_time_std': 100,
                    'final_pop_mean': 0,
                    'final_pop_std': 0,
                    'total_births_mean': 14,
                    'total_births_std': 5
                }
            }
        }
        
        fig = create_phase1_plots(demo_results)
        plt.show()
        sys.exit(0)
    
    visualize_from_file(json_path)
