"""
GENESIS Path B Phase 0: Visualization

Generate plots for experiment results.
"""

import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Warning] matplotlib not available. Visualization disabled.")


def plot_results(results: Dict, output_dir: str, filename: Optional[str] = None):
    """
    Generate 4-subplot figure for Phase 0 results
    
    Subplots:
    1. Population size over time
    2. Average coherence over time
    3. Average energy over time
    4. Cumulative births and deaths
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plot
        filename: Optional custom filename
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Skip] Visualization requires matplotlib")
        return None
    
    history = results['history']
    steps = history['step']
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GENESIS Path B Phase 0: Minimal Artificial Life Results', 
                 fontsize=14, fontweight='bold')
    
    # 1. Population Size
    ax1 = axes[0, 0]
    ax1.plot(steps, history['population_size'], 'b-', linewidth=2, label='Population')
    ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Population Size')
    ax1.set_title('Population Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. Average Coherence
    ax2 = axes[0, 1]
    coherences = history['avg_coherence']
    std_coherences = history['std_coherence']
    ax2.plot(steps, coherences, 'g-', linewidth=2, label='Avg Coherence')
    ax2.fill_between(steps, 
                     np.array(coherences) - np.array(std_coherences),
                     np.array(coherences) + np.array(std_coherences),
                     alpha=0.3, color='green')
    ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Reproduction threshold')
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Death threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Coherence')
    ax2.set_title('Organizational Coherence')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Average Energy
    ax3 = axes[1, 0]
    ax3.plot(steps, history['avg_energy'], 'orange', linewidth=2, label='Avg Energy')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Reproduction energy')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Death threshold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Energy')
    ax3.set_title('Average Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Births and Deaths
    ax4 = axes[1, 1]
    ax4.plot(steps, history['total_births'], 'b-', linewidth=2, label='Cumulative Births')
    ax4.plot(steps, history['total_deaths'], 'r-', linewidth=2, label='Cumulative Deaths')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Count')
    ax4.set_title('Birth/Death Events')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase0_plots_{timestamp}.png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {filepath}")
    return filepath


def plot_coherence_survival(results: Dict, output_dir: str):
    """
    Plot coherence distribution for dead vs living agents
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Skip] Visualization requires matplotlib")
        return None
    
    death_log = results.get('death_log', [])
    
    if len(death_log) < 5:
        print("[Skip] Not enough death data for coherence-survival plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Coherence-Survival Analysis', fontsize=12, fontweight='bold')
    
    # Extract data
    dead_coherences = [d['final_coherence'] for d in death_log]
    dead_ages = [d['age'] for d in death_log]
    
    # 1. Histogram of final coherence for dead agents
    ax1 = axes[0]
    ax1.hist(dead_coherences, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.3, color='black', linestyle='--', linewidth=2, label='Death threshold')
    ax1.set_xlabel('Final Coherence')
    ax1.set_ylabel('Count')
    ax1.set_title('Final Coherence of Dead Agents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter: Age vs Final Coherence
    ax2 = axes[1]
    ax2.scatter(dead_ages, dead_coherences, alpha=0.5, c='red', s=30)
    ax2.axhline(y=0.3, color='black', linestyle='--', linewidth=2, label='Death threshold')
    ax2.set_xlabel('Age at Death')
    ax2.set_ylabel('Final Coherence')
    ax2.set_title('Age vs Coherence at Death')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"coherence_survival_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Coherence-survival plot saved to: {filepath}")
    return filepath


def plot_diversity_evolution(results: Dict, output_dir: str):
    """
    Plot population diversity over time
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[Skip] Visualization requires matplotlib")
        return None
    
    history = results['history']
    steps = history['step']
    diversity = history.get('diversity', [])
    
    if not diversity:
        print("[Skip] No diversity data available")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(steps, diversity, 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Weight Variance (Diversity)')
    ax.set_title('Population Genetic Diversity Over Time')
    ax.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"diversity_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diversity plot saved to: {filepath}")
    return filepath


def create_all_visualizations(results: Dict, output_dir: str):
    """Create all visualization plots"""
    plots = []
    
    p1 = plot_results(results, output_dir)
    if p1:
        plots.append(p1)
    
    p2 = plot_coherence_survival(results, output_dir)
    if p2:
        plots.append(p2)
    
    p3 = plot_diversity_evolution(results, output_dir)
    if p3:
        plots.append(p3)
    
    return plots


if __name__ == "__main__":
    import json
    
    # Try to load most recent results
    results_dir = "/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_b_phase0"
    
    # Find most recent results file
    import glob
    result_files = glob.glob(os.path.join(results_dir, "phase0_results_*.json"))
    
    if result_files:
        latest = max(result_files, key=os.path.getctime)
        print(f"Loading results from: {latest}")
        
        with open(latest, 'r') as f:
            results = json.load(f)
        
        create_all_visualizations(results, results_dir)
    else:
        print("No results files found. Run phase0_experiment.py first.")
