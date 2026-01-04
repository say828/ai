"""
Visualization for Continual Learning Experiments

Creates:
    1. Learning curves (accuracy per task over time)
    2. Forgetting matrix (5x5 heatmap)
    3. Statistical comparison (bar plot with error bars)
    4. Coherence tracking (for autopoietic method)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from typing import Dict, List, Optional
import glob


# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'autopoietic': '#2ecc71',  # Green
    'finetuning': '#e74c3c',   # Red
    'ewc': '#3498db',          # Blue
    'replay': '#9b59b6'        # Purple
}
METHOD_NAMES = {
    'autopoietic': 'Autopoietic',
    'finetuning': 'Fine-tuning',
    'ewc': 'EWC',
    'replay': 'Replay'
}


def load_latest_results(results_dir: str = './results') -> Dict:
    """Load the most recent statistics file."""
    pattern = os.path.join(results_dir, 'statistics_*.json')
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No statistics files found in {results_dir}")
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(statistics: Dict, 
                             save_path: Optional[str] = None):
    """
    Create bar plot comparing average accuracy across methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(statistics.keys())
    means = [statistics[m]['avg_accuracy']['mean'] for m in methods]
    stds = [statistics[m]['avg_accuracy']['std'] for m in methods]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS[m] for m in methods],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Continual Learning: Average Accuracy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=11)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_forgetting_comparison(statistics: Dict,
                               save_path: Optional[str] = None):
    """
    Create bar plot comparing forgetting measure across methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(statistics.keys())
    means = [statistics[m]['forgetting']['mean'] for m in methods]
    stds = [statistics[m]['forgetting']['std'] for m in methods]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS[m] for m in methods],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Forgetting Measure (lower is better)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Continual Learning: Forgetting Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=11)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_forgetting_matrix(statistics: Dict,
                           method: str = 'autopoietic',
                           trial_idx: int = 0,
                           save_path: Optional[str] = None):
    """
    Create heatmap of accuracy matrix (forgetting visualization).
    
    Rows: After training on task i
    Cols: Accuracy on task j
    """
    acc_matrix = np.array(statistics[method]['accuracy_matrices'][trial_idx])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy', rotation=-90, va="bottom", fontsize=11)
    
    # Labels
    task_labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticklabels(task_labels, fontsize=10)
    
    ax.set_xlabel('Evaluated on Task (digits)', fontsize=12)
    ax.set_ylabel('After Training on Task (digits)', fontsize=12)
    ax.set_title(f'Accuracy Matrix: {METHOD_NAMES[method]}', fontsize=14)
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            if j <= i:  # Only show where we have trained
                text = ax.text(j, i, f'{acc_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=10,
                              color='white' if acc_matrix[i, j] < 0.5 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_all_forgetting_matrices(statistics: Dict,
                                 trial_idx: int = 0,
                                 save_path: Optional[str] = None):
    """
    Create 2x2 grid of forgetting matrices for all methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    methods = ['autopoietic', 'finetuning', 'ewc', 'replay']
    
    for ax, method in zip(axes, methods):
        acc_matrix = np.array(statistics[method]['accuracy_matrices'][trial_idx])
        
        im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        task_labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(task_labels, fontsize=9)
        ax.set_yticklabels(task_labels, fontsize=9)
        
        ax.set_xlabel('Eval Task', fontsize=10)
        ax.set_ylabel('After Task', fontsize=10)
        ax.set_title(f'{METHOD_NAMES[method]}', fontsize=12, fontweight='bold',
                    color=COLORS[method])
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                if j <= i:
                    ax.text(j, i, f'{acc_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if acc_matrix[i, j] < 0.5 else 'black')
    
    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Accuracy')
    
    fig.suptitle('Accuracy Matrices: Forgetting Visualization', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_learning_curves(statistics: Dict,
                         trial_idx: int = 0,
                         save_path: Optional[str] = None):
    """
    Plot accuracy on each task over the course of training.
    
    Shows how accuracy on earlier tasks changes as new tasks are learned.
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    
    task_labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
    methods = ['autopoietic', 'finetuning', 'ewc', 'replay']
    
    for task_idx, ax in enumerate(axes):
        ax.set_title(f'Task {task_idx} ({task_labels[task_idx]})', fontsize=11)
        ax.set_xlabel('After Training on Task', fontsize=10)
        if task_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(task_idx, 5))
        ax.set_xticklabels(task_labels[task_idx:], fontsize=9)
        
        for method in methods:
            acc_matrix = np.array(statistics[method]['accuracy_matrices'][trial_idx])
            
            # Get accuracy on this task after training on tasks task_idx, task_idx+1, ...
            accs = acc_matrix[task_idx:, task_idx]
            x = range(task_idx, 5)
            
            ax.plot(x, accs, 'o-', color=COLORS[method], 
                   label=METHOD_NAMES[method], linewidth=2, markersize=6)
    
    # Add legend to last subplot
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
    
    fig.suptitle('Task Accuracy Over Time (Forgetting Curves)', fontsize=13, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_computational_cost(statistics: Dict,
                           save_path: Optional[str] = None):
    """
    Compare computational cost (FLOPs and time) across methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(statistics.keys())
    
    # FLOPs
    ax = axes[0]
    flops_means = [statistics[m]['flops']['mean'] for m in methods]
    flops_stds = [statistics[m]['flops']['std'] for m in methods]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, np.array(flops_means) / 1e9, 
                  yerr=np.array(flops_stds) / 1e9, capsize=5,
                  color=[COLORS[m] for m in methods],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('GFLOPs', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Computational Cost: FLOPs', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=10)
    
    # Time
    ax = axes[1]
    time_means = [statistics[m]['time']['mean'] for m in methods]
    time_stds = [statistics[m]['time']['std'] for m in methods]
    
    bars = ax.bar(x, time_means, yerr=time_stds, capsize=5,
                  color=[COLORS[m] for m in methods],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Computational Cost: Training Time', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_summary_dashboard(statistics: Dict,
                          save_path: Optional[str] = None):
    """
    Create comprehensive summary dashboard.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    methods = list(statistics.keys())
    
    # 1. Accuracy comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    means = [statistics[m]['avg_accuracy']['mean'] for m in methods]
    stds = [statistics[m]['avg_accuracy']['std'] for m in methods]
    x = np.arange(len(methods))
    ax1.bar(x, means, yerr=stds, capsize=4,
            color=[COLORS[m] for m in methods], edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=9, rotation=15)
    ax1.set_ylim(0, 1.0)
    
    # 2. Forgetting comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    means = [statistics[m]['forgetting']['mean'] for m in methods]
    stds = [statistics[m]['forgetting']['std'] for m in methods]
    ax2.bar(x, means, yerr=stds, capsize=4,
            color=[COLORS[m] for m in methods], edgecolor='black')
    ax2.set_ylabel('Forgetting', fontsize=11)
    ax2.set_title('Forgetting Measure (lower = better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=9, rotation=15)
    
    # 3. Forgetting matrices (top right, 2x2)
    ax3 = fig.add_subplot(gs[0, 2])
    # Show autopoietic matrix
    acc_matrix = np.array(statistics['autopoietic']['accuracy_matrices'][0])
    im = ax3.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_title('Autopoietic Accuracy Matrix', fontsize=12, fontweight='bold')
    task_labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
    ax3.set_xticks(np.arange(5))
    ax3.set_yticks(np.arange(5))
    ax3.set_xticklabels(task_labels, fontsize=8)
    ax3.set_yticklabels(task_labels, fontsize=8)
    ax3.set_xlabel('Eval Task', fontsize=10)
    ax3.set_ylabel('After Task', fontsize=10)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Learning curves (bottom, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    task_idx = 0  # Show first task
    for method in methods:
        acc_matrix = np.array(statistics[method]['accuracy_matrices'][0])
        accs = acc_matrix[:, task_idx]
        ax4.plot(range(5), accs, 'o-', color=COLORS[method],
                label=METHOD_NAMES[method], linewidth=2, markersize=8)
    ax4.set_xlabel('After Training on Task', fontsize=11)
    ax4.set_ylabel('Accuracy on Task 0 (0-1)', fontsize=11)
    ax4.set_title('Forgetting Curve: Task 0', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(task_labels, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 1.05)
    
    # 5. Computational cost (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    time_means = [statistics[m]['time']['mean'] for m in methods]
    ax5.bar(x, time_means, color=[COLORS[m] for m in methods], edgecolor='black')
    ax5.set_ylabel('Time (seconds)', fontsize=11)
    ax5.set_title('Training Time', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=9, rotation=15)
    
    fig.suptitle('Continual Learning Experiment: Summary Dashboard', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_visualizations(results_dir: str = './results'):
    """
    Generate all visualizations from the latest results.
    """
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Load results
    statistics = load_latest_results(results_dir)
    
    # Create figures directory
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate all plots
    print("\n1. Accuracy comparison...")
    plot_accuracy_comparison(statistics, 
                            save_path=os.path.join(figures_dir, 'accuracy_comparison.png'))
    
    print("2. Forgetting comparison...")
    plot_forgetting_comparison(statistics,
                              save_path=os.path.join(figures_dir, 'forgetting_comparison.png'))
    
    print("3. Forgetting matrices...")
    plot_all_forgetting_matrices(statistics,
                                save_path=os.path.join(figures_dir, 'forgetting_matrices.png'))
    
    print("4. Learning curves...")
    plot_learning_curves(statistics,
                        save_path=os.path.join(figures_dir, 'learning_curves.png'))
    
    print("5. Computational cost...")
    plot_computational_cost(statistics,
                           save_path=os.path.join(figures_dir, 'computational_cost.png'))
    
    print("6. Summary dashboard...")
    plot_summary_dashboard(statistics,
                          save_path=os.path.join(figures_dir, 'summary_dashboard.png'))
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 60)
    
    return statistics


if __name__ == "__main__":
    generate_all_visualizations()
