"""
Visualization Module for Hybrid Autopoietic-ML Experiments

Creates plots for:
- Ablation bar charts
- Robustness curves
- Training curves
- Coherence evolution
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'Baseline': '#1f77b4',
    '+Coherence': '#ff7f0e',
    '+Plasticity': '#2ca02c',
    '+SelfOrg': '#d62728',
    '+Coh+Plas': '#9467bd',
    '+Coh+Self': '#8c564b',
    '+All': '#e377c2'
}


def plot_ablation_results(
    results: Dict,
    save_path: Optional[str] = None,
    show_error_bars: bool = True
):
    """
    Create bar chart comparing accuracy across ablation conditions.
    
    Args:
        results: Results dictionary from ablation study
        save_path: Path to save figure
        show_error_bars: Whether to show error bars
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    conditions = list(results['conditions'].keys())
    means = [results['conditions'][c]['mean_accuracy'] for c in conditions]
    stds = [results['conditions'][c]['std_accuracy'] for c in conditions]
    
    x = np.arange(len(conditions))
    colors = [COLORS.get(c, '#333333') for c in conditions]
    
    if show_error_bars:
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    else:
        bars = ax.bar(x, means, color=colors, alpha=0.8)
    
    # Add baseline reference line
    baseline_acc = results['conditions']['Baseline']['mean_accuracy']
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    
    # Labels and formatting
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Ablation Study Results - {results["dataset"].upper()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limits
    ax.set_ylim([min(means) - 5, max(means) + 3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_improvement_chart(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Create chart showing improvement over baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = [c for c in results['conditions'].keys() if c != 'Baseline']
    baseline_acc = results['conditions']['Baseline']['mean_accuracy']
    
    improvements = [
        results['conditions'][c]['mean_accuracy'] - baseline_acc
        for c in conditions
    ]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, improvements, color=colors, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Improvement Analysis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        offset = 3 if height >= 0 else -10
        ax.annotate(f'{imp:+.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, offset),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_robustness_curves(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Create robustness curves (accuracy vs noise/attack level).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Noise robustness
    ax1 = axes[0]
    for condition_name, data in results['conditions'].items():
        noise_data = data['noise_robustness']
        noise_levels = sorted([float(k) for k in noise_data.keys()])
        accuracies = [noise_data[str(n) if str(n) in noise_data else n]['accuracy'] for n in noise_levels]
        
        ax1.plot(noise_levels, accuracies, 'o-', 
                label=condition_name, color=COLORS.get(condition_name, '#333333'),
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Noise Level (std)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Robustness to Gaussian Noise', fontsize=14)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Adversarial robustness
    ax2 = axes[1]
    for condition_name, data in results['conditions'].items():
        adv_data = data['adversarial_robustness']
        epsilons = sorted([float(k) for k in adv_data.keys()])
        accuracies = [adv_data[str(e) if str(e) in adv_data else e]['adversarial_accuracy'] for e in epsilons]
        
        ax2.plot(epsilons, accuracies, 'o-',
                label=condition_name, color=COLORS.get(condition_name, '#333333'),
                linewidth=2, markersize=6)
    
    ax2.set_xlabel('FGSM Epsilon', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Robustness to FGSM Attack', fontsize=14)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_training_curves(
    results: Dict,
    condition: str = 'Baseline',
    trial: int = 0,
    save_path: Optional[str] = None
):
    """
    Plot training curves (loss and accuracy over epochs).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get trial data
    trial_data = results['conditions'][condition]['trials'][trial]
    history = trial_data['history']
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss Curves - {condition}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Accuracy Curves - {condition}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_coherence_evolution(
    results: Dict,
    conditions: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot coherence metrics evolution over training.
    """
    if conditions is None:
        conditions = ['+Coherence', '+All']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['predictability', 'stability', 'complexity', 'circularity']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for condition in conditions:
            if condition not in results['conditions']:
                continue
            
            trial_data = results['conditions'][condition]['trials'][0]
            history = trial_data['history']
            coherence_data = history.get('coherence', [])
            
            if not coherence_data:
                continue
            
            values = []
            for coh in coherence_data:
                if isinstance(coh, dict) and metric in coh:
                    values.append(coh[metric])
            
            if values:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, '-', 
                       label=condition, color=COLORS.get(condition, '#333333'),
                       linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} Evolution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Coherence Metrics During Training', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_timing_comparison(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot training time comparison across conditions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = list(results['conditions'].keys())
    times = [results['conditions'][c]['mean_time'] for c in conditions]
    stds = [results['conditions'][c]['std_time'] for c in conditions]
    
    x = np.arange(len(conditions))
    colors = [COLORS.get(c, '#333333') for c in conditions]
    
    bars = ax.bar(x, times, yerr=stds, color=colors, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    
    # Add overhead percentage relative to baseline
    baseline_time = times[0]
    for bar, t in zip(bars[1:], times[1:]):
        overhead = (t - baseline_time) / baseline_time * 100
        ax.annotate(f'+{overhead:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_all_conditions_training(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot test accuracy curves for all conditions on one graph.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for condition in results['conditions']:
        trial_data = results['conditions'][condition]['trials'][0]
        history = trial_data['history']
        epochs = range(1, len(history['test_acc']) + 1)
        
        ax.plot(epochs, history['test_acc'], '-',
               label=condition, color=COLORS.get(condition, '#333333'),
               linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Training Progress - All Conditions', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def create_all_visualizations(
    ablation_results_path: str,
    robustness_results_path: Optional[str] = None,
    output_dir: str = './results'
):
    """
    Create all visualizations from results files.
    
    Args:
        ablation_results_path: Path to ablation study results JSON
        robustness_results_path: Path to robustness study results JSON
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ablation results
    with open(ablation_results_path, 'r') as f:
        ablation_results = json.load(f)
    
    dataset = ablation_results['dataset']
    print(f"Creating visualizations for {dataset}...")
    
    # 1. Ablation bar chart
    plot_ablation_results(
        ablation_results,
        save_path=os.path.join(output_dir, f'ablation_results_{dataset}.png')
    )
    
    # 2. Improvement chart
    plot_improvement_chart(
        ablation_results,
        save_path=os.path.join(output_dir, f'improvement_{dataset}.png')
    )
    
    # 3. Training curves for each condition
    for condition in ablation_results['conditions']:
        plot_training_curves(
            ablation_results,
            condition=condition,
            save_path=os.path.join(output_dir, f'training_{condition.replace("+", "plus_")}_{dataset}.png')
        )
    
    # 4. All conditions training comparison
    plot_all_conditions_training(
        ablation_results,
        save_path=os.path.join(output_dir, f'all_training_{dataset}.png')
    )
    
    # 5. Coherence evolution
    plot_coherence_evolution(
        ablation_results,
        save_path=os.path.join(output_dir, f'coherence_evolution_{dataset}.png')
    )
    
    # 6. Timing comparison
    plot_timing_comparison(
        ablation_results,
        save_path=os.path.join(output_dir, f'timing_{dataset}.png')
    )
    
    # Load and visualize robustness results if available
    if robustness_results_path and os.path.exists(robustness_results_path):
        with open(robustness_results_path, 'r') as f:
            robustness_results = json.load(f)
        
        # 7. Robustness curves
        plot_robustness_curves(
            robustness_results,
            save_path=os.path.join(output_dir, f'robustness_curves_{dataset}.png')
        )
    
    print(f"\nAll visualizations saved to: {output_dir}")


def generate_summary_report(
    ablation_results_path: str,
    robustness_results_path: Optional[str] = None,
    output_path: str = './results/summary_report.md'
):
    """
    Generate a markdown summary report.
    """
    with open(ablation_results_path, 'r') as f:
        ablation = json.load(f)
    
    report = []
    report.append("# Hybrid Autopoietic-ML Experiment Results\n")
    report.append(f"**Dataset:** {ablation['dataset'].upper()}\n")
    report.append(f"**Trials:** {ablation['n_trials']}\n")
    report.append(f"**Epochs:** {ablation['n_epochs']}\n")
    report.append(f"**Timestamp:** {ablation['timestamp']}\n")
    
    report.append("\n## Ablation Study Results\n")
    report.append("| Condition | Accuracy | Std | Improvement | Time (s) |\n")
    report.append("|-----------|----------|-----|-------------|----------|\n")
    
    baseline_acc = ablation['conditions']['Baseline']['mean_accuracy']
    
    for condition in ablation['conditions']:
        data = ablation['conditions'][condition]
        acc = data['mean_accuracy']
        std = data['std_accuracy']
        imp = acc - baseline_acc
        time = data['mean_time']
        sign = '+' if imp >= 0 else ''
        report.append(f"| {condition} | {acc:.2f}% | {std:.2f}% | {sign}{imp:.2f}% | {time:.1f} |\n")
    
    if robustness_results_path and os.path.exists(robustness_results_path):
        with open(robustness_results_path, 'r') as f:
            robustness = json.load(f)
        
        report.append("\n## Robustness Results\n")
        report.append("\n### Noise Robustness\n")
        report.append("| Condition | Clean | N=0.1 | N=0.2 | N=0.5 |\n")
        report.append("|-----------|-------|-------|-------|-------|\n")
        
        for condition in robustness['conditions']:
            data = robustness['conditions'][condition]
            clean = data['clean_accuracy']
            noise_data = data['noise_robustness']
            n01 = noise_data.get('0.1', noise_data.get(0.1, {})).get('accuracy', 'N/A')
            n02 = noise_data.get('0.2', noise_data.get(0.2, {})).get('accuracy', 'N/A')
            n05 = noise_data.get('0.5', noise_data.get(0.5, {})).get('accuracy', 'N/A')
            
            n01_str = f"{n01:.2f}%" if isinstance(n01, (int, float)) else n01
            n02_str = f"{n02:.2f}%" if isinstance(n02, (int, float)) else n02
            n05_str = f"{n05:.2f}%" if isinstance(n05, (int, float)) else n05
            
            report.append(f"| {condition} | {clean:.2f}% | {n01_str} | {n02_str} | {n05_str} |\n")
        
        report.append("\n### Adversarial Robustness (FGSM)\n")
        report.append("| Condition | eps=0.05 | eps=0.1 | eps=0.2 |\n")
        report.append("|-----------|----------|---------|--------|\n")
        
        for condition in robustness['conditions']:
            data = robustness['conditions'][condition]
            adv_data = data['adversarial_robustness']
            e005 = adv_data.get('0.05', adv_data.get(0.05, {})).get('adversarial_accuracy', 'N/A')
            e01 = adv_data.get('0.1', adv_data.get(0.1, {})).get('adversarial_accuracy', 'N/A')
            e02 = adv_data.get('0.2', adv_data.get(0.2, {})).get('adversarial_accuracy', 'N/A')
            
            e005_str = f"{e005:.2f}%" if isinstance(e005, (int, float)) else e005
            e01_str = f"{e01:.2f}%" if isinstance(e01, (int, float)) else e01
            e02_str = f"{e02:.2f}%" if isinstance(e02, (int, float)) else e02
            
            report.append(f"| {condition} | {e005_str} | {e01_str} | {e02_str} |\n")
    
    report.append("\n## Key Findings\n")
    
    # Find best condition
    best_condition = max(
        ablation['conditions'].keys(),
        key=lambda c: ablation['conditions'][c]['mean_accuracy']
    )
    best_acc = ablation['conditions'][best_condition]['mean_accuracy']
    best_imp = best_acc - baseline_acc
    
    report.append(f"1. **Best performing condition:** {best_condition} ({best_acc:.2f}%, +{best_imp:.2f}% over baseline)\n")
    
    # Coherence impact
    coh_acc = ablation['conditions']['+Coherence']['mean_accuracy']
    coh_imp = coh_acc - baseline_acc
    report.append(f"2. **Coherence regularization impact:** {'+' if coh_imp >= 0 else ''}{coh_imp:.2f}%\n")
    
    # Self-organizing impact
    self_acc = ablation['conditions']['+SelfOrg']['mean_accuracy']
    self_imp = self_acc - baseline_acc
    report.append(f"3. **Self-organizing layers impact:** {'+' if self_imp >= 0 else ''}{self_imp:.2f}%\n")
    
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"Summary report saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations')
    parser.add_argument('--ablation', type=str, help='Path to ablation results JSON')
    parser.add_argument('--robustness', type=str, help='Path to robustness results JSON')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--demo', action='store_true', help='Create demo with synthetic data')
    
    args = parser.parse_args()
    
    if args.demo:
        # Create demo data
        print("Creating demo visualizations with synthetic data...")
        
        demo_results = {
            'dataset': 'demo',
            'n_trials': 3,
            'n_epochs': 20,
            'timestamp': datetime.now().isoformat(),
            'conditions': {}
        }
        
        conditions = ['Baseline', '+Coherence', '+Plasticity', '+SelfOrg', '+Coh+Plas', '+Coh+Self', '+All']
        base_acc = 95.0
        improvements = [0, 0.5, 0.3, 0.4, 0.8, 0.7, 1.0]
        
        for cond, imp in zip(conditions, improvements):
            acc = base_acc + imp
            demo_results['conditions'][cond] = {
                'mean_accuracy': acc + np.random.uniform(-0.2, 0.2),
                'std_accuracy': np.random.uniform(0.1, 0.3),
                'mean_time': 60 + imp * 10,
                'std_time': 5,
                'trials': [{
                    'history': {
                        'train_loss': list(np.linspace(2.0, 0.1, 20) + np.random.uniform(-0.05, 0.05, 20)),
                        'test_loss': list(np.linspace(2.0, 0.15, 20) + np.random.uniform(-0.05, 0.05, 20)),
                        'train_acc': list(np.linspace(50, 99, 20) + np.random.uniform(-1, 1, 20)),
                        'test_acc': list(np.linspace(50, acc, 20) + np.random.uniform(-1, 1, 20)),
                        'coherence': [
                            {'predictability': 0.5 + i*0.02, 'stability': 0.6 + i*0.015,
                             'complexity': 0.4 + i*0.01, 'circularity': 0.45 + i*0.01}
                            for i in range(20)
                        ] if 'Coherence' in cond or 'All' in cond else []
                    }
                }]
            }
        
        # Save demo data
        os.makedirs(args.output, exist_ok=True)
        demo_path = os.path.join(args.output, 'demo_ablation_results.json')
        with open(demo_path, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        # Create visualizations
        create_all_visualizations(demo_path, output_dir=args.output)
        generate_summary_report(demo_path, output_path=os.path.join(args.output, 'demo_report.md'))
        
    elif args.ablation:
        create_all_visualizations(
            args.ablation,
            args.robustness,
            args.output
        )
        generate_summary_report(
            args.ablation,
            args.robustness,
            os.path.join(args.output, 'summary_report.md')
        )
    else:
        print("Please provide --ablation path or use --demo for synthetic data")
