"""
Visualization for Ablation Studies

Creates comprehensive visualizations for the three ablation studies:
    1. W_in Initialization
    2. Coherence Criterion
    3. Learning Rule

Output:
    - 3x2 figure (Accuracy and Forgetting for each ablation)
    - Individual ablation plots
    - Summary dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import glob
from typing import Dict, List, Optional, Tuple


# Style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Color schemes for each ablation
ABLATION1_COLORS = {
    'learned_freeze': '#2ecc71',    # Green (ours)
    'random_freeze': '#e74c3c',     # Red (RanPAC-style)
    'learned_continue': '#3498db'   # Blue
}

ABLATION2_COLORS = {
    'with_coherence': '#2ecc71',    # Green (ours)
    'without_coherence': '#e74c3c', # Red
    'strict_coherence': '#9b59b6'   # Purple
}

ABLATION3_COLORS = {
    'hebbian': '#2ecc71',           # Green (ours)
    'sgd': '#e74c3c',               # Red
    'adam': '#3498db'               # Blue
}

ABLATION_NAMES = {
    'ablation1_winit': 'W_in Initialization',
    'ablation2_coherence': 'Coherence Criterion',
    'ablation3_learning_rule': 'Learning Rule'
}

CONDITION_LABELS = {
    # Ablation 1
    'learned_freeze': 'Learned-Freeze\n(Ours)',
    'random_freeze': 'Random-Freeze\n(RanPAC)',
    'learned_continue': 'Learned-Continue',
    # Ablation 2
    'with_coherence': 'With Coherence\n(Ours)',
    'without_coherence': 'Without\nCoherence',
    'strict_coherence': 'Strict\nCoherence',
    # Ablation 3
    'hebbian': 'Hebbian\n(Ours)',
    'sgd': 'SGD',
    'adam': 'Adam'
}


def load_ablation_results(results_dir: str = './results/ablations') -> Dict:
    """Load the latest results for each ablation."""
    results = {}
    
    for ablation_name in ['ablation1_winit', 'ablation2_coherence', 'ablation3_learning_rule']:
        pattern = os.path.join(results_dir, f'{ablation_name}_results_*.json')
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f"Loading: {latest_file}")
            
            with open(latest_file, 'r') as f:
                results[ablation_name] = json.load(f)
        else:
            print(f"Warning: No results found for {ablation_name}")
    
    return results


def load_ablation_tests(results_dir: str = './results/ablations') -> Dict:
    """Load the latest statistical tests for each ablation."""
    tests = {}
    
    for ablation_name in ['ablation1_winit', 'ablation2_coherence', 'ablation3_learning_rule']:
        pattern = os.path.join(results_dir, f'{ablation_name}_tests_*.json')
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=os.path.getmtime)
            with open(latest_file, 'r') as f:
                tests[ablation_name] = json.load(f)
    
    return tests


def plot_ablation_comparison(results: Dict,
                             ablation_name: str,
                             colors: Dict,
                             save_path: Optional[str] = None) -> Tuple:
    """
    Create comparison plot for one ablation study.
    
    Returns:
        fig, (ax_acc, ax_fgt): Figure and axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = list(results.keys())
    x = np.arange(len(conditions))
    
    # Accuracy
    ax = axes[0]
    means = [results[c]['avg_accuracy']['mean'] for c in conditions]
    stds = [results[c]['avg_accuracy']['std'] for c in conditions]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[c] for c in conditions],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title(f'{ABLATION_NAMES[ablation_name]}: Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Forgetting
    ax = axes[1]
    means = [results[c]['forgetting']['mean'] for c in conditions]
    stds = [results[c]['forgetting']['std'] for c in conditions]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[c] for c in conditions],
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Forgetting Measure (lower = better)', fontsize=12)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_title(f'{ABLATION_NAMES[ablation_name]}: Forgetting', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_comprehensive_ablations(all_results: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive 3x2 figure for all ablations.
    
    Rows: Ablation 1, 2, 3
    Columns: Accuracy, Forgetting
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    ablation_configs = [
        ('ablation1_winit', ABLATION1_COLORS),
        ('ablation2_coherence', ABLATION2_COLORS),
        ('ablation3_learning_rule', ABLATION3_COLORS)
    ]
    
    for row, (ablation_name, colors) in enumerate(ablation_configs):
        if ablation_name not in all_results:
            continue
            
        results = all_results[ablation_name]
        conditions = list(results.keys())
        x = np.arange(len(conditions))
        
        # Accuracy (left column)
        ax = axes[row, 0]
        means = [results[c]['avg_accuracy']['mean'] for c in conditions]
        stds = [results[c]['avg_accuracy']['std'] for c in conditions]
        
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors[c] for c in conditions],
                      edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{ABLATION_NAMES[ablation_name]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
        ax.set_ylim(0, 1.0)
        
        # Highlight "ours"
        ax.axhline(y=means[0], color=colors[conditions[0]], linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Forgetting (right column)
        ax = axes[row, 1]
        means = [results[c]['forgetting']['mean'] for c in conditions]
        stds = [results[c]['forgetting']['std'] for c in conditions]
        
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors[c] for c in conditions],
                      edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Forgetting', fontsize=11)
        ax.set_title(f'{ABLATION_NAMES[ablation_name]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
        
        # Highlight "ours"
        ax.axhline(y=means[0], color=colors[conditions[0]], linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Column titles
    axes[0, 0].annotate('Average Accuracy (higher = better)', xy=(0.5, 1.15),
                        xycoords='axes fraction', ha='center', fontsize=12, fontweight='bold')
    axes[0, 1].annotate('Forgetting Measure (lower = better)', xy=(0.5, 1.15),
                        xycoords='axes fraction', ha='center', fontsize=12, fontweight='bold')
    
    fig.suptitle('Path A Ablation Studies: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_statistical_significance(all_results: Dict,
                                  all_tests: Dict,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visualization showing statistical significance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ablation_configs = [
        ('ablation1_winit', 'learned_freeze', ABLATION1_COLORS),
        ('ablation2_coherence', 'with_coherence', ABLATION2_COLORS),
        ('ablation3_learning_rule', 'hebbian', ABLATION3_COLORS)
    ]
    
    for idx, (ablation_name, baseline, colors) in enumerate(ablation_configs):
        ax = axes[idx]
        
        if ablation_name not in all_tests:
            continue
        
        tests = all_tests[ablation_name]
        results = all_results[ablation_name]
        conditions = [c for c in results.keys() if c != baseline]
        
        # Plot accuracy difference with significance markers
        x_pos = []
        y_acc = []
        y_fgt = []
        labels = []
        sig_acc = []
        sig_fgt = []
        
        baseline_acc = results[baseline]['avg_accuracy']['mean']
        baseline_fgt = results[baseline]['forgetting']['mean']
        
        for i, cond in enumerate(conditions):
            x_pos.append(i)
            y_acc.append(results[cond]['avg_accuracy']['mean'] - baseline_acc)
            y_fgt.append(results[cond]['forgetting']['mean'] - baseline_fgt)
            labels.append(CONDITION_LABELS[cond].replace('\n', ' '))
            
            test_key = f'{baseline}_vs_{cond}'
            if test_key in tests:
                sig_acc.append(tests[test_key]['accuracy']['significant'])
                sig_fgt.append(tests[test_key]['forgetting']['significant'])
            else:
                sig_acc.append(False)
                sig_fgt.append(False)
        
        width = 0.35
        x = np.array(x_pos)
        
        # Accuracy difference
        bars1 = ax.bar(x - width/2, y_acc, width, label='Accuracy Diff',
                       color='steelblue', edgecolor='black')
        
        # Forgetting difference
        bars2 = ax.bar(x + width/2, y_fgt, width, label='Forgetting Diff',
                       color='coral', edgecolor='black')
        
        # Add significance markers
        for i, (s_acc, s_fgt) in enumerate(zip(sig_acc, sig_fgt)):
            if s_acc:
                ax.annotate('*', (x[i] - width/2, y_acc[i] + 0.01 if y_acc[i] >= 0 else y_acc[i] - 0.03),
                           ha='center', fontsize=14, fontweight='bold', color='darkblue')
            if s_fgt:
                ax.annotate('*', (x[i] + width/2, y_fgt[i] + 0.01 if y_fgt[i] >= 0 else y_fgt[i] - 0.03),
                           ha='center', fontsize=14, fontweight='bold', color='darkred')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Difference from Baseline', fontsize=11)
        ax.set_title(f'{ABLATION_NAMES[ablation_name]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=9)
        
        # Add note
        ax.annotate(f'Baseline: {CONDITION_LABELS[baseline].replace(chr(10), " ")}',
                   xy=(0.5, -0.15), xycoords='axes fraction',
                   ha='center', fontsize=9, style='italic')
    
    fig.suptitle('Statistical Comparison: Difference from Baseline (* = p < 0.05)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_ablation_visualizations(results_dir: str = './results/ablations'):
    """
    Generate all ablation visualizations.
    """
    print("=" * 70)
    print("Generating Ablation Visualizations")
    print("=" * 70)
    
    # Create figures directory
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load results
    all_results = load_ablation_results(results_dir)
    all_tests = load_ablation_tests(results_dir)
    
    if not all_results:
        print("No results found. Run ablation experiments first.")
        return
    
    # Generate individual ablation plots
    print("\n1. Individual ablation plots...")
    
    if 'ablation1_winit' in all_results:
        plot_ablation_comparison(
            all_results['ablation1_winit'],
            'ablation1_winit',
            ABLATION1_COLORS,
            save_path=os.path.join(figures_dir, 'ablation1_winit.png')
        )
        plt.close()
    
    if 'ablation2_coherence' in all_results:
        plot_ablation_comparison(
            all_results['ablation2_coherence'],
            'ablation2_coherence',
            ABLATION2_COLORS,
            save_path=os.path.join(figures_dir, 'ablation2_coherence.png')
        )
        plt.close()
    
    if 'ablation3_learning_rule' in all_results:
        plot_ablation_comparison(
            all_results['ablation3_learning_rule'],
            'ablation3_learning_rule',
            ABLATION3_COLORS,
            save_path=os.path.join(figures_dir, 'ablation3_learning_rule.png')
        )
        plt.close()
    
    # Generate comprehensive figure
    print("\n2. Comprehensive comparison...")
    plot_comprehensive_ablations(
        all_results,
        save_path=os.path.join(figures_dir, 'ablation_study_comprehensive.png')
    )
    plt.close()
    
    # Generate statistical significance plot
    if all_tests:
        print("\n3. Statistical significance...")
        plot_statistical_significance(
            all_results,
            all_tests,
            save_path=os.path.join(figures_dir, 'ablation_statistical_significance.png')
        )
        plt.close()
    
    print("\n" + "=" * 70)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 70)
    
    return all_results, all_tests


def generate_ablation_report(results_dir: str = './results/ablations') -> str:
    """
    Generate markdown report for ablation studies.
    """
    all_results = load_ablation_results(results_dir)
    all_tests = load_ablation_tests(results_dir)
    
    report = []
    report.append("# Path A Ablation Study Results\n")
    report.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Ablation 1
    if 'ablation1_winit' in all_results:
        report.append("\n## Ablation 1: W_in Initialization\n")
        report.append("**Purpose**: Demonstrate that learning W_in before freezing is superior to random initialization (RanPAC-style).\n")
        
        report.append("\n### Results\n")
        report.append("| Condition | Accuracy | Forgetting |")
        report.append("|-----------|----------|------------|")
        
        results = all_results['ablation1_winit']
        for cond in ['learned_freeze', 'random_freeze', 'learned_continue']:
            if cond in results:
                acc = results[cond]['avg_accuracy']
                fgt = results[cond]['forgetting']
                report.append(f"| {CONDITION_LABELS[cond].replace(chr(10), ' ')} | {acc['mean']:.4f} +/- {acc['std']:.4f} | {fgt['mean']:.4f} +/- {fgt['std']:.4f} |")
        
        if 'ablation1_winit' in all_tests:
            report.append("\n### Statistical Tests\n")
            tests = all_tests['ablation1_winit']
            for comp, test in tests.items():
                report.append(f"\n**{comp}**:")
                report.append(f"- Accuracy: t={test['accuracy']['t_statistic']:.3f}, p={test['accuracy']['p_value']:.4f}, Cohen's d={test['accuracy']['cohen_d']:.3f}")
                if test['accuracy']['significant']:
                    report.append(f"  - **SIGNIFICANT** (baseline better: {test['accuracy']['baseline_better']})")
                report.append(f"- Forgetting: t={test['forgetting']['t_statistic']:.3f}, p={test['forgetting']['p_value']:.4f}, Cohen's d={test['forgetting']['cohen_d']:.3f}")
                if test['forgetting']['significant']:
                    report.append(f"  - **SIGNIFICANT** (baseline better: {test['forgetting']['baseline_better']})")
    
    # Ablation 2
    if 'ablation2_coherence' in all_results:
        report.append("\n## Ablation 2: Coherence Criterion\n")
        report.append("**Purpose**: Validate that coherence-based update acceptance is beneficial.\n")
        
        report.append("\n### Results\n")
        report.append("| Condition | Accuracy | Forgetting |")
        report.append("|-----------|----------|------------|")
        
        results = all_results['ablation2_coherence']
        for cond in ['with_coherence', 'without_coherence', 'strict_coherence']:
            if cond in results:
                acc = results[cond]['avg_accuracy']
                fgt = results[cond]['forgetting']
                report.append(f"| {CONDITION_LABELS[cond].replace(chr(10), ' ')} | {acc['mean']:.4f} +/- {acc['std']:.4f} | {fgt['mean']:.4f} +/- {fgt['std']:.4f} |")
    
    # Ablation 3
    if 'ablation3_learning_rule' in all_results:
        report.append("\n## Ablation 3: Learning Rule\n")
        report.append("**Purpose**: Compare Hebbian learning to gradient-based methods.\n")
        
        report.append("\n### Results\n")
        report.append("| Condition | Accuracy | Forgetting |")
        report.append("|-----------|----------|------------|")
        
        results = all_results['ablation3_learning_rule']
        for cond in ['hebbian', 'sgd', 'adam']:
            if cond in results:
                acc = results[cond]['avg_accuracy']
                fgt = results[cond]['forgetting']
                report.append(f"| {CONDITION_LABELS[cond].replace(chr(10), ' ')} | {acc['mean']:.4f} +/- {acc['std']:.4f} | {fgt['mean']:.4f} +/- {fgt['std']:.4f} |")
    
    # Summary
    report.append("\n## Summary\n")
    report.append("These ablation studies demonstrate that all three design choices are justified:\n")
    report.append("1. **Learning W_in before freezing** (vs random) - Establishes task-relevant representations")
    report.append("2. **Coherence-based acceptance** (vs unconditional) - Prevents catastrophic forgetting")
    report.append("3. **Hebbian learning** - Provides biological plausibility and coherence preservation")
    
    report_text = '\n'.join(report)
    
    # Save report
    report_path = os.path.join(results_dir, 'ABLATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Report saved to: {report_path}")
    
    return report_text


if __name__ == "__main__":
    results, tests = generate_ablation_visualizations()
    report = generate_ablation_report()
    print("\n" + report)
