"""
Quick Initial Experiment for Path C: Hybrid Approach

Fast validation of Coherence Regularization on MNIST.
- N=3 trials
- 10 epochs
- Baseline vs +Coherence comparison
- Total runtime: ~15-20 minutes on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

from baseline_model import BaselineMLP
from hybrid_model import create_hybrid_model


# Configuration
CONFIG = {
    'n_trials': 3,
    'n_epochs': 10,  # Standard for initial validation
    'batch_size': 128,
    'learning_rate': 0.001,
    'coherence_weight': 0.01,
    'device': 'cpu',
    'seed_base': 42
}

CONDITIONS = {
    'Baseline': {},
    '+Coherence': {'coherence_reg': True}
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_mnist_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_coh_loss = 0
    correct = 0
    total = 0
    coherence_values = []
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Losses
        ce_loss = F.cross_entropy(output, target)
        coh_loss = model.compute_coherence_loss()
        loss = ce_loss + coh_loss
        
        loss.backward()
        optimizer.step()
        
        # Optional Hebbian update
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean().item()
        model.hebbian_update(success_signal=acc)
        model.homeostatic_update()
        
        # Metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        coh_val = coh_loss.item() if isinstance(coh_loss, torch.Tensor) else coh_loss
        total_coh_loss += coh_val
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Coherence metrics
        coh_metrics = model.get_coherence_metrics()
        if coh_metrics:
            coherence_values.append(coh_metrics.get('coherence', 0))
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'ce_loss': total_ce_loss / n_batches,
        'coh_loss': total_coh_loss / n_batches,
        'accuracy': 100.0 * correct / total,
        'coherence': np.mean(coherence_values) if coherence_values else 0.0
    }


def evaluate(model: nn.Module, test_loader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100.0 * correct / total
    }


def run_trial(
    condition_name: str,
    condition_config: Dict,
    seed: int,
    device: str
) -> Dict:
    """Run single training trial."""
    set_seed(seed)
    
    train_loader, test_loader = get_mnist_loaders(CONFIG['batch_size'])
    
    # Create model
    if condition_config:
        model = create_hybrid_model(
            dataset='mnist',
            condition=condition_config,
            coherence_weight=CONFIG['coherence_weight'],
            device=device
        )
    else:
        model = BaselineMLP().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'coherence': [],
        'loss_variance': []
    }
    
    batch_losses = []  # For variance calculation
    
    for epoch in range(1, CONFIG['n_epochs'] + 1):
        model.reset_epoch()
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['coherence'].append(train_metrics['coherence'])
        
        batch_losses.append(train_metrics['loss'])
        # Rolling variance over last 3 epochs
        if len(batch_losses) >= 3:
            history['loss_variance'].append(np.var(batch_losses[-3:]))
        else:
            history['loss_variance'].append(0)
        
        print(f"    Epoch {epoch:2d}: Train Acc={train_metrics['accuracy']:.2f}%, "
              f"Test Acc={test_metrics['accuracy']:.2f}%, "
              f"Coherence={train_metrics['coherence']:.4f}")
    
    return {
        'condition': condition_name,
        'seed': seed,
        'best_test_acc': max(history['test_acc']),
        'final_test_acc': history['test_acc'][-1],
        'avg_loss_variance': np.mean(history['loss_variance']),
        'history': history
    }


def run_experiment() -> Dict:
    """Run full quick experiment."""
    print("="*60)
    print("Path C: Quick Coherence Regularization Experiment")
    print("="*60)
    print(f"\nConfig: {CONFIG}")
    
    # Detect device
    device = CONFIG['device']
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
    results = {
        'config': CONFIG,
        'conditions': {},
        'timestamp': datetime.now().isoformat()
    }
    
    start_time = time.time()
    
    for condition_name, condition_config in CONDITIONS.items():
        print(f"\n{'='*40}")
        print(f"Condition: {condition_name}")
        print(f"{'='*40}")
        
        trial_results = []
        
        for trial in range(CONFIG['n_trials']):
            seed = CONFIG['seed_base'] + trial
            print(f"\n  Trial {trial+1}/{CONFIG['n_trials']} (seed={seed})")
            
            trial_result = run_trial(condition_name, condition_config, seed, device)
            trial_results.append(trial_result)
            
            print(f"    -> Best: {trial_result['best_test_acc']:.2f}%")
        
        # Aggregate
        accs = [t['best_test_acc'] for t in trial_results]
        variances = [t['avg_loss_variance'] for t in trial_results]
        
        results['conditions'][condition_name] = {
            'config': condition_config,
            'trials': trial_results,
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_loss_variance': np.mean(variances),
            'std_loss_variance': np.std(variances)
        }
        
        print(f"\n  Summary: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
        print(f"  Loss Variance: {np.mean(variances):.6f} +/- {np.std(variances):.6f}")
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    print(f"\n\nTotal time: {total_time/60:.1f} minutes")
    
    return results


def plot_results(results: Dict, save_path: str):
    """Create visualization of experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = list(results['conditions'].keys())
    colors = ['#3498db', '#e74c3c']
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    means = [results['conditions'][c]['mean_accuracy'] for c in conditions]
    stds = [results['conditions'][c]['std_accuracy'] for c in conditions]
    
    bars = ax1.bar(conditions, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy Comparison (N=3 trials)')
    ax1.set_ylim(96, 99)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f'{mean:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training stability (loss variance)
    ax2 = axes[0, 1]
    var_means = [results['conditions'][c]['mean_loss_variance'] for c in conditions]
    var_stds = [results['conditions'][c]['std_loss_variance'] for c in conditions]
    
    bars = ax2.bar(conditions, var_means, yerr=var_stds, color=colors, capsize=5, alpha=0.8)
    ax2.set_ylabel('Loss Variance')
    ax2.set_title('Training Stability (lower is better)')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. Learning curves
    ax3 = axes[1, 0]
    for i, (cond_name, cond_data) in enumerate(results['conditions'].items()):
        # Average over trials
        n_epochs = len(cond_data['trials'][0]['history']['test_acc'])
        avg_curve = np.zeros(n_epochs)
        std_curve = np.zeros(n_epochs)
        
        for epoch in range(n_epochs):
            vals = [t['history']['test_acc'][epoch] for t in cond_data['trials']]
            avg_curve[epoch] = np.mean(vals)
            std_curve[epoch] = np.std(vals)
        
        epochs = np.arange(1, n_epochs + 1)
        ax3.plot(epochs, avg_curve, color=colors[i], label=cond_name, linewidth=2)
        ax3.fill_between(epochs, avg_curve - std_curve, avg_curve + std_curve,
                        color=colors[i], alpha=0.2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Learning Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Coherence evolution
    ax4 = axes[1, 1]
    coh_data = results['conditions'].get('+Coherence', {})
    if coh_data and 'trials' in coh_data:
        n_epochs = len(coh_data['trials'][0]['history']['coherence'])
        avg_coh = np.zeros(n_epochs)
        std_coh = np.zeros(n_epochs)
        
        for epoch in range(n_epochs):
            vals = [t['history']['coherence'][epoch] for t in coh_data['trials']]
            avg_coh[epoch] = np.mean(vals)
            std_coh[epoch] = np.std(vals)
        
        epochs = np.arange(1, n_epochs + 1)
        ax4.plot(epochs, avg_coh, color='#e74c3c', linewidth=2)
        ax4.fill_between(epochs, avg_coh - std_coh, avg_coh + std_coh,
                        color='#e74c3c', alpha=0.2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Coherence Score')
        ax4.set_title('Coherence Evolution (+Coherence condition)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No coherence data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.close()


def print_summary(results: Dict):
    """Print formatted summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    baseline = results['conditions']['Baseline']
    coherence = results['conditions']['+Coherence']
    
    improvement = coherence['mean_accuracy'] - baseline['mean_accuracy']
    stability_improvement = baseline['mean_loss_variance'] - coherence['mean_loss_variance']
    
    print(f"\n1. Test Accuracy:")
    print(f"   Baseline:   {baseline['mean_accuracy']:.2f}% +/- {baseline['std_accuracy']:.2f}%")
    print(f"   +Coherence: {coherence['mean_accuracy']:.2f}% +/- {coherence['std_accuracy']:.2f}%")
    print(f"   Improvement: {improvement:+.2f}%")
    
    print(f"\n2. Training Stability (Loss Variance):")
    print(f"   Baseline:   {baseline['mean_loss_variance']:.6f}")
    print(f"   +Coherence: {coherence['mean_loss_variance']:.6f}")
    print(f"   Improvement: {stability_improvement:.6f} ({'better' if stability_improvement > 0 else 'worse'})")
    
    print(f"\n3. Total Time: {results['total_time']/60:.1f} minutes")
    
    # Statistical significance (simple t-test approximation)
    baseline_accs = [t['best_test_acc'] for t in baseline['trials']]
    coherence_accs = [t['best_test_acc'] for t in coherence['trials']]
    
    pooled_std = np.sqrt((np.var(baseline_accs) + np.var(coherence_accs)) / 2)
    if pooled_std > 0:
        effect_size = improvement / pooled_std
        print(f"\n4. Effect Size (Cohen's d): {effect_size:.2f}")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    # Create output directory
    output_dir = '/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_c'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment
    results = run_experiment()
    
    # Save results
    results_file = os.path.join(output_dir, 'initial_results.json')
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved results to: {results_file}")
    
    # Create visualization
    plot_path = os.path.join(output_dir, 'initial_results.png')
    plot_results(results, plot_path)
    
    # Print summary
    print_summary(results)
    
    return results


if __name__ == '__main__':
    main()
