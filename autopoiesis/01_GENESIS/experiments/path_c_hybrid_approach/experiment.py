"""
Ablation Study for Hybrid Autopoietic-ML Approach

Tests different combinations of autopoietic mechanisms:
- Baseline (standard ML)
- +Coherence regularization
- +Structural plasticity
- +Self-organizing layers
- Combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import time
import os
from datetime import datetime
from tqdm import tqdm
import random

from hybrid_model import create_hybrid_model, HybridMLP, HybridCNN


# Ablation conditions
ABLATION_CONDITIONS = {
    'Baseline': {},
    '+Coherence': {'coherence_reg': True},
    '+Plasticity': {'structural_plasticity': True},
    '+SelfOrg': {'self_organizing': True},
    '+Coh+Plas': {'coherence_reg': True, 'structural_plasticity': True},
    '+Coh+Self': {'coherence_reg': True, 'self_organizing': True},
    '+All': {'coherence_reg': True, 'structural_plasticity': True, 'self_organizing': True}
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 2,
    subset_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test data loaders.
    
    Args:
        dataset: 'mnist' or 'cifar10'
        batch_size: Batch size
        num_workers: Number of data loading workers
        subset_size: If specified, use only this many training samples
    """
    if dataset.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
    elif dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Use subset if specified
    if subset_size is not None and subset_size < len(train_dataset):
        indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_coh_loss = 0
    correct = 0
    total = 0
    coherence_metrics = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(output, target)
        
        # Coherence loss
        coh_loss = model.compute_coherence_loss()
        
        # Total loss
        loss = ce_loss + coh_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Hebbian update (if applicable)
        # Use accuracy as success signal
        pred = output.argmax(dim=1)
        batch_acc = (pred == target).float().mean().item()
        model.hebbian_update(success_signal=batch_acc)
        
        # Homeostatic update
        model.homeostatic_update()
        
        # Apply structural plasticity
        if hasattr(model, 'apply_plasticity'):
            model.apply_plasticity()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_coh_loss += coh_loss.item() if isinstance(coh_loss, torch.Tensor) else coh_loss
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Get coherence metrics
        coh_metrics = model.get_coherence_metrics()
        if coh_metrics:
            coherence_metrics.append(coh_metrics)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # Aggregate coherence metrics
    avg_coherence = {}
    if coherence_metrics:
        for key in coherence_metrics[0]:
            avg_coherence[key] = np.mean([m[key] for m in coherence_metrics])
    
    return {
        'loss': total_loss / len(train_loader),
        'ce_loss': total_ce_loss / len(train_loader),
        'coh_loss': total_coh_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        'coherence': avg_coherence
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100. * correct / total
    }


def run_single_trial(
    dataset: str,
    condition_name: str,
    condition_config: Dict,
    n_epochs: int,
    coherence_weight: float,
    device: str,
    seed: int
) -> Dict:
    """
    Run a single training trial.
    
    Returns:
        Dictionary with trial results
    """
    set_seed(seed)
    
    # Get data
    train_loader, test_loader = get_data_loaders(dataset, batch_size=128)
    
    # Create model
    model = create_hybrid_model(
        dataset=dataset,
        condition=condition_config,
        coherence_weight=coherence_weight,
        device=device
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'coherence': [],
        'epoch_times': []
    }
    
    best_acc = 0
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        
        # Reset coherence history at epoch start
        model.reset_epoch()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['coherence'].append(train_metrics['coherence'])
        history['epoch_times'].append(epoch_time)
        
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
    
    return {
        'condition': condition_name,
        'seed': seed,
        'best_accuracy': best_acc,
        'final_accuracy': test_metrics['accuracy'],
        'final_loss': test_metrics['loss'],
        'history': history,
        'total_time': sum(history['epoch_times'])
    }


def run_ablation_study(
    dataset: str = 'mnist',
    n_trials: int = 5,
    n_epochs: int = 20,
    coherence_weights: List[float] = [0.01],
    device: str = 'cpu',
    save_dir: str = './results'
) -> Dict:
    """
    Run full ablation study.
    
    Args:
        dataset: Dataset to use
        n_trials: Number of trials per condition
        n_epochs: Number of epochs per trial
        coherence_weights: List of coherence weights to try
        device: Device for training
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'dataset': dataset,
        'n_trials': n_trials,
        'n_epochs': n_epochs,
        'conditions': {},
        'summary': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Run each condition
    for condition_name, condition_config in ABLATION_CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name}")
        print(f"Config: {condition_config}")
        print(f"{'='*60}")
        
        condition_results = []
        
        # Determine coherence weight
        coh_weight = coherence_weights[0] if condition_config.get('coherence_reg', False) else 0
        
        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}")
            seed = 42 + trial
            
            trial_result = run_single_trial(
                dataset=dataset,
                condition_name=condition_name,
                condition_config=condition_config,
                n_epochs=n_epochs,
                coherence_weight=coh_weight,
                device=device,
                seed=seed
            )
            
            condition_results.append(trial_result)
            print(f"    Best accuracy: {trial_result['best_accuracy']:.2f}%")
            print(f"    Training time: {trial_result['total_time']:.1f}s")
        
        # Aggregate results
        accuracies = [r['best_accuracy'] for r in condition_results]
        times = [r['total_time'] for r in condition_results]
        
        results['conditions'][condition_name] = {
            'config': condition_config,
            'trials': condition_results,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
        
        print(f"\n  Summary for {condition_name}:")
        print(f"    Accuracy: {np.mean(accuracies):.2f}% +/- {np.std(accuracies):.2f}%")
        print(f"    Time: {np.mean(times):.1f}s +/- {np.std(times):.1f}s")
    
    # Create summary
    baseline_acc = results['conditions']['Baseline']['mean_accuracy']
    
    for condition_name in results['conditions']:
        acc = results['conditions'][condition_name]['mean_accuracy']
        results['summary'][condition_name] = {
            'accuracy': acc,
            'improvement': acc - baseline_acc
        }
    
    # Save results
    results_file = os.path.join(save_dir, f'ablation_{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def hyperparameter_search(
    dataset: str = 'mnist',
    n_trials: int = 3,
    n_epochs: int = 10,
    device: str = 'cpu'
) -> Dict:
    """
    Grid search for optimal coherence weight.
    
    Returns:
        Dictionary with best hyperparameters
    """
    coherence_weights = [0.001, 0.01, 0.1]
    
    results = {}
    
    for coh_weight in coherence_weights:
        print(f"\nTesting coherence_weight = {coh_weight}")
        
        accuracies = []
        for trial in range(n_trials):
            seed = 42 + trial
            set_seed(seed)
            
            train_loader, test_loader = get_data_loaders(dataset)
            model = create_hybrid_model(
                dataset=dataset,
                condition={'coherence_reg': True},
                coherence_weight=coh_weight,
                device=device
            )
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(1, n_epochs + 1):
                model.reset_epoch()
                train_epoch(model, train_loader, optimizer, device, epoch)
            
            test_metrics = evaluate(model, test_loader, device)
            accuracies.append(test_metrics['accuracy'])
        
        mean_acc = np.mean(accuracies)
        results[coh_weight] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': np.std(accuracies),
            'accuracies': accuracies
        }
        print(f"  Accuracy: {mean_acc:.2f}% +/- {np.std(accuracies):.2f}%")
    
    # Find best
    best_weight = max(results.keys(), key=lambda w: results[w]['mean_accuracy'])
    
    return {
        'results': results,
        'best_weight': best_weight,
        'best_accuracy': results[best_weight]['mean_accuracy']
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer trials/epochs')
    parser.add_argument('--hyperparam_search', action='store_true', help='Run hyperparameter search first')
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Quick mode for testing
    if args.quick:
        args.n_trials = 2
        args.n_epochs = 5
    
    # Hyperparameter search
    if args.hyperparam_search:
        print("\n" + "="*60)
        print("Running hyperparameter search...")
        print("="*60)
        hp_results = hyperparameter_search(
            dataset=args.dataset,
            n_trials=2,
            n_epochs=5,
            device=device
        )
        print(f"\nBest coherence weight: {hp_results['best_weight']}")
        coherence_weights = [hp_results['best_weight']]
    else:
        coherence_weights = [0.01]
    
    # Run ablation study
    print("\n" + "="*60)
    print("Running ablation study...")
    print("="*60)
    
    results = run_ablation_study(
        dataset=args.dataset,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        coherence_weights=coherence_weights,
        device=device,
        save_dir=args.save_dir
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.n_epochs}, Trials: {args.n_trials}")
    print("-"*60)
    
    for condition_name in ABLATION_CONDITIONS:
        summary = results['summary'][condition_name]
        acc = summary['accuracy']
        imp = summary['improvement']
        sign = '+' if imp >= 0 else ''
        print(f"{condition_name:15s}: {acc:6.2f}% ({sign}{imp:.2f}%)")
