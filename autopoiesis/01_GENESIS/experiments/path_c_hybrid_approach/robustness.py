"""
Robustness Evaluation for Hybrid Autopoietic-ML Models

Tests model robustness against:
- Gaussian noise at various levels
- FGSM adversarial attacks
- Out-of-distribution inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm

from hybrid_model import create_hybrid_model
from experiment import set_seed, get_data_loaders, evaluate, ABLATION_CONDITIONS


class NoiseTransform:
    """Add Gaussian noise to input."""
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_level == 0:
            return x
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


def evaluate_with_noise(
    model: nn.Module,
    test_loader: DataLoader,
    noise_level: float,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model with Gaussian noise added to inputs.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        noise_level: Standard deviation of Gaussian noise
        device: Device for computation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    noise_transform = NoiseTransform(noise_level)
    
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Add noise
            noisy_data = noise_transform(data)
            
            # Forward pass
            output = model(noisy_data)
            
            # Metrics
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'accuracy': 100. * correct / total,
        'loss': total_loss / len(test_loader)
    }


def fgsm_attack(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: Model to attack
        data: Input data
        target: True labels
        epsilon: Attack strength
    
    Returns:
        Adversarial examples
    """
    # Enable gradients on input
    data.requires_grad = True
    
    # Forward pass
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples
    sign_grad = data.grad.sign()
    perturbed_data = data + epsilon * sign_grad
    
    # Clamp to valid range (assuming normalized data)
    # For MNIST: normalized with mean=0.1307, std=0.3081
    # For CIFAR: normalized with mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    perturbed_data = torch.clamp(perturbed_data, -3, 3)
    
    return perturbed_data


def evaluate_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    epsilon: float,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model against FGSM adversarial attack.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        epsilon: Attack strength
        device: Device for computation
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    for data, target in tqdm(test_loader, desc=f'FGSM eps={epsilon}', leave=False):
        data, target = data.to(device), target.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            output_clean = model(data)
            pred_clean = output_clean.argmax(dim=1)
            correct_clean += pred_clean.eq(target).sum().item()
        
        # Adversarial accuracy
        adv_data = fgsm_attack(model, data.clone(), target, epsilon)
        
        with torch.no_grad():
            output_adv = model(adv_data)
            pred_adv = output_adv.argmax(dim=1)
            correct_adv += pred_adv.eq(target).sum().item()
        
        total += target.size(0)
    
    return {
        'clean_accuracy': 100. * correct_clean / total,
        'adversarial_accuracy': 100. * correct_adv / total,
        'robustness_drop': (correct_clean - correct_adv) / correct_clean * 100 if correct_clean > 0 else 0
    }


def evaluate_ood(
    model: nn.Module,
    ood_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on out-of-distribution data.
    
    Args:
        model: Model trained on in-distribution data
        ood_loader: OOD data loader
        device: Device for computation
    
    Returns:
        Dictionary with OOD metrics
    """
    model.eval()
    
    all_confidences = []
    all_entropies = []
    
    with torch.no_grad():
        for data, _ in tqdm(ood_loader, desc='OOD Eval', leave=False):
            data = data.to(device)
            
            output = model(data)
            probs = F.softmax(output, dim=1)
            
            # Confidence (max probability)
            confidence = probs.max(dim=1)[0]
            all_confidences.extend(confidence.cpu().numpy())
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            all_entropies.extend(entropy.cpu().numpy())
    
    return {
        'mean_confidence': np.mean(all_confidences),
        'std_confidence': np.std(all_confidences),
        'mean_entropy': np.mean(all_entropies),
        'std_entropy': np.std(all_entropies)
    }


def get_ood_loader(
    in_distribution: str,
    batch_size: int = 128
) -> DataLoader:
    """
    Get OOD data loader based on in-distribution dataset.
    
    Args:
        in_distribution: Name of in-distribution dataset ('mnist' or 'cifar10')
        batch_size: Batch size
    
    Returns:
        DataLoader for OOD dataset
    """
    if in_distribution.lower() == 'mnist':
        # Use Fashion-MNIST as OOD for MNIST-trained models
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as MNIST
        ])
        ood_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif in_distribution.lower() == 'cifar10':
        # Use SVHN as OOD for CIFAR-10 trained models
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        ood_dataset = torchvision.datasets.SVHN(
            root='./data', split='test', download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {in_distribution}")
    
    return DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)


def train_and_evaluate_robustness(
    dataset: str,
    condition_name: str,
    condition_config: Dict,
    n_epochs: int = 20,
    coherence_weight: float = 0.01,
    device: str = 'cpu',
    seed: int = 42
) -> Dict:
    """
    Train a model and evaluate its robustness.
    
    Returns:
        Dictionary with robustness metrics
    """
    from experiment import train_epoch
    import torch.optim as optim
    
    set_seed(seed)
    
    # Get data
    train_loader, test_loader = get_data_loaders(dataset)
    
    # Create and train model
    model = create_hybrid_model(
        dataset=dataset,
        condition=condition_config,
        coherence_weight=coherence_weight,
        device=device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training {condition_name}...")
    for epoch in range(1, n_epochs + 1):
        model.reset_epoch()
        train_epoch(model, train_loader, optimizer, device, epoch)
    
    # Evaluate robustness
    results = {
        'condition': condition_name,
        'config': condition_config
    }
    
    # 1. Clean accuracy
    clean_metrics = evaluate(model, test_loader, device)
    results['clean_accuracy'] = clean_metrics['accuracy']
    print(f"  Clean accuracy: {clean_metrics['accuracy']:.2f}%")
    
    # 2. Noise robustness
    noise_levels = [0, 0.1, 0.2, 0.5]
    results['noise_robustness'] = {}
    print("  Noise robustness:")
    for noise in noise_levels:
        noise_metrics = evaluate_with_noise(model, test_loader, noise, device)
        results['noise_robustness'][noise] = noise_metrics
        print(f"    Noise={noise}: {noise_metrics['accuracy']:.2f}%")
    
    # 3. Adversarial robustness (FGSM)
    epsilons = [0.05, 0.1, 0.2]
    results['adversarial_robustness'] = {}
    print("  Adversarial robustness (FGSM):")
    for eps in epsilons:
        adv_metrics = evaluate_adversarial(model, test_loader, eps, device)
        results['adversarial_robustness'][eps] = adv_metrics
        print(f"    Epsilon={eps}: {adv_metrics['adversarial_accuracy']:.2f}%")
    
    # 4. OOD detection
    try:
        ood_loader = get_ood_loader(dataset)
        ood_metrics = evaluate_ood(model, ood_loader, device)
        results['ood_metrics'] = ood_metrics
        print(f"  OOD confidence: {ood_metrics['mean_confidence']:.4f} +/- {ood_metrics['std_confidence']:.4f}")
    except Exception as e:
        print(f"  OOD evaluation skipped: {e}")
        results['ood_metrics'] = None
    
    return results


def run_robustness_study(
    dataset: str = 'mnist',
    n_epochs: int = 20,
    device: str = 'cpu',
    save_dir: str = './results'
) -> Dict:
    """
    Run full robustness study across all conditions.
    
    Returns:
        Dictionary with all results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'dataset': dataset,
        'n_epochs': n_epochs,
        'conditions': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for condition_name, condition_config in ABLATION_CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name}")
        print(f"{'='*60}")
        
        coh_weight = 0.01 if condition_config.get('coherence_reg', False) else 0
        
        condition_results = train_and_evaluate_robustness(
            dataset=dataset,
            condition_name=condition_name,
            condition_config=condition_config,
            n_epochs=n_epochs,
            coherence_weight=coh_weight,
            device=device
        )
        
        results['conditions'][condition_name] = condition_results
    
    # Save results
    results_file = os.path.join(
        save_dir, 
        f'robustness_{dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def compare_robustness(results: Dict) -> None:
    """
    Print comparison table of robustness metrics.
    """
    print("\n" + "="*80)
    print("ROBUSTNESS COMPARISON")
    print("="*80)
    
    # Headers
    print(f"\n{'Condition':<15} {'Clean':>8} {'N=0.1':>8} {'N=0.2':>8} {'N=0.5':>8} {'FGSM-0.1':>10}")
    print("-"*60)
    
    for condition_name, data in results['conditions'].items():
        clean = data['clean_accuracy']
        n01 = data['noise_robustness']['0.1']['accuracy'] if '0.1' in data['noise_robustness'] else data['noise_robustness'][0.1]['accuracy']
        n02 = data['noise_robustness']['0.2']['accuracy'] if '0.2' in data['noise_robustness'] else data['noise_robustness'][0.2]['accuracy']
        n05 = data['noise_robustness']['0.5']['accuracy'] if '0.5' in data['noise_robustness'] else data['noise_robustness'][0.5]['accuracy']
        fgsm = data['adversarial_robustness']['0.1']['adversarial_accuracy'] if '0.1' in data['adversarial_robustness'] else data['adversarial_robustness'][0.1]['adversarial_accuracy']
        
        print(f"{condition_name:<15} {clean:>7.2f}% {n01:>7.2f}% {n02:>7.2f}% {n05:>7.2f}% {fgsm:>9.2f}%")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run robustness evaluation')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    
    args = parser.parse_args()
    
    # Device detection
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
    
    if args.quick:
        args.n_epochs = 5
    
    results = run_robustness_study(
        dataset=args.dataset,
        n_epochs=args.n_epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    compare_robustness(results)
