"""
Hybrid Autopoietic-ML Model

Integrates:
- Standard neural network architecture (CNN for images)
- Coherence regularization
- Structural plasticity
- Self-organizing layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict

from coherence_regularizer import CoherenceRegularizer, AdaptiveCoherenceRegularizer
from structural_plasticity import StructuralPlasticityModule, AdaptivePlasticityScheduler
from self_organizing_layer import SelfOrganizingLinear, SelfOrganizingBlock


class HybridCNN(nn.Module):
    """
    CNN with optional autopoietic mechanisms for image classification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        use_coherence: bool = False,
        use_plasticity: bool = False,
        use_self_organizing: bool = False,
        coherence_weight: float = 0.01,
        hebbian_lr: float = 0.001,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.use_coherence = use_coherence
        self.use_plasticity = use_plasticity
        self.use_self_organizing = use_self_organizing
        self.device = device
        
        # Convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate flattened size (depends on input size)
        self._flat_size = None
        
        # Fully connected layers (will be initialized in first forward)
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        
        # Autopoietic components
        if use_coherence:
            self.coherence_reg = CoherenceRegularizer(device=device)
            self.coherence_weight = coherence_weight
        
        if use_plasticity:
            self.plasticity = StructuralPlasticityModule(device=device)
            self.plasticity_scheduler = AdaptivePlasticityScheduler(
                self.plasticity,
                warmup_steps=500,
                apply_every=200
            )
        
        self.hebbian_lr = hebbian_lr
        self.num_classes = num_classes
        
        # Activation cache for coherence
        self.activations: Dict[str, torch.Tensor] = {}
    
    def _init_fc_layers(self, flat_size: int):
        """Initialize fully connected layers based on input size."""
        if self.use_self_organizing:
            self.fc1 = SelfOrganizingLinear(flat_size, 256, hebbian_lr=self.hebbian_lr).to(self.device)
            self.fc2 = SelfOrganizingLinear(256, 128, hebbian_lr=self.hebbian_lr).to(self.device)
        else:
            self.fc1 = nn.Linear(flat_size, 256).to(self.device)
            self.fc2 = nn.Linear(256, 128).to(self.device)
        
        self.fc3 = nn.Linear(128, self.num_classes).to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation caching.
        """
        self.activations = {}
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        self.activations['conv1'] = x
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        self.activations['conv2'] = x
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        self.activations['conv3'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        flat_size = x.size(1)
        
        # Initialize FC layers if needed
        if self.fc1 is None:
            self._flat_size = flat_size
            self._init_fc_layers(flat_size)
        
        # FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        self.activations['fc1'] = x
        
        # Update plasticity stats
        if self.use_plasticity:
            self.plasticity.update_activity_stats('fc1', x)
        
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        self.activations['fc2'] = x
        
        if self.use_plasticity:
            self.plasticity.update_activity_stats('fc2', x)
        
        x = self.fc3(x)
        self.activations['output'] = x
        
        return x
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get cached activations for coherence computation."""
        return self.activations
    
    def compute_coherence_loss(self) -> torch.Tensor:
        """Compute coherence penalty to add to loss."""
        if not self.use_coherence:
            return torch.tensor(0.0, device=self.device)
        
        penalty = self.coherence_reg.penalty(self.activations)
        self.coherence_reg.update_history(self.activations)
        return self.coherence_weight * penalty
    
    def hebbian_update(self, success_signal: float = 1.0):
        """Apply Hebbian update to self-organizing layers."""
        if not self.use_self_organizing:
            return
        
        if isinstance(self.fc1, SelfOrganizingLinear):
            self.fc1.hebbian_update(success_signal)
        if isinstance(self.fc2, SelfOrganizingLinear):
            self.fc2.hebbian_update(success_signal)
    
    def homeostatic_update(self):
        """Apply homeostatic update to self-organizing layers."""
        if not self.use_self_organizing:
            return
        
        if isinstance(self.fc1, SelfOrganizingLinear):
            self.fc1.homeostatic_update()
        if isinstance(self.fc2, SelfOrganizingLinear):
            self.fc2.homeostatic_update()
    
    def apply_plasticity(self) -> Dict[str, int]:
        """Apply structural plasticity if scheduled."""
        if not self.use_plasticity:
            return {}
        
        changes = {}
        
        if self.plasticity_scheduler.step():
            # Check for pruning/growing in FC layers
            if self.fc1 is not None and isinstance(self.fc1, nn.Linear):
                new_fc1, new_fc2, n_pruned = self.plasticity.prune_linear_layer(
                    self.fc1, 'fc1', self.fc2
                )
                if n_pruned > 0:
                    self.fc1 = new_fc1
                    self.fc2 = new_fc2
                    changes['fc1_pruned'] = n_pruned
                    self.plasticity_scheduler.mark_change()
        
        return changes
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        """Get current coherence metrics."""
        if not self.use_coherence:
            return {}
        return self.coherence_reg.get_metrics(self.activations)
    
    def reset_epoch(self):
        """Reset state at epoch start."""
        if self.use_coherence:
            self.coherence_reg.reset_history()


class HybridMLP(nn.Module):
    """
    Simple MLP with optional autopoietic mechanisms.
    Useful for simpler datasets like MNIST.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 256, 128],
        num_classes: int = 10,
        use_coherence: bool = False,
        use_plasticity: bool = False,
        use_self_organizing: bool = False,
        coherence_weight: float = 0.01,
        hebbian_lr: float = 0.001,
        dropout: float = 0.2,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.use_coherence = use_coherence
        self.use_plasticity = use_plasticity
        self.use_self_organizing = use_self_organizing
        self.device = device
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if use_self_organizing:
                layers.append((f'fc{i}', SelfOrganizingLinear(
                    prev_dim, hidden_dim, hebbian_lr=hebbian_lr
                )))
            else:
                layers.append((f'fc{i}', nn.Linear(prev_dim, hidden_dim)))
            
            layers.append((f'bn{i}', nn.BatchNorm1d(hidden_dim)))
            layers.append((f'relu{i}', nn.ReLU()))
            layers.append((f'drop{i}', nn.Dropout(dropout)))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Autopoietic components
        if use_coherence:
            self.coherence_reg = CoherenceRegularizer(device=device)
            self.coherence_weight = coherence_weight
        
        if use_plasticity:
            self.plasticity = StructuralPlasticityModule(device=device)
        
        self.activations: Dict[str, torch.Tensor] = {}
        self.hidden_dims = hidden_dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = {}
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Process through layers
        for name, module in self.features.named_children():
            x = module(x)
            
            # Cache activations after ReLU
            if 'relu' in name:
                layer_idx = name.replace('relu', '')
                self.activations[f'fc{layer_idx}'] = x
                
                # Update plasticity stats
                if self.use_plasticity:
                    self.plasticity.update_activity_stats(f'fc{layer_idx}', x)
        
        x = self.classifier(x)
        self.activations['output'] = x
        
        return x
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations
    
    def compute_coherence_loss(self) -> torch.Tensor:
        if not self.use_coherence:
            return torch.tensor(0.0, device=self.device)
        
        penalty = self.coherence_reg.penalty(self.activations)
        self.coherence_reg.update_history(self.activations)
        return self.coherence_weight * penalty
    
    def hebbian_update(self, success_signal: float = 1.0):
        if not self.use_self_organizing:
            return
        
        for name, module in self.features.named_children():
            if isinstance(module, SelfOrganizingLinear):
                module.hebbian_update(success_signal)
    
    def homeostatic_update(self):
        if not self.use_self_organizing:
            return
        
        for name, module in self.features.named_children():
            if isinstance(module, SelfOrganizingLinear):
                module.homeostatic_update()
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        if not self.use_coherence:
            return {}
        return self.coherence_reg.get_metrics(self.activations)
    
    def reset_epoch(self):
        if self.use_coherence:
            self.coherence_reg.reset_history()


def create_hybrid_model(
    dataset: str = 'mnist',
    condition: Dict[str, bool] = None,
    coherence_weight: float = 0.01,
    device: str = 'cpu'
) -> nn.Module:
    """
    Factory function to create hybrid model based on dataset and condition.
    
    Args:
        dataset: 'mnist' or 'cifar10'
        condition: Dict with keys 'coherence_reg', 'structural_plasticity', 'self_organizing'
        coherence_weight: Weight for coherence loss
        device: Device for model
    
    Returns:
        Configured hybrid model
    """
    if condition is None:
        condition = {}
    
    use_coherence = condition.get('coherence_reg', False)
    use_plasticity = condition.get('structural_plasticity', False)
    use_self_organizing = condition.get('self_organizing', False)
    
    if dataset.lower() == 'mnist':
        model = HybridMLP(
            input_dim=784,
            hidden_dims=[512, 256, 128],
            num_classes=10,
            use_coherence=use_coherence,
            use_plasticity=use_plasticity,
            use_self_organizing=use_self_organizing,
            coherence_weight=coherence_weight,
            device=device
        )
    elif dataset.lower() == 'cifar10':
        model = HybridCNN(
            input_channels=3,
            num_classes=10,
            use_coherence=use_coherence,
            use_plasticity=use_plasticity,
            use_self_organizing=use_self_organizing,
            coherence_weight=coherence_weight,
            device=device
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return model.to(device)


if __name__ == '__main__':
    print("Testing Hybrid Models...")
    
    device = 'cpu'
    
    # Test HybridMLP with all features
    print("\n1. Testing HybridMLP with all features...")
    model = HybridMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        num_classes=10,
        use_coherence=True,
        use_plasticity=True,
        use_self_organizing=True,
        device=device
    )
    
    x = torch.randn(32, 784)
    y = model(x)
    print(f"   Input: {x.shape}, Output: {y.shape}")
    
    # Test coherence
    coh_loss = model.compute_coherence_loss()
    print(f"   Coherence loss: {coh_loss.item():.4f}")
    
    metrics = model.get_coherence_metrics()
    print(f"   Coherence: {metrics.get('coherence', 'N/A'):.4f}")
    
    # Test Hebbian update
    model.hebbian_update(success_signal=0.9)
    model.homeostatic_update()
    print("   Hebbian and homeostatic updates applied")
    
    # Test HybridCNN
    print("\n2. Testing HybridCNN with all features...")
    cnn_model = HybridCNN(
        input_channels=3,
        num_classes=10,
        use_coherence=True,
        use_plasticity=True,
        use_self_organizing=True,
        device=device
    )
    
    x = torch.randn(16, 3, 32, 32)  # CIFAR-10 size
    y = cnn_model(x)
    print(f"   Input: {x.shape}, Output: {y.shape}")
    
    coh_loss = cnn_model.compute_coherence_loss()
    print(f"   Coherence loss: {coh_loss.item():.4f}")
    
    # Test factory function
    print("\n3. Testing create_hybrid_model factory...")
    conditions = [
        {'name': 'Baseline', 'config': {}},
        {'name': '+Coherence', 'config': {'coherence_reg': True}},
        {'name': '+All', 'config': {'coherence_reg': True, 'structural_plasticity': True, 'self_organizing': True}}
    ]
    
    for cond in conditions:
        m = create_hybrid_model('mnist', cond['config'], device=device)
        x = torch.randn(8, 784)
        y = m(x)
        print(f"   {cond['name']}: output shape = {y.shape}")
    
    # Test gradient flow
    print("\n4. Testing gradient flow with coherence loss...")
    model = create_hybrid_model('mnist', {'coherence_reg': True}, device=device)
    x = torch.randn(16, 784)
    y = model(x)
    
    ce_loss = F.cross_entropy(y, torch.randint(0, 10, (16,)))
    coh_loss = model.compute_coherence_loss()
    total_loss = ce_loss + coh_loss
    
    total_loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"   Parameters with gradients: {grad_count}")
    
    print("\n[OK] Hybrid Models test passed!")
