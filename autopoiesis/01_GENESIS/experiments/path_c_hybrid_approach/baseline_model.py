"""
Baseline Model for Comparison

Standard PyTorch LeNet-5 style model without any autopoietic mechanisms.
Used as control condition in ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BaselineLeNet(nn.Module):
    """
    LeNet-5 style CNN for MNIST.
    Standard architecture without autopoietic mechanisms.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # For API compatibility with hybrid models
        self.activations: Dict[str, torch.Tensor] = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation caching."""
        self.activations = {}
        
        # Reshape if flattened (784,) -> (1, 28, 28)
        if x.dim() == 2 and x.size(1) == 784:
            x = x.view(-1, 1, 28, 28)
        
        # Conv block 1: (B, 1, 28, 28) -> (B, 6, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        self.activations['conv1'] = x
        
        # Conv block 2: (B, 6, 14, 14) -> (B, 16, 5, 5)
        x = self.pool(F.relu(self.conv2(x)))
        self.activations['conv2'] = x
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        self.activations['fc1'] = x
        
        x = F.relu(self.fc2(x))
        self.activations['fc2'] = x
        
        x = self.fc3(x)
        self.activations['output'] = x
        
        return x
    
    # Compatibility methods (no-ops for baseline)
    def compute_coherence_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def hebbian_update(self, success_signal: float = 1.0):
        pass
    
    def homeostatic_update(self):
        pass
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        return {}
    
    def reset_epoch(self):
        pass


class BaselineMLP(nn.Module):
    """
    Simple MLP for MNIST.
    Matches the hidden dimensions of hybrid model for fair comparison.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        self.activations: Dict[str, torch.Tensor] = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = {}
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.features(x)
        self.activations['features'] = x
        
        x = self.classifier(x)
        self.activations['output'] = x
        
        return x
    
    # Compatibility methods
    def compute_coherence_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def hebbian_update(self, success_signal: float = 1.0):
        pass
    
    def homeostatic_update(self):
        pass
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        return {}
    
    def reset_epoch(self):
        pass


if __name__ == '__main__':
    print("Testing Baseline Models...")
    
    # Test LeNet
    print("\n1. BaselineLeNet:")
    model = BaselineLeNet()
    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test MLP
    print("\n2. BaselineMLP:")
    mlp = BaselineMLP()
    x = torch.randn(32, 784)
    y = mlp(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # Verify compatibility methods
    print("\n3. API compatibility check:")
    print(f"   compute_coherence_loss: {model.compute_coherence_loss()}")
    print(f"   get_coherence_metrics: {model.get_coherence_metrics()}")
    
    print("\n[OK] Baseline models test passed!")
