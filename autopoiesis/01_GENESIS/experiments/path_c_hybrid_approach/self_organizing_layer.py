"""
Self-Organizing Layer for Hybrid Autopoietic-ML Approach

Implements hybrid learning combining:
- Standard gradient-based backpropagation
- Hebbian learning ("neurons that fire together, wire together")
- Homeostatic plasticity (maintain stable activity levels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SelfOrganizingLinear(nn.Module):
    """
    Linear layer with Hebbian learning capability.
    Combines gradient descent with local learning rules.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hebbian_lr: float = 0.001,
        hebbian_decay: float = 0.999,
        homeostatic_target: float = 0.1,
        homeostatic_rate: float = 0.01
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            hebbian_lr: Learning rate for Hebbian updates
            hebbian_decay: Weight decay for Hebbian weights
            homeostatic_target: Target mean activation level
            homeostatic_rate: Rate of homeostatic adjustment
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.homeostatic_target = homeostatic_target
        self.homeostatic_rate = homeostatic_rate
        
        # Standard weights (trained by gradient descent)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Hebbian weights (trained by local rules, not gradient)
        self.register_buffer('hebbian_weight', torch.zeros(out_features, in_features))
        
        # Homeostatic scaling factors
        self.register_buffer('homeostatic_scale', torch.ones(out_features))
        
        # Cache for Hebbian update
        self.cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Activity statistics
        self.register_buffer('running_activity', torch.zeros(out_features))
        self.register_buffer('activity_count', torch.tensor(0.0))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with caching for Hebbian update.
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            Output tensor [batch, out_features]
        """
        # Combine standard and Hebbian weights
        combined_weight = self.weight + self.hebbian_weight
        
        # Standard linear transformation
        y = F.linear(x, combined_weight, self.bias)
        
        # Apply homeostatic scaling
        y = y * self.homeostatic_scale.unsqueeze(0)
        
        # Cache for Hebbian update
        self.cache = (x.detach(), y.detach())
        
        # Update activity statistics (for homeostasis)
        with torch.no_grad():
            activity = y.abs().mean(dim=0)
            self.running_activity = 0.99 * self.running_activity + 0.01 * activity
            self.activity_count += 1
        
        return y
    
    def hebbian_update(self, success_signal: float = 1.0):
        """
        Apply Hebbian learning rule: strengthen co-active connections.
        
        Args:
            success_signal: Modulation based on task success [0, 1]
                           Higher = stronger Hebbian update
        """
        if self.cache is None:
            return
        
        x, y = self.cache
        
        # Hebbian rule: delta_w = lr * post * pre^T
        # Modulated by success signal
        with torch.no_grad():
            # Normalize activations
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
            
            # Outer product: [out_features, in_features]
            delta = torch.einsum('bi,bj->ij', y_norm, x_norm) / x.size(0)
            
            # Modulated update
            self.hebbian_weight *= self.hebbian_decay
            self.hebbian_weight += self.hebbian_lr * success_signal * delta
            
            # Normalize Hebbian weights to prevent explosion
            hebb_norm = self.hebbian_weight.norm()
            weight_norm = self.weight.norm()
            if hebb_norm > 0.1 * weight_norm:
                self.hebbian_weight *= 0.1 * weight_norm / hebb_norm
    
    def homeostatic_update(self):
        """
        Apply homeostatic plasticity to maintain stable activity levels.
        Neurons with too high/low activity get scaled down/up.
        """
        if self.activity_count < 10:
            return
        
        with torch.no_grad():
            # Compare running activity to target
            ratio = self.running_activity / (self.homeostatic_target + 1e-8)
            
            # Adjust scaling factors (inverse relationship)
            adjustment = 1.0 / (ratio + 1e-8)
            adjustment = adjustment.clamp(0.9, 1.1)  # Limit adjustment rate
            
            # Smooth update
            self.homeostatic_scale = (
                (1 - self.homeostatic_rate) * self.homeostatic_scale +
                self.homeostatic_rate * adjustment
            )
    
    def get_effective_weight(self) -> torch.Tensor:
        """Get the combined effective weight matrix."""
        return self.weight + self.hebbian_weight
    
    def get_hebbian_contribution(self) -> float:
        """Get the relative contribution of Hebbian weights."""
        total_norm = self.get_effective_weight().norm().item()
        hebbian_norm = self.hebbian_weight.norm().item()
        return hebbian_norm / (total_norm + 1e-8)
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, hebbian_lr={self.hebbian_lr}'
        )


class SelfOrganizingConv2d(nn.Module):
    """
    Convolutional layer with Hebbian learning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        hebbian_lr: float = 0.0001,
        hebbian_decay: float = 0.999
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        
        # Standard conv layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        
        # Hebbian weights (same shape as conv weights)
        self.register_buffer(
            'hebbian_weight',
            torch.zeros_like(self.conv.weight)
        )
        
        self.cache = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with caching."""
        # Combine weights
        combined_weight = self.conv.weight + self.hebbian_weight
        
        # Manual convolution with combined weights
        y = F.conv2d(x, combined_weight, self.conv.bias,
                     stride=self.stride, padding=self.padding)
        
        # Cache for Hebbian update
        self.cache = (x.detach(), y.detach())
        
        return y
    
    def hebbian_update(self, success_signal: float = 1.0):
        """Apply Hebbian update for convolutional layer."""
        if self.cache is None:
            return
        
        x, y = self.cache
        
        with torch.no_grad():
            # Unfold input for local connectivity
            # This is a simplified version - full implementation would use im2col
            
            # Global pooled correlation
            x_pooled = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C_in]
            y_pooled = F.adaptive_avg_pool2d(y, 1).squeeze(-1).squeeze(-1)  # [B, C_out]
            
            # Outer product
            delta = torch.einsum('bi,bj->ij', y_pooled, x_pooled) / x.size(0)
            
            # Expand to kernel shape
            delta = delta.unsqueeze(-1).unsqueeze(-1).expand_as(self.hebbian_weight)
            
            # Update
            self.hebbian_weight *= self.hebbian_decay
            self.hebbian_weight += self.hebbian_lr * success_signal * delta
            
            # Normalize
            hebb_norm = self.hebbian_weight.norm()
            weight_norm = self.conv.weight.norm()
            if hebb_norm > 0.1 * weight_norm:
                self.hebbian_weight *= 0.1 * weight_norm / hebb_norm


class SelfOrganizingBlock(nn.Module):
    """
    A block combining self-organizing layer with normalization and activation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        hebbian_lr: float = 0.001,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.linear = SelfOrganizingLinear(
            in_features, out_features,
            hebbian_lr=hebbian_lr
        )
        
        self.norm = nn.LayerNorm(out_features) if use_layer_norm else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
    def hebbian_update(self, success_signal: float = 1.0):
        self.linear.hebbian_update(success_signal)
    
    def homeostatic_update(self):
        self.linear.homeostatic_update()


class SelfOrganizingMLP(nn.Module):
    """
    Multi-layer perceptron with self-organizing capabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        hebbian_lr: float = 0.001,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(SelfOrganizingBlock(
                prev_dim, hidden_dim,
                activation='relu',
                hebbian_lr=hebbian_lr,
                dropout=dropout
            ))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def hebbian_update(self, success_signal: float = 1.0):
        """Apply Hebbian update to all self-organizing layers."""
        for layer in self.hidden_layers:
            layer.hebbian_update(success_signal)
    
    def homeostatic_update(self):
        """Apply homeostatic update to all layers."""
        for layer in self.hidden_layers:
            layer.homeostatic_update()
    
    def get_hebbian_contributions(self) -> Dict[str, float]:
        """Get Hebbian contribution for each layer."""
        contributions = {}
        for i, layer in enumerate(self.hidden_layers):
            contributions[f'layer_{i}'] = layer.linear.get_hebbian_contribution()
        return contributions


if __name__ == '__main__':
    print("Testing Self-Organizing Layers...")
    
    # Test SelfOrganizingLinear
    print("\n1. Testing SelfOrganizingLinear...")
    layer = SelfOrganizingLinear(64, 32, hebbian_lr=0.01)
    
    # Forward pass
    x = torch.randn(16, 64)
    y = layer(x)
    print(f"   Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Hebbian update
    layer.hebbian_update(success_signal=1.0)
    hebb_contrib = layer.get_hebbian_contribution()
    print(f"   Hebbian contribution after 1 update: {hebb_contrib:.4f}")
    
    # Multiple updates
    for _ in range(10):
        y = layer(x)
        layer.hebbian_update(success_signal=0.8)
    
    hebb_contrib = layer.get_hebbian_contribution()
    print(f"   Hebbian contribution after 10 updates: {hebb_contrib:.4f}")
    
    # Test homeostatic update
    layer.homeostatic_update()
    print(f"   Homeostatic scale range: [{layer.homeostatic_scale.min():.4f}, {layer.homeostatic_scale.max():.4f}]")
    
    # Test SelfOrganizingMLP
    print("\n2. Testing SelfOrganizingMLP...")
    mlp = SelfOrganizingMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        hebbian_lr=0.001
    )
    
    x = torch.randn(32, 784)
    y = mlp(x)
    print(f"   Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Train-like loop
    for _ in range(5):
        y = mlp(x)
        mlp.hebbian_update(success_signal=0.9)
        mlp.homeostatic_update()
    
    contributions = mlp.get_hebbian_contributions()
    print("   Hebbian contributions:")
    for name, contrib in contributions.items():
        print(f"     {name}: {contrib:.4f}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    mlp.zero_grad()
    y = mlp(x)
    loss = y.sum()
    loss.backward()
    
    grad_norms = []
    for name, param in mlp.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print("   Gradient norms (first 3 params):")
    for name, norm in grad_norms[:3]:
        print(f"     {name}: {norm:.4f}")
    
    print("\n[OK] Self-Organizing Layers test passed!")
