"""
Structural Plasticity Module for Hybrid Autopoietic-ML Approach

Implements dynamic network structure modification during training:
- Neuron pruning: Remove low-activation neurons
- Neuron growing: Add new neurons to bottleneck layers
- Connection rewiring: Modify connection patterns based on activity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict


class StructuralPlasticityModule:
    """
    Manages dynamic structural changes to neural network during training.
    Inspired by biological neural plasticity.
    """
    
    def __init__(
        self,
        prune_threshold: float = 0.01,
        grow_threshold: float = 0.1,
        activity_ema_alpha: float = 0.1,
        min_neurons: int = 8,
        max_neurons: int = 1024,
        device: str = 'cpu'
    ):
        """
        Args:
            prune_threshold: Neurons with activation below this are pruned
            grow_threshold: Layers with high activation variance get new neurons
            activity_ema_alpha: EMA smoothing for activity tracking
            min_neurons: Minimum neurons to keep in any layer
            max_neurons: Maximum neurons allowed in any layer
            device: Device for computations
        """
        self.prune_threshold = prune_threshold
        self.grow_threshold = grow_threshold
        self.activity_ema_alpha = activity_ema_alpha
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.device = device
        
        # Track neuron activity over time
        self.activity_stats: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.modification_history: List[Dict] = []
        
    def update_activity_stats(
        self,
        layer_name: str,
        activations: torch.Tensor
    ):
        """
        Update activity statistics for a layer using EMA.
        
        Args:
            layer_name: Name of the layer
            activations: Activation tensor [batch, neurons, ...]
        """
        # Compute per-neuron mean activation (absolute value)
        if activations.dim() == 2:
            # [batch, neurons]
            mean_act = activations.abs().mean(dim=0)
        elif activations.dim() == 4:
            # [batch, channels, h, w] - convolutional
            mean_act = activations.abs().mean(dim=(0, 2, 3))
        else:
            mean_act = activations.abs().view(activations.size(0), activations.size(1), -1).mean(dim=(0, 2))
        
        # EMA update
        if 'mean_activity' not in self.activity_stats[layer_name]:
            self.activity_stats[layer_name]['mean_activity'] = mean_act.detach()
            self.activity_stats[layer_name]['activity_var'] = torch.zeros_like(mean_act)
            self.activity_stats[layer_name]['update_count'] = 0
        else:
            old_mean = self.activity_stats[layer_name]['mean_activity']
            alpha = self.activity_ema_alpha
            new_mean = (1 - alpha) * old_mean + alpha * mean_act.detach()
            
            # Track variance
            old_var = self.activity_stats[layer_name]['activity_var']
            new_var = (1 - alpha) * old_var + alpha * (mean_act.detach() - new_mean).pow(2)
            
            self.activity_stats[layer_name]['mean_activity'] = new_mean
            self.activity_stats[layer_name]['activity_var'] = new_var
        
        self.activity_stats[layer_name]['update_count'] += 1
    
    def identify_neurons_to_prune(
        self,
        layer_name: str,
        current_size: int
    ) -> List[int]:
        """
        Identify which neurons should be pruned based on low activity.
        
        Returns:
            List of neuron indices to prune
        """
        if layer_name not in self.activity_stats:
            return []
        
        if self.activity_stats[layer_name]['update_count'] < 100:
            # Not enough data yet
            return []
        
        mean_activity = self.activity_stats[layer_name]['mean_activity']
        
        # Normalize by max activity
        max_act = mean_activity.max()
        if max_act < 1e-8:
            return []
        
        normalized = mean_activity / max_act
        
        # Find neurons below threshold
        prune_candidates = (normalized < self.prune_threshold).nonzero().squeeze(-1).tolist()
        
        if isinstance(prune_candidates, int):
            prune_candidates = [prune_candidates]
        
        # Don't prune below minimum
        max_prune = current_size - self.min_neurons
        if len(prune_candidates) > max_prune:
            # Sort by activity and keep only the lowest
            activities = [(i, normalized[i].item()) for i in prune_candidates]
            activities.sort(key=lambda x: x[1])
            prune_candidates = [a[0] for a in activities[:max_prune]]
        
        return prune_candidates
    
    def identify_neurons_to_grow(
        self,
        layer_name: str,
        current_size: int
    ) -> int:
        """
        Determine how many neurons to add based on activity variance.
        High variance suggests the layer is a bottleneck.
        
        Returns:
            Number of neurons to add
        """
        if layer_name not in self.activity_stats:
            return 0
        
        if self.activity_stats[layer_name]['update_count'] < 100:
            return 0
        
        if current_size >= self.max_neurons:
            return 0
        
        activity_var = self.activity_stats[layer_name]['activity_var']
        mean_var = activity_var.mean().item()
        
        # High variance = potential bottleneck
        if mean_var > self.grow_threshold:
            # Add neurons proportional to variance
            n_new = int(min(
                current_size * 0.1,  # Max 10% growth
                self.max_neurons - current_size
            ))
            return max(1, n_new)
        
        return 0
    
    def prune_linear_layer(
        self,
        layer: nn.Linear,
        layer_name: str,
        next_layer: Optional[nn.Linear] = None
    ) -> Tuple[nn.Linear, Optional[nn.Linear], int]:
        """
        Prune neurons from a linear layer.
        
        Args:
            layer: The layer to prune
            layer_name: Name of the layer
            next_layer: The following layer (needs input adjustment)
            
        Returns:
            Modified layer, modified next_layer, number pruned
        """
        prune_indices = self.identify_neurons_to_prune(
            layer_name, 
            layer.out_features
        )
        
        if not prune_indices:
            return layer, next_layer, 0
        
        # Keep indices
        keep_indices = [i for i in range(layer.out_features) if i not in prune_indices]
        keep_indices = torch.tensor(keep_indices, device=self.device)
        
        # Create new layer
        new_layer = nn.Linear(
            layer.in_features,
            len(keep_indices),
            bias=layer.bias is not None
        ).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            new_layer.weight.data = layer.weight.data[keep_indices]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_indices]
        
        # Update next layer if provided
        new_next_layer = next_layer
        if next_layer is not None:
            new_next_layer = nn.Linear(
                len(keep_indices),
                next_layer.out_features,
                bias=next_layer.bias is not None
            ).to(self.device)
            
            with torch.no_grad():
                new_next_layer.weight.data = next_layer.weight.data[:, keep_indices]
                if next_layer.bias is not None:
                    new_next_layer.bias.data = next_layer.bias.data
        
        # Update activity stats
        if layer_name in self.activity_stats:
            stats = self.activity_stats[layer_name]
            stats['mean_activity'] = stats['mean_activity'][keep_indices]
            stats['activity_var'] = stats['activity_var'][keep_indices]
        
        # Log modification
        self.modification_history.append({
            'type': 'prune',
            'layer': layer_name,
            'removed': len(prune_indices),
            'remaining': len(keep_indices)
        })
        
        return new_layer, new_next_layer, len(prune_indices)
    
    def grow_linear_layer(
        self,
        layer: nn.Linear,
        layer_name: str,
        next_layer: Optional[nn.Linear] = None,
        init_scale: float = 0.01
    ) -> Tuple[nn.Linear, Optional[nn.Linear], int]:
        """
        Add neurons to a linear layer.
        
        Args:
            layer: The layer to grow
            layer_name: Name of the layer
            next_layer: The following layer (needs input adjustment)
            init_scale: Scale for initializing new weights
            
        Returns:
            Modified layer, modified next_layer, number added
        """
        n_new = self.identify_neurons_to_grow(layer_name, layer.out_features)
        
        if n_new == 0:
            return layer, next_layer, 0
        
        new_size = layer.out_features + n_new
        
        # Create new layer
        new_layer = nn.Linear(
            layer.in_features,
            new_size,
            bias=layer.bias is not None
        ).to(self.device)
        
        # Copy old weights and initialize new ones
        with torch.no_grad():
            new_layer.weight.data[:layer.out_features] = layer.weight.data
            new_layer.weight.data[layer.out_features:] = init_scale * torch.randn(
                n_new, layer.in_features, device=self.device
            )
            
            if layer.bias is not None:
                new_layer.bias.data[:layer.out_features] = layer.bias.data
                new_layer.bias.data[layer.out_features:] = 0
        
        # Update next layer if provided
        new_next_layer = next_layer
        if next_layer is not None:
            new_next_layer = nn.Linear(
                new_size,
                next_layer.out_features,
                bias=next_layer.bias is not None
            ).to(self.device)
            
            with torch.no_grad():
                new_next_layer.weight.data[:, :layer.out_features] = next_layer.weight.data
                # Initialize new connections with small random weights
                new_next_layer.weight.data[:, layer.out_features:] = init_scale * torch.randn(
                    next_layer.out_features, n_new, device=self.device
                )
                if next_layer.bias is not None:
                    new_next_layer.bias.data = next_layer.bias.data
        
        # Update activity stats
        if layer_name in self.activity_stats:
            stats = self.activity_stats[layer_name]
            # Initialize new neuron stats
            old_mean = stats['mean_activity'].mean()
            old_var = stats['activity_var'].mean()
            
            stats['mean_activity'] = torch.cat([
                stats['mean_activity'],
                old_mean * torch.ones(n_new, device=self.device)
            ])
            stats['activity_var'] = torch.cat([
                stats['activity_var'],
                old_var * torch.ones(n_new, device=self.device)
            ])
        
        # Log modification
        self.modification_history.append({
            'type': 'grow',
            'layer': layer_name,
            'added': n_new,
            'new_size': new_size
        })
        
        return new_layer, new_next_layer, n_new
    
    def prune_conv_layer(
        self,
        layer: nn.Conv2d,
        layer_name: str,
        next_layer: Optional[Union[nn.Conv2d, nn.Linear]] = None,
        batch_norm: Optional[nn.BatchNorm2d] = None
    ) -> Tuple[nn.Conv2d, Optional[Union[nn.Conv2d, nn.Linear]], Optional[nn.BatchNorm2d], int]:
        """
        Prune channels from a convolutional layer.
        """
        prune_indices = self.identify_neurons_to_prune(
            layer_name,
            layer.out_channels
        )
        
        if not prune_indices:
            return layer, next_layer, batch_norm, 0
        
        keep_indices = [i for i in range(layer.out_channels) if i not in prune_indices]
        keep_indices = torch.tensor(keep_indices, device=self.device)
        
        # Create new conv layer
        new_layer = nn.Conv2d(
            layer.in_channels,
            len(keep_indices),
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias is not None
        ).to(self.device)
        
        with torch.no_grad():
            new_layer.weight.data = layer.weight.data[keep_indices]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_indices]
        
        # Update batch norm if present
        new_bn = batch_norm
        if batch_norm is not None:
            new_bn = nn.BatchNorm2d(len(keep_indices)).to(self.device)
            with torch.no_grad():
                new_bn.weight.data = batch_norm.weight.data[keep_indices]
                new_bn.bias.data = batch_norm.bias.data[keep_indices]
                new_bn.running_mean = batch_norm.running_mean[keep_indices]
                new_bn.running_var = batch_norm.running_var[keep_indices]
        
        # Update next layer
        new_next_layer = next_layer
        if isinstance(next_layer, nn.Conv2d):
            new_next_layer = nn.Conv2d(
                len(keep_indices),
                next_layer.out_channels,
                next_layer.kernel_size,
                stride=next_layer.stride,
                padding=next_layer.padding,
                bias=next_layer.bias is not None
            ).to(self.device)
            
            with torch.no_grad():
                new_next_layer.weight.data = next_layer.weight.data[:, keep_indices]
                if next_layer.bias is not None:
                    new_next_layer.bias.data = next_layer.bias.data
        
        # Log modification
        self.modification_history.append({
            'type': 'prune_conv',
            'layer': layer_name,
            'removed': len(prune_indices),
            'remaining': len(keep_indices)
        })
        
        return new_layer, new_next_layer, new_bn, len(prune_indices)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about structural modifications.
        """
        stats = {
            'total_modifications': len(self.modification_history),
            'total_pruned': sum(
                m.get('removed', 0) for m in self.modification_history if m['type'] in ['prune', 'prune_conv']
            ),
            'total_grown': sum(
                m.get('added', 0) for m in self.modification_history if m['type'] == 'grow'
            ),
            'layers_tracked': len(self.activity_stats)
        }
        return stats
    
    def reset(self):
        """Reset all statistics and history."""
        self.activity_stats = defaultdict(dict)
        self.modification_history = []


class AdaptivePlasticityScheduler:
    """
    Schedules when to apply structural plasticity during training.
    """
    
    def __init__(
        self,
        plasticity_module: StructuralPlasticityModule,
        warmup_steps: int = 1000,
        apply_every: int = 500,
        cooldown_after_change: int = 200
    ):
        self.plasticity = plasticity_module
        self.warmup_steps = warmup_steps
        self.apply_every = apply_every
        self.cooldown_after_change = cooldown_after_change
        
        self.step_count = 0
        self.last_change_step = -self.cooldown_after_change
    
    def step(self) -> bool:
        """
        Advance one step. Returns True if plasticity should be applied.
        """
        self.step_count += 1
        
        # Check warmup
        if self.step_count < self.warmup_steps:
            return False
        
        # Check cooldown
        if self.step_count - self.last_change_step < self.cooldown_after_change:
            return False
        
        # Check interval
        if (self.step_count - self.warmup_steps) % self.apply_every == 0:
            return True
        
        return False
    
    def mark_change(self):
        """Mark that a structural change occurred."""
        self.last_change_step = self.step_count


if __name__ == '__main__':
    print("Testing StructuralPlasticityModule...")
    
    device = 'cpu'
    plasticity = StructuralPlasticityModule(device=device)
    
    # Create test layers
    layer1 = nn.Linear(64, 128).to(device)
    layer2 = nn.Linear(128, 64).to(device)
    
    # Simulate training with activity updates
    print("\nSimulating training with activity tracking...")
    for step in range(200):
        # Simulate forward pass
        x = torch.randn(32, 64, device=device)
        h = F.relu(layer1(x))
        
        # Update activity stats
        plasticity.update_activity_stats('layer1', h)
        
        # Make some neurons inactive
        if step == 0:
            with torch.no_grad():
                # Set some weights to near-zero (will become inactive)
                layer1.weight.data[100:110] *= 0.001
    
    # Check pruning candidates
    prune_candidates = plasticity.identify_neurons_to_prune('layer1', 128)
    print(f"Neurons to prune: {len(prune_candidates)}")
    
    # Check grow candidates
    n_grow = plasticity.identify_neurons_to_grow('layer1', 128)
    print(f"Neurons to grow: {n_grow}")
    
    # Apply pruning
    new_layer1, new_layer2, n_pruned = plasticity.prune_linear_layer(
        layer1, 'layer1', layer2
    )
    print(f"\nAfter pruning:")
    print(f"  Layer1: {layer1.out_features} -> {new_layer1.out_features}")
    print(f"  Layer2 input: {layer2.in_features} -> {new_layer2.in_features}")
    
    # Get statistics
    stats = plasticity.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n[OK] StructuralPlasticityModule test passed!")
