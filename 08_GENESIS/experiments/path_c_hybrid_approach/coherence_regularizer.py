"""
Coherence Regularizer for Hybrid Autopoietic-ML Approach

Implements 4D coherence metrics based on autopoietic principles:
- Predictability: How well internal state predicts next state
- Stability: Temporal consistency of activations
- Complexity: Information richness (not too simple, not too chaotic)
- Circularity: Self-referential consistency (outputs inform inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CoherenceRegularizer:
    """
    Adds coherence penalty to loss function.
    Maximizes coherence by penalizing 1 - coherence.
    """
    
    def __init__(
        self,
        pred_weight: float = 0.3,
        stab_weight: float = 0.3,
        comp_weight: float = 0.2,
        circ_weight: float = 0.2,
        history_size: int = 10,
        device: str = 'cpu'
    ):
        """
        Args:
            pred_weight: Weight for predictability component
            stab_weight: Weight for stability component
            comp_weight: Weight for complexity component
            circ_weight: Weight for circularity component
            history_size: Number of past activations to store
            device: Device for computations
        """
        self.pred_weight = pred_weight
        self.stab_weight = stab_weight
        self.comp_weight = comp_weight
        self.circ_weight = circ_weight
        self.history_size = history_size
        self.device = device
        
        # History buffers for temporal metrics
        self.activation_history: List[Dict[str, torch.Tensor]] = []
        self.prediction_model: Optional[nn.Module] = None
        
    def predictability(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measure how predictable current activations are from previous ones.
        High predictability = system has learnable internal dynamics.
        
        Uses exponential moving average prediction.
        """
        if len(self.activation_history) < 2:
            return torch.tensor(0.5, device=self.device)
        
        # Simple prediction: EMA of past activations
        total_pred_score = 0.0
        n_layers = 0
        
        for layer_name, current_act in activations.items():
            if layer_name not in self.activation_history[-1]:
                continue
                
            # Get history for this layer
            history = [h[layer_name] for h in self.activation_history if layer_name in h]
            if len(history) < 2:
                continue
            
            # Get current batch size
            current_batch = current_act.size(0)

            # Filter history to only include same batch size (avoid last-batch issues)
            valid_history = [h for h in history if h.size(0) == current_batch]
            if len(valid_history) < 2:
                continue

            # EMA prediction using only compatible batch sizes
            alpha = 0.3
            predicted = valid_history[-1].clone()
            for i, h in enumerate(reversed(valid_history[:-1])):
                weight = alpha * ((1 - alpha) ** i)
                predicted = predicted + weight * (h - predicted)

            # Prediction error (normalized)
            current_flat = current_act.view(current_batch, -1)
            predicted_flat = predicted.view(current_batch, -1)

            # Cosine similarity as predictability measure
            cos_sim = F.cosine_similarity(current_flat, predicted_flat, dim=1)
            pred_score = (cos_sim + 1) / 2  # Map to [0, 1]

            total_pred_score += pred_score.mean()
            n_layers += 1
        
        if n_layers == 0:
            return torch.tensor(0.5, device=self.device)
            
        return total_pred_score / n_layers
    
    def stability(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measure temporal stability of activations.
        Stable = activations don't fluctuate wildly between steps.
        
        Uses variance of activation norms over time.
        """
        if len(self.activation_history) < 2:
            return torch.tensor(0.5, device=self.device)
        
        total_stability = 0.0
        n_layers = 0
        
        for layer_name, current_act in activations.items():
            history = [h[layer_name] for h in self.activation_history if layer_name in h]
            if len(history) < 2:
                continue
            
            # Compute norm trajectory
            norms = []
            for h in history:
                norm = h.view(h.size(0), -1).norm(dim=1).mean()
                norms.append(norm)
            norms.append(current_act.view(current_act.size(0), -1).norm(dim=1).mean())
            
            # Stack and compute variance
            norms_tensor = torch.stack(norms)
            mean_norm = norms_tensor.mean()
            
            # Coefficient of variation (lower = more stable)
            if mean_norm > 1e-8:
                cv = norms_tensor.std() / mean_norm
                # Map CV to stability score (lower CV = higher stability)
                stability = torch.exp(-cv)
            else:
                stability = torch.tensor(1.0, device=self.device)
            
            total_stability += stability
            n_layers += 1
        
        if n_layers == 0:
            return torch.tensor(0.5, device=self.device)
            
        return total_stability / n_layers
    
    def complexity(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measure information complexity of activations.
        Uses entropy-like measure: not too simple (all same), not too chaotic.
        
        Optimal complexity is in the middle range.
        """
        total_complexity = 0.0
        n_layers = 0
        
        for layer_name, act in activations.items():
            act_flat = act.view(act.size(0), -1)
            
            # Normalize activations to [0, 1] for entropy calculation
            act_min = act_flat.min(dim=1, keepdim=True)[0]
            act_max = act_flat.max(dim=1, keepdim=True)[0]
            act_range = act_max - act_min + 1e-8
            act_norm = (act_flat - act_min) / act_range
            
            # Compute histogram-based entropy
            n_bins = 20
            entropy_sum = 0.0
            
            for i in range(act_norm.size(0)):
                hist = torch.histc(act_norm[i], bins=n_bins, min=0, max=1)
                hist = hist / hist.sum()
                hist = hist + 1e-10  # Avoid log(0)
                entropy = -(hist * torch.log(hist)).sum()
                entropy_sum += entropy
            
            avg_entropy = entropy_sum / act_norm.size(0)
            max_entropy = np.log(n_bins)
            
            # Normalized entropy (0 = no complexity, 1 = maximum entropy)
            norm_entropy = avg_entropy / max_entropy
            
            # Complexity score: prefer middle range (not 0, not 1)
            # Use inverted U-shape: 4 * x * (1 - x) peaks at 0.5
            complexity = 4 * norm_entropy * (1 - norm_entropy)
            
            total_complexity += complexity
            n_layers += 1
        
        if n_layers == 0:
            return torch.tensor(0.5, device=self.device)
            
        return total_complexity / n_layers
    
    def circularity(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measure self-referential consistency.
        How well do later layers maintain information from earlier layers?
        
        Uses correlation between first and last layer activations.
        """
        layer_names = list(activations.keys())
        if len(layer_names) < 2:
            return torch.tensor(0.5, device=self.device)
        
        first_act = activations[layer_names[0]]
        last_act = activations[layer_names[-1]]
        
        # Flatten
        first_flat = first_act.view(first_act.size(0), -1)
        last_flat = last_act.view(last_act.size(0), -1)
        
        # Project to same dimension for comparison
        min_dim = min(first_flat.size(1), last_flat.size(1))
        
        # Use PCA-like projection (first min_dim dimensions)
        first_proj = first_flat[:, :min_dim]
        last_proj = last_flat[:, :min_dim]
        
        # Compute correlation
        first_centered = first_proj - first_proj.mean(dim=0, keepdim=True)
        last_centered = last_proj - last_proj.mean(dim=0, keepdim=True)
        
        # Cosine similarity as circularity measure
        cos_sim = F.cosine_similarity(
            first_centered.mean(dim=0, keepdim=True),
            last_centered.mean(dim=0, keepdim=True),
            dim=1
        )
        
        # Map to [0, 1]
        circularity = (cos_sim + 1) / 2
        
        return circularity.mean()
    
    def compute_coherence(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute overall coherence score as weighted sum of 4 dimensions.
        
        Returns:
            Coherence score in [0, 1]
        """
        pred = self.predictability(activations)
        stab = self.stability(activations)
        comp = self.complexity(activations)
        circ = self.circularity(activations)
        
        coherence = (
            self.pred_weight * pred +
            self.stab_weight * stab +
            self.comp_weight * comp +
            self.circ_weight * circ
        )
        
        return coherence
    
    def penalty(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute coherence penalty for loss function.
        Penalty = 1 - coherence (to maximize coherence).
        
        Returns:
            Penalty value to add to loss
        """
        coherence = self.compute_coherence(activations)
        return 1.0 - coherence
    
    def update_history(self, activations: Dict[str, torch.Tensor]):
        """
        Update activation history for temporal metrics.
        
        Args:
            activations: Current layer activations
        """
        # Detach and clone to avoid memory leaks
        detached = {
            name: act.detach().clone()
            for name, act in activations.items()
        }
        
        self.activation_history.append(detached)
        
        # Keep only recent history
        if len(self.activation_history) > self.history_size:
            self.activation_history.pop(0)
    
    def reset_history(self):
        """Clear activation history (e.g., at epoch start)."""
        self.activation_history = []
    
    def get_metrics(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get individual coherence metrics for logging.
        
        Returns:
            Dictionary with individual metric values
        """
        return {
            'predictability': self.predictability(activations).item(),
            'stability': self.stability(activations).item(),
            'complexity': self.complexity(activations).item(),
            'circularity': self.circularity(activations).item(),
            'coherence': self.compute_coherence(activations).item()
        }


class AdaptiveCoherenceRegularizer(CoherenceRegularizer):
    """
    Adaptive version that adjusts weights based on training progress.
    """
    
    def __init__(self, *args, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.metric_history = {
            'predictability': [],
            'stability': [],
            'complexity': [],
            'circularity': []
        }
    
    def adapt_weights(self, activations: Dict[str, torch.Tensor]):
        """
        Adapt weights to emphasize underperforming metrics.
        """
        metrics = self.get_metrics(activations)
        
        for name in self.metric_history:
            self.metric_history[name].append(metrics[name])
        
        # Only adapt after enough history
        if len(self.metric_history['predictability']) < 10:
            return
        
        # Compute recent averages
        recent = {}
        for name in self.metric_history:
            recent[name] = np.mean(self.metric_history[name][-10:])
        
        # Increase weight for lowest-performing metric
        min_metric = min(recent, key=recent.get)
        total_increase = self.adaptation_rate
        
        # Redistribute weights
        if min_metric == 'predictability':
            self.pred_weight += total_increase
        elif min_metric == 'stability':
            self.stab_weight += total_increase
        elif min_metric == 'complexity':
            self.comp_weight += total_increase
        else:
            self.circ_weight += total_increase
        
        # Normalize weights
        total = self.pred_weight + self.stab_weight + self.comp_weight + self.circ_weight
        self.pred_weight /= total
        self.stab_weight /= total
        self.comp_weight /= total
        self.circ_weight /= total


if __name__ == '__main__':
    # Test the coherence regularizer
    print("Testing CoherenceRegularizer...")
    
    reg = CoherenceRegularizer(device='cpu')
    
    # Simulate activations from a network
    batch_size = 32
    activations = {
        'layer1': torch.randn(batch_size, 64),
        'layer2': torch.randn(batch_size, 128),
        'layer3': torch.randn(batch_size, 64)
    }
    
    # Test without history
    coherence = reg.compute_coherence(activations)
    print(f"Initial coherence (no history): {coherence.item():.4f}")
    
    # Add some history
    for _ in range(5):
        reg.update_history(activations)
        # Slightly modify activations (simulating training)
        activations = {
            name: act + 0.1 * torch.randn_like(act)
            for name, act in activations.items()
        }
    
    # Test with history
    coherence = reg.compute_coherence(activations)
    penalty = reg.penalty(activations)
    metrics = reg.get_metrics(activations)
    
    print(f"\nCoherence with history: {coherence.item():.4f}")
    print(f"Penalty: {penalty.item():.4f}")
    print("\nIndividual metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n[OK] CoherenceRegularizer test passed!")
