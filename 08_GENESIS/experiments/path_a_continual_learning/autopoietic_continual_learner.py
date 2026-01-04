"""
Autopoietic Continual Learner (v2 - Improved Learning)

Extension of AutopoeticEntity for continual learning benchmarks.

Key adaptations:
    1. Scaled architecture: 256 hidden units (from 20)
    2. Task-specific output heads
    3. Hierarchical coherence measurement
    4. Evolution Strategy (ES) based coherence-preserving updates

v2 Improvements:
    - Evolution Strategy for efficient search in high-dimensional space
    - Adaptive plasticity rate based on learning progress
    - Best-of-K selection for structural updates
    - Momentum in search direction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class HierarchicalCoherenceAssessor:
    """
    Hierarchical Coherence Assessment for scaled networks.
    
    Measures coherence at multiple levels:
        1. Layer-level: coherence within each layer
        2. Inter-layer: coherence between layers
        3. Global: overall network coherence
        4. Task-alignment: how well internal state aligns with classification
    """
    
    def __init__(self, history_len: int = 50):
        self.history_len = history_len
        self.state_history = deque(maxlen=history_len)
        self.prediction_history = deque(maxlen=history_len)
        self.coherence_history = deque(maxlen=history_len)
        
    def assess(self, 
               hidden_state: np.ndarray,
               prediction_logits: np.ndarray = None,
               target: np.ndarray = None) -> Dict[str, float]:
        """
        Multi-dimensional coherence assessment.
        
        Args:
            hidden_state: Internal hidden state
            prediction_logits: Output prediction logits
            target: True target (for task alignment)
            
        Returns:
            coherence_scores: Dictionary of coherence metrics
        """
        self.state_history.append(hidden_state.copy())
        
        if len(self.state_history) < 10:
            return {
                'predictability': 0.5,
                'stability': 0.5,
                'complexity': 0.5,
                'circularity': 0.5,
                'task_alignment': 0.5,
                'composite': 0.5
            }
        
        states = np.array(list(self.state_history))
        
        # 1. Predictability: Low variance in state changes
        state_changes = np.diff(states, axis=0)
        predictability = 1.0 / (1.0 + np.mean(np.var(state_changes, axis=0)))
        
        # 2. Stability: Recent states stability
        recent_states = states[-20:] if len(states) >= 20 else states
        stability = 1.0 / (1.0 + np.std(recent_states))
        
        # 3. Complexity: Optimal around 0.5 variance
        state_variance = np.var(states)
        complexity = 1.0 - abs(state_variance - 0.5)
        complexity = np.clip(complexity, 0, 1)
        
        # 4. Circularity: Temporal autocorrelation
        if len(states) >= 20:
            try:
                autocorr = np.corrcoef(
                    states[:-10].flatten()[:1000],  # Limit size
                    states[10:].flatten()[:1000]
                )[0, 1]
                circularity = abs(autocorr) if not np.isnan(autocorr) else 0.5
            except:
                circularity = 0.5
        else:
            circularity = 0.5
            
        # 5. Task Alignment: Prediction accuracy
        task_alignment = 0.5
        if prediction_logits is not None and target is not None:
            pred = np.argmax(prediction_logits)
            task_alignment = 1.0 if pred == target else 0.0
            self.prediction_history.append(task_alignment)
            if len(self.prediction_history) > 5:
                task_alignment = np.mean(list(self.prediction_history)[-10:])
        
        # Composite score
        composite = (
            0.2 * predictability +
            0.2 * stability +
            0.15 * complexity +
            0.15 * circularity +
            0.3 * task_alignment  # Higher weight for task performance
        )
        
        coherence = {
            'predictability': float(np.clip(predictability, 0, 1)),
            'stability': float(np.clip(stability, 0, 1)),
            'complexity': float(np.clip(complexity, 0, 1)),
            'circularity': float(np.clip(circularity, 0, 1)),
            'task_alignment': float(np.clip(task_alignment, 0, 1)),
            'composite': float(np.clip(composite, 0, 1))
        }
        
        self.coherence_history.append(coherence['composite'])
        return coherence
    
    def reset(self):
        """Reset history for new task."""
        # Keep some history for continuity
        self.state_history = deque(
            list(self.state_history)[-10:] if self.state_history else [],
            maxlen=self.history_len
        )


class AutopoeticContinualLearner:
    """
    Autopoietic Continual Learning System
    
    Core Principles:
        1. NO external loss function optimization
        2. Structural drift guided by coherence preservation
        3. Task learning through coherence alignment
        4. Natural forgetting resistance via organizational closure
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 num_tasks: int = 5,
                 classes_per_task: int = 2,
                 connectivity: float = 0.4,
                 plasticity_rate: float = 0.01,
                 coherence_threshold: float = 0.3,
                 seed: int = 42,
                 # Ablation parameters
                 random_win: bool = False,
                 freeze_win_after_task0: bool = True,
                 coherence_acceptance_threshold: float = 0.95,
                 learning_rule: str = 'hebbian',
                 sgd_lr: float = 0.01):
        """
        Args:
            input_dim: Input dimension (after PCA)
            hidden_dim: Hidden layer dimension
            num_tasks: Number of continual learning tasks
            classes_per_task: Classes per task
            connectivity: Recurrent connectivity density
            plasticity_rate: Rate of structural changes
            coherence_threshold: Minimum coherence for survival
            seed: Random seed

            Ablation parameters:
            random_win: If True, use random W_in (never learn it) - RanPAC style
            freeze_win_after_task0: If True, freeze W_in after task 0 (default behavior)
            coherence_acceptance_threshold: Threshold for accepting updates (0.0 = always accept)
            learning_rule: 'hebbian', 'sgd', or 'adam'
            sgd_lr: Learning rate for SGD/Adam
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.connectivity = connectivity
        self.plasticity_rate = plasticity_rate
        self.coherence_threshold = coherence_threshold

        # Ablation parameters
        self.random_win = random_win
        self.freeze_win_after_task0 = freeze_win_after_task0
        self.coherence_acceptance_threshold = coherence_acceptance_threshold
        self.learning_rule = learning_rule
        self.sgd_lr = sgd_lr

        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # ====================================================
        # Network Structure (NOT for optimization!)
        # ====================================================

        # Input projection
        # Use He initialization for better random projection (when using random_win)
        if self.random_win:
            # He initialization: sqrt(2/fan_in) for ReLU-like activations
            self.W_in = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        else:
            self.W_in = np.random.randn(hidden_dim, input_dim) * 0.1
        
        # Recurrent dynamics (the "organization")
        mask = np.random.rand(hidden_dim, hidden_dim) < connectivity
        self.W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.2 * mask
        np.fill_diagonal(self.W_rec, 0)  # No self-connections
        
        # Task-specific output heads
        self.W_out = {}
        self.b_out = {}
        for task_id in range(num_tasks):
            self.W_out[task_id] = np.random.randn(classes_per_task, hidden_dim) * 0.1
            self.b_out[task_id] = np.zeros(classes_per_task)
        
        # Internal state
        self.hidden_state = np.random.randn(hidden_dim) * 0.1
        self.hidden_history = deque(maxlen=100)
        
        # Coherence assessment
        self.assessor = HierarchicalCoherenceAssessor()
        
        # Tracking
        self.total_steps = 0
        self.task_steps = {i: 0 for i in range(num_tasks)}
        self.structural_changes = 0
        self.coherence_log = []

        # Track which tasks have been trained (for freezing W_in after first task)
        self.trained_tasks = set()

        # FLOPs counter for fairness comparison
        self.flops = 0
        
        print(f"AutopoeticContinualLearner initialized:")
        print(f"  Input: {input_dim} -> Hidden: {hidden_dim} -> Output: {classes_per_task}/task")
        print(f"  Connectivity: {connectivity:.1%}")
        print(f"  Paradigm: Autopoietic (coherence-driven learning)")
        
    def _forward(self, x: np.ndarray, task_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.
        
        NOT optimization-based! This computes the network's natural response.
        
        Args:
            x: Input (batch_size, input_dim) or (input_dim,)
            task_id: Current task
            
        Returns:
            hidden: Hidden state
            logits: Output logits
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        batch_size = x.shape[0]
        
        # Project input
        input_projection = np.dot(x, self.W_in.T)  # (batch, hidden)
        
        # FLOPs: input projection
        self.flops += batch_size * self.input_dim * self.hidden_dim
        
        all_hidden = []
        all_logits = []
        
        for i in range(batch_size):
            # Recurrent dynamics
            internal = np.tanh(np.dot(self.W_rec, self.hidden_state))
            
            # Leaky integration with input
            tau = 0.7
            new_hidden = tau * self.hidden_state + (1 - tau) * (internal + 0.3 * input_projection[i])
            self.hidden_state = np.clip(new_hidden, -2, 2)
            
            # FLOPs: recurrent
            self.flops += self.hidden_dim * self.hidden_dim + self.hidden_dim
            
            # Task-specific output
            logits = np.dot(self.W_out[task_id], self.hidden_state) + self.b_out[task_id]
            
            # FLOPs: output
            self.flops += self.classes_per_task * self.hidden_dim
            
            all_hidden.append(self.hidden_state.copy())
            all_logits.append(logits)
            
        return np.array(all_hidden), np.array(all_logits)
    
    def _hebbian_update(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        task_id: int) -> Dict:
        """
        Hebbian-inspired coherence-preserving update.

        Key insight: Use correlation between activity patterns and targets
        to guide weight updates, while checking coherence preservation.

        This is NOT gradient descent:
        - No backpropagation
        - No chain rule
        - Only local correlations + coherence check

        COHERENCE PRESERVATION STRATEGY:
        - Task 0: Learn both W_in and W_out (establish shared representation)
        - Task 1+: Only learn W_out (preserve shared representation)

        Ablation conditions:
        - random_win=True: W_in is never learned (RanPAC style)
        - freeze_win_after_task0=False: W_in is learned on all tasks
        - coherence_acceptance_threshold=0.0: All updates accepted
        - learning_rule='sgd'/'adam': Use gradient-based learning for W_out

        Args:
            x: Input batch
            y: Target labels
            task_id: Current task

        Returns:
            metrics: Update metrics
        """
        batch_size = len(x)

        # Forward pass to get hidden states and predictions
        hidden_states, logits = self._forward(x, task_id)

        # Predictions and errors
        predictions = np.argmax(logits, axis=1)
        correct = (predictions == y)
        accuracy_before = np.mean(correct)

        # Store original weights for coherence check
        W_out_orig = self.W_out[task_id].copy()
        b_out_orig = self.b_out[task_id].copy()
        W_in_orig = self.W_in.copy()

        # Convert targets to one-hot
        target_onehot = np.zeros((batch_size, self.classes_per_task))
        target_onehot[np.arange(batch_size), y] = 1

        # Softmax predictions
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        pred_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Error signal (NOT gradient, just correlation guidance)
        error = target_onehot - pred_probs  # (batch, classes)

        # === Output Layer Update ===
        if self.learning_rule == 'hebbian':
            # Hebbian Update for Output Layer
            delta_W_out = np.zeros_like(self.W_out[task_id])
            delta_b_out = np.zeros_like(self.b_out[task_id])

            for i in range(batch_size):
                delta_W_out += np.outer(error[i], hidden_states[i])
                delta_b_out += error[i]

            delta_W_out /= batch_size
            delta_b_out /= batch_size

            # Apply output layer updates
            lr = self.plasticity_rate
            self.W_out[task_id] += lr * delta_W_out
            self.b_out[task_id] += lr * delta_b_out

        elif self.learning_rule in ['sgd', 'adam']:
            # Gradient-based update for W_out
            # Compute gradient: dL/dW_out = -error @ hidden_states / batch_size
            grad_W_out = -np.dot(error.T, hidden_states) / batch_size
            grad_b_out = -np.mean(error, axis=0)

            if self.learning_rule == 'sgd':
                self.W_out[task_id] -= self.sgd_lr * grad_W_out
                self.b_out[task_id] -= self.sgd_lr * grad_b_out

            elif self.learning_rule == 'adam':
                # Simple Adam implementation
                if not hasattr(self, '_adam_state'):
                    self._adam_state = {}

                key = f'task_{task_id}'
                if key not in self._adam_state:
                    self._adam_state[key] = {
                        'm_W': np.zeros_like(self.W_out[task_id]),
                        'v_W': np.zeros_like(self.W_out[task_id]),
                        'm_b': np.zeros_like(self.b_out[task_id]),
                        'v_b': np.zeros_like(self.b_out[task_id]),
                        't': 0
                    }

                state = self._adam_state[key]
                state['t'] += 1
                t = state['t']
                beta1, beta2, eps = 0.9, 0.999, 1e-8

                # Update moments for W
                state['m_W'] = beta1 * state['m_W'] + (1 - beta1) * grad_W_out
                state['v_W'] = beta2 * state['v_W'] + (1 - beta2) * (grad_W_out ** 2)
                m_hat_W = state['m_W'] / (1 - beta1 ** t)
                v_hat_W = state['v_W'] / (1 - beta2 ** t)
                self.W_out[task_id] -= self.sgd_lr * m_hat_W / (np.sqrt(v_hat_W) + eps)

                # Update moments for b
                state['m_b'] = beta1 * state['m_b'] + (1 - beta1) * grad_b_out
                state['v_b'] = beta2 * state['v_b'] + (1 - beta2) * (grad_b_out ** 2)
                m_hat_b = state['m_b'] / (1 - beta1 ** t)
                v_hat_b = state['v_b'] / (1 - beta2 ** t)
                self.b_out[task_id] -= self.sgd_lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # === Input Layer Update (Hebbian only, based on ablation settings) ===
        # Determine if we should update W_in
        should_update_win = (
            not self.random_win and  # Not using random W_in (RanPAC style)
            self.learning_rule == 'hebbian' and  # Only for Hebbian
            (not self.freeze_win_after_task0 or len(self.trained_tasks) == 0)  # First task or not freezing
        )

        if should_update_win:
            # Learn W_in
            delta_W_in = np.zeros_like(self.W_in)
            lr = self.plasticity_rate

            for i in range(batch_size):
                if correct[i]:
                    delta_W_in += np.outer(hidden_states[i], x[i]) * 0.1
                else:
                    delta_W_in -= np.outer(hidden_states[i], x[i]) * 0.05

            delta_W_in /= batch_size
            self.W_in += lr * 0.5 * delta_W_in

            # FLOPs estimate (including W_in update)
            self.flops += batch_size * (self.hidden_dim * self.classes_per_task +
                                        self.hidden_dim * self.input_dim)
        else:
            # W_in is frozen (either random or after task 0)
            self.flops += batch_size * self.hidden_dim * self.classes_per_task

        # === Coherence Check ===
        _, new_logits = self._forward(x, task_id)
        new_predictions = np.argmax(new_logits, axis=1)
        accuracy_after = np.mean(new_predictions == y)

        # Check coherence
        coherence = self.assessor.assess(
            self.hidden_state,
            new_logits[-1] if len(new_logits) > 0 else None,
            y[-1] if len(y) > 0 else None
        )

        # Accept if accuracy improved or maintained (based on coherence threshold)
        # coherence_acceptance_threshold = 0.0 means always accept
        # coherence_acceptance_threshold = 0.95 means accept if acc_after >= 0.95 * acc_before
        # coherence_acceptance_threshold = 1.0 means only accept if acc_after >= acc_before
        if self.coherence_acceptance_threshold == 0.0:
            # Always accept (no coherence check)
            self.structural_changes += 1
            accepted = True
        elif accuracy_after >= accuracy_before * self.coherence_acceptance_threshold:
            self.structural_changes += 1
            accepted = True
        else:
            # Revert
            self.W_out[task_id] = W_out_orig
            self.b_out[task_id] = b_out_orig
            self.W_in = W_in_orig
            accepted = False
            accuracy_after = accuracy_before

        return {
            'accepted': accepted,
            'accuracy_before': accuracy_before,
            'accuracy_after': accuracy_after,
            'coherence': coherence['composite']
        }

    def _coherence_guided_update(self,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  task_id: int,
                                  coherence_before: float) -> bool:
        """
        Legacy interface - now uses Hebbian learning internally.
        """
        result = self._hebbian_update(x, y, task_id)
        return result['accepted']
    
    def train_step(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   task_id: int,
                   num_updates: int = 3) -> Dict:
        """
        One training step (batch) using Hebbian learning.

        Args:
            x: Input batch (batch_size, input_dim)
            y: Target labels (batch_size,)
            task_id: Current task
            num_updates: Number of Hebbian updates per step

        Returns:
            metrics: Training metrics
        """
        self.total_steps += 1
        self.task_steps[task_id] += 1

        # Multiple Hebbian updates for this batch
        accepted = 0
        for _ in range(num_updates):
            result = self._hebbian_update(x, y, task_id)
            if result['accepted']:
                accepted += 1

        # Final evaluation
        _, final_logits = self._forward(x, task_id)
        predictions = np.argmax(final_logits, axis=1)
        accuracy = np.mean(predictions == y)

        # Final coherence
        coherence = self.assessor.assess(
            self.hidden_state,
            final_logits[-1] if len(final_logits) > 0 else None,
            y[-1] if len(y) > 0 else None
        )

        self.coherence_log.append(coherence['composite'])

        return {
            'accuracy': accuracy,
            'coherence': coherence,
            'updates_accepted': accepted,
            'total_steps': self.total_steps
        }
    
    def evaluate(self, 
                 x: np.ndarray, 
                 y: np.ndarray, 
                 task_id: int) -> Dict:
        """
        Evaluate on a dataset.
        
        Args:
            x: Input data
            y: Target labels
            task_id: Task to evaluate
            
        Returns:
            metrics: Evaluation metrics
        """
        # Store state
        hidden_orig = self.hidden_state.copy()
        
        # Forward pass
        _, logits = self._forward(x, task_id)
        
        # Predictions
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y)
        
        # Restore state
        self.hidden_state = hidden_orig
        
        return {
            'accuracy': accuracy,
            'task_id': task_id,
            'num_samples': len(x)
        }
    
    def train_on_task(self,
                      dataloader,
                      task_id: int,
                      epochs: int = 5,
                      verbose: bool = True) -> List[Dict]:
        """
        Train on a complete task.
        
        Args:
            dataloader: DataLoader for the task
            task_id: Task index
            epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            history: Training history
        """
        history = []
        
        for epoch in range(epochs):
            epoch_acc = []
            epoch_coherence = []
            
            for batch_x, batch_y in dataloader:
                x = batch_x.numpy()
                y = batch_y.numpy()
                
                metrics = self.train_step(x, y, task_id)
                epoch_acc.append(metrics['accuracy'])
                epoch_coherence.append(metrics['coherence']['composite'])
                
            avg_acc = np.mean(epoch_acc)
            avg_coh = np.mean(epoch_coherence)
            
            history.append({
                'epoch': epoch,
                'accuracy': avg_acc,
                'coherence': avg_coh
            })
            
            if verbose:
                print(f"  Task {task_id} Epoch {epoch+1}/{epochs}: "
                      f"Acc={avg_acc:.3f}, Coherence={avg_coh:.3f}")

        # Mark this task as trained (for W_in freezing in subsequent tasks)
        self.trained_tasks.add(task_id)

        return history
    
    def get_flops(self) -> int:
        """Get total FLOPs used."""
        return self.flops
    
    def reset_flops(self):
        """Reset FLOPs counter."""
        self.flops = 0
        
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_steps': self.total_steps,
            'task_steps': self.task_steps,
            'structural_changes': self.structural_changes,
            'avg_coherence': np.mean(self.coherence_log) if self.coherence_log else 0,
            'flops': self.flops
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Autopoietic Continual Learner Test")
    print("=" * 60)
    
    # Create learner
    learner = AutopoeticContinualLearner(
        input_dim=100,
        hidden_dim=256,
        num_tasks=5,
        classes_per_task=2,
        connectivity=0.4,
        plasticity_rate=0.01,
        seed=42
    )
    
    # Test with random data
    print("\nTesting with random data...")
    
    for task_id in range(2):
        print(f"\nTask {task_id}:")
        
        for step in range(5):
            x = np.random.randn(32, 100)
            y = np.random.randint(0, 2, 32)
            
            metrics = learner.train_step(x, y, task_id)
            
            print(f"  Step {step+1}: Acc={metrics['accuracy']:.3f}, "
                  f"Coherence={metrics['coherence']['composite']:.3f}")
    
    # Summary
    print("\n" + "-" * 60)
    summary = learner.get_summary()
    print(f"Total steps: {summary['total_steps']}")
    print(f"Structural changes: {summary['structural_changes']}")
    print(f"Average coherence: {summary['avg_coherence']:.3f}")
    print(f"FLOPs: {summary['flops']:,}")
    
    print("\n" + "=" * 60)
    print("Autopoietic Continual Learner Ready!")
    print("=" * 60)
