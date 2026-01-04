"""
Baseline Methods for Continual Learning Comparison

1. Fine-tuning (SGD): Standard neural network with no forgetting mitigation
2. EWC (Elastic Weight Consolidation): Fisher-based regularization
3. Replay: Experience replay with small buffer

All methods use identical architectures for fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy


class BaselineNetwork(nn.Module):
    """
    Base neural network for continual learning baselines.
    
    Architecture matches AutopoeticContinualLearner for fair comparison:
        - Input: 100 (PCA reduced)
        - Hidden: 256 (with recurrent-like structure)
        - Output: 2 per task
    """
    
    def __init__(self, 
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 num_tasks: int = 5,
                 classes_per_task: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        
        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layer (mimics recurrent structure)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Task-specific output heads
        self.heads = nn.ModuleDict({
            str(i): nn.Linear(hidden_dim, classes_per_task)
            for i in range(num_tasks)
        })
        
        # FLOPs counter
        self.flops = 0
        
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            task_id: Current task
            
        Returns:
            logits: Output logits (batch_size, classes_per_task)
        """
        batch_size = x.shape[0]
        
        # Layer 1
        h = F.relu(self.fc1(x))
        self.flops += batch_size * self.input_dim * self.hidden_dim
        
        # Layer 2
        h = F.relu(self.fc2(h))
        self.flops += batch_size * self.hidden_dim * self.hidden_dim
        
        # Task head
        logits = self.heads[str(task_id)](h)
        self.flops += batch_size * self.hidden_dim * self.classes_per_task
        
        return logits
    
    def get_flops(self) -> int:
        return self.flops
    
    def reset_flops(self):
        self.flops = 0


class FineTuning:
    """
    Fine-tuning baseline (standard SGD).
    
    No forgetting mitigation - represents worst-case forgetting.
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 num_tasks: int = 5,
                 lr: float = 0.01,
                 seed: int = 42):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = BaselineNetwork(input_dim, hidden_dim, num_tasks)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_history = defaultdict(list)
        
        print(f"FineTuning initialized: lr={lr}")
        
    def train_step(self, 
                   x: torch.Tensor, 
                   y: torch.Tensor, 
                   task_id: int) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(x, task_id)
        loss = self.criterion(logits, y)
        
        loss.backward()
        self.optimizer.step()

        # FLOPs for backward pass: approximately 2x forward FLOPs per batch
        batch_flops = x.shape[0] * (self.model.input_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.classes_per_task)
        self.model.flops += batch_flops * 2  # backward is ~2x forward

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()
        
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def train_on_task(self,
                      dataloader,
                      task_id: int,
                      epochs: int = 5,
                      verbose: bool = True) -> List[Dict]:
        """Train on a complete task."""
        history = []
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_acc = []
            
            for batch_x, batch_y in dataloader:
                metrics = self.train_step(batch_x, batch_y, task_id)
                epoch_loss.append(metrics['loss'])
                epoch_acc.append(metrics['accuracy'])
                
            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            
            history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc
            })
            
            if verbose:
                print(f"  Task {task_id} Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")
                
        return history
    
    def evaluate(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 task_id: int) -> Dict:
        """Evaluate on a dataset."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, task_id)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean().item()
            
        return {'accuracy': accuracy, 'task_id': task_id}
    
    def get_flops(self) -> int:
        return self.model.get_flops()


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017)
    
    Protects important weights using Fisher Information.
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 num_tasks: int = 5,
                 lr: float = 0.01,
                 ewc_lambda: float = 400,
                 seed: int = 42):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = BaselineNetwork(input_dim, hidden_dim, num_tasks)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        self.ewc_lambda = ewc_lambda
        
        # Store Fisher and optimal params for each task
        self.fisher = {}
        self.optimal_params = {}
        self.trained_tasks = []
        
        print(f"EWC initialized: lr={lr}, lambda={ewc_lambda}")
        
    def _compute_fisher(self, dataloader, task_id: int):
        """Compute Fisher Information Matrix for a task."""
        self.model.eval()
        
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        num_samples = 0
        for batch_x, batch_y in dataloader:
            self.model.zero_grad()
            
            logits = self.model(batch_x, task_id)
            
            # Use log-likelihood for Fisher
            log_probs = F.log_softmax(logits, dim=1)
            
            for i in range(len(batch_x)):
                self.model.zero_grad()
                log_probs[i, batch_y[i]].backward(retain_graph=True)
                
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                        
                num_samples += 1
                
            if num_samples >= 500:  # Limit samples
                break
                
        # Normalize
        for n in fisher:
            fisher[n] /= num_samples
            
        return fisher
    
    def _ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = 0
        
        for task_id in self.trained_tasks:
            for n, p in self.model.named_parameters():
                loss += (self.fisher[task_id][n] * 
                        (p - self.optimal_params[task_id][n]) ** 2).sum()
                        
        return self.ewc_lambda * loss
    
    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   task_id: int) -> Dict:
        """Single training step with EWC penalty."""
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(x, task_id)
        ce_loss = self.criterion(logits, y)
        
        # Add EWC penalty
        ewc_loss = self._ewc_loss() if self.trained_tasks else torch.tensor(0.0)
        total_loss = ce_loss + ewc_loss
        
        total_loss.backward()
        self.optimizer.step()

        # FLOPs for this step
        batch_flops = x.shape[0] * (self.model.input_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.classes_per_task)
        # Forward + backward + EWC overhead
        ewc_overhead = len(self.trained_tasks) * 50000  # EWC parameter comparison
        self.model.flops += batch_flops * 3 + ewc_overhead

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0,
            'accuracy': accuracy
        }
    
    def train_on_task(self,
                      dataloader,
                      task_id: int,
                      epochs: int = 5,
                      verbose: bool = True) -> List[Dict]:
        """Train on a complete task."""
        history = []
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_acc = []
            
            for batch_x, batch_y in dataloader:
                metrics = self.train_step(batch_x, batch_y, task_id)
                epoch_loss.append(metrics['loss'])
                epoch_acc.append(metrics['accuracy'])
                
            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            
            history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc
            })
            
            if verbose:
                print(f"  Task {task_id} Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")
        
        # After training, compute Fisher and store params
        print(f"  Computing Fisher Information for Task {task_id}...")
        self.fisher[task_id] = self._compute_fisher(dataloader, task_id)
        self.optimal_params[task_id] = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }
        self.trained_tasks.append(task_id)
        
        return history
    
    def evaluate(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 task_id: int) -> Dict:
        """Evaluate on a dataset."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, task_id)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean().item()
            
        return {'accuracy': accuracy, 'task_id': task_id}
    
    def get_flops(self) -> int:
        return self.model.get_flops()


class Replay:
    """
    Experience Replay baseline.
    
    Stores samples from previous tasks and replays during training.
    """
    
    def __init__(self,
                 input_dim: int = 100,
                 hidden_dim: int = 256,
                 num_tasks: int = 5,
                 lr: float = 0.01,
                 buffer_size: int = 200,
                 seed: int = 42):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = BaselineNetwork(input_dim, hidden_dim, num_tasks)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        self.buffer_size = buffer_size
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_task = []
        
        print(f"Replay initialized: lr={lr}, buffer={buffer_size}")
        
    def _add_to_buffer(self, 
                       x: torch.Tensor, 
                       y: torch.Tensor, 
                       task_id: int,
                       samples_per_task: int = None):
        """Add samples to replay buffer."""
        if samples_per_task is None:
            # Evenly distribute
            num_tasks = len(set(self.buffer_task)) + 1
            samples_per_task = self.buffer_size // num_tasks
            
        # Random selection
        n = len(x)
        indices = np.random.choice(n, min(samples_per_task, n), replace=False)
        
        self.buffer_x.append(x[indices])
        self.buffer_y.append(y[indices])
        self.buffer_task.extend([task_id] * len(indices))
        
        # Truncate if over max size
        if len(self.buffer_task) > self.buffer_size:
            all_x = torch.cat(self.buffer_x, dim=0)
            all_y = torch.cat(self.buffer_y, dim=0)
            all_task = np.array(self.buffer_task)
            
            indices = np.random.choice(len(all_x), self.buffer_size, replace=False)
            
            self.buffer_x = [all_x[indices]]
            self.buffer_y = [all_y[indices]]
            self.buffer_task = all_task[indices].tolist()
    
    def _get_replay_batch(self, batch_size: int) -> Tuple:
        """Get a batch from replay buffer."""
        if not self.buffer_x:
            return None, None, None
            
        all_x = torch.cat(self.buffer_x, dim=0)
        all_y = torch.cat(self.buffer_y, dim=0)
        all_task = np.array(self.buffer_task)
        
        n = len(all_x)
        indices = np.random.choice(n, min(batch_size, n), replace=False)
        
        return all_x[indices], all_y[indices], all_task[indices]
    
    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   task_id: int) -> Dict:
        """Single training step with replay."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Current task loss
        logits = self.model(x, task_id)
        loss = self.criterion(logits, y)
        
        # Replay loss
        replay_x, replay_y, replay_tasks = self._get_replay_batch(len(x))
        
        if replay_x is not None:
            for tid in np.unique(replay_tasks):
                mask = replay_tasks == tid
                if mask.sum() > 0:
                    replay_logits = self.model(replay_x[mask], int(tid))
                    loss += self.criterion(replay_logits, replay_y[mask])
        
        loss.backward()
        self.optimizer.step()

        # FLOPs for this step
        batch_flops = x.shape[0] * (self.model.input_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.hidden_dim +
                                     self.model.hidden_dim * self.model.classes_per_task)
        num_replay_tasks = len(np.unique(replay_tasks)) if replay_x is not None else 0
        replay_flops = num_replay_tasks * batch_flops  # Replay forward passes
        self.model.flops += batch_flops * 3 + replay_flops  # forward + backward + replay

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()
        
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def train_on_task(self,
                      dataloader,
                      task_id: int,
                      epochs: int = 5,
                      verbose: bool = True) -> List[Dict]:
        """Train on a complete task."""
        history = []
        
        # Collect data for buffer
        all_x, all_y = [], []
        for batch_x, batch_y in dataloader:
            all_x.append(batch_x)
            all_y.append(batch_y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        for epoch in range(epochs):
            epoch_loss = []
            epoch_acc = []
            
            for batch_x, batch_y in dataloader:
                metrics = self.train_step(batch_x, batch_y, task_id)
                epoch_loss.append(metrics['loss'])
                epoch_acc.append(metrics['accuracy'])
                
            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            
            history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc
            })
            
            if verbose:
                print(f"  Task {task_id} Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")
        
        # Add to replay buffer
        self._add_to_buffer(all_x, all_y, task_id)
        print(f"  Buffer size: {len(self.buffer_task)}")
        
        return history
    
    def evaluate(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 task_id: int) -> Dict:
        """Evaluate on a dataset."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, task_id)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean().item()
            
        return {'accuracy': accuracy, 'task_id': task_id}
    
    def get_flops(self) -> int:
        return self.model.get_flops()


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Methods Test")
    print("=" * 60)
    
    # Test data
    x = torch.randn(32, 100)
    y = torch.randint(0, 2, (32,))
    
    print("\n1. Fine-tuning:")
    ft = FineTuning()
    metrics = ft.train_step(x, y, 0)
    print(f"   Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.3f}")
    
    print("\n2. EWC:")
    ewc = EWC()
    metrics = ewc.train_step(x, y, 0)
    print(f"   Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.3f}")
    
    print("\n3. Replay:")
    replay = Replay()
    metrics = replay.train_step(x, y, 0)
    print(f"   Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.3f}")
    
    print("\n" + "=" * 60)
    print("Baselines Ready!")
    print("=" * 60)
