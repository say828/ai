"""
Split-MNIST Dataset for Continual Learning Experiments

5 sequential tasks:
    Task 0: digits 0, 1
    Task 1: digits 2, 3
    Task 2: digits 4, 5
    Task 3: digits 6, 7
    Task 4: digits 8, 9

Each task is a binary classification problem.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Dict
import os


class SplitMNIST:
    """
    Split-MNIST Dataset Manager
    
    Provides sequential tasks for continual learning experiments.
    """
    
    # Task definitions: which digits belong to each task
    TASK_DIGITS = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9]
    }
    
    def __init__(self, 
                 data_dir: str = './data',
                 use_pca: bool = True,
                 pca_dim: int = 100,
                 batch_size: int = 64,
                 seed: int = 42):
        """
        Args:
            data_dir: Directory to store MNIST data
            use_pca: Whether to apply PCA dimensionality reduction
            pca_dim: Target dimension after PCA
            batch_size: Batch size for DataLoader
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.batch_size = batch_size
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load MNIST
        self._load_mnist()
        
        # Fit PCA if needed
        if use_pca:
            self._fit_pca()
        
        # Create task datasets
        self.task_data = self._create_task_datasets()
        
        print(f"SplitMNIST initialized:")
        print(f"  - 5 tasks, 2 classes each")
        print(f"  - Input dim: {self.pca_dim if use_pca else 784}")
        print(f"  - Batch size: {batch_size}")
        
    def _load_mnist(self):
        """Load and preprocess MNIST dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download/load MNIST
        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        
        # Flatten and convert to numpy
        self.train_data = self.train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        self.train_labels = self.train_dataset.targets.numpy()
        
        self.test_data = self.test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        self.test_labels = self.test_dataset.targets.numpy()
        
        # Normalize
        self.train_mean = self.train_data.mean(axis=0)
        self.train_std = self.train_data.std(axis=0) + 1e-8
        
        self.train_data = (self.train_data - self.train_mean) / self.train_std
        self.test_data = (self.test_data - self.train_mean) / self.train_std
        
    def _fit_pca(self):
        """Fit PCA on training data."""
        from sklearn.decomposition import PCA
        
        print(f"Fitting PCA: 784 -> {self.pca_dim} dimensions...")
        self.pca = PCA(n_components=self.pca_dim, random_state=self.seed)
        self.train_data = self.pca.fit_transform(self.train_data)
        self.test_data = self.pca.transform(self.test_data)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  - Explained variance: {explained_var:.2%}")
        
    def _create_task_datasets(self) -> Dict:
        """Create datasets for each task."""
        task_data = {}
        
        for task_id, digits in self.TASK_DIGITS.items():
            # Train data for this task
            train_mask = np.isin(self.train_labels, digits)
            train_x = self.train_data[train_mask]
            train_y = self.train_labels[train_mask]
            # Convert to binary labels (0 or 1 for this task)
            train_y = (train_y == digits[1]).astype(np.int64)
            
            # Test data for this task
            test_mask = np.isin(self.test_labels, digits)
            test_x = self.test_data[test_mask]
            test_y = self.test_labels[test_mask]
            test_y = (test_y == digits[1]).astype(np.int64)
            
            task_data[task_id] = {
                'train_x': torch.tensor(train_x, dtype=torch.float32),
                'train_y': torch.tensor(train_y, dtype=torch.long),
                'test_x': torch.tensor(test_x, dtype=torch.float32),
                'test_y': torch.tensor(test_y, dtype=torch.long),
                'digits': digits
            }
            
            print(f"  Task {task_id} ({digits}): "
                  f"train={len(train_x)}, test={len(test_x)}")
            
        return task_data
    
    def get_task_dataloader(self, 
                            task_id: int, 
                            train: bool = True,
                            shuffle: bool = None) -> DataLoader:
        """
        Get DataLoader for a specific task.
        
        Args:
            task_id: Task index (0-4)
            train: Whether to get training data
            shuffle: Whether to shuffle (default: True for train, False for test)
            
        Returns:
            DataLoader for the task
        """
        data = self.task_data[task_id]
        
        if train:
            x, y = data['train_x'], data['train_y']
        else:
            x, y = data['test_x'], data['test_y']
            
        if shuffle is None:
            shuffle = train
            
        dataset = torch.utils.data.TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def get_all_test_dataloaders(self) -> List[DataLoader]:
        """Get test dataloaders for all tasks."""
        return [self.get_task_dataloader(i, train=False) for i in range(5)]
    
    def get_input_dim(self) -> int:
        """Get input dimension."""
        return self.pca_dim if self.use_pca else 784
    
    def get_num_tasks(self) -> int:
        """Get number of tasks."""
        return 5
    
    def get_task_info(self, task_id: int) -> Dict:
        """Get information about a specific task."""
        return {
            'task_id': task_id,
            'digits': self.TASK_DIGITS[task_id],
            'num_classes': 2,
            'train_size': len(self.task_data[task_id]['train_x']),
            'test_size': len(self.task_data[task_id]['test_x'])
        }


class ReplayBuffer:
    """
    Experience Replay Buffer for Continual Learning
    
    Stores samples from previous tasks for replay during new task training.
    """
    
    def __init__(self, max_size: int = 200, seed: int = 42):
        """
        Args:
            max_size: Maximum number of samples to store
            seed: Random seed
        """
        self.max_size = max_size
        self.seed = seed
        np.random.seed(seed)
        
        self.samples_x = []
        self.samples_y = []
        self.task_ids = []
        
    def add_task_samples(self, 
                         task_id: int,
                         x: torch.Tensor, 
                         y: torch.Tensor,
                         samples_per_task: int = None):
        """
        Add samples from a task to the buffer.
        
        Args:
            task_id: Task identifier
            x: Input data
            y: Labels
            samples_per_task: Number of samples to store from this task
        """
        if samples_per_task is None:
            # Evenly distribute buffer among tasks
            num_tasks = task_id + 1
            samples_per_task = self.max_size // num_tasks
            
        # Random selection
        n = len(x)
        indices = np.random.choice(n, min(samples_per_task, n), replace=False)
        
        self.samples_x.append(x[indices])
        self.samples_y.append(y[indices])
        self.task_ids.extend([task_id] * len(indices))
        
        # Truncate if over max size
        self._truncate()
        
    def _truncate(self):
        """Truncate buffer if over max size."""
        if len(self.task_ids) <= self.max_size:
            return
            
        # Combine all samples
        all_x = torch.cat(self.samples_x, dim=0)
        all_y = torch.cat(self.samples_y, dim=0)
        all_tasks = np.array(self.task_ids)
        
        # Random selection to keep max_size
        indices = np.random.choice(len(all_x), self.max_size, replace=False)
        
        self.samples_x = [all_x[indices]]
        self.samples_y = [all_y[indices]]
        self.task_ids = all_tasks[indices].tolist()
        
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Get a random batch from the buffer.
        
        Returns:
            x: Input batch
            y: Label batch
            task_ids: Task IDs for each sample
        """
        if not self.samples_x:
            return None, None, None
            
        all_x = torch.cat(self.samples_x, dim=0)
        all_y = torch.cat(self.samples_y, dim=0)
        all_tasks = np.array(self.task_ids)
        
        n = len(all_x)
        indices = np.random.choice(n, min(batch_size, n), replace=False)
        
        return all_x[indices], all_y[indices], all_tasks[indices]
    
    def __len__(self):
        return len(self.task_ids)


if __name__ == "__main__":
    print("=" * 60)
    print("Split-MNIST Dataset Test")
    print("=" * 60)
    
    # Create dataset
    dataset = SplitMNIST(
        data_dir='./data',
        use_pca=True,
        pca_dim=100,
        batch_size=64,
        seed=42
    )
    
    print(f"\nInput dimension: {dataset.get_input_dim()}")
    print(f"Number of tasks: {dataset.get_num_tasks()}")
    
    # Test each task
    print("\n" + "-" * 60)
    for task_id in range(5):
        info = dataset.get_task_info(task_id)
        loader = dataset.get_task_dataloader(task_id, train=True)
        
        # Get one batch
        x, y = next(iter(loader))
        
        print(f"Task {task_id} - Digits {info['digits']}:")
        print(f"  Train size: {info['train_size']}, Test size: {info['test_size']}")
        print(f"  Batch shape: x={x.shape}, y={y.shape}")
        print(f"  Label distribution: {torch.bincount(y).tolist()}")
    
    # Test replay buffer
    print("\n" + "-" * 60)
    print("Replay Buffer Test")
    
    buffer = ReplayBuffer(max_size=200)
    
    for task_id in range(3):
        data = dataset.task_data[task_id]
        buffer.add_task_samples(task_id, data['train_x'], data['train_y'])
        print(f"After task {task_id}: buffer size = {len(buffer)}")
    
    # Get batch
    x, y, tasks = buffer.get_batch(32)
    print(f"Replay batch: x={x.shape}, y={y.shape}, tasks={np.unique(tasks)}")
    
    print("\n" + "=" * 60)
    print("Split-MNIST Dataset Ready!")
    print("=" * 60)
