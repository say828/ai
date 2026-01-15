"""
GENESIS v2.0: Modular Hierarchical Phenotype

Key Features:
- Hierarchical architecture (Shared → Functional → Task-specific)
- Dynamic module addition/removal
- Task routing capability
- Hybrid learning (Gradient + Hebbian)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SharedEncoder:
    """
    Low-level feature extraction layer (task-agnostic)

    Purpose:
    - Extract primitive features from raw input
    - Shared across all tasks
    - Enables positive transfer learning
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # He initialization for ReLU
        self.W = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(hidden_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input (shape: [batch_size, input_size] or [input_size])

        Returns:
            features: Shared features (shape: [batch_size, hidden_size])
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Linear transformation
        z = np.dot(x, self.W) + self.b

        # ReLU activation
        features = np.maximum(0, z)

        return features

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for gradient computation

        Args:
            grad_output: Gradient from next layer
            x: Original input

        Returns:
            grad_W, grad_b, grad_x
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Compute pre-activation
        z = np.dot(x, self.W) + self.b

        # ReLU gradient
        grad_z = grad_output * (z > 0)

        # Parameter gradients
        grad_W = np.dot(x.T, grad_z)
        grad_b = np.sum(grad_z, axis=0)

        # Input gradient
        grad_x = np.dot(grad_z, self.W.T)

        return grad_W, grad_b, grad_x


class FunctionalModule:
    """
    Base class for functional modules

    Functional modules are reusable components that capture
    specific types of patterns (linear, nonlinear, interactions, etc.)
    """

    def __init__(self, input_size: int, output_size: int, module_type: str):
        self.input_size = input_size
        self.output_size = output_size
        self.module_type = module_type

        # He initialization
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

        # Hebbian pathway strength
        self.pathway_strength = np.ones_like(self.W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass - to be implemented by subclasses"""
        raise NotImplementedError

    def hebbian_update(self, x: np.ndarray, output: np.ndarray, success: bool):
        """
        Hebbian-like pathway strengthening

        "Neurons that fire together, wire together"
        """
        if success:
            # Strengthen successful pathways
            activity = np.abs(np.outer(x.flatten(), output.flatten()))
            strength_update = 0.01 * activity * self.pathway_strength
            self.W += strength_update
            self.pathway_strength *= 1.01
            self.pathway_strength = np.clip(self.pathway_strength, 0.5, 2.0)
        else:
            # Mildly weaken failed pathways
            self.pathway_strength *= 0.99
            self.pathway_strength = np.clip(self.pathway_strength, 0.5, 2.0)


class LinearModule(FunctionalModule):
    """
    Captures linear relationships

    y = Wx + b
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, module_type='linear')

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        grad_W = np.dot(x.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        grad_x = np.dot(grad_output, self.W.T)

        return grad_W, grad_b, grad_x


class NonlinearModule(FunctionalModule):
    """
    Captures nonlinear relationships

    y = tanh(Wx + b)
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, module_type='nonlinear')

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        z = np.dot(x, self.W) + self.b
        return np.tanh(z)

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Forward to get activation
        z = np.dot(x, self.W) + self.b
        activation = np.tanh(z)

        # tanh gradient
        grad_z = grad_output * (1 - activation ** 2)

        grad_W = np.dot(x.T, grad_z)
        grad_b = np.sum(grad_z, axis=0)
        grad_x = np.dot(grad_z, self.W.T)

        return grad_W, grad_b, grad_x


class InteractionModule(FunctionalModule):
    """
    Captures multiplicative interactions

    Computes pairwise feature interactions
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, module_type='interaction')

        # For interaction, we need different weight structure
        # W captures which interaction pairs to use
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Compute pairwise interactions
        # For simplicity, use weighted sum of feature products
        interactions = []
        for i in range(x.shape[1]):
            for j in range(i+1, x.shape[1]):
                interactions.append(x[:, i] * x[:, j])

        if len(interactions) == 0:
            # Fallback: squared features
            interactions = [x[:, i] ** 2 for i in range(x.shape[1])]

        interaction_features = np.column_stack(interactions) if len(interactions) > 1 else np.array(interactions).T

        # Project to output size
        if interaction_features.shape[1] > self.input_size:
            interaction_features = interaction_features[:, :self.input_size]
        elif interaction_features.shape[1] < self.input_size:
            padding = np.zeros((interaction_features.shape[0], self.input_size - interaction_features.shape[1]))
            interaction_features = np.column_stack([interaction_features, padding])

        output = np.dot(interaction_features, self.W) + self.b
        return output

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Simplified backward (approximate gradient)
        # Full derivation is complex for interactions
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Approximate gradient through interactions
        grad_W = np.dot(x.T, grad_output)
        if grad_W.shape != self.W.shape:
            grad_W = grad_W[:self.W.shape[0], :self.W.shape[1]]

        grad_b = np.sum(grad_output, axis=0)
        grad_x = np.dot(grad_output, self.W.T[:x.shape[1], :])

        return grad_W, grad_b, grad_x


class TaskHead:
    """
    Task-specific output layer

    Maps module outputs to task predictions
    """

    def __init__(self, input_size: int, output_size: int, task_id: str):
        self.input_size = input_size
        self.output_size = output_size
        self.task_id = task_id

        # Xavier initialization for linear output
        self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        grad_W = np.dot(x.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        grad_x = np.dot(grad_output, self.W.T)

        return grad_W, grad_b, grad_x


class ModularPhenotype_v2_0:
    """
    GENESIS v2.0: Hierarchical Modular Phenotype

    Architecture:
    Input → SharedEncoder → FunctionalModules → TaskHead → Output
           (task-agnostic)  (reusable)         (task-specific)
    """

    def __init__(self, input_size: int = 10, shared_size: int = 32, module_size: int = 16):
        self.input_size = input_size
        self.shared_size = shared_size
        self.module_size = module_size

        # Layer 1: Shared encoder
        self.shared_encoder = SharedEncoder(input_size, shared_size)

        # Layer 2: Functional modules
        self.modules: Dict[str, FunctionalModule] = {
            'linear': LinearModule(shared_size, module_size),
            'nonlinear': NonlinearModule(shared_size, module_size),
            'interaction': InteractionModule(shared_size, module_size)
        }

        # Layer 3: Task-specific heads
        self.task_heads: Dict[str, TaskHead] = {}

        # Activation tracking (for Hebbian learning)
        self.last_shared_features = None
        self.last_module_outputs = {}
        self.last_active_modules = []

    def forward(self, x: np.ndarray, task_id: str, active_modules: Optional[List[str]] = None) -> np.ndarray:
        """
        Hierarchical forward pass

        Args:
            x: Input features
            task_id: Task identifier
            active_modules: Which modules to use (if None, use all)

        Returns:
            prediction: Task output
        """
        # 1. Shared encoding
        shared_features = self.shared_encoder.forward(x)
        self.last_shared_features = shared_features

        # 2. Functional modules
        if active_modules is None:
            active_modules = list(self.modules.keys())

        module_outputs = []
        self.last_module_outputs = {}
        for module_name in active_modules:
            if module_name in self.modules:
                output = self.modules[module_name].forward(shared_features)
                module_outputs.append(output)
                self.last_module_outputs[module_name] = output

        self.last_active_modules = active_modules

        # 3. Concatenate module outputs
        if len(module_outputs) > 0:
            combined = np.concatenate(module_outputs, axis=1)
        else:
            combined = shared_features

        # 4. Task-specific head
        if task_id not in self.task_heads:
            # Create new task head
            self.add_task(task_id, input_size=combined.shape[1])

        prediction = self.task_heads[task_id].forward(combined)

        return prediction

    def backward(self, grad_output: np.ndarray, x: np.ndarray, task_id: str, learning_rate: float = 0.01):
        """
        Backward pass for gradient-based learning

        Args:
            grad_output: Gradient from loss
            x: Original input
            task_id: Task identifier
            learning_rate: Step size
        """
        # 1. Task head gradient
        combined = np.concatenate([self.last_module_outputs[m] for m in self.last_active_modules], axis=1) \
            if len(self.last_module_outputs) > 0 else self.last_shared_features

        grad_W_head, grad_b_head, grad_combined = self.task_heads[task_id].backward(grad_output, combined)

        # Gradient clipping (moderate)
        grad_W_head = np.clip(grad_W_head, -5.0, 5.0)
        grad_b_head = np.clip(grad_b_head, -5.0, 5.0)
        grad_combined = np.clip(grad_combined, -5.0, 5.0)

        # Update task head
        self.task_heads[task_id].W -= learning_rate * grad_W_head
        self.task_heads[task_id].b -= learning_rate * grad_b_head

        # 2. Module gradients
        grad_shared_total = np.zeros_like(self.last_shared_features)

        split_sizes = [self.last_module_outputs[m].shape[1] for m in self.last_active_modules]
        grad_splits = np.split(grad_combined, np.cumsum(split_sizes)[:-1], axis=1)

        for module_name, grad_module in zip(self.last_active_modules, grad_splits):
            grad_W_mod, grad_b_mod, grad_shared = self.modules[module_name].backward(
                grad_module, self.last_shared_features
            )

            # Gradient clipping (moderate)
            grad_W_mod = np.clip(grad_W_mod, -5.0, 5.0)
            grad_b_mod = np.clip(grad_b_mod, -5.0, 5.0)
            grad_shared = np.clip(grad_shared, -5.0, 5.0)

            # Update module
            self.modules[module_name].W -= learning_rate * grad_W_mod
            self.modules[module_name].b -= learning_rate * grad_b_mod

            grad_shared_total += grad_shared

        # 3. Shared encoder gradient
        grad_W_enc, grad_b_enc, grad_x = self.shared_encoder.backward(grad_shared_total, x)

        # Gradient clipping (moderate)
        grad_W_enc = np.clip(grad_W_enc, -5.0, 5.0)
        grad_b_enc = np.clip(grad_b_enc, -5.0, 5.0)

        # Update shared encoder
        self.shared_encoder.W -= learning_rate * grad_W_enc
        self.shared_encoder.b -= learning_rate * grad_b_enc

    def hebbian_update(self, task_id: str, success: bool):
        """
        Hebbian consolidation of successful pathways

        Args:
            task_id: Task identifier
            success: Whether the prediction was successful
        """
        for module_name in self.last_active_modules:
            if module_name in self.modules:
                self.modules[module_name].hebbian_update(
                    x=self.last_shared_features,
                    output=self.last_module_outputs[module_name],
                    success=success
                )

    def add_task(self, task_id: str, input_size: Optional[int] = None, output_size: int = 1):
        """
        Add new task-specific head

        Args:
            task_id: Task identifier
            input_size: Size of combined module outputs
            output_size: Task output dimension
        """
        if input_size is None:
            input_size = len(self.modules) * self.module_size

        self.task_heads[task_id] = TaskHead(input_size, output_size, task_id)

    def add_module(self, module_name: str, module_type: str = 'linear'):
        """
        Dynamically add new functional module

        Args:
            module_name: Name for the new module
            module_type: Type (linear, nonlinear, interaction)
        """
        if module_type == 'linear':
            self.modules[module_name] = LinearModule(self.shared_size, self.module_size)
        elif module_type == 'nonlinear':
            self.modules[module_name] = NonlinearModule(self.shared_size, self.module_size)
        elif module_type == 'interaction':
            self.modules[module_name] = InteractionModule(self.shared_size, self.module_size)
        else:
            raise ValueError(f"Unknown module type: {module_type}")

    def get_module_list(self) -> List[str]:
        """Get list of available modules"""
        return list(self.modules.keys())

    def get_task_list(self) -> List[str]:
        """Get list of known tasks"""
        return list(self.task_heads.keys())

    def __repr__(self):
        return (f"ModularPhenotype_v2_0("
                f"input={self.input_size}, shared={self.shared_size}, "
                f"modules={len(self.modules)}, tasks={len(self.task_heads)})")


if __name__ == "__main__":
    print("Testing ModularPhenotype_v2_0...")

    # Create phenotype
    phenotype = ModularPhenotype_v2_0(input_size=2, shared_size=8, module_size=4)
    print(f"Created: {phenotype}")

    # Test forward pass
    x = np.array([1.0, 2.0])
    task_id = "task_1"

    prediction = phenotype.forward(x, task_id)
    print(f"\nForward pass:")
    print(f"  Input: {x}")
    print(f"  Prediction: {prediction}")

    # Test backward pass
    target = np.array([[5.0]])
    error = prediction - target
    grad_output = 2 * error  # MSE gradient

    phenotype.backward(grad_output, x, task_id, learning_rate=0.01)
    print(f"\nBackward pass completed")

    # Test Hebbian update
    success = (np.abs(error) < 1.0)
    phenotype.hebbian_update(task_id, success)
    print(f"Hebbian update: success={success}")

    # Test adding new task
    task_2 = "task_2"
    prediction_2 = phenotype.forward(x, task_2)
    print(f"\nNew task '{task_2}' created:")
    print(f"  Prediction: {prediction_2}")

    # Test adding new module
    phenotype.add_module("custom_linear", module_type='linear')
    print(f"\nAdded custom module: {phenotype.get_module_list()}")

    print(f"\n✅ ModularPhenotype_v2_0 test complete!")
    print(f"Final state: {phenotype}")
