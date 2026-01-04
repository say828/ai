"""
ULTIMATE: Policy Network
Context → Primitive Weights 매핑을 학습하는 신경망
"""

import numpy as np


class PolicyNetwork:
    """
    Neural network that maps context (12-dim) to primitive weights (10-dim)

    Architecture:
        Input (12) → Hidden (64) → Hidden (32) → Output (10) → Softmax
    """
    def __init__(self, context_dim=12, n_primitives=10, hidden1=64, hidden2=32):
        self.context_dim = context_dim
        self.n_primitives = n_primitives

        # Initialize weights (Xavier initialization)
        scale1 = np.sqrt(2.0 / context_dim)
        scale2 = np.sqrt(2.0 / hidden1)
        scale3 = np.sqrt(2.0 / hidden2)

        self.W1 = np.random.randn(context_dim, hidden1) * scale1
        self.b1 = np.zeros(hidden1)

        self.W2 = np.random.randn(hidden1, hidden2) * scale2
        self.b2 = np.zeros(hidden2)

        self.W3 = np.random.randn(hidden2, n_primitives) * scale3
        self.b3 = np.zeros(n_primitives)

    def forward(self, context):
        """
        Forward pass

        Args:
            context: (12,) numpy array

        Returns:
            weights: (10,) numpy array (softmax output, sums to 1)
        """
        # Layer 1
        h1 = np.dot(context, self.W1) + self.b1
        h1 = np.maximum(0, h1)  # ReLU

        # Layer 2
        h2 = np.dot(h1, self.W2) + self.b2
        h2 = np.maximum(0, h2)  # ReLU

        # Layer 3
        logits = np.dot(h2, self.W3) + self.b3

        # Softmax
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        weights = exp_logits / np.sum(exp_logits)

        return weights

    def __call__(self, context):
        """Shorthand for forward"""
        return self.forward(context)

    def get_parameters(self):
        """Get all parameters as a flat vector"""
        params = []
        params.append(self.W1.flatten())
        params.append(self.b1.flatten())
        params.append(self.W2.flatten())
        params.append(self.b2.flatten())
        params.append(self.W3.flatten())
        params.append(self.b3.flatten())
        return np.concatenate(params)

    def set_parameters(self, params):
        """Set all parameters from a flat vector"""
        idx = 0

        # W1
        size = self.context_dim * 64
        self.W1 = params[idx:idx+size].reshape(self.context_dim, 64)
        idx += size

        # b1
        size = 64
        self.b1 = params[idx:idx+size]
        idx += size

        # W2
        size = 64 * 32
        self.W2 = params[idx:idx+size].reshape(64, 32)
        idx += size

        # b2
        size = 32
        self.b2 = params[idx:idx+size]
        idx += size

        # W3
        size = 32 * self.n_primitives
        self.W3 = params[idx:idx+size].reshape(32, self.n_primitives)
        idx += size

        # b3
        size = self.n_primitives
        self.b3 = params[idx:idx+size]
        idx += size

    def compute_gradient(self, context, target_weights, current_weights):
        """
        Compute gradient of loss w.r.t. policy network parameters

        Loss: weighted MSE between current_weights and target_weights

        Args:
            context: (12,) context vector
            target_weights: (10,) target primitive weights
            current_weights: (10,) current output from forward pass

        Returns:
            grads: dict of gradients for each parameter
        """
        # Forward pass (with caching)
        h1 = np.dot(context, self.W1) + self.b1
        h1_relu = np.maximum(0, h1)

        h2 = np.dot(h1_relu, self.W2) + self.b2
        h2_relu = np.maximum(0, h2)

        logits = np.dot(h2_relu, self.W3) + self.b3

        # Softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        softmax_out = exp_logits / np.sum(exp_logits)

        # Loss gradient w.r.t. softmax output
        # Loss = ||current_weights - target_weights||^2
        d_loss = 2 * (softmax_out - target_weights)

        # Backprop through softmax
        # d_logits = softmax * (d_loss - sum(d_loss * softmax))
        d_logits = softmax_out * (d_loss - np.dot(d_loss, softmax_out))

        # Backprop through layer 3
        d_W3 = np.outer(h2_relu, d_logits)
        d_b3 = d_logits
        d_h2 = np.dot(d_logits, self.W3.T)

        # Backprop through ReLU
        d_h2[h2 <= 0] = 0

        # Backprop through layer 2
        d_W2 = np.outer(h1_relu, d_h2)
        d_b2 = d_h2
        d_h1 = np.dot(d_h2, self.W2.T)

        # Backprop through ReLU
        d_h1[h1 <= 0] = 0

        # Backprop through layer 1
        d_W1 = np.outer(context, d_h1)
        d_b1 = d_h1

        return {
            'W1': d_W1, 'b1': d_b1,
            'W2': d_W2, 'b2': d_b2,
            'W3': d_W3, 'b3': d_b3
        }

    def update_parameters(self, grads, lr=0.001):
        """Update parameters using gradients"""
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
        self.W3 -= lr * grads['W3']
        self.b3 -= lr * grads['b3']
