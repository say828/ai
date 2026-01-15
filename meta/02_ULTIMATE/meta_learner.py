"""
ULTIMATE: Meta-Learner
Experience buffer를 활용한 policy network 학습
"""

import numpy as np
from collections import deque


class ExperienceBuffer:
    """
    Stores (context, weights, improvement) tuples
    """
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, context, weights, improvement):
        """Add experience to buffer"""
        self.buffer.append({
            'context': context.copy(),
            'weights': weights.copy(),
            'improvement': improvement
        })

    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        """Get all experiences"""
        return list(self.buffer)


class MetaLearner:
    """
    Learns policy network from experience buffer

    Training objective:
        Predict weights that led to good improvements
        Loss: weighted MSE (weighted by improvement)
    """
    def __init__(self, policy_network, learning_rate=0.001):
        self.policy_network = policy_network
        self.learning_rate = learning_rate

    def compute_target_weights(self, experiences):
        """
        Compute target weights for each experience

        Strategy:
            - If improvement > 0: Use actual weights (they worked!)
            - If improvement <= 0: Lower weight on those primitives

        Returns:
            List of target weight vectors
        """
        targets = []

        for exp in experiences:
            weights = exp['weights']
            improvement = exp['improvement']

            if improvement > 0:
                # Good weights, use as-is
                target = weights.copy()
            else:
                # Bad weights, dampen them
                target = weights * 0.5
                target = target / (target.sum() + 1e-10)  # Renormalize

            targets.append(target)

        return targets

    def update(self, experience_buffer, n_epochs=10, batch_size=32):
        """
        Update policy network using experience buffer

        Args:
            experience_buffer: ExperienceBuffer object
            n_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if len(experience_buffer) < 10:
            return  # Not enough data

        experiences = experience_buffer.get_all()
        targets = self.compute_target_weights(experiences)

        for epoch in range(n_epochs):
            # Shuffle
            indices = np.random.permutation(len(experiences))

            total_loss = 0.0
            n_batches = 0

            for start_idx in range(0, len(experiences), batch_size):
                end_idx = min(start_idx + batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]

                # Accumulate gradients over batch
                accumulated_grads = {
                    'W1': 0, 'b1': 0,
                    'W2': 0, 'b2': 0,
                    'W3': 0, 'b3': 0
                }

                batch_loss = 0.0

                for idx in batch_indices:
                    exp = experiences[idx]
                    target = targets[idx]

                    context = exp['context']
                    improvement = exp['improvement']

                    # Forward pass
                    predicted_weights = self.policy_network.forward(context)

                    # Compute loss (weighted by improvement)
                    weight_factor = max(0, improvement) + 0.1  # Always positive
                    loss = weight_factor * np.sum((predicted_weights - target) ** 2)
                    batch_loss += loss

                    # Compute gradients
                    grads = self.policy_network.compute_gradient(context, target, predicted_weights)

                    # Accumulate (weighted)
                    for key in accumulated_grads:
                        accumulated_grads[key] += weight_factor * grads[key]

                # Average gradients
                batch_size_actual = end_idx - start_idx
                for key in accumulated_grads:
                    accumulated_grads[key] /= batch_size_actual

                # Update parameters
                self.policy_network.update_parameters(accumulated_grads, lr=self.learning_rate)

                total_loss += batch_loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if epoch % 5 == 0:
                print(f"  Meta-learning epoch {epoch}/{n_epochs}, Loss: {avg_loss:.6f}")

    def evaluate(self, experience_buffer):
        """
        Evaluate policy network on experience buffer

        Returns:
            Average prediction error
        """
        if len(experience_buffer) < 10:
            return float('inf')

        experiences = experience_buffer.get_all()
        targets = self.compute_target_weights(experiences)

        total_error = 0.0

        for exp, target in zip(experiences, targets):
            context = exp['context']
            predicted = self.policy_network.forward(context)
            error = np.sum((predicted - target) ** 2)
            total_error += error

        return total_error / len(experiences)
