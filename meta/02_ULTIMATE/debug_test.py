"""
Debug test for ULTIMATE - find the NaN source
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_ultimate import SimpleNetwork, generate_datasets


# Test 1: Can we create a network and compute gradient?
print("Test 1: Network creation and gradient computation")
network = SimpleNetwork()
datasets = generate_datasets()
X, y = datasets['Linear']

print(f"  Initial loss: {network.loss(X, y)}")
grad = network.compute_gradient(X, y)
print(f"  Gradient norm: {np.linalg.norm(grad)}")
print(f"  Gradient has NaN: {np.any(np.isnan(grad))}")
print(f"  Gradient shape: {grad.shape}")

# Test 2: Can we do a simple SGD step?
print("\nTest 2: Simple SGD step")
theta = network.get_weights()
print(f"  Theta has NaN: {np.any(np.isnan(theta))}")
new_theta = theta - 0.01 * grad
print(f"  New theta has NaN: {np.any(np.isnan(new_theta))}")
network.set_weights(new_theta)
new_loss = network.loss(X, y)
print(f"  New loss: {new_loss}")
print(f"  New loss is NaN: {np.isnan(new_loss)}")

# Test 3: Test each primitive individually
print("\nTest 3: Testing each primitive")
from primitives import get_all_primitives
from context import OptimizationContext

network = SimpleNetwork()  # Fresh network
primitives = get_all_primitives()
context = OptimizationContext(max_iterations=200)

primitive_names = [
    'GradientDescent', 'Momentum', 'Adaptive', 'ParticleSwarm', 'BestAttractor',
    'StochasticJump', 'PathSampling', 'ActionGuided', 'MultiScale', 'EnsembleAverage'
]

for i, (primitive, name) in enumerate(zip(primitives, primitive_names)):
    print(f"\n  Testing {name}...")
    network = SimpleNetwork()  # Fresh network for each test
    try:
        update = primitive.compute_update(network, X, y, context)
        print(f"    Update shape: {update.shape}")
        print(f"    Update norm: {np.linalg.norm(update)}")
        print(f"    Update has NaN: {np.any(np.isnan(update))}")

        # Try applying it
        theta = network.get_weights()
        new_theta = theta + update
        network.set_weights(new_theta)
        new_loss = network.loss(X, y)
        print(f"    Loss after update: {new_loss}")
        print(f"    Loss is NaN: {np.isnan(new_loss)}")

    except Exception as e:
        print(f"    ERROR: {e}")

# Test 4: Test policy network
print("\nTest 4: Testing policy network")
from policy_network import PolicyNetwork

policy = PolicyNetwork(context_dim=12, n_primitives=10)
fake_context = np.random.randn(12)
print(f"  Fake context: {fake_context}")
weights = policy.forward(fake_context)
print(f"  Policy weights: {weights}")
print(f"  Weights sum: {np.sum(weights)}")
print(f"  Weights have NaN: {np.any(np.isnan(weights))}")
