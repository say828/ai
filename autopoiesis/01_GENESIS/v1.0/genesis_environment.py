"""
GENESIS: Environment
- Dynamic environment that entities interact with
- Can be simple (regression) or complex (open world)
"""

import numpy as np
from typing import Dict, List, Any, Optional


class Environment:
    """
    Base environment class
    """

    def probe(self, query: Dict) -> Any:
        """
        Respond to entity's query

        Entity actively asks questions, not passively receives input
        """
        raise NotImplementedError

    def apply(self, action: Dict) -> Dict:
        """
        Apply entity's action and return consequence
        """
        raise NotImplementedError

    def observe_consequence(self) -> Dict:
        """
        Observe consequence of last action
        """
        raise NotImplementedError

    def drift(self):
        """
        Environment changes over time (non-stationarity)
        """
        pass


class RegressionEnvironment(Environment):
    """
    Simple regression environment

    Goal: Learn to predict y from X
    BUT: No explicit loss function given to entity!
    Entity must discover what "success" means
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, noise_level: float = 0.1):
        self.X = X
        self.y = y
        self.noise_level = noise_level

        self.current_idx = 0
        self.last_prediction = None
        self.last_target = None

    def probe(self, query: Dict) -> Any:
        """
        Entity can ask for:
        - Random sample
        - Specific index
        - Pattern queries
        """
        query_type = query.get('type', 'random_sample')

        if query_type == 'random_sample':
            idx = np.random.randint(len(self.X))
            return {
                'input': self.X[idx],
                'feedback': None  # No target given unless action taken
            }

        elif query_type == 'get_index':
            idx = query.get('index', 0)
            idx = max(0, min(idx, len(self.X) - 1))
            return {
                'input': self.X[idx],
                'feedback': None
            }

        elif query_type == 'check_pattern':
            # Entity can check if pattern exists
            pattern = query.get('pattern')
            # Simplified pattern matching
            return {
                'pattern_exists': True,
                'confidence': np.random.rand()
            }

        else:
            return {'error': 'unknown query type'}

    def apply(self, action: Dict) -> Dict:
        """
        Entity takes action (makes prediction)
        """
        action_type = action.get('type', 'predict')

        if action_type == 'predict':
            # Entity makes prediction
            input_data = action.get('input')
            prediction = action.get('prediction')

            # Find corresponding target
            # Simplified: find closest X
            distances = np.linalg.norm(self.X - input_data, axis=1)
            idx = np.argmin(distances)
            target = self.y[idx]

            # Store for consequence
            self.last_prediction = prediction
            self.last_target = target

            return {
                'action_taken': True,
                'prediction': prediction,
                'target': target  # Entity sees target AFTER making prediction
            }

        elif action_type == 'explore':
            # Entity just explores without prediction
            idx = np.random.randint(len(self.X))
            return {
                'input': self.X[idx],
                'target': self.y[idx],
                'exploration': True
            }

        else:
            return {'error': 'unknown action type'}

    def observe_consequence(self) -> Dict:
        """
        Observe consequence of last action
        """
        if self.last_prediction is None or self.last_target is None:
            return {'consequence': None}

        # Calculate error (but don't call it "loss"!)
        error = np.abs(self.last_prediction - self.last_target)

        # Translate to "viability contribution"
        # Good prediction = high viability contribution
        viability_contribution = np.exp(-error)

        return {
            'error': error,
            'viability_contribution': viability_contribution,
            'success': error < 1.0  # Arbitrary success threshold
        }

    def drift(self):
        """Add noise to make environment non-stationary"""
        if np.random.rand() < 0.01:  # 1% chance
            self.y += np.random.randn(*self.y.shape) * self.noise_level


class ClassificationEnvironment(Environment):
    """
    Classification environment
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.last_prediction = None
        self.last_target = None

    def probe(self, query: Dict) -> Any:
        query_type = query.get('type', 'random_sample')

        if query_type == 'random_sample':
            idx = np.random.randint(len(self.X))
            return {'input': self.X[idx]}

        return {'error': 'unknown query'}

    def apply(self, action: Dict) -> Dict:
        if action['type'] == 'classify':
            input_data = action['input']
            prediction = action['prediction']

            # Find target
            distances = np.linalg.norm(self.X - input_data, axis=1)
            idx = np.argmin(distances)
            target = self.y[idx]

            self.last_prediction = prediction
            self.last_target = target

            return {
                'action_taken': True,
                'prediction': prediction,
                'target': target
            }

        return {'error': 'unknown action'}

    def observe_consequence(self) -> Dict:
        if self.last_prediction is None:
            return {'consequence': None}

        correct = (self.last_prediction == self.last_target)

        return {
            'correct': correct,
            'viability_contribution': 1.0 if correct else 0.0,
            'success': correct
        }


class OpenWorldEnvironment(Environment):
    """
    Open-ended environment with no specific task

    Entity must discover what to do on its own!
    """

    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.random.rand(size, size)  # Random landscape
        self.entity_position = [size // 2, size // 2]  # Start in center
        self.resources = np.random.rand(size, size)  # Resources to collect
        self.time = 0

    def probe(self, query: Dict) -> Any:
        """
        Entity can probe environment
        """
        query_type = query.get('type', 'sense_local')

        if query_type == 'sense_local':
            # Sense nearby area
            x, y = self.entity_position
            local_view = self.grid[
                max(0, x-1):min(self.size, x+2),
                max(0, y-1):min(self.size, y+2)
            ]
            local_resources = self.resources[
                max(0, x-1):min(self.size, x+2),
                max(0, y-1):min(self.size, y+2)
            ]

            return {
                'local_view': local_view,
                'local_resources': local_resources,
                'position': self.entity_position.copy()
            }

        elif query_type == 'sense_global':
            # Sense entire environment (costly)
            return {
                'global_view': self.grid.copy(),
                'global_resources': self.resources.copy()
            }

        return {'error': 'unknown query'}

    def apply(self, action: Dict) -> Dict:
        """
        Entity can move, collect, etc.
        """
        action_type = action.get('type', 'idle')

        if action_type == 'move':
            direction = action.get('direction', 'north')

            x, y = self.entity_position

            if direction == 'north':
                x = max(0, x - 1)
            elif direction == 'south':
                x = min(self.size - 1, x + 1)
            elif direction == 'west':
                y = max(0, y - 1)
            elif direction == 'east':
                y = min(self.size - 1, y + 1)

            self.entity_position = [x, y]

            return {
                'action_taken': True,
                'new_position': self.entity_position.copy()
            }

        elif action_type == 'collect':
            # Collect resources at current position
            x, y = self.entity_position
            amount = self.resources[x, y]
            self.resources[x, y] = 0  # Depleted

            return {
                'action_taken': True,
                'collected': amount
            }

        elif action_type == 'idle':
            return {'action_taken': True, 'idle': True}

        return {'error': 'unknown action'}

    def observe_consequence(self) -> Dict:
        """
        Observe world state
        """
        x, y = self.entity_position
        local_resource_density = self.resources[
            max(0, x-1):min(self.size, x+2),
            max(0, y-1):min(self.size, y+2)
        ].mean()

        # Viability contribution: being in resource-rich area is good
        viability_contribution = local_resource_density

        return {
            'viability_contribution': viability_contribution,
            'resource_density': local_resource_density,
            'success': viability_contribution > 0.3
        }

    def drift(self):
        """
        Environment changes over time
        """
        self.time += 1

        # Resources regenerate slowly
        self.resources += np.random.rand(self.size, self.size) * 0.01
        self.resources = np.clip(self.resources, 0, 1)

        # Landscape changes occasionally
        if self.time % 100 == 0:
            self.grid += np.random.randn(self.size, self.size) * 0.1
            self.grid = np.clip(self.grid, 0, 1)


if __name__ == "__main__":
    print("Testing GENESIS Environments...")

    # Test RegressionEnvironment
    print("\n1. RegressionEnvironment")
    X = np.random.randn(100, 2)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1

    env = RegressionEnvironment(X, y)

    # Probe
    response = env.probe({'type': 'random_sample'})
    print(f"Probe response: {response}")

    # Act
    consequence = env.apply({
        'type': 'predict',
        'input': X[0],
        'prediction': 5.0
    })
    print(f"Action consequence: {consequence}")

    # Observe
    observation = env.observe_consequence()
    print(f"Observation: {observation}")

    # Test OpenWorldEnvironment
    print("\n2. OpenWorldEnvironment")
    open_world = OpenWorldEnvironment(size=5)

    response = open_world.probe({'type': 'sense_local'})
    print(f"Sense local: position={response['position']}")

    consequence = open_world.apply({'type': 'move', 'direction': 'north'})
    print(f"Move consequence: {consequence}")

    observation = open_world.observe_consequence()
    print(f"Viability contribution: {observation['viability_contribution']:.3f}")

    print("\n✅ Environments test complete!")
