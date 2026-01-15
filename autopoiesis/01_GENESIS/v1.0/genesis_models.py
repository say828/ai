"""
GENESIS: Foundational Models
- WorldModel: Entity's understanding of environment
- SelfModel: Entity's self-awareness
- Genome: Genetic encoding of entity
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional


class WorldModel:
    """
    Entity's internal model of the environment

    NOT a passive representation, but active understanding
    - Tracks patterns
    - Detects novelty
    - Makes predictions
    """

    def __init__(self):
        self.observations = []
        self.patterns = defaultdict(int)
        self.predictions = {}
        self.novelty_threshold = 0.5

    def update(self, observations: List[Any]):
        """Update world model with new observations"""
        for obs in observations:
            self.observations.append(obs)

            # Track patterns (simplified)
            pattern_key = self._extract_pattern(obs)
            self.patterns[pattern_key] += 1

    def interpret(self, observations: List[Any]) -> Dict:
        """Interpret raw observations"""
        return {
            'observations': observations,
            'inferred_state': self._infer_state(observations),
            'predicted_next': self._predict_next()
        }

    def compute_novelty(self, observations: List[Any]) -> float:
        """
        How novel/surprising are these observations?

        High novelty = worth exploring
        Low novelty = already understand
        """
        novelty_scores = []

        for obs in observations:
            pattern_key = self._extract_pattern(obs)

            # Novel if we haven't seen this pattern much
            frequency = self.patterns.get(pattern_key, 0)
            novelty = 1.0 / (1.0 + frequency)
            novelty_scores.append(novelty)

        return np.mean(novelty_scores) if novelty_scores else 0.5

    def _extract_pattern(self, obs) -> str:
        """Extract pattern from observation (simplified)"""
        if isinstance(obs, (int, float)):
            # Discretize continuous values
            return f"range_{int(obs // 10) * 10}"
        elif isinstance(obs, np.ndarray):
            # Hash array to pattern
            return f"array_mean_{obs.mean():.2f}"
        else:
            return str(type(obs).__name__)

    def _infer_state(self, observations: List[Any]) -> str:
        """Infer current world state"""
        if not observations:
            return "unknown"

        # Simplified state inference
        avg_novelty = self.compute_novelty(observations)

        if avg_novelty > 0.7:
            return "highly_novel"
        elif avg_novelty > 0.3:
            return "moderately_novel"
        else:
            return "familiar"

    def _predict_next(self) -> Optional[Any]:
        """Predict what comes next (simplified)"""
        if len(self.observations) < 2:
            return None

        # Return most common pattern
        most_common = max(self.patterns.items(), key=lambda x: x[1])
        return most_common[0]


class SelfModel:
    """
    Entity's model of itself (Meta-consciousness)

    Key aspects:
    - What can I do?
    - What can't I do?
    - What do I know?
    - What don't I know?
    - Who am I?
    """

    def __init__(self):
        self.capabilities = set()
        self.limitations = set()
        self.knowledge_map = defaultdict(float)  # topic -> confidence
        self.identity_traits = []
        self.success_history = []

    def assess_capabilities(self) -> List[str]:
        """What can I do?"""
        return list(self.capabilities)

    def recognize_limitations(self) -> List[str]:
        """What can't I do?"""
        return list(self.limitations)

    def map_knowledge(self) -> Dict[str, float]:
        """What do I know? (with confidence levels)"""
        # Return high-confidence knowledge
        return {
            topic: confidence
            for topic, confidence in self.knowledge_map.items()
            if confidence > 0.5
        }

    def map_ignorance(self) -> List[str]:
        """What don't I know?"""
        # Return low-confidence areas
        return [
            topic
            for topic, confidence in self.knowledge_map.items()
            if confidence < 0.3
        ]

    def what_am_i_curious_about(self) -> List[str]:
        """Generate curiosity-driven goals"""
        curiosities = []

        # Curious about areas with medium confidence (learning edge)
        for topic, confidence in self.knowledge_map.items():
            if 0.3 <= confidence <= 0.7:
                curiosities.append(topic)

        return curiosities

    def construct_identity(self) -> str:
        """Who am I?"""
        if not self.identity_traits:
            return "undefined"

        return ", ".join(self.identity_traits)

    def infer_purpose(self) -> str:
        """Why do I exist?"""
        if not self.success_history:
            return "seeking purpose"

        # Analyze what I've been successful at
        successes = [s for s in self.success_history if s['success']]

        if not successes:
            return "learning to survive"

        # Find common patterns in successes
        most_common_type = max(
            set(s['type'] for s in successes),
            key=lambda t: sum(1 for s in successes if s['type'] == t)
        )

        return f"specialized in {most_common_type}"

    def update_from_experience(self, experience):
        """Learn about self from experience"""
        # Update capabilities
        if experience.was_successful():
            self.capabilities.add(experience.action_type)
            self.success_history.append({
                'type': experience.action_type,
                'success': True
            })
        else:
            self.limitations.add(experience.action_type)
            self.success_history.append({
                'type': experience.action_type,
                'success': False
            })

        # Update knowledge map
        topic = experience.get_topic()
        if topic:
            current_confidence = self.knowledge_map[topic]
            # Increase confidence on success, decrease on failure
            delta = 0.1 if experience.was_successful() else -0.05
            self.knowledge_map[topic] = np.clip(
                current_confidence + delta,
                0.0,
                1.0
            )


class Genome:
    """
    Genetic encoding of entity

    Encodes:
    - Architecture blueprint
    - Behavioral predispositions
    - Learning strategies
    """

    def __init__(self, genes: Optional[Dict] = None):
        if genes is None:
            # Random initialization
            self.genes = self._random_genes()
        else:
            self.genes = genes

    def _random_genes(self) -> Dict:
        """Generate random genome"""
        return {
            # Architecture genes
            'n_modules': np.random.randint(2, 8),
            'module_sizes': np.random.randint(8, 64, size=5).tolist(),
            'connectivity': np.random.rand(),

            # Behavioral genes
            'curiosity': np.random.rand(),
            'risk_tolerance': np.random.rand(),
            'sociability': np.random.rand(),

            # Learning genes
            'plasticity': np.random.rand(),
            'memory_capacity': np.random.randint(100, 1000),
            'learning_rate': 0.001 + np.random.rand() * 0.01
        }

    def mutate(self, mutation_rate: float = 0.1) -> 'Genome':
        """Create mutated copy"""
        new_genes = self.genes.copy()

        for key, value in new_genes.items():
            if np.random.rand() < mutation_rate:
                if isinstance(value, int):
                    new_genes[key] = max(1, value + np.random.randint(-2, 3))
                elif isinstance(value, float):
                    new_genes[key] = np.clip(
                        value + np.random.randn() * 0.1,
                        0.0,
                        1.0
                    )
                elif isinstance(value, list):
                    idx = np.random.randint(len(value))
                    new_genes[key][idx] = max(1, value[idx] + np.random.randint(-5, 6))

        return Genome(new_genes)

    def crossover(self, other: 'Genome') -> 'Genome':
        """Sexual reproduction: combine two genomes"""
        new_genes = {}

        for key in self.genes.keys():
            # Randomly inherit from either parent
            if np.random.rand() < 0.5:
                new_genes[key] = self.genes[key]
            else:
                new_genes[key] = other.genes[key]

        return Genome(new_genes)

    def express(self) -> Dict:
        """Express genome into phenotype (actual structure)"""
        # This would create actual neural architecture
        # Simplified version returns blueprint
        return {
            'architecture': {
                'n_layers': self.genes['n_modules'],
                'layer_sizes': self.genes['module_sizes'][:self.genes['n_modules']],
                'connectivity': self.genes['connectivity']
            },
            'behavior': {
                'curiosity': self.genes['curiosity'],
                'risk_tolerance': self.genes['risk_tolerance'],
                'sociability': self.genes['sociability']
            },
            'learning': {
                'plasticity': self.genes['plasticity'],
                'memory_capacity': self.genes['memory_capacity'],
                'learning_rate': self.genes['learning_rate']
            }
        }

    def __repr__(self):
        return f"Genome(modules={self.genes['n_modules']}, curiosity={self.genes['curiosity']:.2f})"


class Intention:
    """
    Self-generated goal/objective
    """

    def __init__(self, type: str, goal: str, priority: float, target: Any = None):
        self.type = type  # 'survival', 'exploration', 'growth', 'symbiosis'
        self.goal = goal
        self.priority = priority
        self.target = target
        self.created_at = 0  # Would be timestamp

    def __repr__(self):
        return f"Intention({self.type}: {self.goal}, priority={self.priority:.2f})"


class Experience:
    """
    Record of perception → action → consequence
    """

    def __init__(self, perception, action, consequence, context=None):
        self.perception = perception
        self.action = action
        self.consequence = consequence
        self.context = context or {}
        self.action_type = action.get('type', 'unknown') if isinstance(action, dict) else 'unknown'

    def was_successful(self) -> bool:
        """Was this experience successful?"""
        if isinstance(self.consequence, dict):
            return self.consequence.get('success', False)
        return False

    def get_topic(self) -> Optional[str]:
        """What topic does this experience relate to?"""
        if isinstance(self.action, dict):
            return self.action.get('topic', None)
        return None

    def __repr__(self):
        success = "✓" if self.was_successful() else "✗"
        return f"Experience({self.action_type} {success})"


class ExperienceBuffer:
    """
    Memory of experiences
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences = []

    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.experiences.append(experience)

        # Limit size
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def sample(self, n: int = 32) -> List[Experience]:
        """Sample random experiences"""
        if len(self.experiences) <= n:
            return self.experiences.copy()

        indices = np.random.choice(len(self.experiences), n, replace=False)
        return [self.experiences[i] for i in indices]

    def get_recent(self, n: int = 10) -> List[Experience]:
        """Get recent experiences"""
        return self.experiences[-n:]

    def __len__(self):
        return len(self.experiences)


if __name__ == "__main__":
    # Test
    print("Testing GENESIS Models...")

    # WorldModel
    world = WorldModel()
    world.update([1.5, 2.3, 15.7])
    novelty = world.compute_novelty([100.0])
    print(f"Novelty: {novelty:.2f}")

    # SelfModel
    self_model = SelfModel()
    self_model.capabilities.add("explore")
    self_model.knowledge_map["math"] = 0.8
    self_model.knowledge_map["physics"] = 0.2
    print(f"Capabilities: {self_model.assess_capabilities()}")
    print(f"Ignorance: {self_model.map_ignorance()}")

    # Genome
    genome = Genome()
    print(f"Genome: {genome}")
    phenotype = genome.express()
    print(f"Phenotype: {phenotype['architecture']}")

    # Mutation
    mutated = genome.mutate()
    print(f"Mutated: {mutated}")

    print("\n✅ Models test complete!")
