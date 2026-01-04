"""
GENESIS: Entity
The core of GENESIS - a self-organizing, self-aware, evolving AI entity

NO LOSS FUNCTION
NO EXPLICIT OBJECTIVE
NO GRADIENT DESCENT

Only: Viability, Intentions, Integration, Evolution
"""

import numpy as np
from typing import Dict, List, Any, Optional
from genesis_models import (
    WorldModel, SelfModel, Genome, Intention,
    Experience, ExperienceBuffer
)


class Phenotype:
    """
    Actual neural structure (expressed from genome)

    Simplified: A flexible neural network that can grow/shrink
    """

    def __init__(self, blueprint: Dict):
        self.blueprint = blueprint
        self.layers = self._build_layers(blueprint['architecture'])
        self.parameters = self._initialize_parameters()

    def _build_layers(self, architecture: Dict) -> List:
        """Build neural layers from blueprint"""
        n_layers = architecture['n_layers']
        layer_sizes = architecture['layer_sizes']

        layers = []
        for i in range(n_layers):
            layers.append({
                'size': layer_sizes[i],
                'weights': None,  # Will be initialized
                'active': True
            })

        return layers

    def _initialize_parameters(self) -> Dict:
        """Initialize weights"""
        params = {}

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_size = 10  # Default input
            else:
                input_size = self.layers[i-1]['size']

            # Simple weight initialization
            params[f'layer_{i}'] = np.random.randn(input_size, layer['size']) * 0.01

        return params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Simplified neural computation
        """
        activation = x

        for i, layer in enumerate(self.layers):
            if not layer['active']:
                continue

            W = self.parameters[f'layer_{i}']

            # Match dimensions
            if len(activation.shape) == 1:
                activation = activation.reshape(1, -1)

            if activation.shape[1] != W.shape[0]:
                # Pad or truncate
                if activation.shape[1] < W.shape[0]:
                    padding = np.zeros((activation.shape[0], W.shape[0] - activation.shape[1]))
                    activation = np.concatenate([activation, padding], axis=1)
                else:
                    activation = activation[:, :W.shape[0]]

            # Linear + activation
            activation = np.dot(activation, W)
            activation = np.tanh(activation)  # Nonlinearity

        return activation.flatten()

    def integrate_experience(self, experience: Experience) -> None:
        """
        Integrate experience into structure

        NOT gradient descent! Holistic transformation
        """
        if not experience.was_successful():
            # Failed experience: small random perturbation
            for key in self.parameters:
                self.parameters[key] += np.random.randn(*self.parameters[key].shape) * 0.001
        else:
            # Successful experience: reinforcement
            # Simplified: slight movement toward recent successful pattern
            for key in self.parameters:
                self.parameters[key] *= 1.001  # Amplify

    def add_module(self, module_spec: Dict) -> None:
        """Add new module to structure"""
        new_layer = {
            'size': module_spec['size'],
            'weights': None,
            'active': True
        }
        self.layers.append(new_layer)

        # Initialize new parameters
        if len(self.layers) > 1:
            prev_size = self.layers[-2]['size']
        else:
            prev_size = 10

        self.parameters[f'layer_{len(self.layers)-1}'] = \
            np.random.randn(prev_size, new_layer['size']) * 0.01

    def remove_module(self, module_idx: int) -> None:
        """Remove obsolete module"""
        if 0 <= module_idx < len(self.layers):
            self.layers[module_idx]['active'] = False


class GENESIS_Entity:
    """
    A self-organizing, self-aware, evolving AI entity

    The revolutionary AI that:
    - Has NO loss function
    - Generates its own objectives
    - Evolves its structure
    - Is self-aware
    - Exists as part of ecosystem
    """

    def __init__(self, genome: Optional[Genome] = None, entity_id: int = 0):
        # Identity
        self.id = entity_id

        # Genetics
        self.genome = genome if genome else Genome()

        # Phenotype (actual structure)
        blueprint = self.genome.express()
        self.phenotype = Phenotype(blueprint)
        self.blueprint = blueprint

        # Models
        self.world_model = WorldModel()
        self.self_model = SelfModel()

        # State
        self.intentions = []
        self.experiences = ExperienceBuffer(
            max_size=self.genome.genes['memory_capacity']
        )

        # Viability (NOT loss!)
        self.viability = 1.0

        # Behavioral traits from genome
        self.curiosity = self.genome.genes['curiosity']
        self.risk_tolerance = self.genome.genes['risk_tolerance']
        self.sociability = self.genome.genes['sociability']

        # History
        self.age = 0
        self.viability_history = []

    def live_one_step(self, environment, ecosystem=None) -> float:
        """
        One cycle of existence

        This is where the magic happens!
        """
        self.age += 1

        # 1. PERCEIVE: Active sensing
        perception = self.perceive(environment)

        # 2. REFLECT: Meta-cognition
        self_awareness = self.reflect()

        # 3. INTEND: Generate objectives (self-generated!)
        self.intentions = self.generate_intentions(
            perception,
            self_awareness,
            ecosystem
        )

        # 4. ACT: Execute intention
        if len(self.intentions) > 0:
            action = self.choose_action(self.intentions[0])
            consequence = environment.apply(action)
        else:
            # No intentions: idle
            consequence = {'idle': True, 'success': False}

        # 5. INTEGRATE: Holistic learning
        self.integrate_experience(perception, action if self.intentions else {}, consequence)

        # 6. INTERACT: Symbiotic exchange (if ecosystem exists)
        if ecosystem is not None:
            self.interact_with_ecosystem(ecosystem)

        # 7. EVALUATE: Am I thriving?
        self.viability = self.assess_viability(environment, ecosystem)
        self.viability_history.append(self.viability)

        # 8. EVOLVE: Structural change (occasionally)
        if self.should_metamorphose():
            self.metamorphose()

        return self.viability

    def perceive(self, environment) -> Dict:
        """
        Active perception: Entity asks questions
        """
        # Generate queries based on curiosity
        queries = []

        # Always sense local
        queries.append({'type': 'sense_local' if hasattr(environment, 'entity_position') else 'random_sample'})

        # If curious, probe more
        if np.random.rand() < self.curiosity:
            queries.append({'type': 'sense_global' if hasattr(environment, 'grid') else 'random_sample'})

        # Probe environment
        responses = []
        for query in queries:
            try:
                response = environment.probe(query)
                responses.append(response)
            except:
                pass

        # Update world model
        self.world_model.update(responses)

        return {
            'raw': responses,
            'interpreted': self.world_model.interpret(responses),
            'novelty': self.world_model.compute_novelty(responses)
        }

    def reflect(self) -> Dict:
        """
        Meta-consciousness: Entity examines itself
        """
        return {
            'capabilities': self.self_model.assess_capabilities(),
            'limitations': self.self_model.recognize_limitations(),
            'knowledge': self.self_model.map_knowledge(),
            'ignorance': self.self_model.map_ignorance(),
            'curiosity': self.self_model.what_am_i_curious_about(),
            'identity': self.self_model.construct_identity(),
            'purpose': self.self_model.infer_purpose(),
            'viability': self.viability
        }

    def generate_intentions(self, perception: Dict, self_awareness: Dict,
                           ecosystem) -> List[Intention]:
        """
        Self-generated objectives

        NO human-defined loss function!
        Entity decides what it wants to do
        """
        intentions = []

        # 1. Survival intentions
        if self.viability < 0.5:
            intentions.append(Intention(
                type='survival',
                goal='increase_viability',
                priority=1.0
            ))

        # 2. Curiosity intentions
        novelty = perception['novelty']
        if novelty > 0.6 and np.random.rand() < self.curiosity:
            intentions.append(Intention(
                type='exploration',
                goal='understand_novelty',
                target=novelty,
                priority=0.8
            ))

        # 3. Growth intentions
        gaps = self_awareness['ignorance']
        if len(gaps) > 0:
            intentions.append(Intention(
                type='growth',
                goal='fill_knowledge_gap',
                target=gaps[0] if gaps else None,
                priority=0.6
            ))

        # 4. Social intentions (if ecosystem exists)
        if ecosystem is not None and np.random.rand() < self.sociability:
            intentions.append(Intention(
                type='symbiosis',
                goal='collaborate',
                priority=0.7
            ))

        # Sort by priority
        intentions.sort(key=lambda x: x.priority, reverse=True)

        return intentions

    def choose_action(self, intention: Intention) -> Dict:
        """
        Convert intention to concrete action
        """
        if intention.type == 'survival':
            # Try to improve viability
            return {
                'type': 'predict' if hasattr(self, 'last_input') else 'explore',
                'intention': 'survival'
            }

        elif intention.type == 'exploration':
            return {
                'type': 'explore',
                'intention': 'exploration'
            }

        elif intention.type == 'growth':
            return {
                'type': 'learn',
                'intention': 'growth',
                'topic': intention.target
            }

        elif intention.type == 'symbiosis':
            return {
                'type': 'interact',
                'intention': 'symbiosis'
            }

        else:
            return {'type': 'idle'}

    def integrate_experience(self, perception: Dict, action: Dict,
                            consequence: Dict) -> None:
        """
        Holistic learning

        NOT gradient descent!
        Experience becomes part of entity's being
        """
        # Create experience
        experience = Experience(
            perception=perception,
            action=action,
            consequence=consequence,
            context={'age': self.age, 'viability': self.viability}
        )

        # Add to memory
        self.experiences.add(experience)

        # Update phenotype (NOT via gradients!)
        self.phenotype.integrate_experience(experience)

        # Update self-model
        self.self_model.update_from_experience(experience)

    def interact_with_ecosystem(self, ecosystem) -> None:
        """
        Symbiotic interactions

        Entities can share knowledge, collaborate, compete
        """
        if not hasattr(ecosystem, 'entities'):
            return

        # Find neighbors
        neighbors = [e for e in ecosystem.entities if e.id != self.id]

        if len(neighbors) == 0:
            return

        # Randomly select neighbor
        neighbor = np.random.choice(neighbors)

        # Simple interaction: share viability strategies
        if neighbor.viability > self.viability:
            # Learn from more successful neighbor
            # Copy some behavioral traits
            self.curiosity = 0.9 * self.curiosity + 0.1 * neighbor.curiosity

    def assess_viability(self, environment, ecosystem) -> float:
        """
        Viability metric (replaces loss function!)

        NOT "how accurate?" but "can I thrive?"
        """
        scores = []

        # 1. Survival: Recent performance
        if len(self.experiences) > 0:
            recent = self.experiences.get_recent(10)
            success_rate = sum(1 for e in recent if e.was_successful()) / len(recent)
            scores.append(success_rate)
        else:
            scores.append(0.5)

        # 2. Growth: Am I learning?
        if len(self.viability_history) > 10:
            recent_trend = np.mean(self.viability_history[-10:])
            scores.append(min(1.0, recent_trend))
        else:
            scores.append(0.5)

        # 3. Adaptability: Can I handle novelty?
        adaptability_score = len(self.self_model.capabilities) / 10.0
        scores.append(min(1.0, adaptability_score))

        # 4. Contribution (if ecosystem exists)
        if ecosystem is not None:
            # Simplified: viability relative to average
            if hasattr(ecosystem, 'entities') and len(ecosystem.entities) > 0:
                avg_viability = np.mean([e.viability for e in ecosystem.entities])
                relative_score = self.viability / (avg_viability + 0.1)
                scores.append(min(1.0, relative_score))
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # Aggregate
        viability = np.mean(scores)

        # Add some noise (life is uncertain)
        viability += np.random.randn() * 0.05
        viability = np.clip(viability, 0.0, 1.0)

        return viability

    def should_metamorphose(self) -> bool:
        """
        Should entity undergo structural evolution?
        """
        # Metamorphose if:
        # 1. Viability is low (need change)
        # 2. Stuck (no improvement)
        # 3. Random exploration

        if self.viability < 0.3:
            return True

        if len(self.viability_history) > 20:
            recent = self.viability_history[-20:]
            if np.std(recent) < 0.05:  # Stuck
                return True

        # Random metamorphosis (exploration)
        if np.random.rand() < 0.01:
            return True

        return False

    def metamorphose(self) -> None:
        """
        Structural evolution

        Architecture itself changes!
        Like caterpillar → butterfly
        """
        print(f"Entity {self.id}: Metamorphosis at age {self.age}!")

        # Mutate genome
        self.genome = self.genome.mutate(mutation_rate=0.2)

        # Re-express
        new_blueprint = self.genome.express()

        # Structural changes
        if len(self.phenotype.layers) < new_blueprint['architecture']['n_layers']:
            # Add module
            self.phenotype.add_module({'size': 16})
            print(f"  Added new module!")

        elif len(self.phenotype.layers) > new_blueprint['architecture']['n_layers']:
            # Remove module
            self.phenotype.remove_module(len(self.phenotype.layers) - 1)
            print(f"  Removed obsolete module!")

        # Update behavioral traits
        self.curiosity = new_blueprint['behavior']['curiosity']
        self.risk_tolerance = new_blueprint['behavior']['risk_tolerance']
        self.sociability = new_blueprint['behavior']['sociability']

    def __repr__(self):
        return (f"Entity(id={self.id}, age={self.age}, viability={self.viability:.2f}, "
                f"modules={len(self.phenotype.layers)})")


if __name__ == "__main__":
    print("Testing GENESIS Entity...")

    from genesis_environment import OpenWorldEnvironment

    # Create entity
    entity = GENESIS_Entity(entity_id=1)
    print(f"Created: {entity}")
    print(f"Genome: {entity.genome}")
    print(f"Curiosity: {entity.curiosity:.2f}")

    # Create environment
    env = OpenWorldEnvironment(size=5)

    # Live for 10 steps
    print("\nLiving for 10 steps:")
    for step in range(10):
        viability = entity.live_one_step(env, ecosystem=None)

        if step % 3 == 0:
            print(f"Step {step}: viability={viability:.3f}, intentions={len(entity.intentions)}")

    print(f"\nFinal: {entity}")
    print(f"Capabilities: {entity.self_model.assess_capabilities()}")

    print("\n✅ Entity test complete!")
