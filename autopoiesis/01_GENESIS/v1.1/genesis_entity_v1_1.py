"""
GENESIS v1.1: Improved Entity

Phase 2 improvements:
1. Viability ↔ Performance connection
2. Hebbian-like integration
3. Controlled metamorphosis
"""

import numpy as np
from typing import Dict, List, Any, Optional
from genesis_models import (
    WorldModel, SelfModel, Genome, Intention,
    Experience, ExperienceBuffer
)


class Phenotype_v1_1:
    """
    Improved Phenotype with Hebbian-like integration
    """

    def __init__(self, blueprint: Dict):
        self.blueprint = blueprint
        self.layers = self._build_layers(blueprint['architecture'])
        self.parameters = self._initialize_parameters()

        # v1.1: Track successful pathways
        self.pathway_strengths = {}
        for key in self.parameters:
            self.pathway_strengths[key] = np.ones_like(self.parameters[key])

    def _build_layers(self, architecture: Dict) -> List:
        """Build neural layers from blueprint"""
        n_layers = architecture['n_layers']
        layer_sizes = architecture['layer_sizes']

        layers = []
        for i in range(n_layers):
            # Handle case where layer_sizes is shorter than n_layers
            if i < len(layer_sizes):
                size = layer_sizes[i]
            else:
                size = 16  # Default size

            layers.append({
                'size': size,
                'weights': None,
                'active': True
            })

        return layers

    def _initialize_parameters(self) -> Dict:
        """Initialize weights"""
        params = {}

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_size = 10
            else:
                input_size = self.layers[i-1]['size']

            params[f'layer_{i}'] = np.random.randn(input_size, layer['size']) * 0.01

        return params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        activation = x

        for i, layer in enumerate(self.layers):
            if not layer['active']:
                continue

            W = self.parameters[f'layer_{i}']

            if len(activation.shape) == 1:
                activation = activation.reshape(1, -1)

            if activation.shape[1] != W.shape[0]:
                if activation.shape[1] < W.shape[0]:
                    padding = np.zeros((activation.shape[0], W.shape[0] - activation.shape[1]))
                    activation = np.concatenate([activation, padding], axis=1)
                else:
                    activation = activation[:, :W.shape[0]]

            activation = np.dot(activation, W)
            activation = np.tanh(activation)

        return activation.flatten()

    def integrate_experience(self, experience: Experience) -> None:
        """
        v1.1: Hebbian-like integration

        "Neurons that fire together, wire together"
        Successful pathways get strengthened
        """
        success = experience.was_successful()

        if success:
            # v1.1: Strengthen successful pathways (Hebbian)
            for key in self.parameters:
                # Get activity (simplified: use parameter magnitudes as proxy)
                activity = np.abs(self.parameters[key])

                # Strengthen active connections
                strength_update = 0.01 * activity * self.pathway_strengths[key]
                self.parameters[key] += strength_update

                # Increase pathway strength
                self.pathway_strengths[key] *= 1.01
                self.pathway_strengths[key] = np.clip(self.pathway_strengths[key], 0.5, 2.0)
        else:
            # v1.1: Weaken failed pathways (anti-Hebbian)
            for key in self.parameters:
                # Small random perturbation + pathway weakening
                self.parameters[key] += np.random.randn(*self.parameters[key].shape) * 0.001

                # Decrease pathway strength slightly
                self.pathway_strengths[key] *= 0.99
                self.pathway_strengths[key] = np.clip(self.pathway_strengths[key], 0.5, 2.0)

    def add_module(self, module_spec: Dict) -> None:
        """Add new module to structure"""
        new_layer = {
            'size': module_spec['size'],
            'weights': None,
            'active': True
        }
        self.layers.append(new_layer)

        if len(self.layers) > 1:
            prev_size = self.layers[-2]['size']
        else:
            prev_size = 10

        new_key = f'layer_{len(self.layers)-1}'
        self.parameters[new_key] = np.random.randn(prev_size, new_layer['size']) * 0.01
        self.pathway_strengths[new_key] = np.ones((prev_size, new_layer['size']))

    def remove_module(self, module_idx: int) -> None:
        """Remove obsolete module"""
        if 0 <= module_idx < len(self.layers):
            self.layers[module_idx]['active'] = False


class GENESIS_Entity_v1_1:
    """
    GENESIS v1.1: Improved entity

    Phase 2 improvements:
    - Better viability metric (connected to performance)
    - Hebbian-like integration
    - Controlled metamorphosis
    """

    def __init__(self, genome: Optional[Genome] = None, entity_id: int = 0):
        self.id = entity_id
        self.genome = genome if genome else Genome()

        blueprint = self.genome.express()
        self.phenotype = Phenotype_v1_1(blueprint)  # v1.1!
        self.blueprint = blueprint

        self.world_model = WorldModel()
        self.self_model = SelfModel()

        self.intentions = []
        self.experiences = ExperienceBuffer(
            max_size=self.genome.genes['memory_capacity']
        )

        self.viability = 1.0

        self.curiosity = self.genome.genes['curiosity']
        self.risk_tolerance = self.genome.genes['risk_tolerance']
        self.sociability = self.genome.genes['sociability']

        self.age = 0
        self.viability_history = []

        # v1.1: Track environment feedback
        self.recent_feedback = []

    def live_one_step(self, environment, ecosystem=None) -> float:
        """One cycle of existence"""
        self.age += 1

        # 1. PERCEIVE
        perception = self.perceive(environment)

        # 2. REFLECT
        self_awareness = self.reflect()

        # 3. INTEND
        self.intentions = self.generate_intentions(
            perception,
            self_awareness,
            ecosystem
        )

        # 4. ACT
        if len(self.intentions) > 0:
            action = self.choose_action(self.intentions[0])
            consequence = environment.apply(action)
        else:
            consequence = {'idle': True, 'success': False}

        # v1.1: Store environment feedback
        if 'viability_contribution' in consequence:
            self.recent_feedback.append(consequence['viability_contribution'])
            if len(self.recent_feedback) > 20:
                self.recent_feedback.pop(0)

        # 5. INTEGRATE
        self.integrate_experience(perception, action if self.intentions else {}, consequence)

        # 6. INTERACT
        if ecosystem is not None:
            self.interact_with_ecosystem(ecosystem)

        # 7. EVALUATE (v1.1: improved!)
        self.viability = self.assess_viability(environment, ecosystem)
        self.viability_history.append(self.viability)

        # 8. EVOLVE (v1.1: controlled!)
        if self.should_metamorphose():
            self.metamorphose()

        return self.viability

    def perceive(self, environment) -> Dict:
        """Active perception"""
        queries = []

        queries.append({'type': 'sense_local' if hasattr(environment, 'entity_position') else 'random_sample'})

        if np.random.rand() < self.curiosity:
            queries.append({'type': 'sense_global' if hasattr(environment, 'grid') else 'random_sample'})

        responses = []
        for query in queries:
            try:
                response = environment.probe(query)
                responses.append(response)
            except:
                pass

        self.world_model.update(responses)

        return {
            'raw': responses,
            'interpreted': self.world_model.interpret(responses),
            'novelty': self.world_model.compute_novelty(responses)
        }

    def reflect(self) -> Dict:
        """Meta-consciousness"""
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
        """Self-generated objectives"""
        intentions = []

        if self.viability < 0.5:
            intentions.append(Intention(
                type='survival',
                goal='increase_viability',
                priority=1.0
            ))

        novelty = perception['novelty']
        if novelty > 0.6 and np.random.rand() < self.curiosity:
            intentions.append(Intention(
                type='exploration',
                goal='understand_novelty',
                target=novelty,
                priority=0.8
            ))

        gaps = self_awareness['ignorance']
        if len(gaps) > 0:
            intentions.append(Intention(
                type='growth',
                goal='fill_knowledge_gap',
                target=gaps[0] if gaps else None,
                priority=0.6
            ))

        if ecosystem is not None and np.random.rand() < self.sociability:
            intentions.append(Intention(
                type='symbiosis',
                goal='collaborate',
                priority=0.7
            ))

        intentions.sort(key=lambda x: x.priority, reverse=True)
        return intentions

    def choose_action(self, intention: Intention) -> Dict:
        """Convert intention to action"""
        if intention.type == 'survival':
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
        """Holistic learning"""
        experience = Experience(
            perception=perception,
            action=action,
            consequence=consequence,
            context={'age': self.age, 'viability': self.viability}
        )

        self.experiences.add(experience)

        # v1.1: Improved integration
        self.phenotype.integrate_experience(experience)

        self.self_model.update_from_experience(experience)

    def interact_with_ecosystem(self, ecosystem) -> None:
        """Symbiotic interactions"""
        if not hasattr(ecosystem, 'entities'):
            return

        neighbors = [e for e in ecosystem.entities if e.id != self.id]

        if len(neighbors) == 0:
            return

        neighbor = np.random.choice(neighbors)

        if neighbor.viability > self.viability:
            self.curiosity = 0.9 * self.curiosity + 0.1 * neighbor.curiosity

    def assess_viability(self, environment, ecosystem) -> float:
        """
        v1.1: Improved viability metric

        NOW directly reflects environment feedback!
        """
        scores = []

        # 1. Environment feedback (v1.1: NEW!)
        if len(self.recent_feedback) > 0:
            env_feedback_score = np.mean(self.recent_feedback)
            scores.append(env_feedback_score)
        else:
            scores.append(0.5)

        # 2. Success rate
        if len(self.experiences) > 0:
            recent = self.experiences.get_recent(10)
            success_rate = sum(1 for e in recent if e.was_successful()) / len(recent)
            scores.append(success_rate)
        else:
            scores.append(0.5)

        # 3. Growth trend
        if len(self.viability_history) > 10:
            recent_trend = np.mean(self.viability_history[-10:])
            scores.append(min(1.0, recent_trend))
        else:
            scores.append(0.5)

        # 4. Adaptability
        adaptability_score = len(self.self_model.capabilities) / 10.0
        scores.append(min(1.0, adaptability_score))

        # Weighted average (v1.1: environment feedback weighted more!)
        weights = [0.4, 0.3, 0.2, 0.1]  # Environment feedback = 40%
        viability = np.average(scores, weights=weights)

        # Add noise
        viability += np.random.randn() * 0.05
        viability = np.clip(viability, 0.0, 1.0)

        return viability

    def should_metamorphose(self) -> bool:
        """
        v1.1: Controlled metamorphosis

        Less frequent, more strategic
        """
        # Only metamorphose if truly needed

        # 1. Critical failure
        if self.viability < 0.2:
            return True

        # 2. Prolonged stagnation (v1.1: longer period required)
        if len(self.viability_history) > 50:
            recent = self.viability_history[-50:]
            if np.std(recent) < 0.02:  # Really stuck
                return True

        # 3. Random exploration (v1.1: much rarer)
        if np.random.rand() < 0.005:  # Was 0.01, now 0.005
            return True

        return False

    def metamorphose(self) -> None:
        """Structural evolution"""
        print(f"Entity {self.id}: Metamorphosis at age {self.age}!")

        self.genome = self.genome.mutate(mutation_rate=0.2)

        new_blueprint = self.genome.express()

        if len(self.phenotype.layers) < new_blueprint['architecture']['n_layers']:
            self.phenotype.add_module({'size': 16})
            print(f"  Added new module!")
        elif len(self.phenotype.layers) > new_blueprint['architecture']['n_layers']:
            self.phenotype.remove_module(len(self.phenotype.layers) - 1)
            print(f"  Removed obsolete module!")

        self.curiosity = new_blueprint['behavior']['curiosity']
        self.risk_tolerance = new_blueprint['behavior']['risk_tolerance']
        self.sociability = new_blueprint['behavior']['sociability']

    def __repr__(self):
        return (f"Entity_v1.1(id={self.id}, age={self.age}, viability={self.viability:.2f}, "
                f"modules={len(self.phenotype.layers)})")


if __name__ == "__main__":
    print("Testing GENESIS Entity v1.1...")

    from genesis_environment import OpenWorldEnvironment

    entity = GENESIS_Entity_v1_1(entity_id=1)
    print(f"Created: {entity}")

    env = OpenWorldEnvironment(size=5)

    print("\nLiving for 10 steps:")
    for step in range(10):
        viability = entity.live_one_step(env, ecosystem=None)

        if step % 3 == 0:
            print(f"Step {step}: viability={viability:.3f}, feedback={len(entity.recent_feedback)}")

    print(f"\nFinal: {entity}")
    print(f"Recent feedback: {entity.recent_feedback[-5:] if entity.recent_feedback else 'None'}")

    print("\n✅ Entity v1.1 test complete!")
