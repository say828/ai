"""
GENESIS v1.2: Refined Entity

Phase 3 improvements:
1. Hebbian learning rate increased: 0.01 → 0.05
2. Environment feedback smoothing: moving average over 10 recent feedbacks
3. Better initialization: Xavier/He initialization
4. Metamorphosis threshold decreased: 0.005 → 0.001
5. Network capacity increased: [32, 16] → [64, 32]

Goal: Achieve POSITIVE learning
"""

import numpy as np
from typing import Dict, List, Any, Optional
from genesis_models import (
    WorldModel, SelfModel, Genome, Intention,
    Experience, ExperienceBuffer
)


class Phenotype_v1_2:
    """
    v1.2 Phenotype with stronger integration and better initialization
    """

    def __init__(self, blueprint: Dict):
        self.blueprint = blueprint
        self.layers = self._build_layers(blueprint['architecture'])
        self.parameters = self._initialize_parameters()  # v1.2: Better initialization!

        # Track successful pathways (Hebbian)
        self.pathway_strengths = {}
        for key in self.parameters:
            self.pathway_strengths[key] = np.ones_like(self.parameters[key])

    def _build_layers(self, architecture: Dict) -> List:
        """Build neural layers from blueprint"""
        n_layers = architecture['n_layers']
        layer_sizes = architecture['layer_sizes']

        layers = []
        for i in range(n_layers):
            layers.append({
                'size': layer_sizes[i],
                'weights': None,
                'active': True
            })

        return layers

    def _initialize_parameters(self) -> Dict:
        """
        v1.2: Xavier/He initialization instead of simple * 0.01

        Xavier: sqrt(2 / (n_in + n_out))
        He: sqrt(2 / n_in)
        """
        params = {}

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_size = 10
            else:
                input_size = self.layers[i-1]['size']

            # v1.2: He initialization (better for ReLU/tanh)
            he_scale = np.sqrt(2.0 / input_size)
            params[f'layer_{i}'] = np.random.randn(input_size, layer['size']) * he_scale

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
        v1.2: STRONGER Hebbian-like integration

        Learning rate: 0.01 → 0.05 (5x stronger!)
        """
        success = experience.was_successful()

        if success:
            # v1.2: STRONGER Hebbian learning!
            for key in self.parameters:
                activity = np.abs(self.parameters[key])

                # v1.2: 0.01 → 0.05 (5x increase!)
                strength_update = 0.05 * activity * self.pathway_strengths[key]
                self.parameters[key] += strength_update

                # Increase pathway strength
                self.pathway_strengths[key] *= 1.05  # Was 1.01, now 1.05
                self.pathway_strengths[key] = np.clip(self.pathway_strengths[key], 0.5, 2.0)
        else:
            # Anti-Hebbian: weaken failed pathways
            for key in self.parameters:
                self.parameters[key] += np.random.randn(*self.parameters[key].shape) * 0.001

                # Decrease pathway strength
                self.pathway_strengths[key] *= 0.95  # Was 0.99, now 0.95
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

        # v1.2: He initialization for new modules too
        he_scale = np.sqrt(2.0 / prev_size)
        self.parameters[new_key] = np.random.randn(prev_size, new_layer['size']) * he_scale
        self.pathway_strengths[new_key] = np.ones((prev_size, new_layer['size']))

    def remove_module(self, module_idx: int) -> None:
        """Remove obsolete module"""
        if 0 <= module_idx < len(self.layers):
            self.layers[module_idx]['active'] = False


class GENESIS_Entity_v1_2:
    """
    GENESIS v1.2: Refined entity

    Phase 3 improvements:
    1. Stronger Hebbian learning (0.01 → 0.05)
    2. Smoothed environment feedback (moving average)
    3. Better initialization (Xavier/He)
    4. Less metamorphosis (0.005 → 0.001)
    5. Larger capacity ([32,16] → [64,32])

    Goal: Achieve POSITIVE learning!
    """

    def __init__(self, genome: Optional[Genome] = None, entity_id: int = 0):
        self.id = entity_id
        self.genome = genome if genome else Genome()

        # v1.2: Override genome for larger capacity
        self.genome.genes['architecture'] = {
            'n_layers': 2,
            'layer_sizes': [64, 32]  # Was [32, 16], now doubled!
        }

        blueprint = self.genome.express()
        self.phenotype = Phenotype_v1_2(blueprint)  # v1.2!
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

        # v1.2: Smoothed environment feedback (moving average)
        self.recent_feedback = []
        self.feedback_window = 10  # Use last 10 feedbacks

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

        # v1.2: Store and smooth environment feedback
        if 'viability_contribution' in consequence:
            self.recent_feedback.append(consequence['viability_contribution'])
            if len(self.recent_feedback) > self.feedback_window:
                self.recent_feedback.pop(0)

        # 5. INTEGRATE
        self.integrate_experience(perception, action if self.intentions else {}, consequence)

        # 6. INTERACT
        if ecosystem is not None:
            self.interact_with_ecosystem(ecosystem)

        # 7. EVALUATE
        self.viability = self.assess_viability(environment, ecosystem)
        self.viability_history.append(self.viability)

        # 8. EVOLVE (v1.2: even more controlled!)
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

        # v1.2: Stronger integration
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
        v1.2: Smoothed environment feedback

        Uses moving average over last 10 feedbacks
        """
        scores = []

        # 1. Environment feedback (v1.2: SMOOTHED!)
        if len(self.recent_feedback) >= self.feedback_window:
            # Use full window average
            env_feedback_score = np.mean(self.recent_feedback[-self.feedback_window:])
            scores.append(env_feedback_score)
        elif len(self.recent_feedback) > 0:
            # Use partial window
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

        # Weighted average (environment feedback = 40%)
        weights = [0.4, 0.3, 0.2, 0.1]
        viability = np.average(scores, weights=weights)

        # Add noise
        viability += np.random.randn() * 0.05
        viability = np.clip(viability, 0.0, 1.0)

        return viability

    def should_metamorphose(self) -> bool:
        """
        v1.2: Even MORE controlled metamorphosis

        Threshold: 0.005 → 0.001 (5x reduction)
        """
        # 1. Critical failure
        if self.viability < 0.2:
            return True

        # 2. Prolonged stagnation
        if len(self.viability_history) > 50:
            recent = self.viability_history[-50:]
            if np.std(recent) < 0.02:
                return True

        # 3. Random exploration (v1.2: even rarer!)
        if np.random.rand() < 0.001:  # Was 0.005, now 0.001
            return True

        return False

    def metamorphose(self) -> None:
        """Structural evolution"""
        print(f"Entity {self.id}: Metamorphosis at age {self.age}!")

        self.genome = self.genome.mutate(mutation_rate=0.2)

        new_blueprint = self.genome.express()

        if len(self.phenotype.layers) < new_blueprint['architecture']['n_layers']:
            # v1.2: Larger modules (32 instead of 16)
            self.phenotype.add_module({'size': 32})
            print(f"  Added new module!")
        elif len(self.phenotype.layers) > new_blueprint['architecture']['n_layers']:
            self.phenotype.remove_module(len(self.phenotype.layers) - 1)
            print(f"  Removed obsolete module!")

        self.curiosity = new_blueprint['behavior']['curiosity']
        self.risk_tolerance = new_blueprint['behavior']['risk_tolerance']
        self.sociability = new_blueprint['behavior']['sociability']

    def __repr__(self):
        return (f"Entity_v1.2(id={self.id}, age={self.age}, viability={self.viability:.2f}, "
                f"modules={len(self.phenotype.layers)})")


if __name__ == "__main__":
    print("Testing GENESIS Entity v1.2...")

    from genesis_environment import OpenWorldEnvironment

    entity = GENESIS_Entity_v1_2(entity_id=1)
    print(f"Created: {entity}")

    env = OpenWorldEnvironment(size=5)

    print("\nLiving for 10 steps:")
    for step in range(10):
        viability = entity.live_one_step(env, ecosystem=None)

        if step % 3 == 0:
            print(f"Step {step}: viability={viability:.3f}, feedback_window={len(entity.recent_feedback)}")

    print(f"\nFinal: {entity}")
    print(f"Recent feedback (smoothed): {np.mean(entity.recent_feedback) if entity.recent_feedback else 'None'}")

    print("\nv1.2 improvements:")
    print("1. Hebbian learning rate: 0.01 -> 0.05 (5x)")
    print("2. Feedback smoothing: moving average over 10 steps")
    print("3. Initialization: Xavier/He instead of *0.01")
    print("4. Metamorphosis threshold: 0.005 -> 0.001 (5x reduction)")
    print("5. Network capacity: [32,16] -> [64,32] (2x)")

    print("\nExpected outcome: POSITIVE learning!")
