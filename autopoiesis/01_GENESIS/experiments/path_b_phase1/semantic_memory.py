"""
GENESIS Phase 2: Semantic Memory System

Extracts explicit knowledge from experiences:
- Concepts: "food", "danger", "high coherence zones"
- Relations: "X causes Y", "A correlates with B"
- Rules: "IF condition THEN action ELSE alternative"

This transforms implicit procedural knowledge (Teacher weights) into
explicit declarative knowledge (queryable facts and rules).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json


class SemanticMemory:
    """
    Knowledge Graph for Artificial Life

    Extracts and stores explicit knowledge patterns from agent experiences.
    Unlike Teacher Network (implicit weights), this provides queryable,
    explainable knowledge.

    Architecture:
    - Concepts: Clustered observation patterns with semantic labels
    - Relations: Causal and correlational links between concepts
    - Rules: IF-THEN production rules for decision making
    """

    def __init__(self, embedding_dim: int = 32):
        """
        Args:
            embedding_dim: Dimensionality for concept embeddings
        """
        self.embedding_dim = embedding_dim

        # Knowledge components
        self.concepts = {}  # {concept_id: {name, embedding, frequency, ...}}
        self.relations = []  # [(concept_A, relation_type, concept_B, strength)]
        self.rules = []      # [{condition, action, success_rate, confidence}]

        # Statistics
        self.total_patterns_seen = 0
        self.concepts_discovered = 0
        self.rules_validated = 0

    def extract_concept(self, observation: np.ndarray, action: np.ndarray,
                       outcome: float) -> Optional[str]:
        """
        Extract or update concept from observation-action-outcome pattern

        Uses clustering to identify recurring patterns in high-dimensional
        observation space and assign them semantic meaning.

        Args:
            observation: Sensory input vector
            action: Action taken
            outcome: Result (coherence, reward, etc.)

        Returns:
            Concept ID if pattern recognized
        """
        # Reduce observation to embedding (simple PCA-like projection)
        obs_embedding = self._embed_observation(observation)

        # Find most similar existing concept
        best_match_id = None
        best_similarity = -1.0

        for concept_id, concept in self.concepts.items():
            similarity = self._cosine_similarity(obs_embedding, concept['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = concept_id

        # Decision: New concept or update existing?
        if best_similarity < 0.7:  # Threshold for new concept
            # Discover new concept
            concept_id = f"concept_{self.concepts_discovered}"
            self.concepts[concept_id] = {
                'id': concept_id,
                'embedding': obs_embedding,
                'frequency': 1,
                'associated_actions': [action.copy()],
                'outcomes': [outcome],
                'avg_outcome': outcome,
                'created_at': self.total_patterns_seen
            }
            self.concepts_discovered += 1
            return concept_id
        else:
            # Update existing concept (incremental learning)
            concept = self.concepts[best_match_id]
            concept['frequency'] += 1
            concept['associated_actions'].append(action.copy())
            concept['outcomes'].append(outcome)
            concept['avg_outcome'] = np.mean(concept['outcomes'])

            # Update embedding (running average)
            alpha = 0.1
            concept['embedding'] = (1 - alpha) * concept['embedding'] + alpha * obs_embedding

            return best_match_id

        self.total_patterns_seen += 1

    def discover_causal_relations(self, temporal_window: int = 10):
        """
        Discover causal relationships between concepts

        Uses temporal correlation: If concept A frequently appears before
        concept B with high outcome, there may be a causal link.

        Method: Granger causality + association rule mining

        Args:
            temporal_window: Time steps to look back for causality
        """
        # Build temporal co-occurrence matrix
        concept_pairs = defaultdict(lambda: {'count': 0, 'outcomes': []})

        # This requires temporal sequencing in memory (Phase 2.1 enhancement)
        # For now, use simplified correlation analysis

        for concept_A_id in self.concepts:
            for concept_B_id in self.concepts:
                if concept_A_id == concept_B_id:
                    continue

                # Calculate correlation between concepts' outcomes
                outcomes_A = self.concepts[concept_A_id]['outcomes']
                outcomes_B = self.concepts[concept_B_id]['outcomes']

                if len(outcomes_A) > 5 and len(outcomes_B) > 5:
                    # Correlation strength
                    correlation = np.corrcoef(
                        outcomes_A[:min(len(outcomes_A), len(outcomes_B))],
                        outcomes_B[:min(len(outcomes_A), len(outcomes_B))]
                    )[0, 1]

                    if abs(correlation) > 0.6:  # Strong correlation
                        relation_type = "correlates_with" if correlation > 0 else "anti_correlates_with"
                        relation = (concept_A_id, relation_type, concept_B_id, abs(correlation))

                        # Add if not already present
                        if relation not in self.relations:
                            self.relations.append(relation)

    def generate_survival_rules(self, min_confidence: float = 0.7,
                               min_support: int = 10):
        """
        Generate production rules from concept-action patterns

        Rules take the form:
        IF concept_X observed THEN action_Y (with confidence Z)

        Args:
            min_confidence: Minimum success rate to generate rule
            min_support: Minimum number of observations
        """
        new_rules = []

        for concept_id, concept in self.concepts.items():
            if concept['frequency'] < min_support:
                continue

            # Group by actions
            action_outcomes = defaultdict(list)
            for action, outcome in zip(concept['associated_actions'], concept['outcomes']):
                # Discretize action to 5 bins for grouping
                action_key = tuple(np.digitize(action, bins=[-1, -0.5, 0, 0.5, 1]))
                action_outcomes[action_key].append(outcome)

            # Generate rules for successful action patterns
            for action_key, outcomes in action_outcomes.items():
                if len(outcomes) < min_support:
                    continue

                success_rate = np.mean(np.array(outcomes) > 0.8)  # Success = outcome > 0.8

                if success_rate >= min_confidence:
                    rule = {
                        'condition': f"observe_{concept_id}",
                        'action': np.array(action_key) / 2.0 - 1.0,  # Convert back to [-1, 1]
                        'success_rate': float(success_rate),
                        'confidence': float(success_rate),
                        'support': len(outcomes),
                        'avg_outcome': float(np.mean(outcomes))
                    }

                    # Check if rule already exists
                    if not any(r['condition'] == rule['condition'] and
                             np.allclose(r['action'], rule['action'])
                             for r in self.rules):
                        new_rules.append(rule)
                        self.rules_validated += 1

        self.rules.extend(new_rules)
        return len(new_rules)

    def query_knowledge(self, observation: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Query knowledge base for best action given observation

        Returns explicit reasoning: "In situation X, do action Y with confidence Z"

        Args:
            observation: Current sensory input

        Returns:
            (suggested_action, confidence) or (None, 0.0) if no applicable rule
        """
        if not self.concepts or not self.rules:
            return None, 0.0

        # Find matching concept
        obs_embedding = self._embed_observation(observation)

        best_concept_id = None
        best_similarity = -1.0

        for concept_id, concept in self.concepts.items():
            similarity = self._cosine_similarity(obs_embedding, concept['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept_id = concept_id

        if best_similarity < 0.5:  # Too different from known concepts
            return None, 0.0

        # Find applicable rules
        applicable_rules = [
            r for r in self.rules
            if best_concept_id in r['condition']
        ]

        if not applicable_rules:
            return None, 0.0

        # Select best rule (highest confidence × avg_outcome)
        best_rule = max(applicable_rules,
                       key=lambda r: r['confidence'] * r['avg_outcome'])

        return best_rule['action'], best_rule['confidence']

    def get_concept_summary(self, concept_id: str) -> Dict:
        """Get human-readable summary of a concept"""
        if concept_id not in self.concepts:
            return {}

        concept = self.concepts[concept_id]
        return {
            'id': concept_id,
            'frequency': concept['frequency'],
            'avg_outcome': concept['avg_outcome'],
            'total_observations': len(concept['outcomes']),
            'outcome_distribution': {
                'mean': float(np.mean(concept['outcomes'])),
                'std': float(np.std(concept['outcomes'])),
                'min': float(np.min(concept['outcomes'])),
                'max': float(np.max(concept['outcomes']))
            }
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive semantic memory statistics"""
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'total_rules': len(self.rules),
            'patterns_seen': self.total_patterns_seen,
            'concepts_discovered': self.concepts_discovered,
            'rules_validated': self.rules_validated,
            'avg_concept_frequency': float(np.mean([c['frequency'] for c in self.concepts.values()])) if self.concepts else 0.0,
            'avg_rule_confidence': float(np.mean([r['confidence'] for r in self.rules])) if self.rules else 0.0,
            'knowledge_density': len(self.relations) / max(len(self.concepts), 1)
        }

    def save(self, filepath: str):
        """Save semantic memory to file"""
        data = {
            'concepts': {
                cid: {
                    'id': c['id'],
                    'embedding': c['embedding'].tolist(),
                    'frequency': c['frequency'],
                    'avg_outcome': c['avg_outcome'],
                    'created_at': c['created_at']
                }
                for cid, c in self.concepts.items()
            },
            'relations': self.relations,
            'rules': [
                {
                    'condition': r['condition'],
                    'action': r['action'].tolist(),
                    'success_rate': r['success_rate'],
                    'confidence': r['confidence'],
                    'support': r['support']
                }
                for r in self.rules
            ],
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # Helper methods

    def _embed_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Project high-dimensional observation to low-dimensional embedding

        Uses random projection (simple but effective for concept clustering)
        """
        if not hasattr(self, '_projection_matrix'):
            # Initialize random projection matrix (fixed seed for consistency)
            np.random.seed(42)
            self._projection_matrix = np.random.randn(self.embedding_dim, len(observation))
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0)

        embedding = self._projection_matrix @ observation
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
