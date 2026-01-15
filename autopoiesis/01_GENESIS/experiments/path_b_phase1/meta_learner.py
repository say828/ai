"""
GENESIS Phase 2: Meta-Learning System

Learning to learn - evolving the learning algorithm itself.

Hierarchical learning levels:
- Level 1: Agents learn behaviors (RNN weights)
- Level 2: Teacher learns from agents (EMA of elite)
- Level 3: MetaLearner optimizes learning process (THIS) ← NEW!

The meta-learner adapts hyperparameters and strategies based on
population performance, enabling continuous improvement of the
learning system itself.
"""

import numpy as np
from typing import Dict, List, Optional
import json


class MetaLearner:
    """
    Self-Adaptive Learning System

    Evolves the learning algorithm's hyperparameters:
    - Teacher learning rate
    - Mutation rates and scales
    - Elite selection criteria
    - Memory consolidation thresholds
    - Stigmergy decay rates

    Methods:
    1. Rule-based adaptation (fast, reactive)
    2. Evolutionary meta-search (slow, exploratory)
    3. Performance trend analysis (medium, strategic)
    """

    def __init__(self):
        """Initialize with default meta-parameters"""
        # Learnable hyperparameters
        self.meta_params = {
            'teacher_lr': 0.1,
            'mutation_rate': 0.05,
            'mutation_scale': 0.1,
            'elite_percent': 0.2,
            'memory_consolidation_threshold': 0.85,
            'stigmergy_decay': 0.98,
            'teacher_update_interval': 100
        }

        # Performance tracking
        self.performance_history = []
        self.meta_param_history = []
        self.adaptation_count = 0

        # Bounds for each parameter (prevent extreme values)
        self.param_bounds = {
            'teacher_lr': (0.01, 0.5),
            'mutation_rate': (0.01, 0.2),
            'mutation_scale': (0.01, 0.5),
            'elite_percent': (0.1, 0.4),
            'memory_consolidation_threshold': (0.7, 0.95),
            'stigmergy_decay': (0.9, 0.99),
            'teacher_update_interval': (50, 500)
        }

    def adapt_learning_strategy(self, population_stats: Dict) -> Dict:
        """
        Rule-based adaptation of learning strategy

        Analyzes recent population performance and adjusts hyperparameters
        according to predefined rules.

        Adaptation rules:
        - Stagnation → Increase exploration (higher mutation)
        - Rapid improvement → Increase exploitation (lower mutation)
        - Population crisis → Faster knowledge propagation
        - High diversity → Increase selective pressure

        Args:
            population_stats: Dictionary with population metrics

        Returns:
            Dictionary of adapted parameters
        """
        # Extract recent performance
        coherence_history = population_stats.get('coherence_history', [])
        if len(coherence_history) < 100:
            return self.meta_params  # Not enough data yet

        recent_coherence = coherence_history[-100:]
        coherence_trend = self._compute_trend(recent_coherence)
        coherence_variance = np.var(recent_coherence)

        pop_size = population_stats.get('population_size', 100)
        qd_coverage = population_stats.get('qd_coverage', 0)

        # Adaptation rules
        old_params = self.meta_params.copy()

        # Rule 1: Stagnation detection
        if abs(coherence_trend) < 0.0001:  # Flat trend = stagnation
            # Increase exploration
            self.meta_params['mutation_rate'] *= 1.1
            self.meta_params['mutation_scale'] *= 1.05
            self.meta_params['elite_percent'] *= 0.95  # More diversity

        # Rule 2: Rapid improvement
        elif coherence_trend > 0.002:  # Strong positive trend
            # Increase exploitation (but not too much)
            self.meta_params['mutation_rate'] *= 0.95
            self.meta_params['mutation_scale'] *= 0.98
            self.meta_params['elite_percent'] = min(0.3, self.meta_params['elite_percent'] * 1.05)

        # Rule 3: Population crisis
        if pop_size < 100:
            # Emergency: faster knowledge transfer
            self.meta_params['teacher_lr'] = min(0.3, self.meta_params['teacher_lr'] * 1.2)
            self.meta_params['teacher_update_interval'] = max(50, int(self.meta_params['teacher_update_interval'] * 0.8))

        # Rule 4: High variance (unstable)
        if coherence_variance > 0.05:
            # Stabilize with stronger teacher influence
            self.meta_params['teacher_lr'] = min(0.3, self.meta_params['teacher_lr'] * 1.1)

        # Rule 5: Low diversity (QD coverage not growing)
        if len(self.performance_history) > 10:
            recent_qd = [p['qd_coverage'] for p in self.performance_history[-10:]]
            qd_growth = self._compute_trend(recent_qd)
            if qd_growth < 100:  # QD not growing much
                # Encourage diversity
                self.meta_params['mutation_rate'] = min(0.15, self.meta_params['mutation_rate'] * 1.15)

        # Apply bounds
        self._apply_bounds()

        # Track changes
        self.performance_history.append({
            'step': population_stats.get('step', len(self.performance_history)),
            'avg_coherence': population_stats.get('avg_coherence', 0.5),
            'population_size': pop_size,
            'qd_coverage': qd_coverage,
            'coherence_trend': coherence_trend
        })
        self.meta_param_history.append(self.meta_params.copy())
        self.adaptation_count += 1

        # Log significant changes
        changes = {k: self.meta_params[k] - old_params[k]
                  for k in self.meta_params
                  if abs(self.meta_params[k] - old_params[k]) > 1e-6}

        return self.meta_params

    def evolutionary_meta_search(self, n_variants: int = 20,
                                 test_steps: int = 1000) -> Dict:
        """
        Evolutionary search in meta-parameter space

        Tests multiple hyperparameter combinations and selects the best.
        More expensive but finds better configurations.

        Args:
            n_variants: Number of parameter variants to test
            test_steps: Simulation length for each variant

        Returns:
            Best meta-parameters found
        """
        # This is a placeholder for full implementation
        # In practice, would run short simulations with each variant

        variants = []
        for _ in range(n_variants):
            variant = {
                k: v * np.random.uniform(0.7, 1.3)
                for k, v in self.meta_params.items()
            }
            # Apply bounds
            for k, v in variant.items():
                bounds = self.param_bounds[k]
                variant[k] = np.clip(v, bounds[0], bounds[1])

            variants.append(variant)

        # In full implementation: run test simulations and rank by performance
        # For now: return a slightly perturbed version
        best_variant = variants[np.random.randint(n_variants)]
        return best_variant

    def predict_optimal_interval(self, population_size: int,
                                 teacher_knowledge: float) -> int:
        """
        Predict optimal teacher update interval

        Uses heuristic: smaller population → more frequent updates
        Higher teacher knowledge → less frequent updates (already good)

        Args:
            population_size: Current population size
            teacher_knowledge: Teacher's knowledge level (0-1)

        Returns:
            Recommended update interval (steps)
        """
        # Base interval
        base_interval = 100

        # Adjust for population size (smaller pop → more frequent)
        pop_factor = np.clip(population_size / 100.0, 0.5, 2.0)

        # Adjust for teacher quality (better teacher → less frequent updates)
        knowledge_factor = 1.0 + teacher_knowledge * 0.5

        optimal_interval = int(base_interval * pop_factor * knowledge_factor)

        # Clamp to bounds
        return int(np.clip(optimal_interval, 50, 500))

    def get_statistics(self) -> Dict:
        """Get meta-learner statistics"""
        stats = {
            'current_params': self.meta_params.copy(),
            'adaptation_count': self.adaptation_count,
            'performance_trajectory': {
                'coherence': [p['avg_coherence'] for p in self.performance_history[-50:]],
                'population': [p['population_size'] for p in self.performance_history[-50:]],
                'qd_coverage': [p['qd_coverage'] for p in self.performance_history[-50:]]
            }
        }

        if self.performance_history:
            recent_perf = self.performance_history[-100:]
            stats['performance_summary'] = {
                'avg_coherence': float(np.mean([p['avg_coherence'] for p in recent_perf])),
                'coherence_trend': float(self._compute_trend([p['avg_coherence'] for p in recent_perf])),
                'avg_population': float(np.mean([p['population_size'] for p in recent_perf])),
                'qd_growth': float(self._compute_trend([p['qd_coverage'] for p in recent_perf]))
            }

        return stats

    def save(self, filepath: str):
        """Save meta-learner state"""
        data = {
            'meta_params': self.meta_params,
            'performance_history': self.performance_history,
            'adaptation_count': self.adaptation_count
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load meta-learner state"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.meta_params = data['meta_params']
        self.performance_history = data['performance_history']
        self.adaptation_count = data['adaptation_count']

    # Helper methods

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of time series"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _apply_bounds(self):
        """Ensure all parameters are within valid bounds"""
        for param_name, value in self.meta_params.items():
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                self.meta_params[param_name] = float(np.clip(value, bounds[0], bounds[1]))
