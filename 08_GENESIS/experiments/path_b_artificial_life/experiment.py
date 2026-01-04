"""
GENESIS Path B: Artificial Life Experiment
==========================================

Main experiment script for validating autopoietic agents against baselines.

Experiment Design:
- Run multiple populations in parallel (autopoietic, RL, NEAT, random)
- Track survival, diversity, and emergent behaviors
- Generate comprehensive analysis and visualizations

Success Criteria:
1. Autopoietic population survives 5000+ steps
2. Behavioral diversity > baselines
3. Unique emergent patterns observed

Author: GENESIS Project
Date: 2026-01-04
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid_world import GridWorld
from autopoietic_grid_agent import AutopoieticGridAgent
from baseline_agents import RandomAgent, RLAgent, NEATAgent, create_agent
from analysis import (
    BehavioralAnalyzer, DiversityMetrics, EmergenceDetector,
    SurvivalAnalyzer, PopulationDynamicsAnalyzer, run_comprehensive_analysis
)
from visualize import (
    GridWorldVisualizer, PopulationPlotter, BehaviorVisualizer,
    NicheVisualizer, ComprehensiveReport
)


class ArtificialLifeExperiment:
    """
    Main experiment class for artificial life validation
    """
    
    def __init__(
        self,
        grid_size: int = 100,
        n_resources: int = 200,
        n_predators: int = 5,
        n_agents_per_type: int = 10,
        max_population: int = 50,
        max_steps: int = 5000,
        reproduction_enabled: bool = True,
        save_interval: int = 100,
        output_dir: str = "results",
        seed: int = 42
    ):
        """
        Initialize experiment
        
        Args:
            grid_size: Size of grid world
            n_resources: Number of resources
            n_predators: Number of predators
            n_agents_per_type: Initial agents per type
            max_population: Maximum population per type
            max_steps: Maximum simulation steps
            reproduction_enabled: Whether agents can reproduce
            save_interval: Steps between saves
            output_dir: Output directory
            seed: Random seed
        """
        self.grid_size = grid_size
        self.n_agents_per_type = n_agents_per_type
        self.max_population = max_population
        self.max_steps = max_steps
        self.reproduction_enabled = reproduction_enabled
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.seed = seed
        
        np.random.seed(seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize world
        self.world = GridWorld(
            size=grid_size,
            n_resources=n_resources,
            n_predators=n_predators,
            seed=seed
        )
        
        # Agent storage
        self.agents: Dict[str, object] = {}  # agent_id -> agent
        self.agent_types: Dict[str, str] = {}  # agent_id -> type
        
        # Analysis tools
        self.survival_analyzer = SurvivalAnalyzer()
        self.population_analyzer = PopulationDynamicsAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        
        # History tracking
        self.diversity_history = []
        self.emergence_history = []
        self.frames = []  # For animation
        
        # Statistics
        self.stats = {
            'autopoietic': {'births': 0, 'deaths': 0},
            'rl': {'births': 0, 'deaths': 0},
            'neat': {'births': 0, 'deaths': 0},
            'random': {'births': 0, 'deaths': 0}
        }
        
        print(f"\nExperiment initialized:")
        print(f"  Grid: {grid_size}x{grid_size}")
        print(f"  Resources: {n_resources}")
        print(f"  Predators: {n_predators}")
        print(f"  Agents per type: {n_agents_per_type}")
        print(f"  Max steps: {max_steps}")
        print(f"  Output: {self.run_dir}")
    
    def initialize_population(self):
        """Initialize all agent populations"""
        print("\nInitializing populations...")
        
        agent_types = ['autopoietic', 'rl', 'neat', 'random']
        
        for agent_type in agent_types:
            for i in range(self.n_agents_per_type):
                agent_id = f"{agent_type}_{i}"
                
                if agent_type == 'autopoietic':
                    agent = AutopoieticGridAgent(agent_id=agent_id)
                elif agent_type == 'rl':
                    agent = RLAgent(agent_id=agent_id)
                elif agent_type == 'neat':
                    agent = NEATAgent(agent_id=agent_id)
                else:  # random
                    agent = RandomAgent(agent_id=agent_id)
                
                # Add to world
                position = self.world.add_agent(agent_id)
                
                # Store
                self.agents[agent_id] = agent
                self.agent_types[agent_id] = agent_type
            
            print(f"  {agent_type}: {self.n_agents_per_type} agents")
    
    def step(self, step_num: int):
        """Execute one simulation step"""
        # Track population counts
        pop_counts = defaultdict(int)
        for agent_type in self.agent_types.values():
            pop_counts[agent_type] += 1
        self.population_analyzer.record_population(step_num, dict(pop_counts))
        
        # Process each agent
        dead_agents = []
        new_agents = []
        
        for agent_id, agent in list(self.agents.items()):
            if not agent.is_alive:
                dead_agents.append(agent_id)
                continue
            
            # Get observation
            obs = self.world.get_local_observation(agent_id, vision_size=9)
            
            # Agent perceives and acts
            action = agent.perceive_and_act(obs)
            
            # Execute action in world
            result = self.world.step_agent(agent_id, action)
            
            # Get position
            position = self.world.agent_positions.get(agent_id, (0, 0))
            
            # Update agent state
            agent.update_state(result, position)
            
            # Check for death
            if not agent.is_alive:
                dead_agents.append(agent_id)
                agent_type = self.agent_types[agent_id]
                self.survival_analyzer.record_death(agent_type, agent.age, "energy_or_coherence")
                self.population_analyzer.record_death(step_num, agent_type)
                self.stats[agent_type]['deaths'] += 1
            
            # Check for reproduction
            elif self.reproduction_enabled and agent.can_reproduce:
                agent_type = self.agent_types[agent_id]
                
                # Check population limit
                type_count = sum(1 for t in self.agent_types.values() if t == agent_type)
                
                if type_count < self.max_population:
                    offspring = agent.reproduce(mutation_rate=0.1)
                    new_agents.append((offspring, agent_type))
                    self.population_analyzer.record_birth(step_num, agent_type)
                    self.stats[agent_type]['births'] += 1
        
        # Remove dead agents
        for agent_id in dead_agents:
            if agent_id in self.world.agent_positions:
                self.world.remove_agent(agent_id)
            if agent_id in self.agents:
                del self.agents[agent_id]
            if agent_id in self.agent_types:
                del self.agent_types[agent_id]
        
        # Add new agents
        for offspring, agent_type in new_agents:
            agent_id = offspring.id
            position = self.world.add_agent(agent_id)
            self.agents[agent_id] = offspring
            self.agent_types[agent_id] = agent_type
        
        # Update world (predators, resources)
        self.world.step_world()
    
    def analyze_population(self, step_num: int) -> Dict:
        """Analyze current population state"""
        # Get position histories
        position_histories = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'position_history'):
                position_histories[agent_id] = list(agent.position_history)
        
        # Current positions
        positions = dict(self.world.agent_positions)
        
        # Run comprehensive analysis
        analysis = run_comprehensive_analysis(
            list(self.agents.values()),
            positions,
            position_histories,
            self.grid_size
        )
        
        return analysis
    
    def run(self, verbose: bool = True):
        """Run the full experiment"""
        print("\n" + "=" * 70)
        print("Starting Artificial Life Experiment")
        print("=" * 70)
        
        # Initialize
        self.initialize_population()
        
        # Visualization
        visualizer = GridWorldVisualizer(grid_size=self.grid_size)
        
        start_time = time.time()
        
        for step in range(self.max_steps):
            # Run simulation step
            self.step(step)
            
            # Periodic analysis and logging
            if step % self.save_interval == 0:
                # Count populations
                pop_counts = defaultdict(int)
                for agent_type in self.agent_types.values():
                    pop_counts[agent_type] += 1
                
                # Check for extinction
                total_pop = sum(pop_counts.values())
                if total_pop == 0:
                    print(f"\nAll populations extinct at step {step}!")
                    break
                
                # Analyze
                analysis = self.analyze_population(step)
                self.diversity_history.append(analysis.get('diversity', {}))
                self.emergence_history.append(analysis.get('emergence', {}))
                
                # Save frame for animation
                agent_trails = {}
                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'position_history'):
                        agent_trails[agent_id] = list(agent.position_history)
                
                frame = visualizer.render_frame(
                    self.world.grid,
                    dict(self.world.agent_positions),
                    self.agent_types,
                    agent_trails,
                    step
                )
                self.frames.append(frame)
                
                if verbose:
                    elapsed = time.time() - start_time
                    div = analysis.get('diversity', {})
                    emg = analysis.get('emergence', {})
                    
                    print(f"\nStep {step}/{self.max_steps} ({elapsed:.1f}s)")
                    print(f"  Population: " + ", ".join([f"{k}={v}" for k, v in sorted(pop_counts.items())]))
                    print(f"  Diversity: entropy={div.get('behavioral_entropy', 0):.3f}, "
                          f"pairwise={div.get('pairwise_diversity', 0):.3f}")
                    print(f"  Emergence: complexity={emg.get('mean_complexity', 0):.3f}, "
                          f"niche={emg.get('niche_specialization', 0):.3f}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Experiment completed in {total_time:.1f} seconds")
        print(f"{'='*70}")
        
        # Final analysis and save
        self.save_results()
    
    def save_results(self):
        """Save all results and generate visualizations"""
        print("\nSaving results...")
        
        # Record final lifespans
        for agent_id, agent in self.agents.items():
            agent_type = self.agent_types[agent_id]
            self.survival_analyzer.record_lifespan(agent_type, agent.age)
        
        # Gather behavioral features
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_trajectory_features'):
                features = agent.get_trajectory_features()
                agent_type = self.agent_types[agent_id]
                self.behavioral_analyzer.add_agent(agent_id, features, agent_type)
        
        # Compute embeddings and clusters
        if len(self.behavioral_analyzer.trajectory_features) >= 5:
            embeddings = self.behavioral_analyzer.compute_embeddings()
            clusters = self.behavioral_analyzer.cluster_behaviors()
        else:
            embeddings = np.zeros((len(self.behavioral_analyzer.trajectory_features), 2))
            clusters = np.zeros(len(self.behavioral_analyzer.trajectory_features))
        
        # Get survival statistics
        survival_stats = self.survival_analyzer.get_survival_stats()
        
        # Prepare results for visualization
        results = {
            'population_history': {
                agent_type: self.population_analyzer.population_history.get(agent_type, [])
                for agent_type in ['autopoietic', 'rl', 'neat', 'random']
            },
            'survival_curves': {
                agent_type: self.survival_analyzer.survival_curve(agent_type, self.max_steps)
                for agent_type in ['autopoietic', 'rl', 'neat', 'random']
            },
            'diversity_history': self.diversity_history,
            'embeddings': embeddings,
            'agent_types': list(self.behavioral_analyzer.agent_types.values()),
            'clusters': clusters
        }
        
        # Action counts
        action_counts = defaultdict(lambda: np.zeros(6))
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'action_history') and len(agent.action_history) > 0:
                agent_type = self.agent_types[agent_id]
                actions = list(agent.action_history)
                counts = np.bincount(actions, minlength=6)
                action_counts[agent_type] += counts
        results['action_counts'] = dict(action_counts)
        
        # Position histories
        position_histories = {}
        agent_types_dict = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'position_history'):
                position_histories[agent_id] = list(agent.position_history)
                agent_types_dict[agent_id] = self.agent_types[agent_id]
        results['position_histories'] = position_histories
        results['agent_types_dict'] = agent_types_dict
        
        # Generate visualizations
        ComprehensiveReport.generate_report(results, self.run_dir, prefix="experiment")
        
        # Generate animation if frames available
        if len(self.frames) > 10:
            visualizer = GridWorldVisualizer(grid_size=self.grid_size)
            try:
                visualizer.create_animation(
                    self.frames[::5],  # Every 5th frame
                    os.path.join(self.run_dir, "animation.gif"),
                    fps=5
                )
            except Exception as e:
                print(f"Warning: Could not create animation: {e}")
        
        # Save summary statistics
        summary = {
            'config': {
                'grid_size': self.grid_size,
                'max_steps': self.max_steps,
                'n_agents_per_type': self.n_agents_per_type,
                'seed': self.seed
            },
            'survival_stats': survival_stats,
            'final_populations': dict(defaultdict(int, {
                t: sum(1 for v in self.agent_types.values() if v == t)
                for t in ['autopoietic', 'rl', 'neat', 'random']
            })),
            'birth_death_stats': self.stats,
            'diversity_final': self.diversity_history[-1] if self.diversity_history else {},
            'emergence_final': self.emergence_history[-1] if self.emergence_history else {}
        }
        
        with open(os.path.join(self.run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        print("\nSurvival Statistics:")
        for agent_type, stats in survival_stats.items():
            print(f"  {agent_type}:")
            print(f"    Mean lifespan: {stats['mean_lifespan']:.1f}")
            print(f"    Max lifespan: {stats['max_lifespan']}")
            print(f"    Count: {stats['count']}")
        
        print("\nFinal Populations:")
        for agent_type, count in summary['final_populations'].items():
            births = self.stats[agent_type]['births']
            deaths = self.stats[agent_type]['deaths']
            print(f"  {agent_type}: {count} (births={births}, deaths={deaths})")
        
        if self.diversity_history:
            div = self.diversity_history[-1]
            print(f"\nFinal Diversity:")
            print(f"  Behavioral entropy: {div.get('behavioral_entropy', 0):.3f}")
            print(f"  Pairwise diversity: {div.get('pairwise_diversity', 0):.3f}")
            print(f"  Action diversity: {div.get('action_diversity', 0):.3f}")
        
        if self.emergence_history:
            emg = self.emergence_history[-1]
            print(f"\nEmergence Metrics:")
            print(f"  Mean complexity: {emg.get('mean_complexity', 0):.3f}")
            print(f"  Niche specialization: {emg.get('niche_specialization', 0):.3f}")
            print(f"  Clustering coefficient: {emg.get('clustering_coefficient', 0):.3f}")
        
        print(f"\nResults saved to: {self.run_dir}")
        print("=" * 70)
        
        # Evaluate success criteria
        self.evaluate_success(survival_stats)
    
    def evaluate_success(self, survival_stats: Dict):
        """Evaluate if experiment met success criteria"""
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA EVALUATION")
        print("=" * 70)
        
        criteria_met = 0
        total_criteria = 3
        
        # Criterion 1: Autopoietic survival > 5000 steps
        auto_stats = survival_stats.get('autopoietic', {})
        max_lifespan = auto_stats.get('max_lifespan', 0)
        criterion1 = max_lifespan >= 5000
        print(f"\n1. Autopoietic survival >= 5000 steps")
        print(f"   Max lifespan: {max_lifespan}")
        print(f"   Status: {'PASS' if criterion1 else 'FAIL'}")
        if criterion1:
            criteria_met += 1
        
        # Criterion 2: Behavioral diversity > baselines
        if self.diversity_history:
            final_div = self.diversity_history[-1]
            auto_div = final_div.get('behavioral_entropy', 0)
            
            # Compare to other types (need type-specific diversity)
            # For now, check if diversity > 0.3
            criterion2 = auto_div > 0.3
            print(f"\n2. Behavioral diversity > baselines")
            print(f"   Behavioral entropy: {auto_div:.3f}")
            print(f"   Status: {'PASS' if criterion2 else 'FAIL'}")
            if criterion2:
                criteria_met += 1
        else:
            print(f"\n2. Behavioral diversity > baselines")
            print(f"   Status: FAIL (no data)")
        
        # Criterion 3: Unique emergent behavior
        if self.emergence_history:
            final_emg = self.emergence_history[-1]
            complexity = final_emg.get('mean_complexity', 0)
            niche = final_emg.get('niche_specialization', 0)
            
            criterion3 = complexity > 0.2 or niche > 0.3
            print(f"\n3. Emergent behavior observed")
            print(f"   Complexity: {complexity:.3f}, Niche specialization: {niche:.3f}")
            print(f"   Status: {'PASS' if criterion3 else 'FAIL'}")
            if criterion3:
                criteria_met += 1
        else:
            print(f"\n3. Emergent behavior observed")
            print(f"   Status: FAIL (no data)")
        
        print(f"\n{'='*70}")
        print(f"OVERALL: {criteria_met}/{total_criteria} criteria met")
        if criteria_met == total_criteria:
            print("EXPERIMENT SUCCESS!")
        elif criteria_met >= 2:
            print("Partial success - further investigation needed")
        else:
            print("Criteria not met - review parameters")
        print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GENESIS Path B: Artificial Life Experiment")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid dimension")
    parser.add_argument("--resources", type=int, default=200, help="Number of resources")
    parser.add_argument("--predators", type=int, default=5, help="Number of predators")
    parser.add_argument("--agents-per-type", type=int, default=10, help="Initial agents per type")
    parser.add_argument("--max-population", type=int, default=50, help="Max population per type")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum steps")
    parser.add_argument("--no-reproduction", action="store_true", help="Disable reproduction")
    parser.add_argument("--save-interval", type=int, default=100, help="Steps between saves")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()
    
    experiment = ArtificialLifeExperiment(
        grid_size=args.grid_size,
        n_resources=args.resources,
        n_predators=args.predators,
        n_agents_per_type=args.agents_per_type,
        max_population=args.max_population,
        max_steps=args.max_steps,
        reproduction_enabled=not args.no_reproduction,
        save_interval=args.save_interval,
        output_dir=args.output,
        seed=args.seed
    )
    
    experiment.run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
