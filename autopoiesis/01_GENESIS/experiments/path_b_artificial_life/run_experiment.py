"""
GENESIS Path B: Simplified Experiment Runner
=============================================

Lightweight version for testing without full RL overhead.

Author: GENESIS Project
Date: 2026-01-04
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid_world import GridWorld
from autopoietic_grid_agent import AutopoieticGridAgent
from baseline_agents import RandomAgent, NEATAgent
from analysis import (
    BehavioralAnalyzer, DiversityMetrics, EmergenceDetector,
    SurvivalAnalyzer, run_comprehensive_analysis
)
from visualize import (
    GridWorldVisualizer, PopulationPlotter, BehaviorVisualizer,
    NicheVisualizer
)


def run_experiment(
    max_steps: int = 5000,
    n_agents_per_type: int = 10,
    grid_size: int = 100,
    n_resources: int = 200,
    n_predators: int = 5,
    max_population: int = 30,
    save_interval: int = 200,
    output_dir: str = "results",
    seed: int = 42,
    use_rl: bool = False  # Disable RL by default for stability
):
    """Run simplified experiment"""
    
    np.random.seed(seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENESIS Path B: Artificial Life Experiment")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Resources: {n_resources}")
    print(f"  Predators: {n_predators}")
    print(f"  Agents per type: {n_agents_per_type}")
    print(f"  Max steps: {max_steps}")
    print(f"  Output: {run_dir}")
    
    # Initialize world
    world = GridWorld(
        size=grid_size,
        n_resources=n_resources,
        n_predators=n_predators,
        seed=seed
    )
    
    # Agent storage
    agents: Dict[str, object] = {}
    agent_types: Dict[str, str] = {}
    
    # Create agents
    print("\nInitializing populations...")
    
    agent_classes = {
        'autopoietic': AutopoieticGridAgent,
        'neat': NEATAgent,
        'random': RandomAgent
    }
    
    if use_rl:
        from baseline_agents import RLAgent
        agent_classes['rl'] = RLAgent
    
    for agent_type, AgentClass in agent_classes.items():
        for i in range(n_agents_per_type):
            agent_id = f"{agent_type}_{i}"
            agent = AgentClass(agent_id=agent_id)
            world.add_agent(agent_id)
            agents[agent_id] = agent
            agent_types[agent_id] = agent_type
        print(f"  {agent_type}: {n_agents_per_type} agents")
    
    # Analysis tools
    survival_analyzer = SurvivalAnalyzer()
    
    # History
    population_history = defaultdict(list)
    diversity_history = []
    emergence_history = []
    frames = []
    
    # Stats
    stats = {t: {'births': 0, 'deaths': 0} for t in agent_classes.keys()}
    
    # Visualization
    visualizer = GridWorldVisualizer(grid_size=grid_size)
    
    print("\nRunning simulation...")
    start_time = time.time()
    
    for step in range(max_steps):
        # Count populations
        pop_counts = defaultdict(int)
        for t in agent_types.values():
            pop_counts[t] += 1
        
        for agent_type, count in pop_counts.items():
            population_history[agent_type].append((step, count))
        
        # Check extinction
        if sum(pop_counts.values()) == 0:
            print(f"\nAll populations extinct at step {step}!")
            break
        
        # Process agents
        dead_agents = []
        new_agents = []
        
        for agent_id, agent in list(agents.items()):
            if not agent.is_alive:
                dead_agents.append(agent_id)
                continue
            
            # Perceive and act
            obs = world.get_local_observation(agent_id)
            action = agent.perceive_and_act(obs)
            result = world.step_agent(agent_id, action)
            pos = world.agent_positions.get(agent_id, (0, 0))
            agent.update_state(result, pos)
            
            # Check death
            if not agent.is_alive or result.get('hit_predator'):
                dead_agents.append(agent_id)
                agent_type = agent_types[agent_id]
                survival_analyzer.record_death(agent_type, agent.age, "energy_or_predator")
                stats[agent_type]['deaths'] += 1
            
            # Check reproduction
            elif agent.can_reproduce:
                agent_type = agent_types[agent_id]
                type_count = sum(1 for t in agent_types.values() if t == agent_type)
                
                if type_count < max_population:
                    offspring = agent.reproduce(mutation_rate=0.1)
                    new_agents.append((offspring, agent_type))
                    stats[agent_type]['births'] += 1
        
        # Remove dead
        for agent_id in set(dead_agents):
            if agent_id in world.agent_positions:
                world.remove_agent(agent_id)
            if agent_id in agents:
                del agents[agent_id]
            if agent_id in agent_types:
                del agent_types[agent_id]
        
        # Add new
        for offspring, agent_type in new_agents:
            world.add_agent(offspring.id)
            agents[offspring.id] = offspring
            agent_types[offspring.id] = agent_type
        
        # Update world
        world.step_world()
        
        # Periodic analysis
        if step % save_interval == 0:
            elapsed = time.time() - start_time
            
            # Analyze
            position_histories = {}
            for agent_id, agent in agents.items():
                if hasattr(agent, 'position_history'):
                    position_histories[agent_id] = list(agent.position_history)
            
            analysis = run_comprehensive_analysis(
                list(agents.values()),
                dict(world.agent_positions),
                position_histories,
                grid_size
            )
            
            diversity_history.append(analysis.get('diversity', {}))
            emergence_history.append(analysis.get('emergence', {}))
            
            # Save frame
            agent_trails = position_histories
            frame = visualizer.render_frame(
                world.grid,
                dict(world.agent_positions),
                agent_types,
                agent_trails,
                step
            )
            frames.append(frame)
            
            div = analysis.get('diversity', {})
            emg = analysis.get('emergence', {})
            
            print(f"\nStep {step}/{max_steps} ({elapsed:.1f}s)")
            pop_str = ", ".join([f"{k}={v}" for k, v in sorted(pop_counts.items())])
            print(f"  Population: {pop_str}")
            print(f"  Diversity: entropy={div.get('behavioral_entropy', 0):.3f}")
            print(f"  Emergence: complexity={emg.get('mean_complexity', 0):.3f}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Experiment completed in {total_time:.1f} seconds")
    print(f"{'='*70}")
    
    # Save results
    print("\nSaving results...")
    
    # Record final lifespans
    for agent_id, agent in agents.items():
        agent_type = agent_types.get(agent_id, 'unknown')
        survival_analyzer.record_lifespan(agent_type, agent.age)
    
    survival_stats = survival_analyzer.get_survival_stats()
    
    # Behavioral analysis
    behavioral = BehavioralAnalyzer()
    for agent_id, agent in agents.items():
        if hasattr(agent, 'get_trajectory_features'):
            features = agent.get_trajectory_features()
            agent_type = agent_types.get(agent_id, 'unknown')
            behavioral.add_agent(agent_id, features, agent_type)
    
    embeddings = None
    clusters = None
    if len(behavioral.trajectory_features) >= 5:
        embeddings = behavioral.compute_embeddings()
        clusters = behavioral.cluster_behaviors()
    
    # Generate plots
    if len(population_history) > 0:
        PopulationPlotter.plot_population_curves(
            dict(population_history),
            os.path.join(run_dir, 'population.png')
        )
    
    if len(survival_stats) > 0:
        survival_curves = {
            t: survival_analyzer.survival_curve(t, max_steps)
            for t in survival_stats.keys()
        }
        PopulationPlotter.plot_survival_curves(
            survival_curves,
            os.path.join(run_dir, 'survival.png')
        )
    
    if embeddings is not None and len(embeddings) > 0:
        agent_type_list = list(behavioral.agent_types.values())
        BehaviorVisualizer.plot_behavioral_embedding(
            embeddings,
            agent_type_list,
            clusters,
            os.path.join(run_dir, 'embedding.png')
        )
    
    # Action distribution
    action_counts = defaultdict(lambda: np.zeros(6))
    for agent_id, agent in agents.items():
        if hasattr(agent, 'action_history') and len(agent.action_history) > 0:
            agent_type = agent_types.get(agent_id, 'unknown')
            actions = list(agent.action_history)
            counts = np.bincount(actions, minlength=6)
            action_counts[agent_type] += counts
    
    if len(action_counts) > 0:
        BehaviorVisualizer.plot_action_distribution(
            dict(action_counts),
            os.path.join(run_dir, 'actions.png')
        )
    
    # Niche heatmap
    position_histories = {}
    agent_types_dict = {}
    for agent_id, agent in agents.items():
        if hasattr(agent, 'position_history') and len(agent.position_history) > 0:
            position_histories[agent_id] = list(agent.position_history)
            agent_types_dict[agent_id] = agent_types.get(agent_id, 'unknown')
    
    if len(position_histories) > 0:
        NicheVisualizer.plot_niche_heatmap(
            position_histories,
            agent_types_dict,
            grid_size,
            os.path.join(run_dir, 'niches.png')
        )
    
    # Animation
    if len(frames) > 5:
        try:
            visualizer.create_animation(
                frames,
                os.path.join(run_dir, 'animation.gif'),
                fps=3
            )
        except Exception as e:
            print(f"Could not create animation: {e}")
    
    # Summary JSON
    final_pops = defaultdict(int)
    for t in agent_types.values():
        final_pops[t] += 1
    
    summary = {
        'config': {
            'grid_size': grid_size,
            'max_steps': max_steps,
            'n_agents_per_type': n_agents_per_type,
            'seed': seed
        },
        'survival_stats': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) 
                               for kk, vv in v.items()} 
                          for k, v in survival_stats.items()},
        'final_populations': dict(final_pops),
        'birth_death_stats': stats
    }
    
    if diversity_history:
        summary['final_diversity'] = diversity_history[-1]
    if emergence_history:
        summary['final_emergence'] = emergence_history[-1]
    
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print("\nSurvival Statistics:")
    for agent_type, s in survival_stats.items():
        print(f"  {agent_type}: mean={s['mean_lifespan']:.0f}, max={s['max_lifespan']}")
    
    print("\nFinal Populations:")
    for agent_type, count in sorted(final_pops.items()):
        b = stats.get(agent_type, {}).get('births', 0)
        d = stats.get(agent_type, {}).get('deaths', 0)
        print(f"  {agent_type}: {count} (births={b}, deaths={d})")
    
    if diversity_history:
        div = diversity_history[-1]
        print(f"\nFinal Diversity:")
        print(f"  Behavioral entropy: {div.get('behavioral_entropy', 0):.3f}")
        print(f"  Pairwise diversity: {div.get('pairwise_diversity', 0):.3f}")
    
    print(f"\nResults saved to: {run_dir}")
    
    # Evaluate success
    print("\n" + "=" * 70)
    print("SUCCESS EVALUATION")
    print("=" * 70)
    
    success_count = 0
    
    # Criterion 1: Survival
    auto_stats = survival_stats.get('autopoietic', {})
    max_lifespan = auto_stats.get('max_lifespan', 0)
    crit1 = max_lifespan >= max_steps * 0.5
    print(f"\n1. Autopoietic survival >= {max_steps * 0.5:.0f} steps")
    print(f"   Max lifespan: {max_lifespan}")
    print(f"   Status: {'PASS' if crit1 else 'FAIL'}")
    if crit1:
        success_count += 1
    
    # Criterion 2: Diversity
    if diversity_history:
        final_div = diversity_history[-1]
        entropy = final_div.get('behavioral_entropy', 0)
        crit2 = entropy > 0.2
        print(f"\n2. Behavioral diversity > 0.2")
        print(f"   Entropy: {entropy:.3f}")
        print(f"   Status: {'PASS' if crit2 else 'FAIL'}")
        if crit2:
            success_count += 1
    
    # Criterion 3: Emergence
    if emergence_history:
        final_emg = emergence_history[-1]
        complexity = final_emg.get('mean_complexity', 0)
        crit3 = complexity > 0.1
        print(f"\n3. Emergent complexity > 0.1")
        print(f"   Complexity: {complexity:.3f}")
        print(f"   Status: {'PASS' if crit3 else 'FAIL'}")
        if crit3:
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {success_count}/3 criteria met")
    print("=" * 70)
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GENESIS Path B Experiment")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--agents-per-type", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--resources", type=int, default=200)
    parser.add_argument("--predators", type=int, default=5)
    parser.add_argument("--max-population", type=int, default=30)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-rl", action="store_true", help="Include RL agents")
    
    args = parser.parse_args()
    
    run_experiment(
        max_steps=args.max_steps,
        n_agents_per_type=args.agents_per_type,
        grid_size=args.grid_size,
        n_resources=args.resources,
        n_predators=args.predators,
        max_population=args.max_population,
        save_interval=args.save_interval,
        output_dir=args.output,
        seed=args.seed,
        use_rl=args.with_rl
    )
