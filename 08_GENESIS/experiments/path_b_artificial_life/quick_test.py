"""
Quick integration test for Path B
"""

import numpy as np
import sys
import os

# Add current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Quick Integration Test")
print("=" * 70)

# Test 1: Grid World + Agents
print("\n1. Testing GridWorld + Agents Integration...")

from grid_world import GridWorld
from autopoietic_grid_agent import AutopoieticGridAgent
from baseline_agents import RandomAgent, NEATAgent

world = GridWorld(size=50, n_resources=50, n_predators=2, seed=42)

# Add one agent of each type
agents = {
    'auto_1': AutopoieticGridAgent(agent_id='auto_1'),
    'random_1': RandomAgent(agent_id='random_1'),
    'neat_1': NEATAgent(agent_id='neat_1')
}

agent_types = {
    'auto_1': 'autopoietic',
    'random_1': 'random', 
    'neat_1': 'neat'
}

# Place agents
for agent_id in agents:
    world.add_agent(agent_id)

print(f"   World created: {world.size}x{world.size}")
print(f"   Agents: {len(agents)}")

# Run 200 steps
print("\n2. Running 200 steps...")
for step in range(200):
    dead = []
    
    for agent_id, agent in agents.items():
        if not agent.is_alive:
            dead.append(agent_id)
            continue
            
        # Get observation and act
        obs = world.get_local_observation(agent_id)
        action = agent.perceive_and_act(obs)
        result = world.step_agent(agent_id, action)
        pos = world.agent_positions.get(agent_id, (0, 0))
        agent.update_state(result, pos)
        
        if result.get('hit_predator'):
            dead.append(agent_id)
    
    # Remove dead
    for agent_id in dead:
        if agent_id in world.agent_positions:
            world.remove_agent(agent_id)
    
    world.step_world()
    
    if step % 50 == 0:
        alive = sum(1 for a in agents.values() if a.is_alive)
        print(f"   Step {step}: {alive} agents alive")

# Summary
print("\n3. Results:")
for agent_id, agent in agents.items():
    summary = agent.get_summary()
    print(f"   {agent_id} ({agent_types[agent_id]}): "
          f"alive={summary['is_alive']}, age={summary['age']}, "
          f"energy={summary.get('energy', 0):.3f}, consumed={summary['total_consumed']}")

# Test 3: Analysis
print("\n4. Testing Analysis...")
from analysis import BehavioralAnalyzer, DiversityMetrics

behavioral = BehavioralAnalyzer()
for agent_id, agent in agents.items():
    if hasattr(agent, 'get_trajectory_features'):
        features = agent.get_trajectory_features()
        behavioral.add_agent(agent_id, features, agent_types[agent_id])

if len(behavioral.trajectory_features) >= 3:
    features = np.array(list(behavioral.trajectory_features.values()))
    entropy = DiversityMetrics.behavioral_entropy(features)
    diversity = DiversityMetrics.pairwise_diversity(features)
    print(f"   Behavioral entropy: {entropy:.3f}")
    print(f"   Pairwise diversity: {diversity:.3f}")

print("\n" + "=" * 70)
print("Quick test PASSED!")
print("=" * 70)
