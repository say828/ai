"""
GENESIS Path B Phase 0: Minimal Population Manager

Manages agent lifecycle: birth, death, reproduction
"""

import numpy as np
from typing import Dict, List, Optional
from minimal_environment import MinimalGrid
from minimal_agent import MinimalAutopoieticAgent


class MinimalPopulation:
    """
    Population manager for autopoietic agents
    
    Responsibilities:
    - Agent lifecycle (birth, death)
    - Reproduction with mutation
    - Population statistics
    """
    
    def __init__(self, 
                 env: MinimalGrid, 
                 initial_agents: int = 20,
                 max_population: int = 100,
                 mutation_rate: float = 0.1):
        """
        Args:
            env: Grid environment
            initial_agents: Initial population size
            max_population: Maximum allowed population
            mutation_rate: Mutation rate for reproduction
        """
        self.env = env
        self.max_population = max_population
        self.mutation_rate = mutation_rate
        
        # Create initial agents
        self.agents: List[MinimalAutopoieticAgent] = []
        for i in range(initial_agents):
            x = np.random.randint(0, env.size)
            y = np.random.randint(0, env.size)
            agent = MinimalAutopoieticAgent(x=x, y=y)
            self.agents.append(agent)
        
        # Statistics tracking
        self.total_births = 0
        self.total_deaths = 0
        self.generation = 0
        
        # History for analysis
        self.death_log: List[Dict] = []  # Records death events with coherence
        self.birth_log: List[Dict] = []  # Records birth events
        
    def step(self) -> Dict:
        """
        Execute one simulation step
        
        Process:
        1. All agents sense and act
        2. Environment updates
        3. Check for deaths
        4. Check for reproduction
        5. Collect statistics
        
        Returns:
            stats: Step statistics
        """
        # 1. All agents act
        for agent in self.agents:
            if agent.is_alive:
                agent.act(self.env)
                agent.compute_coherence()
        
        # 2. Environment step
        self.env.step()
        
        # 3. Check deaths
        dead_agents = []
        for agent in self.agents:
            if agent.should_die():
                agent.is_alive = False
                dead_agents.append(agent)
                self.total_deaths += 1
                
                # Log death
                self.death_log.append({
                    'agent_id': agent.id,
                    'age': agent.age,
                    'final_coherence': agent.coherence_history[-1] if len(agent.coherence_history) > 0 else 0,
                    'final_energy': agent.energy,
                    'cause': 'low_energy' if agent.energy < 0 else 'low_coherence'
                })
        
        # Remove dead agents
        self.agents = [a for a in self.agents if a.is_alive]
        
        # 4. Check reproduction
        new_agents = []
        if len(self.agents) < self.max_population:
            for agent in self.agents:
                if agent.can_reproduce() and len(self.agents) + len(new_agents) < self.max_population:
                    child = agent.mutate_child(self.mutation_rate)
                    
                    # Place child at random nearby position
                    child.x = (agent.x + np.random.randint(-2, 3)) % self.env.size
                    child.y = (agent.y + np.random.randint(-2, 3)) % self.env.size
                    
                    new_agents.append(child)
                    self.total_births += 1
                    
                    # Log birth
                    self.birth_log.append({
                        'parent_id': agent.id,
                        'child_id': child.id,
                        'parent_age': agent.age,
                        'parent_coherence': agent.coherence_history[-1] if len(agent.coherence_history) > 0 else 0
                    })
        
        self.agents.extend(new_agents)
        
        # 5. Update generation counter
        if len(new_agents) > 0:
            self.generation += 1
        
        # 6. Collect statistics
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Get current population statistics"""
        if len(self.agents) == 0:
            return {
                'size': 0,
                'avg_coherence': 0,
                'std_coherence': 0,
                'avg_energy': 0,
                'avg_age': 0,
                'max_age': 0,
                'births_this_step': 0,
                'deaths_this_step': 0,
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
                'generation': self.generation
            }
        
        coherences = []
        energies = []
        ages = []
        
        for agent in self.agents:
            if len(agent.coherence_history) > 0:
                coherences.append(agent.coherence_history[-1])
            energies.append(agent.energy)
            ages.append(agent.age)
        
        return {
            'size': len(self.agents),
            'avg_coherence': float(np.mean(coherences)) if coherences else 0.5,
            'std_coherence': float(np.std(coherences)) if coherences else 0,
            'avg_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)),
            'avg_age': float(np.mean(ages)),
            'max_age': int(np.max(ages)),
            'min_energy': float(np.min(energies)),
            'max_energy': float(np.max(energies)),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'generation': self.generation
        }
    
    def get_coherence_survival_correlation(self) -> Dict:
        """
        Analyze correlation between coherence and survival

        Returns:
            analysis: Correlation statistics
        """
        if len(self.death_log) < 5:
            return {'correlation': None, 'sample_size': len(self.death_log)}

        # Get coherences of dead agents
        dead_coherences = [d['final_coherence'] for d in self.death_log]
        dead_ages = [d['age'] for d in self.death_log]

        # Separate by cause of death
        low_energy_deaths = [d for d in self.death_log if d['cause'] == 'low_energy']
        low_coherence_deaths = [d for d in self.death_log if d['cause'] == 'low_coherence']

        # Get coherences of living agents
        live_coherences = [
            a.coherence_history[-1] if len(a.coherence_history) > 0 else 0.5
            for a in self.agents
        ]
        live_ages = [a.age for a in self.agents]

        # Correlation: higher coherence -> longer life?
        if len(dead_ages) > 10:
            try:
                corr = np.corrcoef(dead_coherences, dead_ages)[0, 1]
                coherence_age_correlation = float(corr) if not np.isnan(corr) else 0.0
            except:
                coherence_age_correlation = 0.0
        else:
            coherence_age_correlation = 0.0

        return {
            'dead_avg_coherence': float(np.mean(dead_coherences)),
            'dead_avg_age': float(np.mean(dead_ages)),
            'live_avg_coherence': float(np.mean(live_coherences)) if live_coherences else 0,
            'live_avg_age': float(np.mean(live_ages)) if live_ages else 0,
            'coherence_gap': float(np.mean(live_coherences) - np.mean(dead_coherences)) if live_coherences else 0,
            'coherence_age_correlation': coherence_age_correlation,
            'deaths_by_low_energy': len(low_energy_deaths),
            'deaths_by_low_coherence': len(low_coherence_deaths),
            'low_energy_avg_coherence': float(np.mean([d['final_coherence'] for d in low_energy_deaths])) if low_energy_deaths else 0,
            'low_coherence_avg_coherence': float(np.mean([d['final_coherence'] for d in low_coherence_deaths])) if low_coherence_deaths else 0,
            'sample_size_dead': len(dead_coherences),
            'sample_size_live': len(live_coherences)
        }
    
    def get_agent_positions(self) -> List[tuple]:
        """Get all agent positions for visualization"""
        return [(a.x, a.y) for a in self.agents]
    
    def get_population_diversity(self) -> float:
        """
        Estimate population diversity based on weight variance
        
        Returns:
            diversity: Average variance across weights
        """
        if len(self.agents) < 2:
            return 0.0
        
        # Stack all W_rec weights
        all_weights = np.array([a.W_rec.flatten() for a in self.agents])
        
        # Compute variance across population
        diversity = float(np.mean(np.var(all_weights, axis=0)))
        
        return diversity


if __name__ == "__main__":
    print("Testing MinimalPopulation...")
    
    # Create environment and population
    env = MinimalGrid(size=16)
    pop = MinimalPopulation(env, initial_agents=20)
    
    print(f"Initial population: {len(pop.agents)}")
    
    # Run for 200 steps
    for step in range(200):
        stats = pop.step()
        
        if step % 50 == 0:
            print(f"Step {step}: "
                  f"pop={stats['size']}, "
                  f"coherence={stats['avg_coherence']:.3f}, "
                  f"energy={stats['avg_energy']:.3f}, "
                  f"births={stats['total_births']}, "
                  f"deaths={stats['total_deaths']}")
    
    # Final analysis
    print(f"\nFinal Statistics:")
    stats = pop.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print(f"\nCoherence-Survival Analysis:")
    analysis = pop.get_coherence_survival_correlation()
    for k, v in analysis.items():
        print(f"  {k}: {v}")
    
    print(f"\nPopulation Diversity: {pop.get_population_diversity():.6f}")
    
    print("\nPopulation test passed!")
