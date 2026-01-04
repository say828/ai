"""
GENESIS Path B Phase 1: Full Population Manager

Complete population management with:
- Agent lifecycle (birth, death, reproduction)
- Spatial collision handling
- Phylogenetic tree tracking
- Quality-Diversity metrics
- Comprehensive statistics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import json

from full_environment import FullALifeEnvironment
from full_agent import FullAutopoieticAgent
from teacher_network import TeacherNetwork, EpisodicMemory


@dataclass
class PhylogenyNode:
    """Node in phylogenetic tree"""
    agent_id: int
    parent_id: Optional[int]
    birth_time: int
    death_time: Optional[int] = None
    final_age: int = 0
    final_coherence: float = 0.5
    offspring_ids: List[int] = field(default_factory=list)
    behavior_descriptor: Optional[np.ndarray] = None


class FullPopulationManager:
    """
    Complete population manager for Phase 1
    
    Features:
    - Manages 100+ agents on 64x64 grid
    - Tracks phylogenetic relationships
    - Computes Quality-Diversity metrics
    - Handles spatial collisions and interactions
    """
    
    def __init__(self,
                 env: FullALifeEnvironment,
                 initial_pop: int = 100,
                 max_population: int = 500,
                 mutation_rate: float = 0.05,
                 mutation_scale: float = 0.1,
                 min_population: int = 50,
                 enable_teacher: bool = True,
                 teacher_update_interval: int = 100,
                 teacher_learning_rate: float = 0.1):
        """
        Args:
            env: Environment instance
            initial_pop: Initial population size
            max_population: Maximum population cap
            mutation_rate: Probability of mutating each weight
            mutation_scale: Standard deviation of mutations
            min_population: Minimum population to maintain (prevents extinction)
            enable_teacher: Enable Teacher Network for infinite learning
            teacher_update_interval: Steps between teacher updates
            teacher_learning_rate: Teacher EMA learning rate
        """
        self.env = env
        self.max_population = max_population
        self.min_population = min_population
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.enable_teacher = enable_teacher
        self.teacher_update_interval = teacher_update_interval

        # Teacher Network for infinite learning
        if self.enable_teacher:
            self.teacher = TeacherNetwork(
                state_dim=128,
                sensor_dim=370,
                action_dim=5,
                learning_rate=teacher_learning_rate
            )
            self.memory = EpisodicMemory(capacity=10000)
        else:
            self.teacher = None
            self.memory = None
        
        # Agent tracking
        self.agents: List[FullAutopoieticAgent] = []
        self.next_id = 0
        self.current_step = 0
        
        # Initialize population
        for _ in range(initial_pop):
            x = np.random.randint(0, env.size)
            y = np.random.randint(0, env.size)
            agent = FullAutopoieticAgent(x=x, y=y, agent_id=self.next_id)
            agent.birth_time = 0
            self.agents.append(agent)
            self.next_id += 1
        
        # Phylogeny tracking
        self.phylogeny: Dict[int, PhylogenyNode] = {}
        for agent in self.agents:
            self.phylogeny[agent.id] = PhylogenyNode(
                agent_id=agent.id,
                parent_id=None,
                birth_time=0
            )
        
        # Statistics
        self.total_births = 0
        self.total_deaths = 0
        self.generation_counter = 0
        
        # History logs
        self.death_log: List[Dict] = []
        self.birth_log: List[Dict] = []
        self.lineage_extinction_log: List[Dict] = []
        
        # QD Archive (behavior space discretization)
        self.bd_grid_resolution = 10  # 10x10x... grid for each BD dimension
        self.qd_archive: Dict[Tuple, Dict] = {}  # (cell_indices) -> {fitness, agent_snapshot}
        
    def step(self) -> Dict:
        """
        Execute one simulation step
        
        Order:
        1. Update occupation grid
        2. All agents sense & act
        3. Environment step (resource regeneration)
        4. Metabolism & death check
        5. Reproduction
        6. Update QD archive
        7. Compute statistics
        
        Returns:
            Statistics dictionary
        """
        self.current_step += 1
        
        # 1. Update occupation grid
        positions = [(a.x, a.y) for a in self.agents]
        self.env.set_occupation(positions)
        
        # Build spatial index for nearby queries
        spatial_index = self._build_spatial_index()
        
        # 2. All agents sense & act
        for agent in self.agents:
            # Get nearby agents for sensing
            nearby = self._get_nearby_agents(agent, spatial_index, radius=7)
            
            # Sense environment
            sensor = agent.sense(self.env, nearby)
            
            # Forward pass (compute action)
            action = agent.forward(sensor)
            
            # Execute action
            self._execute_action(agent, action)
            
            # Compute coherence
            agent.compute_coherence()
        
        # 3. Environment step
        self.env.step()
        
        # 4. Metabolism & death check
        deaths = []
        for agent in self.agents:
            # Metabolic cost
            last_action = agent.action_history[-1] if agent.action_history else np.zeros(5)
            cost = agent.metabolic_cost(last_action)
            agent.energy -= cost
            
            # Age
            agent.age += 1
            
            # Death check
            if agent.should_die():
                agent.is_alive = False
                deaths.append(agent)
        
        # Process deaths
        for agent in deaths:
            self._process_death(agent)
        
        self.agents = [a for a in self.agents if a.is_alive]
        
        # 5. Reproduction
        offspring = []
        if len(self.agents) < self.max_population:
            for agent in self.agents:
                if agent.can_reproduce() and len(self.agents) + len(offspring) < self.max_population:
                    child = self._process_reproduction(agent)
                    offspring.append(child)
        
        self.agents.extend(offspring)

        # Increment generation if births occurred
        if offspring:
            self.generation_counter += 1

        # 6. Update Teacher Network from elite agents (if enabled)
        if self.enable_teacher and self.current_step % self.teacher_update_interval == 0:
            elite_agents = self._get_elite_agents(top_k_percent=0.2)  # Top 20%
            if elite_agents:
                teacher_stats = self.teacher.distill_from_elite(elite_agents, verbose=False)

                # Log teacher update (only on significant intervals)
                if self.current_step % (self.teacher_update_interval * 10) == 0:
                    print(f"  📚 Teacher Update #{teacher_stats['update_count']}: "
                          f"Elite={teacher_stats['n_elite']}, "
                          f"Coherence={teacher_stats['avg_elite_coherence']:.3f}, "
                          f"Knowledge={teacher_stats['teacher_knowledge_level']:.3f}")

        # 7. Maintain minimum population (prevent extinction)
        if len(self.agents) < self.min_population:
            needed = self.min_population - len(self.agents)
            spawned = self._spawn_agents_from_teacher(needed)
            if spawned and self.current_step % 100 == 0:
                print(f"  🔄 Spawned {len(spawned)} agents from Teacher (pop: {len(self.agents)} → {len(self.agents) + len(spawned)})")
            self.agents.extend(spawned)

        # 8. Update QD archive (every 10 steps)
        if self.current_step % 10 == 0:
            self._update_qd_archive()

        # 9. Compute and return statistics
        return self.get_statistics()
    
    def _build_spatial_index(self) -> Dict[Tuple[int, int], List[FullAutopoieticAgent]]:
        """Build spatial hash for efficient nearby queries"""
        index = defaultdict(list)
        cell_size = 8  # 8x8 cells
        
        for agent in self.agents:
            cell_x = agent.x // cell_size
            cell_y = agent.y // cell_size
            index[(cell_x, cell_y)].append(agent)
        
        return index
    
    def _get_nearby_agents(self, agent: FullAutopoieticAgent, 
                          spatial_index: Dict,
                          radius: int = 7) -> List[FullAutopoieticAgent]:
        """Get agents within radius using spatial index"""
        nearby = []
        cell_size = 8
        cell_radius = (radius // cell_size) + 1
        
        agent_cell_x = agent.x // cell_size
        agent_cell_y = agent.y // cell_size
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_x = (agent_cell_x + dx) % (self.env.size // cell_size)
                cell_y = (agent_cell_y + dy) % (self.env.size // cell_size)
                
                for other in spatial_index.get((cell_x, cell_y), []):
                    if other.id != agent.id:
                        dist = self.env.toroidal_distance(
                            agent.x, agent.y, other.x, other.y
                        )
                        if dist <= radius:
                            nearby.append(other)
        
        return nearby
    
    def _execute_action(self, agent: FullAutopoieticAgent, action: np.ndarray):
        """Execute agent action in environment"""
        # Movement: scale by 2 for larger steps
        dx = int(np.round(action[0] * 2))
        dy = int(np.round(action[1] * 2))
        dx = np.clip(dx, -2, 2)
        dy = np.clip(dy, -2, 2)
        
        new_x = (agent.x + dx) % self.env.size
        new_y = (agent.y + dy) % self.env.size
        
        # Update position
        old_x, old_y = agent.x, agent.y
        agent.x = new_x
        agent.y = new_y
        agent.position_history.append((agent.x, agent.y))
        
        # Track distance
        agent.total_distance_traveled += abs(dx) + abs(dy)
        
        # Resource consumption
        # action[2]: energy consumption intensity [−1, 1] -> [0, 0.4]
        # action[3]: material consumption intensity [−1, 1] -> [0, 0.3]
        consume_e = max(0, (action[2] + 1) / 2) * 0.4
        consume_m = max(0, (action[3] + 1) / 2) * 0.3
        
        gained_e, gained_m = self.env.consume(agent.x, agent.y, consume_e, consume_m)
        
        agent.energy += gained_e
        agent.material += gained_m
        agent.total_energy_consumed += gained_e
        agent.total_material_consumed += gained_m
        
        # Cap resources
        agent.energy = min(agent.energy, 2.0)
        agent.material = min(agent.material, 1.0)
    
    def _process_death(self, agent: FullAutopoieticAgent):
        """Process agent death: update phylogeny and logs"""
        self.total_deaths += 1
        
        # Determine cause of death
        if agent.energy <= 0:
            cause = 'energy_depletion'
        else:
            cause = 'low_coherence'
        
        # Update phylogeny
        if agent.id in self.phylogeny:
            node = self.phylogeny[agent.id]
            node.death_time = self.current_step
            node.final_age = agent.age
            node.final_coherence = agent.coherence_history[-1] if agent.coherence_history else 0.5
            node.behavior_descriptor = agent.get_behavior_descriptor()
        
        # Log death
        self.death_log.append({
            'agent_id': agent.id,
            'parent_id': agent.parent_id,
            'step': self.current_step,
            'age': agent.age,
            'final_coherence': agent.coherence_history[-1] if agent.coherence_history else 0.5,
            'final_energy': agent.energy,
            'cause': cause,
            'total_energy_consumed': agent.total_energy_consumed,
            'offspring_count': agent.offspring_count
        })
    
    def _process_reproduction(self, parent: FullAutopoieticAgent) -> FullAutopoieticAgent:
        """Process reproduction: create child, update phylogeny"""
        child = parent.create_offspring(
            child_id=self.next_id,
            mutation_rate=self.mutation_rate,
            mutation_scale=self.mutation_scale
        )
        child.birth_time = self.current_step
        
        self.next_id += 1
        self.total_births += 1
        
        # Update phylogeny
        self.phylogeny[child.id] = PhylogenyNode(
            agent_id=child.id,
            parent_id=parent.id,
            birth_time=self.current_step
        )
        
        if parent.id in self.phylogeny:
            self.phylogeny[parent.id].offspring_ids.append(child.id)
        
        # Log birth
        self.birth_log.append({
            'parent_id': parent.id,
            'child_id': child.id,
            'step': self.current_step,
            'parent_age': parent.age,
            'parent_coherence': parent.coherence_history[-1] if parent.coherence_history else 0.5
        })
        
        return child
    
    def _update_qd_archive(self):
        """Update Quality-Diversity archive with current population"""
        for agent in self.agents:
            if agent.age < 50:  # Skip young agents
                continue
            
            # Get behavior descriptor
            bd = agent.get_behavior_descriptor()
            
            # Discretize to grid cell
            cell_indices = tuple(
                min(int((bd[i] + 1) / 2 * self.bd_grid_resolution), self.bd_grid_resolution - 1)
                for i in range(len(bd))
            )
            
            # Fitness: coherence + age bonus
            fitness = (
                np.mean(list(agent.coherence_history)[-20:]) if agent.coherence_history else 0.5
            ) + agent.age / 10000
            
            # Update archive if better
            if cell_indices not in self.qd_archive or fitness > self.qd_archive[cell_indices]['fitness']:
                self.qd_archive[cell_indices] = {
                    'fitness': fitness,
                    'agent_id': agent.id,
                    'behavior_descriptor': bd.copy(),
                    'step': self.current_step
                }

    def _get_elite_agents(self, top_k_percent: float = 0.2) -> List[FullAutopoieticAgent]:
        """
        Get elite agents based on coherence and age

        Elite agents are those with:
        1. High coherence (organizational quality)
        2. Sufficient age (proven survival)

        These agents donate their genomes to the Teacher Network.

        Args:
            top_k_percent: Fraction of population to consider elite (default 20%)

        Returns:
            List of elite agents
        """
        if not self.agents:
            return []

        # Filter agents with sufficient age (at least 20 steps old)
        mature_agents = [a for a in self.agents if a.age >= 20]

        if not mature_agents:
            return []

        # Sort by recent coherence (moving average of last 20 steps)
        def agent_fitness(agent):
            if agent.coherence_history:
                recent_coherence = np.mean(list(agent.coherence_history)[-20:])
            else:
                recent_coherence = 0.5
            # Add small age bonus to favor experienced agents
            age_bonus = min(agent.age / 1000, 0.1)
            return recent_coherence + age_bonus

        mature_agents.sort(key=agent_fitness, reverse=True)

        # Take top k%
        k = max(1, int(len(mature_agents) * top_k_percent))
        elite = mature_agents[:k]

        return elite

    def _spawn_agents_from_teacher(self, n: int) -> List[FullAutopoieticAgent]:
        """
        Spawn new agents initialized from Teacher Network

        This is the KEY INNOVATION that prevents knowledge loss!

        Without Teacher:
          - New agent = random weights → starts at coherence ~0.5
          - Must relearn everything from scratch

        With Teacher:
          - New agent = teacher weights + small mutation
          - Starts with accumulated population knowledge
          - Coherence may start at ~0.7+ (inherited wisdom!)

        Args:
            n: Number of agents to spawn

        Returns:
            List of newly spawned agents
        """
        spawned = []

        for _ in range(n):
            # Random spawn position
            x = np.random.randint(0, self.env.size)
            y = np.random.randint(0, self.env.size)

            # Initialize from Teacher (if available) or random
            if self.enable_teacher and self.teacher is not None:
                genome = self.teacher.initialize_student(
                    mutation_rate=self.mutation_rate,
                    mutation_scale=self.mutation_scale
                )
            else:
                genome = None  # Will use random init

            # Create agent
            agent = FullAutopoieticAgent(
                x=x,
                y=y,
                agent_id=self.next_id,
                genome=genome,
                parent_id=None  # Teacher-spawned agents have no parent
            )
            agent.birth_time = self.current_step

            # Add to phylogeny
            self.phylogeny[agent.id] = PhylogenyNode(
                agent_id=agent.id,
                parent_id=None,
                birth_time=self.current_step
            )

            spawned.append(agent)
            self.next_id += 1
            self.total_births += 1

        return spawned

    def get_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        if not self.agents:
            return {
                'step': self.current_step,
                'population_size': 0,
                'avg_coherence': 0,
                'std_coherence': 0,
                'avg_energy': 0,
                'avg_material': 0,
                'avg_age': 0,
                'max_age': 0,
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
                'generation': self.generation_counter,
                'qd_coverage': len(self.qd_archive),
                'extinct': True
            }
        
        coherences = []
        energies = []
        materials = []
        ages = []
        
        for agent in self.agents:
            if agent.coherence_history:
                coherences.append(agent.coherence_history[-1])
            energies.append(agent.energy)
            materials.append(agent.material)
            ages.append(agent.age)
        
        stats = {
            'step': self.current_step,
            'population_size': len(self.agents),
            'avg_coherence': float(np.mean(coherences)) if coherences else 0.5,
            'std_coherence': float(np.std(coherences)) if coherences else 0,
            'avg_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)),
            'avg_material': float(np.mean(materials)),
            'avg_age': float(np.mean(ages)),
            'max_age': int(np.max(ages)),
            'min_energy': float(np.min(energies)),
            'max_energy': float(np.max(energies)),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'generation': self.generation_counter,
            'qd_coverage': len(self.qd_archive),
            'extinct': False
        }

        # Add teacher statistics if enabled
        if self.enable_teacher and self.teacher is not None:
            stats['teacher_knowledge_level'] = self.teacher.knowledge_level
            stats['teacher_update_count'] = self.teacher.update_count

        return stats
    
    def get_qd_metrics(self) -> Dict:
        """Compute Quality-Diversity metrics"""
        if not self.qd_archive:
            return {
                'coverage': 0,
                'qd_score': 0,
                'max_fitness': 0,
                'avg_fitness': 0
            }
        
        # Total possible cells (simplified: 8D BD with 10 resolution each)
        # But most cells are empty, so use actual cells
        total_cells = self.bd_grid_resolution ** 8
        
        coverage = len(self.qd_archive)
        qd_score = sum(entry['fitness'] for entry in self.qd_archive.values())
        fitnesses = [entry['fitness'] for entry in self.qd_archive.values()]
        
        return {
            'coverage': coverage,
            'coverage_ratio': coverage / total_cells,
            'qd_score': qd_score,
            'max_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses)
        }
    
    def get_phylogeny_metrics(self) -> Dict:
        """Compute phylogenetic diversity metrics"""
        if len(self.phylogeny) < 2:
            return {
                'total_lineages': len(self.phylogeny),
                'active_lineages': len(self.agents),
                'max_depth': 0,
                'avg_offspring': 0
            }
        
        # Find root lineages (no parent)
        roots = [nid for nid, node in self.phylogeny.items() if node.parent_id is None]
        
        # Compute tree depths
        def get_depth(node_id, visited=None):
            if visited is None:
                visited = set()
            if node_id in visited:
                return 0
            visited.add(node_id)
            
            node = self.phylogeny.get(node_id)
            if not node or not node.offspring_ids:
                return 0
            return 1 + max(get_depth(cid, visited) for cid in node.offspring_ids)
        
        max_depth = max(get_depth(rid) for rid in roots) if roots else 0
        
        # Average offspring per agent that reproduced
        reproducing = [n for n in self.phylogeny.values() if n.offspring_ids]
        avg_offspring = np.mean([len(n.offspring_ids) for n in reproducing]) if reproducing else 0
        
        # Active lineages (agents still alive)
        active_ids = {a.id for a in self.agents}
        
        return {
            'total_nodes': len(self.phylogeny),
            'root_lineages': len(roots),
            'active_agents': len(active_ids),
            'max_depth': max_depth,
            'avg_offspring': float(avg_offspring),
            'total_reproducing': len(reproducing)
        }
    
    def get_coherence_survival_analysis(self) -> Dict:
        """Analyze correlation between coherence and survival"""
        if len(self.death_log) < 10:
            return {'sample_size': len(self.death_log), 'correlation': None}
        
        # Dead agent coherences and ages
        dead_coherences = [d['final_coherence'] for d in self.death_log]
        dead_ages = [d['age'] for d in self.death_log]
        
        # Living agent coherences
        live_coherences = [
            a.coherence_history[-1] if a.coherence_history else 0.5
            for a in self.agents
        ]
        live_ages = [a.age for a in self.agents]
        
        # Correlation
        try:
            corr = np.corrcoef(dead_coherences, dead_ages)[0, 1]
            coherence_age_correlation = float(corr) if not np.isnan(corr) else 0.0
        except:
            coherence_age_correlation = 0.0
        
        # Death cause breakdown
        energy_deaths = sum(1 for d in self.death_log if d['cause'] == 'energy_depletion')
        coherence_deaths = sum(1 for d in self.death_log if d['cause'] == 'low_coherence')
        
        return {
            'dead_avg_coherence': float(np.mean(dead_coherences)),
            'dead_std_coherence': float(np.std(dead_coherences)),
            'dead_avg_age': float(np.mean(dead_ages)),
            'live_avg_coherence': float(np.mean(live_coherences)) if live_coherences else 0,
            'live_avg_age': float(np.mean(live_ages)) if live_ages else 0,
            'coherence_gap': float(np.mean(live_coherences) - np.mean(dead_coherences)) if live_coherences else 0,
            'coherence_age_correlation': coherence_age_correlation,
            'deaths_by_energy': energy_deaths,
            'deaths_by_coherence': coherence_deaths,
            'sample_size_dead': len(dead_coherences),
            'sample_size_live': len(live_coherences)
        }
    
    def get_population_diversity(self) -> float:
        """Estimate genetic diversity via weight variance"""
        if len(self.agents) < 2:
            return 0.0
        
        # Stack W_rec weights (most indicative of behavior)
        all_weights = np.array([a.W_rec.flatten() for a in self.agents])
        diversity = float(np.mean(np.var(all_weights, axis=0)))
        
        return diversity
    
    def save_state(self, filepath: str):
        """Save population state to file"""
        state = {
            'current_step': self.current_step,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'generation': self.generation_counter,
            'agent_count': len(self.agents),
            'phylogeny_size': len(self.phylogeny),
            'qd_archive_size': len(self.qd_archive),
            'death_log': self.death_log[-100:],  # Last 100 deaths
            'birth_log': self.birth_log[-100:]   # Last 100 births
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)


if __name__ == "__main__":
    print("Testing FullPopulationManager...")
    
    # Create environment and population
    env = FullALifeEnvironment(size=64, seed=42)
    pop = FullPopulationManager(env, initial_pop=50, max_population=200)
    
    print(f"Initial population: {len(pop.agents)}")
    print(f"Environment size: {env.size}x{env.size}")
    
    # Run for 500 steps
    print("\nRunning 500 steps...")
    for step in range(500):
        stats = pop.step()
        
        if step % 100 == 0:
            print(f"  Step {step}: pop={stats['population_size']}, "
                  f"coherence={stats['avg_coherence']:.3f}, "
                  f"births={stats['total_births']}, "
                  f"deaths={stats['total_deaths']}, "
                  f"QD={stats['qd_coverage']}")
        
        if stats['population_size'] == 0:
            print(f"  EXTINCTION at step {step}")
            break
    
    # Final statistics
    print("\nFinal Statistics:")
    stats = pop.get_statistics()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nQD Metrics:")
    qd = pop.get_qd_metrics()
    for k, v in qd.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nPhylogeny Metrics:")
    phylo = pop.get_phylogeny_metrics()
    for k, v in phylo.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nCoherence-Survival Analysis:")
    analysis = pop.get_coherence_survival_analysis()
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif v is not None:
            print(f"  {k}: {v}")
    
    print(f"\nPopulation Diversity: {pop.get_population_diversity():.6f}")
    
    print("\nPopulation test PASSED!")
