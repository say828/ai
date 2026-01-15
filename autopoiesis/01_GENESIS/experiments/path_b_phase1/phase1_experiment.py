"""
GENESIS Path B Phase 1: Full Experiment

Main experiment script for the complete Artificial Life system.

Usage:
    python phase1_experiment.py --steps 10000 --trials 3 --seed 42
    
Expected runtime: ~30-60 minutes for full experiment
"""

import numpy as np
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

from full_environment import FullALifeEnvironment, ResourceConfig
from full_agent import FullAutopoieticAgent
from full_population import FullPopulationManager
from baselines import RandomAgent, FixedPolicyAgent, RLAgent, BaselinePopulationManager


class Phase1Experiment:
    """
    Phase 1 Experiment Manager
    
    Runs the full artificial life simulation with:
    - Autopoietic agents (main condition)
    - Baseline comparisons (Random, Fixed, RL)
    - Comprehensive metrics and logging
    """
    
    def __init__(self,
                 n_steps: int = 10000,
                 initial_pop: int = 100,
                 max_pop: int = 500,
                 grid_size: int = 64,
                 seed: int = 42,
                 log_interval: int = 100,
                 results_dir: Optional[str] = None):
        """
        Args:
            n_steps: Number of simulation steps
            initial_pop: Initial population size
            max_pop: Maximum population cap
            grid_size: Environment grid size
            seed: Random seed
            log_interval: Steps between logging
            results_dir: Directory for results
        """
        self.n_steps = n_steps
        self.initial_pop = initial_pop
        self.max_pop = max_pop
        self.grid_size = grid_size
        self.seed = seed
        self.log_interval = log_interval
        
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent.parent / 'results' / 'path_b_phase1'
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.history = {
            'steps': [],
            'population_size': [],
            'avg_coherence': [],
            'std_coherence': [],
            'avg_energy': [],
            'avg_material': [],
            'avg_age': [],
            'max_age': [],
            'total_births': [],
            'total_deaths': [],
            'qd_coverage': [],
            'generation': []
        }
        
    def run_autopoietic(self, seed: int) -> Dict:
        """Run autopoietic agent experiment"""
        print(f"\n{'='*60}")
        print(f"Running AUTOPOIETIC experiment (seed={seed})")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        
        # Create environment and population
        env = FullALifeEnvironment(size=self.grid_size, seed=seed)
        pop = FullPopulationManager(
            env,
            initial_pop=self.initial_pop,
            max_population=self.max_pop,
            min_population=self.initial_pop // 2,  # Maintain at least 50% of initial pop
            enable_teacher=True,  # Enable infinite learning
            teacher_update_interval=100,
            teacher_learning_rate=0.1
        )
        
        # Reset history
        history = {k: [] for k in self.history.keys()}
        
        start_time = time.time()
        extinction_step = None
        
        for step in range(self.n_steps):
            stats = pop.step()
            
            # Log at intervals
            if step % self.log_interval == 0:
                history['steps'].append(step)
                history['population_size'].append(stats['population_size'])
                history['avg_coherence'].append(stats['avg_coherence'])
                history['std_coherence'].append(stats['std_coherence'])
                history['avg_energy'].append(stats['avg_energy'])
                history['avg_material'].append(stats.get('avg_material', 0))
                history['avg_age'].append(stats['avg_age'])
                history['max_age'].append(stats['max_age'])
                history['total_births'].append(stats['total_births'])
                history['total_deaths'].append(stats['total_deaths'])
                history['qd_coverage'].append(stats['qd_coverage'])
                history['generation'].append(stats['generation'])
                
                elapsed = time.time() - start_time
                rate = step / elapsed if elapsed > 0 else 0
                
                teacher_str = f" | Teacher: {stats['teacher_knowledge_level']:.3f}" if 'teacher_knowledge_level' in stats else ""
                print(f"Step {step:5d} | Pop: {stats['population_size']:3d} | "
                      f"Coh: {stats['avg_coherence']:.3f} | "
                      f"Births: {stats['total_births']:4d} | "
                      f"Deaths: {stats['total_deaths']:4d} | "
                      f"QD: {stats['qd_coverage']:3d} | "
                      f"Rate: {rate:.1f} steps/s{teacher_str}")
            
            # Check extinction
            if stats['population_size'] == 0:
                print(f"\n*** EXTINCTION at step {step} ***")
                extinction_step = step
                break
        
        elapsed = time.time() - start_time
        
        # Final analysis
        final_stats = pop.get_statistics()
        qd_metrics = pop.get_qd_metrics()
        phylo_metrics = pop.get_phylogeny_metrics()
        survival_analysis = pop.get_coherence_survival_analysis()
        diversity = pop.get_population_diversity()
        
        result = {
            'condition': 'autopoietic',
            'seed': seed,
            'n_steps': self.n_steps,
            'elapsed_time': elapsed,
            'extinction_step': extinction_step,
            'history': history,
            'final_stats': final_stats,
            'qd_metrics': qd_metrics,
            'phylo_metrics': phylo_metrics,
            'survival_analysis': survival_analysis,
            'diversity': diversity
        }

        # Add teacher statistics if available
        if pop.teacher:
            result['teacher_stats'] = pop.teacher.get_statistics()
        
        print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Final population: {final_stats['population_size']}")
        print(f"Total births: {final_stats['total_births']}")
        print(f"Total deaths: {final_stats['total_deaths']}")
        print(f"QD Coverage: {qd_metrics['coverage']}")
        print(f"Coherence-Age Correlation: {survival_analysis.get('coherence_age_correlation', 'N/A')}")

        # Add teacher statistics
        if 'teacher_knowledge_level' in final_stats:
            print(f"\n📚 Teacher Network:")
            print(f"  Knowledge Level: {final_stats['teacher_knowledge_level']:.3f}")
            print(f"  Updates: {final_stats['teacher_update_count']}")
            if pop.teacher:
                teacher_stats = pop.teacher.get_statistics()
                if 'coherence_history' in teacher_stats:
                    hist = teacher_stats['coherence_history']
                    print(f"  Coherence Progress: {hist['min']:.3f} → {hist['max']:.3f} (Δ={hist['max']-hist['min']:.3f})")

        return result
    
    def run_baseline(self, agent_class, name: str, seed: int) -> Dict:
        """Run baseline agent experiment"""
        print(f"\n{'='*60}")
        print(f"Running {name.upper()} baseline (seed={seed})")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        
        env = FullALifeEnvironment(size=self.grid_size, seed=seed)
        pop = BaselinePopulationManager(
            env,
            agent_class=agent_class,
            initial_pop=self.initial_pop,
            max_population=self.max_pop
        )
        
        history = {k: [] for k in ['steps', 'population_size', 'avg_coherence', 
                                    'avg_energy', 'avg_age', 'total_births', 'total_deaths']}
        
        start_time = time.time()
        extinction_step = None
        
        for step in range(self.n_steps):
            stats = pop.step()
            
            if step % self.log_interval == 0:
                history['steps'].append(step)
                history['population_size'].append(stats['population_size'])
                history['avg_coherence'].append(stats['avg_coherence'])
                history['avg_energy'].append(stats['avg_energy'])
                history['avg_age'].append(stats['avg_age'])
                history['total_births'].append(stats['total_births'])
                history['total_deaths'].append(stats['total_deaths'])
                
                print(f"Step {step:5d} | Pop: {stats['population_size']:3d} | "
                      f"Energy: {stats['avg_energy']:.3f} | "
                      f"Births: {stats['total_births']:4d}")
            
            if stats['population_size'] == 0:
                print(f"\n*** EXTINCTION at step {step} ***")
                extinction_step = step
                break
        
        elapsed = time.time() - start_time
        final_stats = pop.get_statistics()
        
        result = {
            'condition': name,
            'seed': seed,
            'n_steps': self.n_steps,
            'elapsed_time': elapsed,
            'extinction_step': extinction_step,
            'history': history,
            'final_stats': final_stats
        }
        
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Final population: {final_stats['population_size']}")
        
        return result
    
    def run_full_experiment(self, n_trials: int = 3, 
                           run_baselines: bool = True) -> Dict:
        """
        Run complete experiment with multiple trials
        
        Args:
            n_trials: Number of trials per condition
            run_baselines: Whether to run baseline comparisons
            
        Returns:
            Complete results dictionary
        """
        all_results = {
            'experiment_info': {
                'n_steps': self.n_steps,
                'initial_pop': self.initial_pop,
                'max_pop': self.max_pop,
                'grid_size': self.grid_size,
                'n_trials': n_trials,
                'timestamp': datetime.now().isoformat()
            },
            'autopoietic': [],
            'random': [],
            'fixed': [],
            'rl': []
        }
        
        # Run autopoietic trials
        for trial in range(n_trials):
            seed = self.seed + trial * 100
            result = self.run_autopoietic(seed)
            all_results['autopoietic'].append(result)
        
        # Run baseline trials
        if run_baselines:
            baselines = [
                (RandomAgent, 'random'),
                (FixedPolicyAgent, 'fixed'),
                (RLAgent, 'rl')
            ]
            
            for agent_class, name in baselines:
                for trial in range(n_trials):
                    seed = self.seed + trial * 100
                    result = self.run_baseline(agent_class, name, seed)
                    all_results[name].append(result)
        
        # Compute summary statistics
        all_results['summary'] = self._compute_summary(all_results)
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _compute_summary(self, results: Dict) -> Dict:
        """Compute summary statistics across trials"""
        summary = {}
        
        for condition in ['autopoietic', 'random', 'fixed', 'rl']:
            if not results[condition]:
                continue
            
            trials = results[condition]
            
            # Survival time (steps until extinction or max)
            survival_times = [
                t['extinction_step'] if t['extinction_step'] else self.n_steps
                for t in trials
            ]
            
            # Final population
            final_pops = [t['final_stats']['population_size'] for t in trials]
            
            # Total births
            total_births = [t['final_stats']['total_births'] for t in trials]
            
            # Coherence (for autopoietic only)
            if condition == 'autopoietic':
                coherence_corrs = [
                    t['survival_analysis'].get('coherence_age_correlation', 0)
                    for t in trials if 'survival_analysis' in t
                ]
                qd_coverages = [
                    t['qd_metrics']['coverage'] for t in trials if 'qd_metrics' in t
                ]
            else:
                coherence_corrs = []
                qd_coverages = []
            
            summary[condition] = {
                'n_trials': len(trials),
                'survival_time_mean': float(np.mean(survival_times)),
                'survival_time_std': float(np.std(survival_times)),
                'final_pop_mean': float(np.mean(final_pops)),
                'final_pop_std': float(np.std(final_pops)),
                'total_births_mean': float(np.mean(total_births)),
                'total_births_std': float(np.std(total_births)),
                'extinction_rate': sum(1 for s in survival_times if s < self.n_steps) / len(trials)
            }
            
            if coherence_corrs:
                summary[condition]['coherence_corr_mean'] = float(np.mean(coherence_corrs))
                summary[condition]['coherence_corr_std'] = float(np.std(coherence_corrs))
            
            if qd_coverages:
                summary[condition]['qd_coverage_mean'] = float(np.mean(qd_coverages))
                summary[condition]['qd_coverage_std'] = float(np.std(qd_coverages))
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = self.results_dir / f'phase1_results_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(json_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\nResults saved to: {json_path}")
        
        return json_path


def run_quick_test():
    """Quick test run (500 steps, 1 trial)"""
    print("="*60)
    print("GENESIS Phase 1 - Quick Test")
    print("="*60)
    
    experiment = Phase1Experiment(
        n_steps=500,
        initial_pop=50,
        max_pop=200,
        log_interval=50
    )
    
    results = experiment.run_full_experiment(n_trials=1, run_baselines=False)
    
    print("\n" + "="*60)
    print("Quick Test Summary")
    print("="*60)
    
    if results['autopoietic']:
        trial = results['autopoietic'][0]
        print(f"Final population: {trial['final_stats']['population_size']}")
        print(f"Total births: {trial['final_stats']['total_births']}")
        print(f"Total deaths: {trial['final_stats']['total_deaths']}")
        if 'survival_analysis' in trial:
            corr = trial['survival_analysis'].get('coherence_age_correlation', None)
            if corr is not None and isinstance(corr, (int, float)):
                print(f"Coherence-Age Correlation: {corr:.3f}")
            else:
                print(f"Coherence-Age Correlation: N/A")
    
    return results


def run_full_experiment():
    """Full experiment run (10000 steps, 3 trials, all conditions)"""
    print("="*60)
    print("GENESIS Phase 1 - Full Experiment")
    print("="*60)
    
    experiment = Phase1Experiment(
        n_steps=10000,
        initial_pop=100,
        max_pop=500,
        log_interval=100
    )
    
    results = experiment.run_full_experiment(n_trials=3, run_baselines=True)
    
    print("\n" + "="*60)
    print("Full Experiment Summary")
    print("="*60)
    
    summary = results['summary']
    for condition in ['autopoietic', 'random', 'fixed', 'rl']:
        if condition in summary:
            s = summary[condition]
            print(f"\n{condition.upper()}:")
            print(f"  Survival time: {s['survival_time_mean']:.0f} +/- {s['survival_time_std']:.0f}")
            print(f"  Final pop: {s['final_pop_mean']:.1f} +/- {s['final_pop_std']:.1f}")
            print(f"  Total births: {s['total_births_mean']:.0f} +/- {s['total_births_std']:.0f}")
            print(f"  Extinction rate: {s['extinction_rate']*100:.0f}%")
            if 'coherence_corr_mean' in s:
                print(f"  Coherence corr: {s['coherence_corr_mean']:.3f}")
            if 'qd_coverage_mean' in s:
                print(f"  QD coverage: {s['qd_coverage_mean']:.1f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GENESIS Phase 1 Experiment')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--no-baselines', action='store_true', help='Skip baseline runs')
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_quick_test()
    else:
        experiment = Phase1Experiment(
            n_steps=args.steps,
            seed=args.seed
        )
        results = experiment.run_full_experiment(
            n_trials=args.trials,
            run_baselines=not args.no_baselines
        )
