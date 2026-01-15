"""
GENESIS Path B Phase 0: Main Experiment

Run the minimal artificial life simulation and validate core mechanisms.

Core Questions:
1. Do high-coherence agents survive longer?
2. Does reproduction/selection work?
3. Does the population evolve?
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

from minimal_environment import MinimalGrid
from minimal_agent import MinimalAutopoieticAgent
from minimal_population import MinimalPopulation


def run_phase0(
    n_steps: int = 1000,
    initial_agents: int = 20,
    grid_size: int = 16,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run Phase 0 experiment
    
    Args:
        n_steps: Number of simulation steps
        initial_agents: Initial population size
        grid_size: Grid dimensions
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        results: Complete experiment results
    """
    # Set seed
    np.random.seed(seed)
    
    if verbose:
        print("=" * 70)
        print("GENESIS Path B - Phase 0: Minimal Artificial Life")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Steps: {n_steps}")
        print(f"  Initial agents: {initial_agents}")
        print(f"  Grid size: {grid_size}x{grid_size}")
        print(f"  Seed: {seed}")
        print("=" * 70)
    
    # Create environment and population
    env = MinimalGrid(size=grid_size, growth_rate=0.1)
    pop = MinimalPopulation(env, initial_agents=initial_agents, max_population=100)
    
    # History tracking
    history = {
        'step': [],
        'population_size': [],
        'avg_coherence': [],
        'std_coherence': [],
        'avg_energy': [],
        'avg_age': [],
        'max_age': [],
        'total_births': [],
        'total_deaths': [],
        'generation': [],
        'diversity': [],
        'env_mean_resource': [],
    }
    
    # Run simulation
    start_time = datetime.now()
    
    for step in range(n_steps):
        stats = pop.step()
        
        # Record every 10 steps
        if step % 10 == 0:
            env_stats = env.get_statistics()
            diversity = pop.get_population_diversity()
            
            history['step'].append(step)
            history['population_size'].append(stats['size'])
            history['avg_coherence'].append(stats['avg_coherence'])
            history['std_coherence'].append(stats['std_coherence'])
            history['avg_energy'].append(stats['avg_energy'])
            history['avg_age'].append(stats['avg_age'])
            history['max_age'].append(stats['max_age'])
            history['total_births'].append(stats['total_births'])
            history['total_deaths'].append(stats['total_deaths'])
            history['generation'].append(stats['generation'])
            history['diversity'].append(diversity)
            history['env_mean_resource'].append(env_stats['mean_resource'])
        
        # Print progress
        if verbose and step % 100 == 0:
            print(f"Step {step:4d} | "
                  f"Pop: {stats['size']:3d} | "
                  f"Coherence: {stats['avg_coherence']:.3f} +/- {stats['std_coherence']:.3f} | "
                  f"Energy: {stats['avg_energy']:.3f} | "
                  f"Births: {stats['total_births']:3d} | "
                  f"Deaths: {stats['total_deaths']:3d}")
        
        # Check for extinction
        if stats['size'] == 0:
            if verbose:
                print(f"\n[!] EXTINCTION at step {step}")
            break
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if verbose:
        print("=" * 70)
        print(f"Simulation completed in {duration:.2f} seconds")
        print("=" * 70)
    
    # Collect final analysis
    final_stats = pop.get_statistics()
    coherence_survival = pop.get_coherence_survival_correlation()
    
    # Compile results
    results = {
        'config': {
            'n_steps': n_steps,
            'initial_agents': initial_agents,
            'grid_size': grid_size,
            'seed': seed
        },
        'history': history,
        'final_stats': final_stats,
        'coherence_survival_analysis': coherence_survival,
        'death_log': pop.death_log[-100:],  # Last 100 deaths
        'birth_log': pop.birth_log[-100:],  # Last 100 births
        'duration_seconds': duration,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def analyze_results(results: Dict) -> Dict:
    """
    Analyze experiment results
    
    Returns:
        analysis: Key findings
    """
    history = results['history']
    coherence_survival = results['coherence_survival_analysis']
    
    analysis = {
        'success_criteria': {},
        'key_findings': [],
        'recommendations': []
    }
    
    # 1. Code runs without errors
    analysis['success_criteria']['runs_without_error'] = True
    
    # 2. Coherence-survival correlation
    # Check multiple indicators
    gap = coherence_survival.get('coherence_gap', 0)
    corr = coherence_survival.get('coherence_age_correlation', 0)

    # Either: living agents have higher coherence, OR coherence correlates with age at death
    coherence_matters = gap > 0.03 or corr > 0.1

    if coherence_matters:
        analysis['success_criteria']['coherence_survival_correlation'] = True
        analysis['key_findings'].append(
            f"Coherence affects survival: gap={gap:.3f}, age_correlation={corr:.3f}"
        )
    else:
        analysis['success_criteria']['coherence_survival_correlation'] = False
        analysis['key_findings'].append(
            f"Weak coherence-survival correlation (gap={gap:.3f}, corr={corr:.3f})"
        )

    # Add death cause breakdown
    low_e = coherence_survival.get('deaths_by_low_energy', 0)
    low_c = coherence_survival.get('deaths_by_low_coherence', 0)
    analysis['key_findings'].append(
        f"Death causes: low_energy={low_e}, low_coherence={low_c}"
    )
    
    # 3. Population evolution
    total_births = history['total_births'][-1] if history['total_births'] else 0
    total_deaths = history['total_deaths'][-1] if history['total_deaths'] else 0
    
    if total_births > 5 and total_deaths > 5:
        analysis['success_criteria']['population_evolves'] = True
        analysis['key_findings'].append(
            f"Population evolved: {total_births} births, {total_deaths} deaths"
        )
    else:
        analysis['success_criteria']['population_evolves'] = False
        analysis['key_findings'].append(
            f"Limited evolution: only {total_births} births, {total_deaths} deaths"
        )
    
    # 4. Runtime under 2 minutes
    duration = results['duration_seconds']
    analysis['success_criteria']['runtime_acceptable'] = duration < 120
    analysis['key_findings'].append(f"Runtime: {duration:.2f} seconds")
    
    # Population dynamics
    pop_sizes = history['population_size']
    if len(pop_sizes) > 10:
        pop_start = np.mean(pop_sizes[:10])
        pop_end = np.mean(pop_sizes[-10:])
        pop_trend = "increasing" if pop_end > pop_start * 1.1 else "decreasing" if pop_end < pop_start * 0.9 else "stable"
        analysis['key_findings'].append(f"Population trend: {pop_trend} ({pop_start:.0f} -> {pop_end:.0f})")
    
    # Coherence dynamics
    coherences = history['avg_coherence']
    if len(coherences) > 10:
        coh_start = np.mean(coherences[:10])
        coh_end = np.mean(coherences[-10:])
        coh_trend = "increasing" if coh_end > coh_start * 1.05 else "decreasing" if coh_end < coh_start * 0.95 else "stable"
        analysis['key_findings'].append(f"Coherence trend: {coh_trend} ({coh_start:.3f} -> {coh_end:.3f})")
    
    # Overall success
    all_criteria = analysis['success_criteria'].values()
    analysis['overall_success'] = all(all_criteria)
    
    # Recommendations
    if not analysis['success_criteria'].get('coherence_survival_correlation', False):
        analysis['recommendations'].append(
            "Consider adjusting death threshold or coherence calculation"
        )
    if not analysis['success_criteria'].get('population_evolves', False):
        analysis['recommendations'].append(
            "Consider lowering reproduction threshold or increasing resources"
        )
    
    return analysis


def save_results(results: Dict, analysis: Dict, output_dir: str):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    results_file = os.path.join(output_dir, f"phase0_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")
    
    # Save analysis summary
    summary_file = os.path.join(output_dir, "phase0_summary.md")
    with open(summary_file, 'w') as f:
        f.write("# GENESIS Path B Phase 0 - Results Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Success Criteria\n\n")
        for criterion, passed in analysis['success_criteria'].items():
            status = "PASS" if passed else "FAIL"
            f.write(f"- [{status}] {criterion}\n")
        
        f.write(f"\n**Overall: {'SUCCESS' if analysis['overall_success'] else 'NEEDS WORK'}**\n\n")
        
        f.write("## Key Findings\n\n")
        for finding in analysis['key_findings']:
            f.write(f"- {finding}\n")
        
        if analysis['recommendations']:
            f.write("\n## Recommendations\n\n")
            for rec in analysis['recommendations']:
                f.write(f"- {rec}\n")
        
        f.write("\n## Configuration\n\n")
        for k, v in results['config'].items():
            f.write(f"- {k}: {v}\n")
        
        f.write("\n## Coherence-Survival Analysis\n\n")
        for k, v in results['coherence_survival_analysis'].items():
            f.write(f"- {k}: {v}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return results_file, summary_file


def print_analysis(analysis: Dict):
    """Print analysis to console"""
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    print("\n[Success Criteria]")
    for criterion, passed in analysis['success_criteria'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {criterion}")
    
    print(f"\n[Overall: {'SUCCESS' if analysis['overall_success'] else 'NEEDS WORK'}]")
    
    print("\n[Key Findings]")
    for finding in analysis['key_findings']:
        print(f"  * {finding}")
    
    if analysis['recommendations']:
        print("\n[Recommendations]")
        for rec in analysis['recommendations']:
            print(f"  -> {rec}")


if __name__ == "__main__":
    # Run experiment
    results = run_phase0(
        n_steps=1000,
        initial_agents=20,
        grid_size=16,
        seed=42,
        verbose=True
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print analysis
    print_analysis(analysis)
    
    # Save results
    output_dir = "/Users/say/Documents/GitHub/ai/08_GENESIS/results/path_b_phase0"
    save_results(results, analysis, output_dir)
    
    # Try visualization
    try:
        from visualize_phase0 import plot_results
        plot_results(results, output_dir)
    except ImportError:
        print("\n[Note] Visualization skipped (visualize_phase0.py not available)")
    except Exception as e:
        print(f"\n[Warning] Visualization failed: {e}")
    
    print("\n" + "=" * 70)
    print("Phase 0 experiment complete!")
    print("=" * 70)
