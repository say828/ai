"""
Comprehensive Ablation Studies for Path A Continual Learning

Three ablation studies to validate design choices:
    1. W_in Initialization: Learned-Freeze vs Random-Freeze vs Learned-Continue
    2. Coherence Criterion: With vs Without vs Strict
    3. Learning Rule: Hebbian vs SGD vs Adam

Usage:
    python ablation_experiments.py --all
    python ablation_experiments.py --ablation1
    python ablation_experiments.py --ablation2
    python ablation_experiments.py --ablation3

Expected runtime: ~30-60 minutes for all ablations with N=5 trials
"""

import numpy as np
import torch
import json
import time
import os
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime
from dataclasses import dataclass

from split_mnist import SplitMNIST
from autopoietic_continual_learner import AutopoeticContinualLearner


@dataclass
class AblationConfig:
    """Configuration for an ablation condition."""
    name: str
    random_win: bool = False
    freeze_win_after_task0: bool = True
    coherence_acceptance_threshold: float = 0.95
    learning_rule: str = 'hebbian'
    sgd_lr: float = 0.01
    plasticity_rate: float = 0.5


class AblationExperiment:
    """
    Ablation experiment runner.
    """
    
    def __init__(self,
                 ablation_name: str,
                 conditions: Dict[str, AblationConfig],
                 n_trials: int = 5,
                 epochs_per_task: int = 5,
                 batch_size: int = 64,
                 results_dir: str = './results/ablations'):
        """
        Args:
            ablation_name: Name of this ablation study
            conditions: Dict of condition_name -> AblationConfig
            n_trials: Number of independent trials
            epochs_per_task: Epochs per task
            batch_size: Batch size
            results_dir: Directory to save results
        """
        self.ablation_name = ablation_name
        self.conditions = conditions
        self.n_trials = n_trials
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        print("=" * 70)
        print(f"Ablation Study: {ablation_name}")
        print("=" * 70)
        print(f"Conditions: {list(conditions.keys())}")
        print(f"Trials: {n_trials}")
        print(f"Epochs per task: {epochs_per_task}")
        
    def _create_model(self, config: AblationConfig, seed: int) -> AutopoeticContinualLearner:
        """Create a model with the given ablation configuration."""
        return AutopoeticContinualLearner(
            input_dim=100,
            hidden_dim=256,
            num_tasks=5,
            classes_per_task=2,
            connectivity=0.4,
            plasticity_rate=config.plasticity_rate,
            coherence_threshold=0.3,
            seed=seed,
            # Ablation parameters
            random_win=config.random_win,
            freeze_win_after_task0=config.freeze_win_after_task0,
            coherence_acceptance_threshold=config.coherence_acceptance_threshold,
            learning_rule=config.learning_rule,
            sgd_lr=config.sgd_lr
        )
    
    def _compute_forgetting(self, accuracy_matrix: np.ndarray) -> float:
        """Compute forgetting measure."""
        num_tasks = accuracy_matrix.shape[0]
        forgetting = 0.0
        count = 0
        
        for j in range(num_tasks - 1):
            max_acc = accuracy_matrix[j, j]
            final_acc = accuracy_matrix[-1, j]
            forgetting += max(0, max_acc - final_acc)
            count += 1
            
        return forgetting / count if count > 0 else 0.0
    
    def run_single_trial(self,
                         condition_name: str,
                         config: AblationConfig,
                         seed: int) -> Dict:
        """Run a single trial for one condition."""
        # Create dataset
        dataset = SplitMNIST(
            data_dir='./data',
            use_pca=True,
            pca_dim=100,
            batch_size=self.batch_size,
            seed=seed
        )
        
        # Create model
        model = self._create_model(config, seed)
        
        # Accuracy matrix
        accuracy_matrix = np.zeros((5, 5))
        training_time = 0.0
        
        # Train on each task
        for task_id in range(5):
            train_loader = dataset.get_task_dataloader(task_id, train=True)
            
            start_time = time.time()
            model.train_on_task(
                train_loader,
                task_id,
                epochs=self.epochs_per_task,
                verbose=False
            )
            training_time += time.time() - start_time
            
            # Evaluate on all tasks seen so far
            for eval_task in range(task_id + 1):
                data = dataset.task_data[eval_task]
                result = model.evaluate(
                    data['test_x'].numpy(),
                    data['test_y'].numpy(),
                    eval_task
                )
                accuracy_matrix[task_id, eval_task] = result['accuracy']
        
        # Compute metrics
        avg_accuracy = np.mean(accuracy_matrix[-1, :])
        forgetting = self._compute_forgetting(accuracy_matrix)
        
        return {
            'accuracy_matrix': accuracy_matrix.tolist(),
            'avg_accuracy': avg_accuracy,
            'forgetting': forgetting,
            'training_time': training_time,
            'flops': model.get_flops()
        }
    
    def run_condition(self,
                      condition_name: str,
                      config: AblationConfig) -> Dict:
        """Run all trials for one condition."""
        print(f"\n--- Condition: {condition_name} ---")
        
        results = []
        base_seed = 42
        
        for trial in range(self.n_trials):
            seed = base_seed + trial * 100
            print(f"  Trial {trial + 1}/{self.n_trials} (seed={seed})...", end=" ")
            
            trial_result = self.run_single_trial(condition_name, config, seed)
            results.append(trial_result)
            
            print(f"Acc={trial_result['avg_accuracy']:.3f}, "
                  f"Fgt={trial_result['forgetting']:.3f}")
        
        # Aggregate results
        avg_accs = [r['avg_accuracy'] for r in results]
        forgettings = [r['forgetting'] for r in results]
        times = [r['training_time'] for r in results]
        flops_list = [r['flops'] for r in results]
        
        return {
            'condition_name': condition_name,
            'config': {
                'random_win': config.random_win,
                'freeze_win_after_task0': config.freeze_win_after_task0,
                'coherence_acceptance_threshold': config.coherence_acceptance_threshold,
                'learning_rule': config.learning_rule,
                'sgd_lr': config.sgd_lr,
                'plasticity_rate': config.plasticity_rate
            },
            'n_trials': self.n_trials,
            'avg_accuracy': {
                'mean': np.mean(avg_accs),
                'std': np.std(avg_accs),
                'values': avg_accs
            },
            'forgetting': {
                'mean': np.mean(forgettings),
                'std': np.std(forgettings),
                'values': forgettings
            },
            'training_time': {
                'mean': np.mean(times),
                'std': np.std(times)
            },
            'flops': {
                'mean': np.mean(flops_list),
                'std': np.std(flops_list)
            },
            'trial_results': results
        }
    
    def run(self) -> Dict:
        """Run all conditions."""
        all_results = {}
        
        for condition_name, config in self.conditions.items():
            all_results[condition_name] = self.run_condition(condition_name, config)
        
        return all_results
    
    def statistical_tests(self, results: Dict, baseline_condition: str) -> Dict:
        """Perform statistical tests comparing conditions to baseline."""
        tests = {}
        
        baseline_acc = results[baseline_condition]['avg_accuracy']['values']
        baseline_fgt = results[baseline_condition]['forgetting']['values']
        
        for condition_name, condition_results in results.items():
            if condition_name == baseline_condition:
                continue
                
            cond_acc = condition_results['avg_accuracy']['values']
            cond_fgt = condition_results['forgetting']['values']
            
            # T-tests
            t_acc, p_acc = stats.ttest_ind(baseline_acc, cond_acc)
            t_fgt, p_fgt = stats.ttest_ind(baseline_fgt, cond_fgt)
            
            # Cohen's d
            pooled_std_acc = np.sqrt((np.var(baseline_acc) + np.var(cond_acc)) / 2)
            cohen_d_acc = (np.mean(baseline_acc) - np.mean(cond_acc)) / pooled_std_acc if pooled_std_acc > 0 else 0
            
            pooled_std_fgt = np.sqrt((np.var(baseline_fgt) + np.var(cond_fgt)) / 2)
            cohen_d_fgt = (np.mean(baseline_fgt) - np.mean(cond_fgt)) / pooled_std_fgt if pooled_std_fgt > 0 else 0
            
            tests[f'{baseline_condition}_vs_{condition_name}'] = {
                'accuracy': {
                    't_statistic': float(t_acc) if not np.isnan(t_acc) else 0.0,
                    'p_value': float(p_acc) if not np.isnan(p_acc) else 1.0,
                    'cohen_d': float(cohen_d_acc) if not np.isnan(cohen_d_acc) else 0.0,
                    'significant': bool(p_acc < 0.05) if not np.isnan(p_acc) else False,
                    'baseline_better': bool(np.mean(baseline_acc) > np.mean(cond_acc))
                },
                'forgetting': {
                    't_statistic': float(t_fgt) if not np.isnan(t_fgt) else 0.0,
                    'p_value': float(p_fgt) if not np.isnan(p_fgt) else 1.0,
                    'cohen_d': float(cohen_d_fgt) if not np.isnan(cohen_d_fgt) else 0.0,
                    'significant': bool(p_fgt < 0.05) if not np.isnan(p_fgt) else False,
                    'baseline_better': bool(np.mean(baseline_fgt) < np.mean(cond_fgt))
                }
            }
        
        return tests
    
    def save_results(self, results: Dict, tests: Dict):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = os.path.join(
            self.results_dir, 
            f'{self.ablation_name}_results_{timestamp}.json'
        )
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save tests
        tests_file = os.path.join(
            self.results_dir,
            f'{self.ablation_name}_tests_{timestamp}.json'
        )
        with open(tests_file, 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Tests saved to: {tests_file}")
        
        return results_file, tests_file


# =============================================================================
# Ablation 1: W_in Initialization
# =============================================================================

def ablation1_winit(n_trials: int = 5, epochs: int = 5) -> Tuple[Dict, Dict]:
    """
    Ablation 1: Random W_in vs Learned W_in
    
    Purpose: Demonstrate that learning W_in before freezing (our method)
    is superior to random initialization (RanPAC-style).
    
    Conditions:
        1. LEARNED_FREEZE (ours): Learn W_in on task 0, then freeze
        2. RANDOM_FREEZE (RanPAC): Random W_in, frozen from start
        3. LEARNED_CONTINUE (baseline): Learn W_in on all tasks (no freeze)
    """
    conditions = {
        'learned_freeze': AblationConfig(
            name='Learned-Freeze (Ours)',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.95,
            learning_rule='hebbian',
            plasticity_rate=0.5
        ),
        'random_freeze': AblationConfig(
            name='Random-Freeze (RanPAC-style)',
            random_win=True,
            freeze_win_after_task0=True,  # Already frozen since random
            coherence_acceptance_threshold=0.95,
            learning_rule='hebbian',
            plasticity_rate=0.5
        ),
        'learned_continue': AblationConfig(
            name='Learned-Continue',
            random_win=False,
            freeze_win_after_task0=False,  # Never freeze
            coherence_acceptance_threshold=0.95,
            learning_rule='hebbian',
            plasticity_rate=0.5
        )
    }
    
    experiment = AblationExperiment(
        ablation_name='ablation1_winit',
        conditions=conditions,
        n_trials=n_trials,
        epochs_per_task=epochs
    )
    
    results = experiment.run()
    tests = experiment.statistical_tests(results, 'learned_freeze')
    experiment.save_results(results, tests)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION 1 SUMMARY: W_in Initialization")
    print("=" * 70)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['avg_accuracy']['mean']:.4f} +/- {res['avg_accuracy']['std']:.4f}")
        print(f"  Forgetting: {res['forgetting']['mean']:.4f} +/- {res['forgetting']['std']:.4f}")
    
    print("\n--- Statistical Tests (vs learned_freeze) ---")
    for comparison, test in tests.items():
        print(f"\n{comparison}:")
        print(f"  Accuracy: p={test['accuracy']['p_value']:.4f}, d={test['accuracy']['cohen_d']:.3f}")
        if test['accuracy']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['accuracy']['baseline_better']})")
        print(f"  Forgetting: p={test['forgetting']['p_value']:.4f}, d={test['forgetting']['cohen_d']:.3f}")
        if test['forgetting']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['forgetting']['baseline_better']})")
    
    return results, tests


# =============================================================================
# Ablation 2: Coherence Criterion
# =============================================================================

def ablation2_coherence(n_trials: int = 5, epochs: int = 5) -> Tuple[Dict, Dict]:
    """
    Ablation 2: Coherence Criterion On/Off
    
    Purpose: Validate that coherence-based update acceptance is beneficial.
    
    Conditions:
        1. WITH_COHERENCE (ours): Accept if acc >= 0.95 * prev_acc
        2. WITHOUT_COHERENCE: Always accept updates (threshold=0.0)
        3. STRICT_COHERENCE: Accept only if acc >= prev_acc (threshold=1.0)
    """
    conditions = {
        'with_coherence': AblationConfig(
            name='With Coherence (Ours)',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.95,
            learning_rule='hebbian',
            plasticity_rate=0.5
        ),
        'without_coherence': AblationConfig(
            name='Without Coherence',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.0,  # Always accept
            learning_rule='hebbian',
            plasticity_rate=0.5
        ),
        'strict_coherence': AblationConfig(
            name='Strict Coherence',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=1.0,  # Only accept if no degradation
            learning_rule='hebbian',
            plasticity_rate=0.5
        )
    }
    
    experiment = AblationExperiment(
        ablation_name='ablation2_coherence',
        conditions=conditions,
        n_trials=n_trials,
        epochs_per_task=epochs
    )
    
    results = experiment.run()
    tests = experiment.statistical_tests(results, 'with_coherence')
    experiment.save_results(results, tests)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION 2 SUMMARY: Coherence Criterion")
    print("=" * 70)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['avg_accuracy']['mean']:.4f} +/- {res['avg_accuracy']['std']:.4f}")
        print(f"  Forgetting: {res['forgetting']['mean']:.4f} +/- {res['forgetting']['std']:.4f}")
    
    print("\n--- Statistical Tests (vs with_coherence) ---")
    for comparison, test in tests.items():
        print(f"\n{comparison}:")
        print(f"  Accuracy: p={test['accuracy']['p_value']:.4f}, d={test['accuracy']['cohen_d']:.3f}")
        if test['accuracy']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['accuracy']['baseline_better']})")
        print(f"  Forgetting: p={test['forgetting']['p_value']:.4f}, d={test['forgetting']['cohen_d']:.3f}")
        if test['forgetting']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['forgetting']['baseline_better']})")
    
    return results, tests


# =============================================================================
# Ablation 3: Learning Rule
# =============================================================================

def ablation3_learning_rule(n_trials: int = 5, epochs: int = 5) -> Tuple[Dict, Dict]:
    """
    Ablation 3: Hebbian vs Gradient-based Learning
    
    Purpose: Compare Hebbian learning to standard gradient-based methods.
    
    Conditions:
        1. HEBBIAN (ours): Correlation-based updates
        2. SGD: Gradient descent with momentum
        3. ADAM: Adam optimizer
    """
    conditions = {
        'hebbian': AblationConfig(
            name='Hebbian (Ours)',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.95,
            learning_rule='hebbian',
            plasticity_rate=0.5
        ),
        'sgd': AblationConfig(
            name='SGD',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.95,
            learning_rule='sgd',
            sgd_lr=0.01,
            plasticity_rate=0.5  # Not used for SGD
        ),
        'adam': AblationConfig(
            name='Adam',
            random_win=False,
            freeze_win_after_task0=True,
            coherence_acceptance_threshold=0.95,
            learning_rule='adam',
            sgd_lr=0.001,  # Lower LR for Adam
            plasticity_rate=0.5  # Not used for Adam
        )
    }
    
    experiment = AblationExperiment(
        ablation_name='ablation3_learning_rule',
        conditions=conditions,
        n_trials=n_trials,
        epochs_per_task=epochs
    )
    
    results = experiment.run()
    tests = experiment.statistical_tests(results, 'hebbian')
    experiment.save_results(results, tests)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION 3 SUMMARY: Learning Rule")
    print("=" * 70)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {res['avg_accuracy']['mean']:.4f} +/- {res['avg_accuracy']['std']:.4f}")
        print(f"  Forgetting: {res['forgetting']['mean']:.4f} +/- {res['forgetting']['std']:.4f}")
    
    print("\n--- Statistical Tests (vs hebbian) ---")
    for comparison, test in tests.items():
        print(f"\n{comparison}:")
        print(f"  Accuracy: p={test['accuracy']['p_value']:.4f}, d={test['accuracy']['cohen_d']:.3f}")
        if test['accuracy']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['accuracy']['baseline_better']})")
        print(f"  Forgetting: p={test['forgetting']['p_value']:.4f}, d={test['forgetting']['cohen_d']:.3f}")
        if test['forgetting']['significant']:
            print(f"    *** SIGNIFICANT *** (baseline better: {test['forgetting']['baseline_better']})")
    
    return results, tests


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Path A Ablation Studies')
    parser.add_argument('--all', action='store_true', help='Run all ablations')
    parser.add_argument('--ablation1', action='store_true', help='Run W_in initialization ablation')
    parser.add_argument('--ablation2', action='store_true', help='Run coherence criterion ablation')
    parser.add_argument('--ablation3', action='store_true', help='Run learning rule ablation')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials (default: 5)')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per task (default: 5)')
    
    args = parser.parse_args()
    
    # If no specific ablation selected, show help
    if not (args.all or args.ablation1 or args.ablation2 or args.ablation3):
        parser.print_help()
        return
    
    all_results = {}
    start_time = time.time()
    
    if args.all or args.ablation1:
        print("\n" + "#" * 70)
        print("# Running Ablation 1: W_in Initialization")
        print("#" * 70)
        results1, tests1 = ablation1_winit(n_trials=args.n_trials, epochs=args.epochs)
        all_results['ablation1'] = {'results': results1, 'tests': tests1}
    
    if args.all or args.ablation2:
        print("\n" + "#" * 70)
        print("# Running Ablation 2: Coherence Criterion")
        print("#" * 70)
        results2, tests2 = ablation2_coherence(n_trials=args.n_trials, epochs=args.epochs)
        all_results['ablation2'] = {'results': results2, 'tests': tests2}
    
    if args.all or args.ablation3:
        print("\n" + "#" * 70)
        print("# Running Ablation 3: Learning Rule")
        print("#" * 70)
        results3, tests3 = ablation3_learning_rule(n_trials=args.n_trials, epochs=args.epochs)
        all_results['ablation3'] = {'results': results3, 'tests': tests3}
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"All ablations completed in {elapsed/60:.1f} minutes")
    print("=" * 70)
    
    # Generate visualizations
    if args.all or (args.ablation1 and args.ablation2 and args.ablation3):
        try:
            from visualize_ablations import generate_ablation_visualizations
            generate_ablation_visualizations()
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
    
    return all_results


if __name__ == "__main__":
    main()
