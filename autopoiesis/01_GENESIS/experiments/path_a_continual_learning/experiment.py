"""
Continual Learning Experiment Framework

Compares:
    1. Autopoietic Continual Learner
    2. Fine-tuning (SGD)
    3. EWC (Elastic Weight Consolidation)
    4. Replay (Experience Replay)

Metrics:
    - Average accuracy across all tasks
    - Forgetting measure
    - FLOPs (computational cost)
"""

import numpy as np
import torch
import json
import time
import os
from typing import Dict, List, Tuple
from scipy import stats
from datetime import datetime

from split_mnist import SplitMNIST
from autopoietic_continual_learner import AutopoeticContinualLearner
from baselines import FineTuning, EWC, Replay


class ContinualLearningExperiment:
    """
    Experiment framework for continual learning comparison.
    """
    
    def __init__(self,
                 num_trials: int = 5,
                 epochs_per_task: int = 5,
                 batch_size: int = 64,
                 results_dir: str = './results',
                 use_pca: bool = True,
                 pca_dim: int = 100):
        """
        Args:
            num_trials: Number of independent runs
            epochs_per_task: Training epochs per task
            batch_size: Batch size
            results_dir: Directory to save results
            use_pca: Whether to use PCA
            pca_dim: PCA dimension
        """
        self.num_trials = num_trials
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Method names
        self.methods = ['autopoietic', 'finetuning', 'ewc', 'replay']
        
        print("=" * 70)
        print("Continual Learning Experiment")
        print("=" * 70)
        print(f"Trials: {num_trials}")
        print(f"Epochs per task: {epochs_per_task}")
        print(f"Methods: {self.methods}")
        print(f"Results dir: {results_dir}")
        
    def _create_models(self, seed: int) -> Dict:
        """Create all models with the same seed."""
        input_dim = self.pca_dim if self.use_pca else 784
        
        models = {
            'autopoietic': AutopoeticContinualLearner(
                input_dim=input_dim,
                hidden_dim=256,
                num_tasks=5,
                plasticity_rate=0.5,  # Higher plasticity for Hebbian learning
                seed=seed
            ),
            'finetuning': FineTuning(
                input_dim=input_dim,
                hidden_dim=256,
                num_tasks=5,
                lr=0.01,
                seed=seed
            ),
            'ewc': EWC(
                input_dim=input_dim,
                hidden_dim=256,
                num_tasks=5,
                lr=0.01,
                ewc_lambda=400,
                seed=seed
            ),
            'replay': Replay(
                input_dim=input_dim,
                hidden_dim=256,
                num_tasks=5,
                lr=0.01,
                buffer_size=200,
                seed=seed
            )
        }
        
        return models
    
    def _evaluate_all_tasks(self,
                            model,
                            dataset: SplitMNIST,
                            method_name: str,
                            current_task: int) -> Dict[int, float]:
        """
        Evaluate a model on all tasks seen so far.
        
        Returns:
            accuracies: Dict mapping task_id -> accuracy
        """
        accuracies = {}
        
        for task_id in range(current_task + 1):
            data = dataset.task_data[task_id]
            
            if method_name == 'autopoietic':
                result = model.evaluate(
                    data['test_x'].numpy(),
                    data['test_y'].numpy(),
                    task_id
                )
            else:
                result = model.evaluate(
                    data['test_x'],
                    data['test_y'],
                    task_id
                )
                
            accuracies[task_id] = result['accuracy']
            
        return accuracies
    
    def _compute_forgetting(self, accuracy_matrix: np.ndarray) -> float:
        """
        Compute forgetting measure.
        
        Forgetting = average drop in accuracy on previous tasks
        
        Args:
            accuracy_matrix: (num_tasks, num_tasks) matrix
                             [i,j] = accuracy on task j after training on task i
                             
        Returns:
            forgetting: Average forgetting measure
        """
        num_tasks = accuracy_matrix.shape[0]
        forgetting = 0.0
        count = 0
        
        for j in range(num_tasks - 1):  # For each previous task
            # Maximum accuracy achieved on task j
            max_acc = accuracy_matrix[j, j]
            
            # Final accuracy on task j
            final_acc = accuracy_matrix[-1, j]
            
            # Forgetting
            forgetting += max(0, max_acc - final_acc)
            count += 1
            
        return forgetting / count if count > 0 else 0.0
    
    def run_single_trial(self, 
                         trial_id: int, 
                         seed: int) -> Dict:
        """
        Run a single trial.
        
        Args:
            trial_id: Trial identifier
            seed: Random seed
            
        Returns:
            results: Trial results
        """
        print(f"\n{'='*70}")
        print(f"Trial {trial_id + 1}/{self.num_trials} (seed={seed})")
        print(f"{'='*70}")
        
        # Create dataset
        dataset = SplitMNIST(
            data_dir='./data',
            use_pca=self.use_pca,
            pca_dim=self.pca_dim,
            batch_size=self.batch_size,
            seed=seed
        )
        
        # Create models
        models = self._create_models(seed)
        
        # Results storage
        results = {
            method: {
                'accuracy_matrix': np.zeros((5, 5)),  # [trained_up_to, eval_task]
                'training_time': [],
                'flops': 0
            }
            for method in self.methods
        }
        
        # Sequential task learning
        for task_id in range(5):
            print(f"\n--- Task {task_id} (digits {dataset.TASK_DIGITS[task_id]}) ---")
            
            train_loader = dataset.get_task_dataloader(task_id, train=True)
            
            for method in self.methods:
                print(f"\n[{method.upper()}]")
                
                model = models[method]
                start_time = time.time()
                
                # Train
                if method == 'autopoietic':
                    # Convert to numpy for autopoietic
                    model.train_on_task(
                        train_loader,
                        task_id,
                        epochs=self.epochs_per_task,
                        verbose=True
                    )
                else:
                    model.train_on_task(
                        train_loader,
                        task_id,
                        epochs=self.epochs_per_task,
                        verbose=True
                    )
                    
                elapsed = time.time() - start_time
                results[method]['training_time'].append(elapsed)
                results[method]['flops'] = model.get_flops()
                
                # Evaluate on all tasks seen so far
                accuracies = self._evaluate_all_tasks(model, dataset, method, task_id)
                
                for eval_task, acc in accuracies.items():
                    results[method]['accuracy_matrix'][task_id, eval_task] = acc
                    
                print(f"  Accuracies: {[f'{v:.3f}' for v in accuracies.values()]}")
                print(f"  Time: {elapsed:.1f}s, FLOPs: {results[method]['flops']:,}")
        
        # Compute final metrics
        for method in self.methods:
            acc_matrix = results[method]['accuracy_matrix']
            
            # Average accuracy (final row)
            results[method]['avg_accuracy'] = np.mean(acc_matrix[-1, :])
            
            # Forgetting
            results[method]['forgetting'] = self._compute_forgetting(acc_matrix)
            
            # Total time
            results[method]['total_time'] = sum(results[method]['training_time'])
            
        return results
    
    def run_all_trials(self) -> Dict:
        """
        Run all trials.
        
        Returns:
            all_results: Results from all trials
        """
        all_results = {method: [] for method in self.methods}
        
        base_seed = 42
        
        for trial_id in range(self.num_trials):
            seed = base_seed + trial_id * 100
            
            trial_results = self.run_single_trial(trial_id, seed)
            
            for method in self.methods:
                all_results[method].append(trial_results[method])
                
        return all_results
    
    def compute_statistics(self, all_results: Dict) -> Dict:
        """
        Compute statistics across trials.
        
        Args:
            all_results: Results from all trials
            
        Returns:
            statistics: Statistical summary
        """
        statistics = {}
        
        for method in self.methods:
            results = all_results[method]
            
            # Extract metrics
            avg_accs = [r['avg_accuracy'] for r in results]
            forgettings = [r['forgetting'] for r in results]
            times = [r['total_time'] for r in results]
            flops = [r['flops'] for r in results]
            
            statistics[method] = {
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
                'time': {
                    'mean': np.mean(times),
                    'std': np.std(times)
                },
                'flops': {
                    'mean': np.mean(flops),
                    'std': np.std(flops)
                },
                'accuracy_matrices': [r['accuracy_matrix'].tolist() for r in results]
            }
            
        return statistics
    
    def statistical_tests(self, statistics: Dict) -> Dict:
        """
        Perform statistical tests comparing autopoietic vs baselines.
        
        Args:
            statistics: Statistics from compute_statistics
            
        Returns:
            tests: Statistical test results
        """
        tests = {}
        
        auto_acc = statistics['autopoietic']['avg_accuracy']['values']
        auto_fgt = statistics['autopoietic']['forgetting']['values']
        
        for baseline in ['finetuning', 'ewc', 'replay']:
            base_acc = statistics[baseline]['avg_accuracy']['values']
            base_fgt = statistics[baseline]['forgetting']['values']
            
            # T-test for accuracy
            t_acc, p_acc = stats.ttest_ind(auto_acc, base_acc)
            
            # T-test for forgetting (lower is better)
            t_fgt, p_fgt = stats.ttest_ind(auto_fgt, base_fgt)
            
            # Cohen's d for effect size
            pooled_std_acc = np.sqrt((np.var(auto_acc) + np.var(base_acc)) / 2)
            cohen_d_acc = (np.mean(auto_acc) - np.mean(base_acc)) / pooled_std_acc if pooled_std_acc > 0 else 0
            
            pooled_std_fgt = np.sqrt((np.var(auto_fgt) + np.var(base_fgt)) / 2)
            cohen_d_fgt = (np.mean(auto_fgt) - np.mean(base_fgt)) / pooled_std_fgt if pooled_std_fgt > 0 else 0
            
            tests[f'autopoietic_vs_{baseline}'] = {
                'accuracy': {
                    't_statistic': float(t_acc) if not np.isnan(t_acc) else 0.0,
                    'p_value': float(p_acc) if not np.isnan(p_acc) else 1.0,
                    'cohen_d': float(cohen_d_acc) if not np.isnan(cohen_d_acc) else 0.0,
                    'significant': bool(p_acc < 0.05) if not np.isnan(p_acc) else False
                },
                'forgetting': {
                    't_statistic': float(t_fgt) if not np.isnan(t_fgt) else 0.0,
                    'p_value': float(p_fgt) if not np.isnan(p_fgt) else 1.0,
                    'cohen_d': float(cohen_d_fgt) if not np.isnan(cohen_d_fgt) else 0.0,
                    'significant': bool(p_fgt < 0.05) if not np.isnan(p_fgt) else False
                }
            }
            
        return tests
    
    def save_results(self, 
                     all_results: Dict,
                     statistics: Dict,
                     tests: Dict):
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save statistics
        stats_file = os.path.join(self.results_dir, f'statistics_{timestamp}.json')
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
        
        # Save tests
        tests_file = os.path.join(self.results_dir, f'statistical_tests_{timestamp}.json')
        with open(tests_file, 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Statistical tests saved to: {tests_file}")
        
        # Save summary
        summary = self._create_summary(statistics, tests)
        summary_file = os.path.join(self.results_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"Summary saved to: {summary_file}")
        
        return stats_file, tests_file, summary_file
    
    def _create_summary(self, statistics: Dict, tests: Dict) -> str:
        """Create human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("CONTINUAL LEARNING EXPERIMENT RESULTS")
        lines.append("=" * 70)
        lines.append(f"\nTrials: {self.num_trials}")
        lines.append(f"Epochs per task: {self.epochs_per_task}")
        lines.append(f"Tasks: 5 (Split-MNIST)")
        
        lines.append("\n" + "-" * 70)
        lines.append("AVERAGE ACCURACY (higher is better)")
        lines.append("-" * 70)
        
        for method in self.methods:
            stats = statistics[method]['avg_accuracy']
            lines.append(f"  {method:15s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
            
        lines.append("\n" + "-" * 70)
        lines.append("FORGETTING MEASURE (lower is better)")
        lines.append("-" * 70)
        
        for method in self.methods:
            stats = statistics[method]['forgetting']
            lines.append(f"  {method:15s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
            
        lines.append("\n" + "-" * 70)
        lines.append("COMPUTATIONAL COST")
        lines.append("-" * 70)
        
        for method in self.methods:
            time_stats = statistics[method]['time']
            flops_stats = statistics[method]['flops']
            lines.append(f"  {method:15s}: Time={time_stats['mean']:.1f}s, FLOPs={flops_stats['mean']:.2e}")
            
        lines.append("\n" + "-" * 70)
        lines.append("STATISTICAL TESTS (Autopoietic vs Baselines)")
        lines.append("-" * 70)
        
        for comparison, results in tests.items():
            lines.append(f"\n  {comparison}:")
            acc = results['accuracy']
            fgt = results['forgetting']
            
            lines.append(f"    Accuracy:   t={acc['t_statistic']:.3f}, p={acc['p_value']:.4f}, d={acc['cohen_d']:.3f}")
            lines.append(f"                {'*** SIGNIFICANT ***' if acc['significant'] else 'not significant'}")
            
            lines.append(f"    Forgetting: t={fgt['t_statistic']:.3f}, p={fgt['p_value']:.4f}, d={fgt['cohen_d']:.3f}")
            lines.append(f"                {'*** SIGNIFICANT ***' if fgt['significant'] else 'not significant'}")
        
        lines.append("\n" + "=" * 70)
        lines.append("CONCLUSION")
        lines.append("=" * 70)
        
        # Check if autopoietic beats finetuning
        auto_fgt = statistics['autopoietic']['forgetting']['mean']
        ft_fgt = statistics['finetuning']['forgetting']['mean']
        ewc_fgt = statistics['ewc']['forgetting']['mean']
        
        if auto_fgt < ft_fgt:
            lines.append("\n[PASS] Autopoietic has LOWER forgetting than Fine-tuning")
        else:
            lines.append("\n[FAIL] Autopoietic has HIGHER forgetting than Fine-tuning")
            
        if auto_fgt <= ewc_fgt * 1.1:  # Within 10%
            lines.append("[PASS] Autopoietic forgetting is comparable to or better than EWC")
        else:
            lines.append("[PARTIAL] Autopoietic forgetting is higher than EWC")
            
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def run(self) -> Tuple[Dict, Dict, Dict]:
        """
        Run complete experiment.
        
        Returns:
            all_results, statistics, tests
        """
        print("\nStarting experiment...")
        start_time = time.time()
        
        # Run trials
        all_results = self.run_all_trials()
        
        # Compute statistics
        print("\nComputing statistics...")
        statistics = self.compute_statistics(all_results)
        
        # Statistical tests
        print("Running statistical tests...")
        tests = self.statistical_tests(statistics)
        
        # Save results
        print("\nSaving results...")
        self.save_results(all_results, statistics, tests)
        
        # Print summary
        summary = self._create_summary(statistics, tests)
        print("\n" + summary)
        
        elapsed = time.time() - start_time
        print(f"\nTotal experiment time: {elapsed/60:.1f} minutes")
        
        return all_results, statistics, tests


def main():
    """Run the experiment."""
    experiment = ContinualLearningExperiment(
        num_trials=3,  # N=3 for initial validation
        epochs_per_task=3,  # Reduced for faster iteration
        batch_size=64,
        results_dir='./results',
        use_pca=True,
        pca_dim=100
    )

    all_results, statistics, tests = experiment.run()

    return all_results, statistics, tests


if __name__ == "__main__":
    main()
