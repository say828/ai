"""
GENESIS Experiment 3: Multi-Task Learning

Question: Can GENESIS learn multiple tasks simultaneously and exhibit transfer learning?

Traditional Multi-Task Learning:
  L_total = sum(L_task_i) + regularization
  Shared representations learned via joint optimization

GENESIS Multi-Task:
  Each task affects viability
  Entity must balance multiple objectives
  No explicit shared representations - must emerge naturally

Key Questions:
1. Can entity learn multiple tasks simultaneously?
2. Does transfer learning occur between tasks?
3. Does catastrophic forgetting happen?
4. Do task-specific adaptations emerge?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from genesis_entity_v1_1 import GENESIS_Entity_v1_1
from genesis_environment import Environment


class MultiTaskEnvironment(Environment):
    """
    Environment with multiple regression tasks

    Tasks:
    1. Linear: y = 2*x1 + 3*x2
    2. Quadratic: y = x1^2 + x2^2
    3. Nonlinear: y = sin(x1) + cos(x2)
    4. Interaction: y = x1 * x2
    """

    def __init__(self, n_samples: int = 100, noise_level: float = 0.1):
        self.n_samples = n_samples
        self.noise_level = noise_level

        # Generate data for all tasks
        np.random.seed(42)
        self.X = np.random.randn(n_samples, 2) * 2  # Scale for better task diversity

        # Task 1: Linear
        self.y_linear = (2 * self.X[:, 0] + 3 * self.X[:, 1] +
                        np.random.randn(n_samples) * noise_level)

        # Task 2: Quadratic
        self.y_quadratic = (self.X[:, 0]**2 + self.X[:, 1]**2 +
                           np.random.randn(n_samples) * noise_level)

        # Task 3: Nonlinear (trigonometric)
        self.y_nonlinear = (np.sin(self.X[:, 0]) + np.cos(self.X[:, 1]) +
                           np.random.randn(n_samples) * noise_level)

        # Task 4: Interaction
        self.y_interaction = (self.X[:, 0] * self.X[:, 1] +
                             np.random.randn(n_samples) * noise_level)

        self.tasks = {
            'linear': self.y_linear,
            'quadratic': self.y_quadratic,
            'nonlinear': self.y_nonlinear,
            'interaction': self.y_interaction
        }

        self.current_task = 'linear'
        self.last_prediction = None
        self.last_target = None
        self.last_task = None

    def set_task(self, task_name: str):
        """Switch to specific task"""
        if task_name in self.tasks:
            self.current_task = task_name

    def probe(self, query: Dict):
        """Entity probes for information"""
        query_type = query.get('type', 'random_sample')

        if query_type == 'random_sample':
            idx = np.random.randint(self.n_samples)
            return {
                'input': self.X[idx],
                'task': self.current_task,
                'feedback': None
            }

        return {'error': 'unknown query'}

    def apply(self, action: Dict) -> Dict:
        """Entity takes action"""
        action_type = action.get('type', 'predict')

        if action_type == 'predict':
            input_data = action.get('input')
            prediction = action.get('prediction')

            # Find corresponding target
            if input_data is not None:
                distances = np.linalg.norm(self.X - input_data, axis=1)
                idx = np.argmin(distances)
                target = self.tasks[self.current_task][idx]
            else:
                # Random sample
                idx = np.random.randint(self.n_samples)
                target = self.tasks[self.current_task][idx]
                input_data = self.X[idx]

            self.last_prediction = prediction
            self.last_target = target
            self.last_task = self.current_task

            # Calculate error
            if prediction is not None and len(prediction) > 0:
                pred_val = prediction[0] if hasattr(prediction, '__len__') else prediction
                error = np.abs(pred_val - target)
            else:
                error = 100.0

            # Viability contribution
            viability_contribution = np.exp(-error / 5.0)  # Scale for multiple tasks

            return {
                'action_taken': True,
                'prediction': prediction,
                'target': target,
                'error': error,
                'viability_contribution': viability_contribution,
                'success': error < 2.0,
                'task': self.current_task
            }

        elif action_type == 'explore':
            idx = np.random.randint(self.n_samples)
            return {
                'input': self.X[idx],
                'target': self.tasks[self.current_task][idx],
                'task': self.current_task,
                'exploration': True,
                'viability_contribution': 0.5
            }

        return {'error': 'unknown action', 'viability_contribution': 0.3}

    def observe_consequence(self) -> Dict:
        """Observe consequence of action"""
        if self.last_prediction is None or self.last_target is None:
            return {
                'consequence': None,
                'viability_contribution': 0.5
            }

        error = np.abs(self.last_prediction - self.last_target) if hasattr(self.last_prediction, '__len__') else 100.0
        viability_contribution = np.exp(-error / 5.0)

        return {
            'error': error,
            'viability_contribution': viability_contribution,
            'success': error < 2.0,
            'task': self.last_task
        }


def evaluate_on_task(entity, env: MultiTaskEnvironment, task_name: str, n_samples: int = 20) -> float:
    """
    Evaluate entity's performance on specific task
    Returns average error
    """
    env.set_task(task_name)
    errors = []

    for _ in range(n_samples):
        idx = np.random.randint(len(env.X))
        input_data = env.X[idx]
        target = env.tasks[task_name][idx]

        try:
            prediction = entity.phenotype.forward(input_data)
            if len(prediction) > 0:
                error = np.abs(prediction[0] - target)
            else:
                error = 100.0
        except:
            error = 100.0

        errors.append(error)

    return np.mean(errors)


def scenario_A_single_task(steps_per_task: int = 200) -> Dict:
    """
    Scenario A: Single entity, single task (baseline)

    Train on each task independently to establish baseline performance
    """
    print("\n" + "="*70)
    print("SCENARIO A: Single Entity, Single Task (Baseline)")
    print("="*70)

    env = MultiTaskEnvironment()
    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    results = {}

    for task_name in task_names:
        print(f"\nTraining on task: {task_name}")
        env.set_task(task_name)

        entity = GENESIS_Entity_v1_1(entity_id=1)

        error_history = []
        viability_history = []

        for step in range(steps_per_task):
            viability = entity.live_one_step(env, ecosystem=None)
            viability_history.append(viability)

            # Evaluate
            error = evaluate_on_task(entity, env, task_name, n_samples=5)
            error_history.append(error)

            if step % 50 == 0 and step > 0:
                print(f"  Step {step}: error={np.mean(error_history[-10:]):.3f}, viability={viability:.3f}")

        final_error = np.mean(error_history[-20:])
        initial_error = np.mean(error_history[:20])

        results[task_name] = {
            'error_history': error_history,
            'viability_history': viability_history,
            'final_error': final_error,
            'initial_error': initial_error,
            'improvement': (initial_error - final_error) / initial_error if initial_error > 0 else 0
        }

        print(f"  Final error: {final_error:.3f} (improvement: {results[task_name]['improvement']*100:.1f}%)")

    return results


def scenario_B_sequential(steps_per_task: int = 200) -> Dict:
    """
    Scenario B: Single entity, multi-task sequential

    Train on tasks one after another to measure catastrophic forgetting
    """
    print("\n" + "="*70)
    print("SCENARIO B: Single Entity, Sequential Multi-Task")
    print("="*70)

    env = MultiTaskEnvironment()
    entity = GENESIS_Entity_v1_1(entity_id=1)

    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    results = {
        'task_sequence': [],
        'all_tasks_performance': {task: [] for task in task_names}
    }

    for task_idx, task_name in enumerate(task_names):
        print(f"\n[Phase {task_idx+1}] Training on: {task_name}")
        env.set_task(task_name)

        error_history = []

        for step in range(steps_per_task):
            entity.live_one_step(env, ecosystem=None)

            # Evaluate current task
            error = evaluate_on_task(entity, env, task_name, n_samples=5)
            error_history.append(error)

            # Evaluate ALL tasks to track forgetting
            if step % 50 == 0:
                print(f"  Step {step}: current_task_error={np.mean(error_history[-10:]):.3f}")
                for eval_task in task_names:
                    eval_error = evaluate_on_task(entity, env, eval_task, n_samples=5)
                    results['all_tasks_performance'][eval_task].append(eval_error)

        results['task_sequence'].append({
            'task': task_name,
            'error_history': error_history,
            'final_error': np.mean(error_history[-20:])
        })

    # Final evaluation on all tasks
    print("\nFinal evaluation on all tasks:")
    final_performance = {}
    for task_name in task_names:
        final_error = evaluate_on_task(entity, env, task_name, n_samples=50)
        final_performance[task_name] = final_error
        print(f"  {task_name}: {final_error:.3f}")

    results['final_performance'] = final_performance

    return results


def scenario_C_interleaved(steps_total: int = 800) -> Dict:
    """
    Scenario C: Single entity, interleaved multi-task

    Randomly switch between tasks to encourage shared representations
    """
    print("\n" + "="*70)
    print("SCENARIO C: Single Entity, Interleaved Multi-Task")
    print("="*70)

    env = MultiTaskEnvironment()
    entity = GENESIS_Entity_v1_1(entity_id=1)

    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    task_error_histories = {task: [] for task in task_names}
    viability_history = []
    task_sequence = []

    for step in range(steps_total):
        # Randomly select task
        current_task = np.random.choice(task_names)
        env.set_task(current_task)
        task_sequence.append(current_task)

        viability = entity.live_one_step(env, ecosystem=None)
        viability_history.append(viability)

        # Evaluate all tasks periodically
        if step % 20 == 0:
            for task_name in task_names:
                error = evaluate_on_task(entity, env, task_name, n_samples=5)
                task_error_histories[task_name].append(error)

        if step % 200 == 0 and step > 0:
            print(f"\nStep {step}:")
            for task_name in task_names:
                if len(task_error_histories[task_name]) > 0:
                    recent_error = np.mean(task_error_histories[task_name][-5:])
                    print(f"  {task_name}: {recent_error:.3f}")

    # Final evaluation
    print("\nFinal evaluation:")
    final_performance = {}
    for task_name in task_names:
        final_error = evaluate_on_task(entity, env, task_name, n_samples=50)
        final_performance[task_name] = final_error
        print(f"  {task_name}: {final_error:.3f}")

    return {
        'task_error_histories': task_error_histories,
        'viability_history': viability_history,
        'task_sequence': task_sequence,
        'final_performance': final_performance
    }


def scenario_D_multiple_entities(steps_per_task: int = 200) -> Dict:
    """
    Scenario D: Multiple entities, each specializing in one task

    Compare with multi-task single entity
    """
    print("\n" + "="*70)
    print("SCENARIO D: Multiple Entities, Task Specialization")
    print("="*70)

    env = MultiTaskEnvironment()
    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    # Create entities one at a time to avoid initialization issues
    entities = {}
    for i, task in enumerate(task_names):
        try:
            entities[task] = GENESIS_Entity_v1_1(entity_id=i)
        except Exception as e:
            print(f"Warning: Failed to create entity for {task}: {e}")
            print(f"Retrying...")
            entities[task] = GENESIS_Entity_v1_1(entity_id=i)

    results = {}

    for task_name in task_names:
        print(f"\nTraining specialist for: {task_name}")
        env.set_task(task_name)
        entity = entities[task_name]

        error_history = []

        for step in range(steps_per_task):
            entity.live_one_step(env, ecosystem=None)

            error = evaluate_on_task(entity, env, task_name, n_samples=5)
            error_history.append(error)

            if step % 50 == 0 and step > 0:
                print(f"  Step {step}: error={np.mean(error_history[-10:]):.3f}")

        results[task_name] = {
            'error_history': error_history,
            'final_error': np.mean(error_history[-20:]),
            'entity': entity
        }

        print(f"  Final error: {results[task_name]['final_error']:.3f}")

    # Cross-evaluation: test each specialist on other tasks
    print("\nCross-evaluation (transfer test):")
    cross_performance = {}
    for specialist_task, entity in entities.items():
        cross_performance[specialist_task] = {}
        for test_task in task_names:
            error = evaluate_on_task(entity, env, test_task, n_samples=50)
            cross_performance[specialist_task][test_task] = error
            marker = " <--" if specialist_task == test_task else ""
            print(f"  {specialist_task}_specialist on {test_task}: {error:.3f}{marker}")

    results['cross_performance'] = cross_performance

    return results


def compute_transfer_matrix(results_B: Dict, results_A: Dict) -> np.ndarray:
    """
    Compute transfer learning matrix

    transfer[i,j] = (baseline_error_j - sequential_error_j_after_training_i) / baseline_error_j

    Positive = positive transfer
    Negative = negative transfer (interference)
    """
    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']
    n_tasks = len(task_names)

    transfer_matrix = np.zeros((n_tasks, n_tasks))

    # Use final performance from sequential training
    for i, source_task in enumerate(task_names):
        for j, target_task in enumerate(task_names):
            baseline_error = results_A[target_task]['final_error']
            sequential_error = results_B['final_performance'][target_task]

            if baseline_error > 0:
                transfer = (baseline_error - sequential_error) / baseline_error
                transfer_matrix[i, j] = transfer

    return transfer_matrix


def visualize_all_results(results_A, results_B, results_C, results_D):
    """
    Comprehensive visualization of all scenarios
    """
    print("\nGenerating comprehensive visualizations...")

    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    fig = plt.figure(figsize=(20, 12))

    # 1. Scenario A: Single task baselines
    ax1 = plt.subplot(3, 4, 1)
    for task in task_names:
        ax1.plot(results_A[task]['error_history'], label=task, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Error')
    ax1.set_title('Scenario A: Single Task Baselines')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Scenario A: Final performance
    ax2 = plt.subplot(3, 4, 2)
    final_errors_A = [results_A[task]['final_error'] for task in task_names]
    ax2.bar(range(len(task_names)), final_errors_A, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(task_names)))
    ax2.set_xticklabels(task_names, rotation=45)
    ax2.set_ylabel('Final Error')
    ax2.set_title('Scenario A: Final Performance')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Scenario B: Sequential learning
    ax3 = plt.subplot(3, 4, 3)
    for task_info in results_B['task_sequence']:
        ax3.plot(task_info['error_history'], label=task_info['task'], alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Error')
    ax3.set_title('Scenario B: Sequential Learning')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Scenario B: Catastrophic forgetting
    ax4 = plt.subplot(3, 4, 4)
    for task in task_names:
        if len(results_B['all_tasks_performance'][task]) > 0:
            ax4.plot(results_B['all_tasks_performance'][task], label=task, marker='o')
    ax4.set_xlabel('Evaluation Point')
    ax4.set_ylabel('Error')
    ax4.set_title('Scenario B: Forgetting (all tasks)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Scenario C: Interleaved learning
    ax5 = plt.subplot(3, 4, 5)
    for task in task_names:
        ax5.plot(results_C['task_error_histories'][task], label=task, alpha=0.7, linewidth=2)
    ax5.set_xlabel('Evaluation Point (every 20 steps)')
    ax5.set_ylabel('Error')
    ax5.set_title('Scenario C: Interleaved Learning')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')

    # 6. Scenario C: Task distribution
    ax6 = plt.subplot(3, 4, 6)
    task_counts = {task: results_C['task_sequence'].count(task) for task in task_names}
    ax6.bar(range(len(task_names)), list(task_counts.values()), color='coral', alpha=0.7)
    ax6.set_xticks(range(len(task_names)))
    ax6.set_xticklabels(task_names, rotation=45)
    ax6.set_ylabel('Count')
    ax6.set_title('Scenario C: Task Distribution')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Scenario D: Specialist performance
    ax7 = plt.subplot(3, 4, 7)
    for task in task_names:
        ax7.plot(results_D[task]['error_history'], label=task, alpha=0.7)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Error')
    ax7.set_title('Scenario D: Specialist Training')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')

    # 8. Transfer matrix (from Scenario B)
    ax8 = plt.subplot(3, 4, 8)
    transfer_matrix = compute_transfer_matrix(results_B, results_A)
    im = ax8.imshow(transfer_matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    ax8.set_xticks(range(len(task_names)))
    ax8.set_yticks(range(len(task_names)))
    ax8.set_xticklabels(task_names, rotation=45)
    ax8.set_yticklabels(task_names)
    ax8.set_xlabel('Target Task')
    ax8.set_ylabel('After Training')
    ax8.set_title('Transfer Matrix (B vs A)')
    plt.colorbar(im, ax=ax8)

    # 9. Comparison: All scenarios final performance
    ax9 = plt.subplot(3, 4, 9)
    x = np.arange(len(task_names))
    width = 0.2
    ax9.bar(x - width*1.5, [results_A[t]['final_error'] for t in task_names],
            width, label='A: Single', alpha=0.7)
    ax9.bar(x - width*0.5, [results_B['final_performance'][t] for t in task_names],
            width, label='B: Sequential', alpha=0.7)
    ax9.bar(x + width*0.5, [results_C['final_performance'][t] for t in task_names],
            width, label='C: Interleaved', alpha=0.7)
    ax9.bar(x + width*1.5, [results_D[t]['final_error'] for t in task_names],
            width, label='D: Specialist', alpha=0.7)
    ax9.set_xticks(x)
    ax9.set_xticklabels(task_names, rotation=45)
    ax9.set_ylabel('Final Error')
    ax9.set_title('Final Performance Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')

    # 10. Improvement rates
    ax10 = plt.subplot(3, 4, 10)
    improvements_A = [results_A[t]['improvement'] * 100 for t in task_names]
    ax10.bar(range(len(task_names)), improvements_A, color='green', alpha=0.7)
    ax10.set_xticks(range(len(task_names)))
    ax10.set_xticklabels(task_names, rotation=45)
    ax10.set_ylabel('Improvement (%)')
    ax10.set_title('Learning Improvement Rate')
    ax10.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax10.grid(True, alpha=0.3, axis='y')

    # 11. Cross-task transfer (Scenario D)
    ax11 = plt.subplot(3, 4, 11)
    cross_matrix = np.zeros((len(task_names), len(task_names)))
    for i, specialist in enumerate(task_names):
        for j, test_task in enumerate(task_names):
            cross_matrix[i, j] = results_D['cross_performance'][specialist][test_task]
    im = ax11.imshow(cross_matrix, cmap='YlOrRd')
    ax11.set_xticks(range(len(task_names)))
    ax11.set_yticks(range(len(task_names)))
    ax11.set_xticklabels(task_names, rotation=45)
    ax11.set_yticklabels(task_names)
    ax11.set_xlabel('Test Task')
    ax11.set_ylabel('Specialist')
    ax11.set_title('Cross-Task Performance Matrix')
    plt.colorbar(im, ax=ax11)

    # 12. Generalization score
    ax12 = plt.subplot(3, 4, 12)

    # Compute average performance across all tasks
    avg_A = np.mean([results_A[t]['final_error'] for t in task_names])
    avg_B = np.mean([results_B['final_performance'][t] for t in task_names])
    avg_C = np.mean([results_C['final_performance'][t] for t in task_names])
    avg_D = np.mean([results_D[t]['final_error'] for t in task_names])

    scenarios = ['Single\n(A)', 'Sequential\n(B)', 'Interleaved\n(C)', 'Specialist\n(D)']
    averages = [avg_A, avg_B, avg_C, avg_D]

    colors = ['steelblue', 'orange', 'green', 'red']
    ax12.bar(range(len(scenarios)), averages, color=colors, alpha=0.7)
    ax12.set_xticks(range(len(scenarios)))
    ax12.set_xticklabels(scenarios)
    ax12.set_ylabel('Average Error')
    ax12.set_title('Overall Generalization Score')
    ax12.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = '/Users/say/Documents/GitHub/ai/08_GENESIS/experiment_3_multitask_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return save_path


def run_experiment_3():
    """
    Main experiment runner
    """
    print("="*70)
    print("GENESIS EXPERIMENT 3: Multi-Task Learning")
    print("="*70)
    print("\nTesting GENESIS's ability to:")
    print("1. Learn multiple tasks simultaneously")
    print("2. Transfer knowledge between tasks")
    print("3. Avoid catastrophic forgetting")
    print("4. Develop task-specific adaptations")

    # Run all scenarios
    results_A = scenario_A_single_task(steps_per_task=200)
    results_B = scenario_B_sequential(steps_per_task=200)
    results_C = scenario_C_interleaved(steps_total=800)
    results_D = scenario_D_multiple_entities(steps_per_task=200)

    # Analyze results
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)

    task_names = ['linear', 'quadratic', 'nonlinear', 'interaction']

    # 1. Single-task baseline
    print("\n1. Single-Task Baselines (Scenario A):")
    for task in task_names:
        improvement = results_A[task]['improvement'] * 100
        print(f"   {task:12s}: error={results_A[task]['final_error']:.3f}, "
              f"improvement={improvement:.1f}%")

    # 2. Sequential multi-task
    print("\n2. Sequential Multi-Task (Scenario B):")
    for task in task_names:
        error = results_B['final_performance'][task]
        baseline = results_A[task]['final_error']
        diff = ((error - baseline) / baseline * 100) if baseline > 0 else 0
        marker = "WORSE" if diff > 10 else "BETTER" if diff < -10 else "SIMILAR"
        print(f"   {task:12s}: error={error:.3f} ({marker} vs baseline, {diff:+.1f}%)")

    # 3. Interleaved multi-task
    print("\n3. Interleaved Multi-Task (Scenario C):")
    for task in task_names:
        error = results_C['final_performance'][task]
        baseline = results_A[task]['final_error']
        diff = ((error - baseline) / baseline * 100) if baseline > 0 else 0
        print(f"   {task:12s}: error={error:.3f} (vs baseline: {diff:+.1f}%)")

    # 4. Specialist comparison
    print("\n4. Specialists (Scenario D):")
    for task in task_names:
        error = results_D[task]['final_error']
        baseline = results_A[task]['final_error']
        diff = ((error - baseline) / baseline * 100) if baseline > 0 else 0
        print(f"   {task:12s}: error={error:.3f} (vs baseline: {diff:+.1f}%)")

    # 5. Transfer learning analysis
    print("\n5. Transfer Learning Analysis:")
    transfer_matrix = compute_transfer_matrix(results_B, results_A)
    avg_transfer = np.mean(transfer_matrix)
    print(f"   Average transfer: {avg_transfer:.3f}")
    if avg_transfer > 0.1:
        print("   --> POSITIVE transfer detected!")
    elif avg_transfer < -0.1:
        print("   --> NEGATIVE transfer (interference)")
    else:
        print("   --> Minimal transfer")

    # 6. Catastrophic forgetting
    print("\n6. Catastrophic Forgetting Check:")
    forgetting_detected = False
    for i, task_info in enumerate(results_B['task_sequence']):
        task = task_info['task']
        final_on_task = results_B['final_performance'][task]
        immediate_after = task_info['final_error']

        if final_on_task > immediate_after * 1.5:  # 50% degradation
            print(f"   {task}: FORGOT (immediate: {immediate_after:.3f}, "
                  f"final: {final_on_task:.3f})")
            forgetting_detected = True

    if not forgetting_detected:
        print("   No significant catastrophic forgetting detected!")

    # 7. Best scenario
    print("\n7. Best Multi-Task Approach:")
    avg_errors = {
        'Scenario A (baseline)': np.mean([results_A[t]['final_error'] for t in task_names]),
        'Scenario B (sequential)': np.mean([results_B['final_performance'][t] for t in task_names]),
        'Scenario C (interleaved)': np.mean([results_C['final_performance'][t] for t in task_names]),
        'Scenario D (specialists)': np.mean([results_D[t]['final_error'] for t in task_names])
    }

    best_scenario = min(avg_errors.items(), key=lambda x: x[1])
    print(f"   Winner: {best_scenario[0]} (avg error: {best_scenario[1]:.3f})")

    for scenario, error in avg_errors.items():
        print(f"   {scenario}: {error:.3f}")

    # Visualization
    visualize_all_results(results_A, results_B, results_C, results_D)

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print("\nCan GENESIS learn multiple tasks?")
    all_improved = all(results_A[t]['improvement'] > 0 for t in task_names)
    print(f"   {'YES' if all_improved else 'PARTIALLY'} - "
          f"{sum(1 for t in task_names if results_A[t]['improvement'] > 0)}/{len(task_names)} "
          f"tasks improved")

    print("\nDoes transfer learning occur?")
    if avg_transfer > 0.1:
        print(f"   YES - Average transfer: {avg_transfer:.3f}")
    else:
        print(f"   LIMITED - Average transfer: {avg_transfer:.3f}")

    print("\nDoes catastrophic forgetting happen?")
    print(f"   {'YES' if forgetting_detected else 'NO'}")

    print("\nBest multi-task strategy?")
    print(f"   {best_scenario[0]}")

    return {
        'scenario_A': results_A,
        'scenario_B': results_B,
        'scenario_C': results_C,
        'scenario_D': results_D,
        'transfer_matrix': transfer_matrix,
        'avg_transfer': avg_transfer,
        'forgetting_detected': forgetting_detected,
        'best_scenario': best_scenario
    }


if __name__ == "__main__":
    print("Starting GENESIS Multi-Task Learning Experiment...")
    print("This will take several minutes...\n")

    results = run_experiment_3()

    print("\n" + "="*70)
    print("EXPERIMENT 3 COMPLETE!")
    print("="*70)
    print("\nResults saved to: experiment_3_multitask_results.png")
    print("Documentation will be generated in: GENESIS_Multitask_결과.md")
