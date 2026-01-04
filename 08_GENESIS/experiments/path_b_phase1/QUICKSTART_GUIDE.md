# GENESIS Phase 4: Quick-Start Guide

**Last Updated**: 2026-01-04
**For**: Researchers and developers using GENESIS Phase 4

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Tests](#quick-tests)
3. [Running Experiments](#running-experiments)
4. [Performance Optimization](#performance-optimization)
5. [Visualization](#visualization)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

```bash
Python 3.8+
numpy
torch (PyTorch)
matplotlib
scikit-learn
```

### Setup

```bash
# Navigate to Phase 4 directory
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_phase1

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy torch matplotlib scikit-learn scipy
```

---

## Quick Tests

### 1. Test Phase 4B (Open-Ended Learning)

**What it does**: Tests Novelty Search and MAP-Elites

```bash
python test_phase4b_quick.py
```

**Expected output**:
```
✓ Novelty Archive: 3,984 unique behaviors (99.6% unique)
✓ MAP-Elites Coverage: 82.4%
⏱ Runtime: ~2-3 minutes
```

### 2. Test Phase 4C (Emergent Communication)

**What it does**: Tests neural communication protocols

```bash
python test_phase4c_quick.py
```

**Expected output**:
```
✓ 1,599 messages sent
✓ 100% agent participation
✓ Signal Diversity: 0.96
⏱ Runtime: ~2-3 minutes
```

### 3. Full System Benchmark

**What it does**: Compares baseline vs full system

```bash
python benchmark_comparison.py
```

**Expected output**:
```
Performance Improvement: +31.3%
Behavioral Diversity: 266x improvement
⏱ Runtime: ~5-10 minutes
```

---

## Running Experiments

### Basic Experiment (100 steps)

```bash
python long_term_experiment.py --steps 100 --population 50
```

### Medium Experiment (1,000 steps)

```bash
python long_term_experiment.py \
  --steps 1000 \
  --population 100 \
  --checkpoint-interval 100 \
  --log-interval 50
```

### Long-Term Experiment (10,000 steps)

```bash
python long_term_experiment.py \
  --steps 10000 \
  --population 300 \
  --env-size 50 \
  --checkpoint-interval 1000 \
  --log-interval 100 \
  --output-dir results/long_term
```

**Runtime estimate**: ~2-4 hours (unoptimized), ~15-30 minutes (optimized)

### Custom Configuration

```bash
python long_term_experiment.py \
  --steps 5000 \
  --population 200 \
  --env-size 40 \
  --phase4a \          # Enable Phase 4A
  --phase4b \          # Enable Phase 4B
  --phase4c \          # Enable Phase 4C
  --novelty-search \   # Enable Novelty Search
  --map-elites \       # Enable MAP-Elites
  --checkpoint-interval 500
```

### Resume from Checkpoint

```bash
python long_term_experiment.py \
  --resume results/long_term/20260104_120000/checkpoints/checkpoint_005000.pkl \
  --steps 10000
```

**Note**: Resume functionality is a placeholder, full implementation pending.

---

## Performance Optimization

### Quick Test: Baseline vs Optimized

```bash
# Run full benchmark (5 configurations)
python benchmark_optimizations.py
```

**This will test**:
1. Baseline (original implementation)
2. Batch processing only
3. Cached coherence only
4. Sparse MAP-Elites only
5. All optimizations combined

**Expected results**: 9-18x speedup with all optimizations

### Using Optimized System in Your Code

```python
from optimized_phase4c import create_optimized_phase4c_system

# Create optimized system
manager = create_optimized_phase4c_system(
    env_size=30,
    initial_population=100,
    use_batch_processing=True,      # 2-3x speedup
    use_cached_coherence=True,      # 1.5-2x speedup
    use_sparse_map_elites=True,     # 2-3x speedup
    device='cpu'  # or 'cuda' if GPU available
)

# Run experiment
for step in range(1000):
    stats = manager.step()

    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}: Coherence = {stats['avg_coherence']:.3f}")

# Get optimization report
print(manager.get_optimization_report())
```

### GPU Acceleration (If Available)

```python
manager = create_optimized_phase4c_system(
    env_size=50,
    initial_population=500,
    device='cuda'  # Use GPU
)
```

**Expected speedup with GPU**: 20-50x for neural network operations

---

## Visualization

### Generate All Plots for an Experiment

```python
from visualization_tools import ExperimentVisualizer, load_experiment_data

# Load experiment data
data = load_experiment_data('results/long_term/20260104_120000')
stats_history = data['statistics']

# Create visualizer
viz = ExperimentVisualizer()

# Generate all plots
viz.plot_learning_curves(stats_history, save_path='learning_curves.png')
viz.plot_phase_comparison(stats_history, save_path='phase_comparison.png')

# If you have a manager with MAP-Elites
viz.plot_map_elites_heatmap(manager, save_path='map_elites.png')

# Save all figures
viz.save_all_figures('results/figures/')
```

### Quick Visualization from Command Line

```bash
# Run experiment with built-in visualization
python long_term_experiment.py --steps 1000 --population 100

# Then visualize results
python -c "
from visualization_tools import ExperimentVisualizer, load_experiment_data
import glob

# Find most recent experiment
results_dir = sorted(glob.glob('results/long_term/20*'))[-1]
data = load_experiment_data(results_dir)

viz = ExperimentVisualizer()
viz.plot_learning_curves(data['statistics'], save_path='latest_results.png')
print(f'Saved: latest_results.png')
"
```

---

## Common Workflows

### Workflow 1: Quick Exploration

**Goal**: Quickly test if the system works

```bash
# 1. Test individual phases (5 min)
python test_phase4b_quick.py
python test_phase4c_quick.py

# 2. Run short experiment (5 min)
python long_term_experiment.py --steps 100 --population 50

# 3. Check results
ls results/long_term/*/logs/statistics.json
```

### Workflow 2: Performance Benchmarking

**Goal**: Measure optimization speedup

```bash
# 1. Run comprehensive benchmark (30 min)
python benchmark_optimizations.py > benchmark_results.txt

# 2. Review results
cat benchmark_results.txt | grep "Speedup:"

# 3. Generate comparison plots
# (visualization included in benchmark script)
```

### Workflow 3: Long-Term Research Experiment

**Goal**: Run extended experiment with analysis

```bash
# 1. Start long experiment (4 hours)
python long_term_experiment.py \
  --steps 10000 \
  --population 300 \
  --checkpoint-interval 1000 \
  --output-dir results/research_run_001

# 2. Monitor progress (in another terminal)
tail -f results/research_run_001/logs/statistics.json

# 3. After completion, analyze
python -c "
from visualization_tools import ExperimentVisualizer, load_experiment_data

data = load_experiment_data('results/research_run_001/TIMESTAMP')
viz = ExperimentVisualizer()
viz.plot_learning_curves(data['statistics'])
viz.plot_phase_comparison(data['statistics'])
viz.save_all_figures('results/research_run_001/figures/')
"

# 4. Review final analysis
cat results/research_run_001/analysis/final_analysis.txt
```

### Workflow 4: Comparing Configurations

**Goal**: Test different system configurations

```bash
# Configuration A: All phases enabled
python long_term_experiment.py \
  --steps 1000 --population 100 \
  --phase4a --phase4b --phase4c \
  --output-dir results/config_A

# Configuration B: Phase 4B only
python long_term_experiment.py \
  --steps 1000 --population 100 \
  --phase4b \
  --output-dir results/config_B

# Configuration C: Phase 4C only
python long_term_experiment.py \
  --steps 1000 --population 100 \
  --phase4c \
  --output-dir results/config_C

# Compare results
python -c "
import json
from pathlib import Path

configs = ['config_A', 'config_B', 'config_C']
for config in configs:
    stats_file = Path(f'results/{config}').glob('*/logs/statistics.json').__next__()
    with open(stats_file) as f:
        stats = json.load(f)
    final_coherence = stats[-1]['avg_coherence']
    print(f'{config}: Final Coherence = {final_coherence:.3f}')
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'phase4c_integration'"

**Solution**: Make sure you're in the correct directory

```bash
cd /Users/say/Documents/GitHub/ai/08_GENESIS/experiments/path_b_phase1
python test_phase4c_quick.py
```

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: Use CPU or reduce population size

```python
# Option 1: Use CPU
manager = create_optimized_phase4c_system(device='cpu')

# Option 2: Reduce population
manager = create_optimized_phase4c_system(
    initial_population=100,  # Instead of 500
    device='cuda'
)
```

### Issue: Experiment runs too slowly

**Solution**: Use optimized implementation

```bash
# Instead of:
python test_phase4c_quick.py

# Use:
python benchmark_optimizations.py  # This uses optimized version
```

Or in code:
```python
from optimized_phase4c import create_optimized_phase4c_system  # Not phase4c_integration

manager = create_optimized_phase4c_system(
    use_batch_processing=True,
    use_cached_coherence=True,
    use_sparse_map_elites=True
)
```

### Issue: Checkpoint file not found when resuming

**Solution**: Check the exact path

```bash
# List available checkpoints
ls results/long_term/*/checkpoints/

# Use full path
python long_term_experiment.py \
  --resume results/long_term/20260104_120000/checkpoints/checkpoint_005000.pkl
```

### Issue: Plots not showing up

**Solution**: Make sure matplotlib backend is configured

```python
import matplotlib
matplotlib.use('Agg')  # For saving to file
# or
matplotlib.use('TkAgg')  # For interactive display
```

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch

```bash
# CPU version
pip install torch

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Tips

### 1. Start Small, Scale Up

```bash
# First test with small configuration
python long_term_experiment.py --steps 100 --population 50

# If that works, scale up
python long_term_experiment.py --steps 1000 --population 100

# Then go for long-term
python long_term_experiment.py --steps 10000 --population 300
```

### 2. Use Checkpoints

```bash
# Save checkpoints frequently for long experiments
python long_term_experiment.py \
  --steps 10000 \
  --checkpoint-interval 500  # Save every 500 steps
```

This allows you to:
- Resume if experiment crashes
- Analyze intermediate results
- Stop and continue later

### 3. Monitor Resource Usage

```bash
# In another terminal
watch -n 5 'ps aux | grep python'  # Linux/Mac

# Or use htop
htop
```

### 4. Optimize Before Long Runs

```bash
# First, verify optimizations work
python benchmark_optimizations.py

# Then use optimized version for long experiments
# (Modify long_term_experiment.py to use create_optimized_phase4c_system)
```

---

## Next Steps

### For Research

1. **Read the documentation**:
   - `FINAL_SYSTEM_REPORT.md` - Comprehensive system description
   - `OPTIMIZATION_GUIDE.md` - Performance analysis
   - `FINAL_EVALUATION.md` - Overall project evaluation

2. **Run experiments**:
   - Start with quick tests
   - Scale up to long-term experiments
   - Compare different configurations

3. **Analyze results**:
   - Use visualization tools
   - Review checkpoint data
   - Study behavioral diversity

### For Development

1. **Understand the architecture**:
   - Read source files with comments
   - Trace through one step execution
   - Understand phase integration

2. **Extend the system**:
   - Add new behavioral descriptors
   - Implement additional communication protocols
   - Create custom fitness functions

3. **Optimize further**:
   - Implement spatial indexing (5-10x speedup)
   - Add GPU acceleration (20-50x speedup)
   - Parallelize population processing (2-4x speedup)

### For Application

1. **Adapt to your domain**:
   - Replace environment with your task
   - Define appropriate behavioral descriptors
   - Customize agent architecture

2. **Tune hyperparameters**:
   - Population size
   - Environment complexity
   - Communication parameters

3. **Validate**:
   - Run multiple trials
   - Statistical analysis
   - Compare with baselines

---

## Additional Resources

- **Main README**: `/Users/say/Documents/GitHub/ai/README.md`
- **GENESIS README**: `/Users/say/Documents/GitHub/ai/08_GENESIS/README.md`
- **Research Achievements**: `/Users/say/Documents/GitHub/ai/RESEARCH_ACHIEVEMENTS.md`
- **Optimization Guide**: `OPTIMIZATION_GUIDE.md`
- **Final Evaluation**: `/Users/say/Documents/GitHub/ai/FINAL_EVALUATION.md`

---

## Support

For issues, questions, or contributions:
- Check `TROUBLESHOOTING` section above
- Review documentation files
- Examine example code in test files

---

**Last Updated**: 2026-01-04
**Version**: Phase 4 Complete with Optimizations

Happy Experimenting! 🚀
