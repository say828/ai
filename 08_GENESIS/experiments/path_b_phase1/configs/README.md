# Experiment Configuration Templates

This directory contains pre-configured experiment templates for common use cases.

## Available Configurations

### 1. `quick_test.json`
**Purpose**: Rapid validation and testing
**Duration**: 5-10 minutes
**Use for**: Verifying system works, testing changes, debugging

**Specifications**:
- Steps: 100
- Population: 50
- Environment: 30x30
- Optimizations: Disabled (for baseline testing)

### 2. `medium_experiment.json`
**Purpose**: Development and medium-scale testing
**Duration**: 30-60 minutes (optimized)
**Use for**: Feature development, parameter tuning, preliminary research

**Specifications**:
- Steps: 1,000
- Population: 100
- Environment: 40x40
- Optimizations: Enabled (all quick-wins)

### 3. `long_term.json`
**Purpose**: Research-grade long-term experiments
**Duration**: 30-60 minutes (optimized), 4-8 hours (unoptimized)
**Use for**: Publication-quality results, deep analysis, final validation

**Specifications**:
- Steps: 10,000
- Population: 300
- Environment: 50x50
- Optimizations: Enabled (all quick-wins)

## Usage

### Method 1: Load in Python

```python
from experiment_utils import ExperimentConfig
import json

# Load configuration
with open('configs/medium_experiment.json', 'r') as f:
    config_dict = json.load(f)

# Use in your code
steps = config_dict['experiment']['steps']
population = config_dict['environment']['initial_population']

# Or use ExperimentConfig class
config = ExperimentConfig('medium')
config.set(**config_dict['experiment'])
```

### Method 2: Modify for Your Needs

1. Copy a template:
   ```bash
   cp configs/medium_experiment.json configs/my_experiment.json
   ```

2. Edit the file:
   ```json
   {
     "name": "my_custom_experiment",
     "experiment": {
       "steps": 5000,
       ...
     }
   }
   ```

3. Load and use:
   ```python
   with open('configs/my_experiment.json') as f:
       config = json.load(f)
   ```

## Configuration Schema

### Top-level fields

```json
{
  "name": "string",              // Configuration name
  "description": "string",       // What this config is for

  "experiment": {...},           // Experiment parameters
  "environment": {...},          // Environment setup
  "phases": {...},               // Which phases to enable
  "phase4b_options": {...},      // Phase 4B specific options
  "phase4c_options": {...},      // Phase 4C specific options
  "optimization": {...},         // Performance optimizations
  "output": {...}                // Output configuration
}
```

### Experiment parameters

```json
"experiment": {
  "steps": 1000,                    // Total simulation steps
  "checkpoint_interval": 100,       // Save checkpoint every N steps
  "log_interval": 50                // Print log every N steps
}
```

### Environment parameters

```json
"environment": {
  "size": 40,                       // Environment grid size (40x40)
  "initial_population": 100         // Starting number of agents
}
```

### Phase toggles

```json
"phases": {
  "phase4a": true,                  // Advanced Intelligence
  "phase4b": true,                  // Open-Ended Learning
  "phase4c": true                   // Emergent Communication
}
```

### Phase 4B options

```json
"phase4b_options": {
  "novelty_search": true,           // Enable Novelty Search
  "map_elites": true,               // Enable MAP-Elites
  "poet": false                     // Enable POET (expensive!)
}
```

### Phase 4C options

```json
"phase4c_options": {
  "message_dim": 8,                 // Message signal dimension
  "influence_dim": 32,              // Influence vector dimension
  "local_radius": 5.0               // Communication radius
}
```

### Optimization options

```json
"optimization": {
  "use_batch_processing": true,       // Batch neural networks (2-3x)
  "use_cached_coherence": true,       // Cache coherence (1.5-2x)
  "use_sparse_map_elites": true,      // Sparse updates (2-3x)
  "device": "cpu"                     // "cpu" or "cuda"
}
```

### Output configuration

```json
"output": {
  "output_dir": "results/medium_experiments"
}
```

## Creating Custom Configurations

### Example: High-Performance Configuration

```json
{
  "name": "high_performance",
  "description": "Maximum performance with GPU",

  "experiment": {
    "steps": 50000,
    "checkpoint_interval": 5000,
    "log_interval": 500
  },

  "environment": {
    "size": 60,
    "initial_population": 500
  },

  "phases": {
    "phase4a": true,
    "phase4b": true,
    "phase4c": true
  },

  "phase4b_options": {
    "novelty_search": true,
    "map_elites": true,
    "poet": false
  },

  "phase4c_options": {
    "message_dim": 8,
    "influence_dim": 32,
    "local_radius": 5.0
  },

  "optimization": {
    "use_batch_processing": true,
    "use_cached_coherence": true,
    "use_sparse_map_elites": true,
    "device": "cuda"
  },

  "output": {
    "output_dir": "results/high_performance"
  }
}
```

### Example: Minimal Configuration (Phase 4B only)

```json
{
  "name": "phase4b_only",
  "description": "Test Phase 4B in isolation",

  "experiment": {
    "steps": 1000,
    "checkpoint_interval": 100,
    "log_interval": 50
  },

  "environment": {
    "size": 30,
    "initial_population": 100
  },

  "phases": {
    "phase4a": false,
    "phase4b": true,
    "phase4c": false
  },

  "phase4b_options": {
    "novelty_search": true,
    "map_elites": true,
    "poet": false
  },

  "optimization": {
    "use_batch_processing": false,
    "use_cached_coherence": false,
    "use_sparse_map_elites": true,
    "device": "cpu"
  },

  "output": {
    "output_dir": "results/phase4b_only"
  }
}
```

## Tips

### Performance
- Enable all optimizations for long experiments
- Use GPU ("cuda") if available for 20-50x speedup on neural networks
- Larger batch sizes benefit more from GPU acceleration

### Debugging
- Disable optimizations for debugging (easier to trace)
- Use quick_test configuration to verify changes quickly
- Enable checkpointing frequently during development

### Research
- Use consistent configurations across comparison experiments
- Document all configuration changes
- Save configurations with experiment results

## Validation

You can validate your configuration before running:

```python
import json

# Load config
with open('configs/my_experiment.json') as f:
    config = json.load(f)

# Check required fields
required = ['name', 'experiment', 'environment']
for field in required:
    assert field in config, f"Missing required field: {field}"

# Validate ranges
assert config['experiment']['steps'] > 0
assert config['environment']['initial_population'] > 0
assert config['environment']['size'] > 0

print("✓ Configuration valid")
```

## Best Practices

1. **Start Small**: Begin with `quick_test.json`, then scale up
2. **Document**: Add clear descriptions to custom configurations
3. **Version Control**: Keep configurations in git
4. **Consistency**: Use same config for comparable experiments
5. **Backups**: Save configurations with experiment results

## See Also

- [QUICKSTART_GUIDE.md](../QUICKSTART_GUIDE.md) - How to run experiments
- [experiment_utils.py](../experiment_utils.py) - Config loading utilities
- [INDEX.md](../INDEX.md) - Master navigation guide
