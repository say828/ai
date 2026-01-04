# GENESIS Path B Phase 0 - Results Summary

Generated: 2026-01-04T03:29:48.942053

## Success Criteria

- [PASS] runs_without_error
- [PASS] coherence_survival_correlation
- [PASS] population_evolves
- [PASS] runtime_acceptable

**Overall: SUCCESS**

## Key Findings

- Coherence affects survival: gap=-0.008, age_correlation=0.265
- Death causes: low_energy=539, low_coherence=0
- Population evolved: 523 births, 539 deaths
- Runtime: 4.72 seconds
- Population trend: decreasing (42 -> 4)
- Coherence trend: stable (0.804 -> 0.827)

## Configuration

- n_steps: 1000
- initial_agents: 20
- grid_size: 16
- seed: 42

## Coherence-Survival Analysis

- dead_avg_coherence: 0.8477581254420561
- dead_avg_age: 122.07235621521336
- live_avg_coherence: 0.8394789489360728
- live_avg_age: 162.5
- coherence_gap: -0.00827917650598331
- coherence_age_correlation: 0.26547275248177915
- deaths_by_low_energy: 539
- deaths_by_low_coherence: 0
- low_energy_avg_coherence: 0.8477581254420561
- low_coherence_avg_coherence: 0
- sample_size_dead: 539
- sample_size_live: 4
