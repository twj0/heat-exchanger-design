# Experiment Report: Carbon-Aware Building Control

Generated: 2025-11-30 13:25:18

## Executive Summary

This experiment evaluates the Carbon-Aware Building Control system using 
Deep Reinforcement Learning (SAC) with Transformer-based feature extraction.

## Key Findings

### Performance Comparison

| Controller | Cost (CNY) | Carbon (kgCO2) | Violation Rate |
|------------|-----------|----------------|----------------|
| Transformer-SAC | 4650.31 | 4779.99 | 18.1% |
| MLP-SAC | 4635.08 | 4753.43 | 19.2% |
| Rule-Based | 4677.59 | 4784.82 | 18.5% |
| Carbon-Aware-Rule | 4648.31 | 4772.21 | 17.9% |
| Fixed-Setpoint | 4624.73 | 4758.68 | 18.4% |

### Improvement vs Rule-Based Baseline

- **Cost Reduction**: 0.6%
- **Carbon Reduction**: 0.1%
- **Target**: Cost ≤ 90% of baseline (10% reduction)

⚠️ **Target Not Met** - Consider more training or hyperparameter tuning

## Methodology

### Environment
- Building Model: 5-Zone University Classroom (EnergyPlus)
- Location: Shanghai, China
- Weather: 2024 TMY Data
- Timestep: 15 minutes

### RL Configuration
- Algorithm: Soft Actor-Critic (SAC)
- Feature Extractor: Transformer (Innovation C)
- Reward Function: Cost + λ_carbon × Carbon + λ_comfort × Comfort Violation (Innovation B)

### Baselines
1. **Rule-Based**: Time-of-use setpoint scheduling
2. **Carbon-Aware-Rule**: Rule-based with carbon intensity awareness
3. **MLP-SAC**: SAC with MLP feature extractor (ablation)
4. **Fixed-Setpoint**: Constant thermostat settings

## Files Generated

- `evaluation/`: Detailed evaluation results
- `figures/`: Visualization plots
- `experiment_config.json`: Experiment configuration

## Next Steps

1. Run full training (500k+ timesteps) if quick mode was used
2. Add TES (Thermal Energy Storage) component
3. Test on different scenarios (summer/winter peaks)
4. Prepare manuscript with final results
