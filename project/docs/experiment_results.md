# Experiment Results Template

This document provides a template for recording experimental results.

## Experiment Configuration

**Date:** YYYY-MM-DD  
**Experiment ID:** experiment_001  
**Configuration File:** `configs/default.yaml`  
**Random Seed:** 42

### System Parameters

- **Storage Type:** Sensible heat
- **Storage Mass:** 5000 kg
- **Temperature Range:** 40-50°C
- **Heat Exchanger Area:** 50 m²
- **Maximum Heater Power:** 100 kW

### TOU Pricing

- **Peak Price:** 1.2 CNY/kWh (10-12, 18-21)
- **Shoulder Price:** 0.7 CNY/kWh (8-10, 12-18)
- **Off-peak Price:** 0.3 CNY/kWh (21-8)

### Training Configuration

- **Algorithm:** PPO / SAC / DQN
- **Total Timesteps:** 200,000
- **Learning Rate:** 0.0003
- **Batch Size:** 64
- **Evaluation Frequency:** 10,000 steps

## Training Results

### Learning Curves

*Insert training reward curve plot here*

**Key Observations:**
- Convergence achieved after ~150,000 timesteps
- Final mean reward: -XXX
- Training stability: Good/Moderate/Poor

### Baseline Performance

**Simple TOU Controller:**
- Total Cost: XXX CNY
- Violation Rate: X.XX%
- Demand Satisfaction: XX.X%
- Storage Utilization: X.XX

## Evaluation Results

### Performance Comparison

| Metric | Baseline | PPO | SAC | Best |
|--------|----------|-----|-----|------|
| Total Cost (CNY) | XXX | XXX | XXX | SAC |
| Cost Savings (%) | 0.0 | XX.X | XX.X | SAC |
| Violation Rate (%) | X.XX | X.XX | X.XX | PPO |
| Demand Satisfaction (%) | XX.X | XX.X | XX.X | Tie |
| Storage Utilization | X.XX | X.XX | X.XX | SAC |

### Detailed Analysis

#### Cost Breakdown

**Baseline Controller:**
- Peak Period Cost: XXX CNY (XX%)
- Shoulder Period Cost: XXX CNY (XX%)
- Off-peak Period Cost: XXX CNY (XX%)
- **Total:** XXX CNY

**RL Controller (Best):**
- Peak Period Cost: XXX CNY (XX%)
- Shoulder Period Cost: XXX CNY (XX%)
- Off-peak Period Cost: XXX CNY (XX%)
- **Total:** XXX CNY
- **Savings:** XXX CNY (XX.X%)

#### Temperature Performance

*Insert temperature trajectory plot*

**Observations:**
- Baseline: More conservative, stays near mid-range
- RL: More aggressive utilization, closer to limits
- Both: Violations < 1%

#### Energy Management

*Insert SoC trajectory plot*

**Charging/Discharging Patterns:**
- Baseline: Simple peak-valley strategy
- RL: Adaptive strategy based on price forecast
- RL shows better anticipation of demand peaks

## Scenario Analysis

### Scenario 1: Baseline Operation

Standard operation with normal demand and pricing.

**Results:**
- Cost: XXX CNY
- Performance: As expected

### Scenario 2: High Demand (+20%)

Increased heat demand to test system response.

**Results:**
- Baseline Cost: XXX CNY (+XX%)
- RL Cost: XXX CNY (+XX%)
- RL still maintains XX% savings

### Scenario 3: Price Spike (+50% peak)

Elevated peak prices to test cost optimization.

**Results:**
- Baseline Cost: XXX CNY (+XX%)
- RL Cost: XXX CNY (+XX%)
- RL savings increase to XX%

## Key Findings

1. **Cost Optimization:** RL achieves XX-XX% cost reduction compared to baseline
2. **Constraint Satisfaction:** Both controllers maintain >99% compliance with temperature limits
3. **Demand Meeting:** Both achieve >95% demand satisfaction
4. **Adaptability:** RL shows better performance under varied scenarios

## Ablation Studies

### Reward Function Components

Testing importance of different reward components:

| Configuration | Cost Weight | Temp Penalty | Demand Penalty | Total Cost | Violations |
|---------------|-------------|--------------|----------------|------------|------------|
| Config 1 | -1.0 | 10.0 | 20.0 | XXX | X.XX% |
| Config 2 | -1.0 | 20.0 | 20.0 | XXX | X.XX% |
| Config 3 | -1.0 | 10.0 | 40.0 | XXX | X.XX% |

**Best Configuration:** Config X with balanced penalties

### Observation Space

Testing impact of observation components:

| Features | Cost | Training Time |
|----------|------|---------------|
| Full (Temp + SoC + Price + Demand + Time) | XXX | Baseline |
| Without price forecast | XXX (+XX%) | -20% |
| Without time features | XXX (+XX%) | -10% |

**Conclusion:** Price forecast most critical for performance

## Computational Performance

- **Training Time:** XX hours on [hardware specs]
- **Inference Time:** <1ms per action
- **Memory Usage:** ~XXX MB
- **Suitable for:** Real-time control

## Limitations and Future Work

### Current Limitations

1. **Simplified heat exchanger model** - Could be more detailed
2. **No weather uncertainty** - Deterministic weather data
3. **Single storage unit** - No multi-tank optimization
4. **No equipment degradation** - Assumes perfect components

### Future Improvements

1. **Transfer learning** - Train once, deploy to different sites
2. **Multi-agent RL** - Coordinate multiple TES units
3. **Robust RL** - Handle sensor noise and failures
4. **Model-based RL** - Combine physics models with learning

## Conclusions

This experiment demonstrates that:

- **RL-based control significantly outperforms rule-based baselines** (XX% cost savings)
- **Physical constraints are reliably maintained** (<1% violations)
- **The approach is computationally feasible** for real-time deployment
- **Further optimization is possible** through hyperparameter tuning

## Appendix

### Hardware Configuration

- CPU: [specs]
- GPU: [specs]
- RAM: [specs]
- OS: [specs]

### Software Versions

- Python: 3.10
- PyTorch: 2.0.0
- Gymnasium: 0.29.0
- Stable-Baselines3: 2.0.0

### Reproducibility

All experiments can be reproduced using:
```bash
python rl_algorithms/train.py --config configs/default.yaml --seed 42
```

Configuration file and trained models available at: [location]
