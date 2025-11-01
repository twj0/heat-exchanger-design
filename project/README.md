# Heat Exchanger and Thermal Energy Storage Optimization

A Python-based simulation and optimization system for heat exchangers integrated with thermal energy storage (TES) that operates under time-of-use (TOU) electricity pricing, using deep reinforcement learning.

## Project Overview

This project aims to optimize energy costs while maintaining thermal comfort and meeting heating demands through intelligent control of thermal energy storage systems. The system uses reinforcement learning algorithms (PPO/SAC) to learn optimal charging and discharging strategies based on electricity pricing patterns.

## Features

- **Physical Modeling**: Accurate thermal dynamics models for sensible and latent heat storage
- **Heat Exchanger Models**: Implementation of Effectiveness-NTU and LMTD methods
- **Economic Optimization**: TOU pricing integration with cost minimization
- **RL-based Control**: Advanced control using PPO and SAC algorithms
- **Baseline Comparison**: Rule-based controllers for performance benchmarking
- **Comprehensive Evaluation**: Multiple scenarios and performance metrics

## Project Structure

```
project/
├── env/                    # RL Environment
│   ├── __init__.py
│   ├── tes_heatex_env.py   # Main Gym environment
│   └── utils.py            # Helper functions
├── models/                 # Physical models
│   ├── __init__.py
│   ├── thermal_storage.py  # TES model
│   ├── heat_exchanger.py   # Heat exchanger model
│   └── economic_model.py   # Economic calculations
├── baselines/              # Baseline controllers
│   ├── __init__.py
│   └── rule_based.py       # Rule-based controller
├── rl_algorithms/          # RL training
│   ├── __init__.py
│   └── train.py            # Training script
├── simulate/               # Evaluation
│   ├── __init__.py
│   └── run_eval.py         # Evaluation script
├── metrics/                # Performance metrics
│   ├── __init__.py
│   └── calculator.py       # Metrics calculation
├── data/                   # Data files
│   └── tou_prices.csv
├── configs/                # Configuration files
│   └── default.yaml
├── tests/                  # Unit tests
│   └── test_models.py
├── docs/                   # Documentation
│   ├── technical_doc.md
│   └── experiment_results.md
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a baseline controller

```bash
python baselines/rule_based.py --config configs/default.yaml
```

### 2. Train an RL agent

```bash
# Train with PPO (discrete actions)
python rl_algorithms/train.py \
    --config configs/default.yaml \
    --algo PPO \
    --timesteps 200000 \
    --save-path my_models

# Train with SAC (continuous actions)
# Use the continuous-action config for SAC
python rl_algorithms/train.py \
    --config configs/sac_continuous.yaml \
    --algo SAC \
    --timesteps 200000 \
    --save-path my_models

# Run from any directory (path resolution handled automatically)
python D:/学习/化能任选/换热器设计/workspace/project/rl_algorithms/train.py \
    --config D:/学习/化能任选/换热器设计/workspace/project/configs/sac_continuous.yaml \
    --algo SAC \
    --timesteps 200000 \
    --save-path D:/tmp/my_models
```

### 3. Evaluate and compare

```bash
# Evaluate an RL model (explicitly specify algorithm for robust loading)
python rl_algorithms/train.py --eval \
    --config configs/sac_continuous.yaml \
    --algo SAC \
    --model-path my_models/SAC_YYYYMMDD_HHMMSS_final.zip

# Unified comparison with baseline
python simulate/run_eval.py --config configs/sac_continuous.yaml \
    --baseline simple_tou \
    --rl-model my_models/SAC_YYYYMMDD_HHMMSS_final.zip \
    --algo SAC \
    --episodes 10 \
    --output results
```

## Configuration

All system parameters can be adjusted in `configs/default.yaml`:

- **TES parameters**: Storage capacity, temperature limits, heat losses
- **Heat exchanger**: Type, effectiveness, heat transfer coefficient
- **TOU pricing**: Peak/shoulder/off-peak prices and time periods
- **RL training**: Algorithm, learning rate, batch size, etc.
- **Evaluation**: Scenarios, metrics to calculate

## Key Components

### Thermal Energy Storage Model

Implements both sensible and latent (phase change) heat storage:

```python
# Sensible heat
dT/dt = (P_charge * η - Q_discharge - Q_losses) / (m * c_p)

# Energy balance
E(t+1) = E(t) + η_charge * P_in * Δt - P_out/η_discharge * Δt - Q_loss * Δt
```

### Heat Exchanger Model

Supports multiple methods:
- **Effectiveness-NTU**: For varying flow conditions
- **LMTD**: For steady-state analysis

### Economic Model

```python
Cost = Price_buy * P_grid⁺ * Δt - Price_sell * P_grid⁻ * Δt
```

### RL Environment

- **State space**: Temperature, SoC, electricity price forecast, demand, time features
- **Action space**: Charge/discharge/idle (discrete) or continuous power
- **Reward**: Negative cost with penalties for constraint violations

## Performance Metrics

- **Cost savings**: Compared to baseline controller
- **Energy efficiency**: Storage utilization and losses
- **Temperature violations**: Percentage of time outside safe range
- **Demand satisfaction**: Load matching performance
- **Peak demand reduction**: Contribution to grid stability

## Example Results

After training, you can expect:
- 10-20% cost reduction compared to rule-based baseline
- >95% demand satisfaction rate
- <1% temperature violation time
- Effective peak shifting behavior

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Documentation

- `docs/technical_doc.md`: Detailed model descriptions and equations
- `docs/experiment_results.md`: Experimental setup and results analysis

## References

1. "Deep reinforcement learning-based control of thermal energy storage for university classrooms: Co-Simulation with TRNSYS-Python and transfer learning across operational scenarios"
2. "Optimal scheduling strategy of electricity and thermal energy storage based on SAC approach"
3. "Recent advances in the applications of machine learning methods for heat exchanger modeling"

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository.
