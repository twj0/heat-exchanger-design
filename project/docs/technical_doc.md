# Technical Documentation

## System Overview

This project implements a thermal energy storage (TES) system with heat exchanger, optimized for operation under time-of-use (TOU) electricity pricing using deep reinforcement learning.

## Physical Models

### 1. Thermal Energy Storage (TES)

#### Sensible Heat Storage

The sensible heat storage model uses the energy balance equation:

```
dE/dt = η_charge * P_in - P_out/η_discharge - Q_loss
```

For sensible heat storage:
```
E = m * c_p * (T - T_ref)
dE/dt = m * c_p * dT/dt
```

Therefore:
```
dT/dt = (η_charge * P_in - P_out/η_discharge - Q_loss) / (m * c_p)
```

**Parameters:**
- `m`: Mass of storage material (kg)
- `c_p`: Specific heat capacity (kJ/kg·K)
- `T`: Storage temperature (°C)
- `η_charge`: Charging efficiency (typically 0.98)
- `η_discharge`: Discharging efficiency (typically 0.95)
- `P_in`: Charging power (kW)
- `P_out`: Discharging power (kW)
- `Q_loss`: Heat losses (kW)

**Heat Loss Model:**
```
Q_loss = U_loss * A * (T_storage - T_ambient)
```

**State of Charge (SoC):**
```
SoC = (T - T_min) / (T_max - T_min)
```

#### Phase Change Material (PCM) Storage

PCM storage uses a three-zone model:

1. **Solid phase** (T < T_melt): `E = m * c_p,solid * T`
2. **Phase change** (T = T_melt): `E = m * c_p,solid * T_melt + m * h_latent * f_liquid`
3. **Liquid phase** (T > T_melt): `E = m * (c_p,solid * T_melt + h_latent + c_p,liquid * (T - T_melt))`

**Parameters:**
- `c_p,solid`: Specific heat in solid phase (kJ/kg·K)
- `c_p,liquid`: Specific heat in liquid phase (kJ/kg·K)
- `h_latent`: Latent heat of fusion (kJ/kg)
- `T_melt`: Melting point temperature (°C)
- `f_liquid`: Liquid fraction (0-1)

### 2. Heat Exchanger

#### Effectiveness-NTU Method

The effectiveness-NTU method is used for varying flow conditions:

**Heat Capacity Rates:**
```
C_hot = m_hot * c_p,hot
C_cold = m_cold * c_p,cold
C_min = min(C_hot, C_cold)
C_max = max(C_hot, C_cold)
C_ratio = C_min / C_max
```

**Number of Transfer Units (NTU):**
```
NTU = U * A / C_min
```

**Effectiveness (for counterflow):**
```
ε = (1 - exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * exp(-NTU * (1 - C_ratio)))
```

Special case when C_hot ≈ C_cold:
```
ε = NTU / (1 + NTU)
```

**Heat Transfer:**
```
Q = ε * C_min * (T_hot,in - T_cold,in)
```

**Outlet Temperatures:**
```
T_hot,out = T_hot,in - Q / C_hot
T_cold,out = T_cold,in + Q / C_cold
```

#### LMTD Method

The Log Mean Temperature Difference method uses:

```
Q = U * A * LMTD
```

For counterflow:
```
ΔT_1 = T_hot,in - T_cold,out
ΔT_2 = T_hot,out - T_cold,in
LMTD = (ΔT_1 - ΔT_2) / ln(ΔT_1 / ΔT_2)
```

### 3. Economic Model

#### TOU Electricity Pricing

Three pricing periods:
- **Peak**: High demand periods (e.g., 10-12, 18-21)
- **Shoulder**: Moderate demand periods
- **Off-peak**: Low demand periods (e.g., 21-8)

#### Cost Calculation

```
Cost = Price_buy * E_grid^+ - Price_sell * E_grid^- + Price_gas * V_gas
```

Where:
- `E_grid^+`: Electricity purchased from grid (kWh)
- `E_grid^-`: Electricity sold to grid (kWh)
- `V_gas`: Gas consumed (m³)

## Reinforcement Learning Environment

### State Space

The observation vector includes:

1. **Storage temperature** (normalized): `(T - T_min) / (T_max - T_min)`
2. **State of charge** (0-1)
3. **Electricity price forecast** (next N hours, normalized)
4. **Current heat demand** (normalized)
5. **Time features** (cyclical encoding):
   - `hour_sin = sin(2π * hour / 24)`
   - `hour_cos = cos(2π * hour / 24)`
   - `day_sin = sin(2π * day / 365)`
   - `day_cos = cos(2π * day / 365)`

### Action Space

#### Discrete (default):
- Action 0: Idle
- Action 1: Charge at maximum power
- Action 2: Discharge at maximum power

#### Continuous:
- Single continuous value: power ∈ [-P_max, +P_max]
  - Positive: charging
  - Negative: discharging

### Reward Function

```
R = w_cost * Cost + w_temp * Penalty_temp + w_demand * Penalty_demand + w_cycle * Penalty_cycle
```

**Cost component:**
```
w_cost = -1.0 (negative because we minimize cost)
```

**Temperature violation penalty:**
```
Penalty_temp = -λ_temp * |T - T_limit| if T violates limits
```

**Demand violation penalty:**
```
Penalty_demand = -λ_demand * (Demand - Delivered) if demand not met
```

**Cycling penalty (optional):**
```
Penalty_cycle = -λ_cycle * |P_t - P_{t-1}|
```

### Training Algorithms

#### PPO (Proximal Policy Optimization)

**Advantages:**
- Stable training
- Good sample efficiency
- Works well with both discrete and continuous actions

**Key hyperparameters:**
- Learning rate: 3e-4
- Batch size: 64
- n_steps: 2048
- n_epochs: 10

#### SAC (Soft Actor-Critic)

**Advantages:**
- Maximum entropy framework
- Excellent for continuous control
- Off-policy learning

**Key hyperparameters:**
- Learning rate: 3e-4
- Batch size: 64
- Buffer size: 100,000
- τ (soft update): 0.005

## Performance Metrics

### Economic Metrics

1. **Total operational cost** (CNY)
2. **Cost savings** vs baseline (CNY and %)
3. **Peak/off-peak cost breakdown**
4. **Average cost per hour**

### Energy Metrics

1. **Storage efficiency**: `η = E_discharged / E_charged`
2. **Demand satisfaction rate**: `DSR = E_delivered / E_demanded`
3. **Total energy charged/discharged** (kWh)

### Temperature Metrics

1. **Violation count**: Number of steps outside limits
2. **Violation rate**: Percentage of time violated
3. **Mean temperature and standard deviation**
4. **Temperature range**

### Storage Utilization

1. **Mean SoC**
2. **SoC utilization**: `max(SoC) - min(SoC)`
3. **Estimated charge/discharge cycles**

## Baseline Controllers

### Simple TOU Controller

**Strategy:**
- Charge during off-peak hours when SoC < threshold
- Discharge during peak hours when SoC > threshold
- Idle during shoulder hours

**Hysteresis control** prevents rapid switching:
```
if T < T_min + Δhyst: charge
if T > T_max - Δhyst: stop charging
```

### Predictive Controller

Uses price forecast to anticipate peak periods and pre-charge storage.

## Validation and Testing

### Physical Consistency Checks

1. **Energy conservation**: `E(t+1) = E(t) + E_in - E_out - E_loss`
2. **Temperature limits**: `T_min ≤ T ≤ T_max`
3. **Power limits**: `|P| ≤ P_max`
4. **SoC bounds**: `0 ≤ SoC ≤ 1`

### Unit Tests

Located in `tests/`:
- `test_models.py`: Physical model tests
- `test_env.py`: Environment tests (to be added)
- `test_controllers.py`: Controller tests (to be added)

## Configuration

All parameters are configurable via YAML files in `configs/`:

```yaml
simulation:
  timestep: 3600  # seconds
  duration: 8760  # hours
  
tes:
  type: "sensible"
  mass: 5000  # kg
  specific_heat: 4.18  # kJ/(kg·K)
  
tou_pricing:
  peak_price: 1.2  # CNY/kWh
  offpeak_price: 0.3
```

See `configs/default.yaml` for complete configuration.

## Model Serialization Format

This project uses Stable-Baselines3 (SB3) model serialization with the `.zip` format.

### Files and Paths

- Best checkpoint during training (saved by evaluation callback):
  - `models/{ALGO}_{TIMESTAMP}/best_model.zip`
- Final model after training (explicit save):
  - `models/{ALGO}_{TIMESTAMP}_final.zip`
- Run configuration snapshot (saved alongside best checkpoints):
  - `models/{ALGO}_{TIMESTAMP}/config.yaml`

Where `{ALGO}` ∈ {`PPO`, `SAC`, `DQN`} and `{TIMESTAMP}` is `YYYYMMDD_HHMMSS`.

### Contents of `.zip`

- PyTorch policy parameters (weights)
- SB3 pickled metadata (hyperparameters, normalization stats if used, replay buffer for off-policy when applicable)

Note: Exact internal filenames and metadata are managed by SB3 and may vary by SB3 version. We do not modify SB3’s internal layout. The training run’s YAML config is saved separately as `config.yaml` in the run folder.

### Metadata and Versioning

- Model version: `1.0`
- Algorithm type: encoded in the file path/name (e.g., `PPO_20250101_123000_final.zip`) and can be specified explicitly via CLI during evaluation.
- Environment/config snapshot: `config.yaml` next to `best_model.zip` for reproducibility.

### Loading Behavior

- Preferred: specify algorithm explicitly when evaluating/using a model:
  - `--algo {PPO|SAC|DQN}`
- Fallback: if `--algo` is not provided, code attempts to infer from the model path; if not inferable, defaults to PPO with a warning.

### Custom Save Directory

- Use `--save-path` in the training CLI to choose the base directory (default: `models`).

## References

1. Buscemi et al. (2025). "Deep reinforcement learning-based control of thermal energy storage for university classrooms"
2. Zhang et al. (2024). "Optimal scheduling strategy of electricity and thermal energy storage based on SAC approach"
3. Zou et al. (2023). "Recent advances in the applications of machine learning methods for heat exchanger modeling"
4. Incropera & DeWitt. "Fundamentals of Heat and Mass Transfer"
