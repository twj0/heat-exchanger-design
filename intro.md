# 换热器课程设计项目深度解析

如果说 README 是对入门新手的说明书，那么这份文档就是为希望深入了解深度学习设计的教科书。本文将详细讲解我们的代码设计、对研究论文的理解，以及真实物理世界的建模。

## 1. 项目概述与研究背景

本项目基于 Buscemi 等人发表在《Energy Reports》期刊上的论文《Deep reinforcement learning-based control of thermal energy storage for university classrooms: Co-Simulation with TRNSYS-Python and transfer learning across operational scenarios》进行开发。该论文提出了一个创新框架，通过将 TRNSYS 与 Python 进行协同仿真，利用深度强化学习（特别是 Soft Actor-Critic 算法）优化空调系统中热能存储设备的运行策略。

在我们的实现中，我们将这一技术应用到中国的电力系统环境中，特别是针对峰谷电价政策进行了优化调整。通过智能调度算法，我们旨在实现能源利用效率的最大化和运行成本的最小化。

## 2. 真实物理世界的建模

### 2.1 热储能系统 (TES) 建模

热储能系统是本项目的核心组件之一，我们实现了两种类型的储能模型：

#### 2.1.1 显热储能模型

显热储能基于能量守恒原理，使用以下能量平衡方程：

```
dE/dt = η_charge * P_in - P_out/η_discharge - Q_loss
```

其中：
- E: 储能系统的内能
- η_charge: 充电效率
- P_in: 输入功率
- P_out: 输出功率
- η_discharge: 放电效率
- Q_loss: 热损失

对于显热储能，内能与温度的关系为：
```
E = m * c_p * (T - T_ref)
```

因此温度变化率为：
```
dT/dt = (η_charge * P_in - P_out/η_discharge - Q_loss) / (m * c_p)
```

热损失通过牛顿冷却定律计算：
```
Q_loss = U_loss * A * (T_storage - T_ambient)
```

状态量(SoC)计算为：
```
SoC = (T - T_min) / (T_max - T_min)
```

#### 2.1.2 相变材料 (PCM) 储能模型

PCM 储能使用三区域模型：

1. **固相区** (T < T_melt): `E = m * c_p,solid * T`
2. **相变区** (T = T_melt): `E = m * c_p,solid * T_melt + m * h_latent * f_liquid`
3. **液相区** (T > T_melt): `E = m * (c_p,solid * T_melt + h_latent + c_p,liquid * (T - T_melt))`

其中：
- c_p,solid: 固相比热容
- c_p,liquid: 液相比热容
- h_latent: 潜热
- T_melt: 熔点温度
- f_liquid: 液相比例

### 2.2 换热器建模

#### 2.2.1 有效度-传热单元数法 (ε-NTU)

对于变流量工况，我们采用 ε-NTU 方法：

热容量率计算：
```
C_hot = m_hot * c_p,hot
C_cold = m_cold * c_p,cold
C_min = min(C_hot, C_cold)
C_max = max(C_hot, C_cold)
C_ratio = C_min / C_max
```

传热单元数：
```
NTU = U * A / C_min
```

对于逆流换热器的有效度：
```
ε = (1 - exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * exp(-NTU * (1 - C_ratio)))
```

热流量：
```
Q = ε * C_min * (T_hot,in - T_cold,in)
```

出口温度：
```
T_hot,out = T_hot,in - Q / C_hot
T_cold,out = T_cold,in + Q / C_cold
```

#### 2.2.2 对数平均温差法 (LMTD)

对数平均温差法使用以下公式：

```
Q = U * A * LMTD
```

对于逆流：
```
ΔT_1 = T_hot,in - T_cold,out
ΔT_2 = T_hot,out - T_cold,in
LMTD = (ΔT_1 - ΔT_2) / ln(ΔT_1 / ΔT_2)
```

### 2.3 经济模型

#### 2.3.1 分时电价 (TOU) 模型

我们实现了三时段电价模型：
- **峰时**: 高需求时段 (如 10-12, 18-21)
- **平时**: 中等需求时段
- **谷时**: 低需求时段 (如 23-7)

#### 2.3.2 成本计算模型

总成本计算公式：
```
Cost = Price_buy * E_grid^+ - Price_sell * E_grid^- + Price_gas * V_gas
```

其中：
- E_grid^+: 从电网购买的电量
- E_grid^-: 向电网出售的电量
- V_gas: 消耗的燃气量

## 3. 代码设计与实现

### 3.1 模块化架构设计

我们的代码采用模块化设计，每个模块负责特定的功能：

```
project/
├── models/                 # 物理模型实现
│   ├── thermal_storage.py   # 热储能模型
│   ├── heat_exchanger.py    # 换热器模型
│   └── economic_model.py    # 经济模型
├── env/                    # 强化学习环境
│   ├── tes_heatex_env.py    # Gym环境实现
│   └── utils.py             # 辅助工具
├── rl_algorithms/          # 强化学习算法
│   └── train.py             # 训练脚本
├── baselines/              # 基线控制器
│   └── rule_based.py        # 规则基线策略
├── simulate/               # 仿真与评估
│   └── run_eval.py          # 评估脚本
├── metrics/                # 指标计算
│   └── calculator.py        # 指标计算器
└── configs/                # 配置文件
    ├── default.yaml         # 默认配置
    └── sac_continuous.yaml  # SAC算法配置
```

### 3.2 核心模型实现详解

#### 3.2.1 热储能模型 (thermal_storage.py)

该模块实现了 SensibleHeatStorage 和 PCMHeatStorage 两个类：

```python
class SensibleHeatStorage:
    def __init__(self, mass, specific_heat, initial_temperature, ...):
        # 初始化储能参数
    
    def update_state(self, charging_power, discharging_power, ...):
        # 更新储能状态
    
    def get_temperature(self):
        # 获取当前温度
    
    def get_soc(self):
        # 获取当前状态量
```

SensibleHeatStorage 的输入参数与物理意义一一对应：

- **`mass`**：储能介质的实际质量，对应实体蓄热罐内水或石蜡的吨位。
- **`specific_heat`**：比热容，直接决定单位温差所能存储的热量，来自材料手册或实验测定。
- **`min_temperature` / `max_temperature`**：工艺允许运行区间，对应设备设计规范中的“防结露温度”和“安全上限温度”。
- **`loss_coefficient`**：通过实际设备的保温层导热系数与外表面积折算得到，约束了 `get_heat_losses()` 中的牛顿冷却模型。
- **`ambient_temperature`**：现场环境温度的小时序列，可与 `weather_data` 生成的气象条件对接。

`step()` 方法中的净功率 `p_net` 通过 `power_in`、`power_out`、热损失三者的平衡刻画了真实蓄热罐的能量守恒；`temperature_violation` 字段则提供了违反硬约束时可在奖励函数中处罚的指标。`get_state_of_charge()` 被环境 (`TESHeatExEnv`) 用于构建观测空间，是强化学习代理判定可用储能的直接信号。

#### 3.2.2 换热器模型 (heat_exchanger.py)

该模块实现了多种换热器模型：

```python
class HeatExchanger:
    def __init__(self, area, heat_transfer_coeff, ...):
        # 初始化换热器参数
    
    def calculate_heat_transfer(self, hot_fluid, cold_fluid):
        # 计算换热器热交换量
```

实际使用中我们重点关注 `EffectivenessNTU` 与 `LMTD` 两个子类：

- `EffectivenessNTU` 通过 `ntu = U·A / C_min` 将设备名义参数 (`heat_transfer_area`, `overall_heat_transfer_coefficient`) 与流体状态 (`mass_flow_hot`, `cp_hot`) 联系起来。输出的 `heat_transfer` 与 `temp_hot_out`、`temp_cold_out` 被用于验证蓄热出力能否满足 `heat_demand`。
- `LMTD` 子类利用迭代求解反推换热器出口温度，适用于需要精确核算换热驱动力的情形，例如后续将项目与 TRNSYS 或 EES 联合仿真时。

`create_heat_exchanger()` 工厂函数读取 `configs/default.yaml` 中 `heat_exchanger` 字段，从而保证配置—模型—仿真的一致性。

#### 3.2.3 经济模型 (economic_model.py)

该模块实现了电价和成本计算：

```python
class EconomicModel:
    def __init__(self, pricing_config):
        # 初始化电价配置
    
    def get_electricity_price(self, time):
        # 获取指定时间的电价
    
    def calculate_operational_cost(self, power_consumption, ...):
        # 计算运行成本
```

`TOUPricing` 会在初始化时生成 24 小时价格查找表 `hourly_prices`，`EconomicModel.calculate_step_cost()` 读取该表并按时间步长 (timestep) 将功率转换成电量。输出的 `net_cost`、`electricity_price` 会被环境写入 `episode_data`，从而支撑后续的指标计算 (`metrics/calculator.py`) 与经济性分析。

为了贴合真实峰谷电价，`configs/default.yaml` 中的 `peak_hours`、`shoulder_hours`、`offpeak_hours` 采取了区间数组的写法，可直接映射到国家电网公布的 TOU 方案。若需要适配其他城市，只需替换该 YAML 字段即可，无需改动 Python 代码。

### 3.3 强化学习环境设计 (tes_heatex_env.py)

我们基于 Gymnasium 框架实现了强化学习环境：

#### 3.3.1 状态空间设计

状态向量包括：
1. 储能温度（归一化）
2. 状态量 (SoC)
3. 未来N小时电价预测（归一化）
4. 当前热需求（归一化）
5. 时间特征（循环编码）

#### 3.3.2 动作空间设计

支持两种动作空间：
- **离散动作**: [空闲, 充电, 放电]
- **连续动作**: 功率 ∈ [-P_max, +P_max]

#### 3.3.3 奖励函数设计

奖励函数综合考虑多个因素：
```
R = w_cost * Cost + w_temp * Penalty_temp + w_demand * Penalty_demand + w_cycle * Penalty_cycle
```

其中 `w_cost` 对应 `configs/default.yaml` → `rl_env.reward.cost_weight`，实际为负值（例如 -1.0），用于将“成本越低奖励越高”的工程目标转化成最大化问题；`temperature_violation_penalty` 与 `demand_violation_penalty` 体现了对供热安全性的重视，其量纲分别是 ¥/°C 与 ¥/kW。通过修改这些权重可以实现不同控制策略之间的折中，例如在供暖季以用户舒适度为先，可适当提高温度惩罚系数。

环境还会记录 `episode_data` 中的 `power_command`，使得 `cycling_penalty` 可以约束频繁启停，从设备寿命的角度体现真实工程需求。

### 3.4 强化学习算法实现

我们实现了三种主流强化学习算法：

#### 3.4.1 PPO (Proximal Policy Optimization)

策略梯度方法，使用裁剪机制确保更新稳定性。

#### 3.4.2 SAC (Soft Actor-Critic)

基于最大熵框架的 Actor-Critic 算法，最大化预期回报与策略熵的加权和。

#### 3.4.3 DQN (Deep Q-Network)

深度 Q 学习算法，使用神经网络近似 Q 函数。

#### 3.4.4 训练与评估脚本的协同

- **训练入口**：`project/main.py` 的 `train` 子命令会加载 YAML 配置并调用 `rl_algorithms/train.py` 的 `train_rl_agent()`。该函数负责：
  - 创建 `TESHeatExEnv` 实例并根据 `training.total_timesteps` 安排采样。
  - 依据 `--algo` 参数构造 Stable-Baselines3 中的 `PPO`, `SAC`, `DQN` 模型实例。
  - 将 `log_dir` 与 `model_dir` 传递给算法以便断点续训。
- **评估流程**：`main.py eval` 会根据 `--baseline` 或 `--model` 选择调用 `baselines/rule_based.py` 或 `rl_algorithms/train.py` 中的 `evaluate_rl_agent()`，并生成 `results/` 下的 CSV、图表，用于和论文中的节能数据对比。
- **自动化演示**：`demo.py` 串联训练、评估、可视化，便于快速复现论文中的性能指标，同时也用于教学展示策略学习的全过程。

该脚本级别的设计保证了“配置即实验”，方便在不同工况、不同电价政策下做批量实验或迁移学习。

### 4. 设计意义与用意

### 4.1 工程实践意义

1. **能源效率优化**: 通过智能调度算法，最大化利用谷电，减少峰电使用，实现削峰填谷。
2. **经济效益提升**: 降低运行成本，提高系统的经济性。
3. **系统稳定性**: 通过储能系统提供热能缓冲，提高供热系统的稳定性。

### 4.2 学术研究价值

1. **跨学科融合**: 将热工学、控制理论、强化学习等多学科知识融合。
2. **技术创新**: 将先进的深度强化学习技术应用于传统热能工程领域。
3. **方法论贡献**: 提供了一套完整的基于数据驱动的热能系统优化方法。

### 4.3 教育价值

1. **理论与实践结合**: 学生可以将课堂学习的热工学理论知识与实际工程问题相结合。
2. **现代技术应用**: 接触并学习当前热门的深度强化学习技术。
3. **系统思维培养**: 通过完整项目的实现，培养学生的系统性思维和工程设计能力。

## 5. 项目参数详解

### 5.1 物理模型参数

#### 5.1.1 热储能系统参数
- `mass`: 储能材料质量 (kg)
- `specific_heat`: 比热容 (kJ/kg·K)
- `initial_temperature`: 初始温度 (°C)
- `min_temperature`/`max_temperature`: 温度约束范围 (°C)
- `charging_efficiency`/`discharging_efficiency`: 充放电效率

#### 5.1.2 换热器参数
- `area`: 换热面积 (m²)
- `heat_transfer_coeff`: 总传热系数 (W/m²·K)
- `flow_type`: 流动类型（逆流、并流等）

#### 5.1.3 经济模型参数
- `peak_price`/`offpeak_price`: 峰谷电价 (元/kWh)
- `time_periods`: 各时段划分

### 5.2 强化学习参数

#### 5.2.1 通用训练参数
- `learning_rate`: 学习率
- `batch_size`: 批大小
- `gamma`: 折扣因子
- `total_timesteps`: 总训练步数

#### 5.2.2 算法特定参数
- PPO: `n_steps`, `n_epochs`, `gae_lambda`, `clip_range`
- SAC: `buffer_size`, `tau`, `ent_coef`
- DQN: `exploration_initial_eps`, `target_update_interval`

### 5.3 参数—物理世界—代码映射表

| 参数块 | YAML 路径 | 典型取值 | 物理/业务含义 | 代码读取位置 |
| --- | --- | --- | --- | --- |
| 仿真时间 | `simulation.timestep` | 3600 s | 单次决策的物理时长，决定能量积分精度 | `TESHeatExEnv.__init__` → `self.timestep` |
| 储能规模 | `tes.mass` | 15,000 kg | 代表水罐或 PCM 模块的实际吨位 | `SensibleHeatStorage.__init__`
| 换热能力 | `heat_exchanger.overall_heat_transfer_coefficient` | 0.5 kW/(m²·K) | 与设备铭牌数据对应，决定 `NTU` | `EffectivenessNTU.calculate_heat_transfer`
| 电价结构 | `tou_pricing.peak_hours` | [[10,12],[18,21]] | 反映地方 TOU 政策 | `TOUPricing._create_hourly_lookup`
| 加热设备 | `electric_heater.max_power` | 120 kW | 对应实际电加热器额定功率 | `TESHeatExEnv._action_to_power`
| 奖励系数 | `rl_env.reward.temperature_violation_penalty` | 50 ¥/°C | 温度越轨代价 | `TESHeatExEnv._calculate_reward`
| 训练配置 | `training.total_timesteps` | 500,000 | 总采样步数，影响收敛 | `train_rl_agent()`

通过调整这些参数，可以精确模拟大中型教学楼、实验室或宿舍楼的不同热负荷场景，并与电网的峰谷套利策略联动。

### 5.4 配置文件与算法协同

配置位于 `project/configs/` 目录，采用层次化 YAML 结构控制仿真、环境与强化学习算法：

- **`default.yaml`**：面向通用实验流程，默认使用 SAC 算法与离散动作空间（`rl_env.action_space.type: "discrete"`）。
- **`sac_continuous.yaml`**：专为持续动作空间的 SAC 调优，将动作空间切换为 `continuous`，确保 `train_rl_agent()` 创建的 Stable-Baselines3 `SAC` 模型与环境动作维度一致。
- **`ppo_config.yaml`** 与 **`dqn_config.yaml`**：分别对接 `PPO`、`DQN` 训练入口，保持物理建模一致，同时在 `training` 区块内提供各自的核心超参数。

`main.py train` 会读取指定 YAML，将以下区块注入具体模块：

- **`simulation`**：传递给 `TESHeatExEnv` 的构造函数，决定 `self.timestep`、仿真步数 `max_steps` 及随机种子，直接影响储能积分精度和需求序列生成。
- **`tes`、`heat_exchanger`、`electric_heater`、`heat_pump`**：分别用于初始化 `SensibleHeatStorage`/`PCMStorage`、`create_heat_exchanger()`、加热器功率约束以及可选的热泵模型，使物理边界条件与现场设备规格对齐。
- **`tou_pricing` 与 `heat_demand`**：交由 `create_tou_pricing()` 和 `generate_demand_profile()`，驱动经济模型与需求侧负荷，体现外部市场与气候因素。
- **`rl_env`**：定义观测、动作、奖励三大结构。离散模式下 `n_actions`=3 映射 `idle/charge/discharge`，连续模式则通过 `min_power`、`max_power` 限制策略输出；奖励权重与 `TESHeatExEnv._calculate_reward()` 一一对应。
- **`baseline`**：供 `baselines/rule_based.py` 读取，简单规则控制器使用 `temperature_hysteresis` 建立充放电判据，作为与智能策略对比的参考线。
- **`training`**：传入 `train_rl_agent()`，决定 Stable-Baselines3 算法实例化方式。公共字段（`learning_rate`、`gamma`、`batch_size`）控制采样与梯度更新节奏；算法特定字段映射如下表。

| 算法配置 | 关键 YAML 字段 | SB3 对应参数 | 行为影响 |
| --- | --- | --- | --- |
| SAC (`default.yaml`, `sac_continuous.yaml`) | `buffer_size`, `learning_starts`, `tau`, `ent_coef` | `SAC` 缓冲区大小、启动步数、软更新系数、熵系数 | 决定经验回放容量、目标网络更新平滑度与探索强度 |
| PPO (`ppo_config.yaml`) | `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `n_steps`, `n_epochs` | `PPO` 的优势估计、策略裁剪、熵/价值系数以及批采样长度 | 控制策略更新稳定性与样本效率 |
| DQN (`dqn_config.yaml`) | `exploration_initial_eps`, `exploration_final_eps`, `exploration_fraction`, `buffer_size`, `train_freq`, `target_update_interval`, `policy_kwargs.net_arch` | `DQN` 的 ε-贪婪退火、经验回放、目标网络同步频率、网络结构 | 平衡探索与利用、设定 Q 网络容量 |

此外，`evaluation` 与 `metrics` 区块为评估脚本 (`simulate/run_eval.py`, `metrics/calculator.py`) 提供情景参数与输出路径，使成本节约、温度越轨等指标保持与论文复现实验一致。

#### 5.4.1 算法参数详解

**SAC (`default.yaml` / `sac_continuous.yaml`)**

- **`learning_rate`**：控制策略与 Q 网络的梯度步长，数值越大收敛越快但易震荡；默认 `3e-4`。
- **`batch_size`**：每次从回放缓冲区采样的批量大小，决定单次更新的统计稳定性；默认 `64`。
- **`gamma`**：回报折扣因子，`0.99` 对应约 100 小时的有效记忆窗口。
- **`buffer_size`**：经验回放容量 (`100000`)，影响历史数据覆盖范围；容量越大越能平滑训练但占用更多内存。
- **`learning_starts`**：在收集多少步数据后开始训练 (`1000`)，避免初期样本不足导致策略崩溃。
- **`tau`**：软更新系数 (`0.005`)，决定目标网络跟随主网络的速度，数值越小越平滑。
- **`ent_coef`**：熵系数，若未配置则由 SB3 自动调节，维持策略探索度。

**PPO (`ppo_config.yaml`)**

- **`learning_rate` / `batch_size` / `gamma`**：与 SAC 含义一致。
- **`n_steps`**：每次策略更新前在环境中滚动的步数 (`2048`)，决定单个轨迹批大小。
- **`n_epochs`**：对同一批数据重复优化次数 (`10`)，提高样本利用率。
- **`gae_lambda`**：广义优势估计的平滑因子 (`0.95`)，平衡偏差与方差。
- **`clip_range`**：策略裁剪半径 (`0.2`)，限制新旧策略差异保证稳定性。
- **`ent_coef`**：熵正则项 (`0.01`)，鼓励策略保持探索。
- **`vf_coef`**：价值函数损失权重 (`0.5`)，控制价值网络与策略网络的优化比重。

**DQN (`dqn_config.yaml`)**

- **`learning_rate`**：Q 网络更新步长 (`1e-4`)，需更小数值以确保收敛。
- **`batch_size` / `gamma`**：与前述含义一致，默认 `32`/`0.99`。
- **`buffer_size`**：回放容量 (`10000`)，在离散动作空间下保持较小以加速更新。
- **`learning_starts`**：开始训练前的随机探索步数 (`1000`)，确保 Q 网络有足够样本。
- **`train_freq`**：每收集多少步执行一次梯度更新 (`4`)，影响训练与采样节奏。
- **`gradient_steps`**：每次训练迭代的梯度更新次数 (`-1` 表示与 `train_freq` 一致)。
- **`target_update_interval`**：目标网络同步间隔 (`1000`)，控制估计稳定性。
- **`exploration_initial_eps` / `exploration_final_eps` / `exploration_fraction`**：定义 ε-贪婪策略的初始、最终探索率与衰减比例，默认从 `1.0` 衰减到 `0.05`，在 10% 训练步内完成过渡。
- **`policy_kwargs.net_arch`**：隐藏层结构 (`[256, 256]`)，可调节以匹配环境复杂度。

为了便于查阅，下表总结了关键训练参数在各算法中的作用：

| 参数名称 | YAML 路径 | 所属算法 | Stable-Baselines3 参数 | 默认值 | 主要作用 |
| --- | --- | --- | --- | --- | --- |
| `learning_rate` | `training.learning_rate` | SAC/PPO | `learning_rate` | `3e-4` | 决定策略更新步长，影响收敛速度与稳定性 |
| `learning_rate` | `training.learning_rate` | DQN | `learning_rate` | `1e-4` | 控制 Q 网络权重调整幅度，过大易震荡 |
| `batch_size` | `training.batch_size` | SAC/PPO | `batch_size` | `64` | 影响梯度估计方差与显存占用 |
| `batch_size` | `training.batch_size` | DQN | `batch_size` | `32` | 较小批量提升更新频率，适配离散动作 |
| `gamma` | `training.gamma` | 全部 | `gamma` | `0.99` | 折扣未来回报，平衡短期与长期收益 |
| `buffer_size` | `training.buffer_size` | SAC | `buffer_size` | `100000` | 存储连续动作经验，增强样本多样性 |
| `buffer_size` | `training.buffer_size` | DQN | `buffer_size` | `10000` | 控制经验库大小，防止过时策略干扰 |
| `learning_starts` | `training.learning_starts` | SAC/DQN | `learning_starts` | `1000` | 延迟训练避免早期样本偏差 |
| `tau` | `training.tau` | SAC | `tau` | `0.005` | 软更新目标网络，平滑价值估计 |
| `ent_coef` | `training.ent_coef` | SAC/PPO | `ent_coef` | 自动/`0.01` | 调节策略熵，控制探索程度 |
| `n_steps` | `training.n_steps` | PPO | `n_steps` | `2048` | 单次采样轨迹长度，影响时序依赖建模 |
| `n_epochs` | `training.n_epochs` | PPO | `n_epochs` | `10` | 每批数据重复优化次数，提高样本效率 |
| `gae_lambda` | `training.gae_lambda` | PPO | `gae_lambda` | `0.95` | 平衡优势估计偏差与方差 |
| `clip_range` | `training.clip_range` | PPO | `clip_range` | `0.2` | 限制策略更新幅度，防止性能崩溃 |
| `vf_coef` | `training.vf_coef` | PPO | `vf_coef` | `0.5` | 调整价值函数损失权重 |
| `train_freq` | `training.train_freq` | DQN | `train_freq` | `4` | 控制训练与采样节奏 |
| `gradient_steps` | `training.gradient_steps` | DQN | `gradient_steps` | `-1` | 配合 `train_freq` 决定每次更新次数 |
| `target_update_interval` | `training.target_update_interval` | DQN | `target_update_interval` | `1000` | 同步目标网络，稳定 Q 值估计 |
| `exploration_initial_eps` | `training.exploration_initial_eps` | DQN | `exploration_initial_eps` | `1.0` | 初始随机探索率 |
| `exploration_final_eps` | `training.exploration_final_eps` | DQN | `exploration_final_eps` | `0.05` | 收敛阶段探索率 |
| `exploration_fraction` | `training.exploration_fraction` | DQN | `exploration_fraction` | `0.1` | 探索率衰减所占训练比例 |
| `policy_kwargs.net_arch` | `training.policy_kwargs.net_arch` | DQN | `policy_kwargs` | `[256, 256]` | Q 网络隐藏层尺寸 |
| `eval_freq` | `training.eval_freq` | 全部 | EvalCallback `eval_freq` | `10000` | 控制评估频率，监控训练表现 |
| `n_eval_episodes` | `training.n_eval_episodes` | 全部 | EvalCallback `n_eval_episodes` | `10` | 评估统计稳定性 |
| `save_freq` | `training.save_freq` | 全部 | Checkpoint `save_freq` | `50000` | 定期保存模型以防中断 |

### 5.5 默认配置与标准实验设置

项目默认加载 `configs/default.yaml`，其关键物理含义如下：

- **仿真步长 (`simulation.timestep = 3600 s`)**：代表 1 小时决策步长，与论文中基于小时级时间序列的 TRNSYS 职能模型保持一致，保证功率↔能量积分的量纲一致性。
- **年内持续 (`simulation.duration = 8760 h`)**：覆盖完整年度的供暖/供冷工况，符合 Buscemi 等人将 DRL 控制器部署于整年负荷序列的研究设定。
- **储能初始状态 (`tes.initial_temperature = 45 ℃`)**：位于 `min_temperature=40 ℃` 与 `max_temperature=50 ℃` 之间，对应论文中热罐安全温度窗口的中点，避免初期越界。
- **换热器名义参数 (`heat_exchanger.heat_transfer_area = 50 m²`, `overall_heat_transfer_coefficient = 0.5 kW/(m²·K)`)**：在 ε-NTU 框架中确保 `NTU≈25`，模拟中等规模教学楼的板式换热器性能。
- **电加热器额定功率 (`electric_heater.max_power = 100 kW`)**：作为可调节补热装置，与论文中“通过附加电功率调度实现削峰填谷”的设定一致。
- **峰谷电价结构 (`tou_pricing.peak_price = 1.2`, `shoulder_price = 0.7`, `offpeak_price = 0.3` CNY/kWh)**：映射国内常见 TOU 政策，峰段时段设置为 `[10,12]`、`[18,21]`，为强化学习提供套利动机。
- **合成热负荷 (`heat_demand.base_load = 30 kW`, `peak_load = 80 kW`)**：通过内建生成器构造日内波动，模仿论文中的课堂负荷上下限。
- **奖励权重 (`rl_env.reward.cost_weight = -1.0`, `temperature_violation_penalty = 10`, `demand_violation_penalty = 20`)**：延续论文“以经济性为主、舒适性为约束”的理念，成本项系数为负以鼓励节约。

当用户未指定配置时，`main.py` 与 `simulate/run_eval.py` 将采用上述默认设置执行训练与评估。若需复现论文中的标准场景，建议保留 1h 时间步长、全年持续仿真、峰谷电价与 40–50 ℃ 的 TES 温度窗口，仅在 `training` 区块内调整算法相关超参数以探索不同控制策略。

## 6. 总结

本项目通过深度强化学习技术对热储能与换热系统进行优化控制，不仅在工程实践中具有重要意义，在学术研究和教育方面也具有很高价值。通过模块化的代码设计和详细的物理建模，我们构建了一个完整、可扩展的系统，为后续的研究和应用奠定了坚实基础。