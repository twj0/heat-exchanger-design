# 换热器课程设计项目

## 项目简介

本项目是为换热器课程设计开发的学术研究项目，旨在通过复现和扩展《Deep reinforcement learning-based control of thermal energy storage for university classrooms: Co-Simulation with TRNSYS-Python and transfer learning across operational scenarios》这篇论文的研究成果，结合我国电力系统的峰谷电价政策，探索换热与储热系统的优化设计方案。

该项目将强化学习技术应用于热能存储系统的控制优化，通过智能调度算法实现能源利用效率的最大化和运行成本的最小化。

## 原论文贡献致谢

本项目基于 Buscemi 等人在《Energy Reports》期刊上发表的论文《Deep reinforcement learning-based control of thermal energy storage for university classrooms: Co-Simulation with TRNSYS-Python and transfer learning across operational scenarios》进行开发。该论文提出了一个创新的框架，通过将TRNSYS与Python进行协同仿真，利用深度强化学习（特别是Soft Actor-Critic算法）优化空调系统中热能存储设备的运行策略。

论文的主要贡献包括：
- 建立了TRNSYS-Python协同仿真框架，实现了建筑热环境与控制算法的有效集成
- 应用Soft Actor-Critic（SAC）深度强化学习算法优化热能存储系统的运行策略
- 通过零样本迁移学习验证了控制器在不同气候条件、电价结构和系统故障场景下的适应性
- 实现了显著的节能效果（年一次能源消耗减少4.73 MWh）和成本降低（运行成本降低3.2%）

我们对原论文作者的开创性工作表示诚挚的感谢，他们的研究为我们在换热器课程设计中应用先进控制策略提供了重要理论基础和实践指导。

## 课程设计要求

根据《换热器课程设计选题方面.md》文档，本课程设计需要重点关注以下几个方面：

### 核心要求
- **热能利用与节能经济性**：将热工学知识与现实能源政策（特别是峰谷电价机制）相结合
- **创新性思维**：通过创新性地设计换热与蓄热系统来体现工程思维的整合能力
- **工程设计**：完成从理论分析到实际系统设计的完整过程

### 技术要点
1. **分时电价特性分析**：根据不同地区的峰谷电价时段制定优化策略
2. **蓄热方式对比**：分析水蓄热、固体蓄热和相变蓄热等不同方式的优缺点
3. **系统设计与工作原理**：设计电制热蓄热系统、混凝土蓄热模块或相变热库系统
4. **换热器设计**：设计空气-水管翅式换热器，满足特定的热负荷和流量要求

### 创新方向
- **节能优化**：结合实时电价自动启停储热系统，建立控制模型
- **结构创新**：开发新型换热材料（复合相变介质）
- **经济分析**：量化峰谷差价带来的收益，进行静态投资回收期分析
- **多能协同**：结合太阳能、地热能，实现多能源互补系统

## 项目使用说明

### 环境准备
1. 确保已安装Python 3.8+
2. 安装PowerShell（Windows系统自带）
3. 推荐使用uv[强烈推荐](https://docs.astral.sh/uv/)、conda[miniconda作为推荐](https://docs.conda.io/en/latest/miniconda.html)、Poetry[比较推荐](https://python-poetry.org/docs/)等包管理工具，进行依赖管理

### 使用 start.ps1 脚本

项目根目录下的 [start.ps1](file:///d:/学习/化能任选/换热器设计/workspace/start.ps1) 文件是一个交互式PowerShell脚本，提供了便捷的命令行导航界面：

1. 在项目根目录打开PowerShell终端
2. 运行以下命令启动脚本：
   ```powershell
   .\start.ps1
   ```

### 主菜单功能

```
Available optimization workflows:
  [1] Navigate between project directories - 在项目目录间导航
  [2] Execute Python script - 执行Python脚本
  [3] Train RL agent (Stable-Baselines3 / PyTorch) - 训练强化学习代理
  [4] Evaluate or run inference - 评估或运行推理
  [5] Run demo scenarios - 运行演示场景
  [6] UV environment tools - UV环境工具
  [7] General utilities - 通用工具
  [0] Quit - 退出
```

通过选择相应的选项，您可以方便地执行各种项目操作，而无需记忆复杂的命令行参数。

### 目录结构说明

- `project/` - 主要代码目录
  - `configs/` - 配置文件，包括默认配置和SAC算法配置
  - `env/` - 环境模拟器
  - `models/` - 换热器和储热设备模型
  - `rl_algorithms/` - 强化学习算法实现
  - `simulate/` - 仿真和评估脚本
- `2025-course-design-topics-and-references/` - 课程设计主题和参考文献
- `latex/` - 论文LaTeX模板

## 数据保存与可视化

### 自动保存与生成

1. **演示模式（自动）**：
   ```bash
   python demo.py
   ```
   运行演示脚本会自动：
   - 执行完整的训练和评估流程
   - 将数据保存到 `data/` 文件夹
   - 生成可视化图表并保存为PNG格式

2. **评估脚本（自动）**：
   ```bash
   python simulate/run_eval.py --config configs/default.yaml --rl-model my_models/SAC_xxx/best_model.zip
   ```
   会自动：
   - 运行评估并保存结果到CSV
   - 生成对比图表并保存

### 手动CLI操作

1. **训练模型**：
   ```bash
   python main.py train --algo SAC --config configs/default.yaml
   ```

2. **单独评估**：
   ```bash
   python main.py eval --baseline --config configs/default.yaml
   python main.py eval --model models/best.zip --config configs/default.yaml
   ```

3. **使用PowerShell脚本**：
   ```powershell
   .\start.ps1
   ```
   然后选择相应的工作流程选项

### 数据和图片保存位置

- **数据文件**：默认保存到 `data/` 文件夹
- **图片文件**：默认保存到 `results/` 或 `logs/` 文件夹
- **模型文件**：保存到 `my_models/` 或 `models/` 文件夹

### 配置保存选项

在 [configs/default.yaml](file:///d:/学习/化能任选/换热器设计/workspace/project/configs/default.yaml) 中可以配置：
```yaml
metrics:
  output_dir: "results"
  save_plots: true
  save_csv: true

training:
  save_freq: 50000
  log_dir: "logs"
  model_dir: "my_models"
```

### 总结

- **自动保存**：演示脚本和评估脚本会自动保存数据和生成图片
- **手动控制**：通过CLI命令可以更灵活地控制训练和评估过程
- **PowerShell界面**：提供友好的菜单式操作，适合不熟悉命令行的用户

建议先运行 `python demo.py` 体验完整的自动化流程，然后根据需要使用CLI命令进行更精细的控制。

## 配置文件参数详解

### 物理模型参数

#### 仿真参数
- `simulation.timestep`: 仿真时间步长（秒），默认3600秒（1小时）
- `simulation.duration`: 仿真总时长（小时），默认8760小时（1年）
- `simulation.seed`: 随机种子，用于结果复现

#### 热储能系统参数
- `tes.type`: 储能类型，支持"sensible"（显热）或"pcm"（相变材料）
- `tes.mass`: 储能介质质量（kg）
- `tes.specific_heat`: 比热容（kJ/(kg·K)）
- `tes.initial_temperature`: 初始温度（°C）
- `tes.min_temperature`: 最低工作温度（°C）
- `tes.max_temperature`: 最高工作温度（°C）
- `tes.melting_point`: 相变材料熔点（°C），仅对PCM有效
- `tes.latent_heat`: 相变潜热（kJ/kg），仅对PCM有效
- `tes.loss_coefficient`: 热损失系数（W/K）
- `tes.ambient_temperature`: 环境温度（°C）

#### 换热器参数
- `heat_exchanger.type`: 换热器类型，支持"effectiveness_ntu"或"lmtd"
- `heat_exchanger.heat_transfer_area`: 换热面积（m²）
- `heat_exchanger.overall_heat_transfer_coefficient`: 总传热系数（kW/(m²·K)）
- `heat_exchanger.effectiveness`: 有效度
- `heat_exchanger.flow_arrangement`: 流动布置方式，支持"counterflow"（逆流）、"parallel"（顺流）、"crossflow"（叉流）

#### 电加热器参数
- `electric_heater.max_power`: 最大功率（kW）
- `electric_heater.efficiency`: 加热效率

#### 热泵参数
- `heat_pump.enabled`: 是否启用热泵
- `heat_pump.cop_heating`: 制热性能系数
- `heat_pump.cop_cooling`: 制冷性能系数
- `heat_pump.max_power`: 最大功率（kW）

#### 分时电价参数
- `tou_pricing.peak_price`: 峰时电价（元/kWh）
- `tou_pricing.shoulder_price`: 平时电价（元/kWh）
- `tou_pricing.offpeak_price`: 谷时电价（元/kWh）
- `tou_pricing.peak_hours`: 峰时时间段
- `tou_pricing.shoulder_hours`: 平时时间段
- `tou_pricing.offpeak_hours`: 谷时时间段

### 强化学习算法参数

#### 训练参数
- `training.learning_rate`: 学习率
- `training.batch_size`: 批处理大小
- `training.gamma`: 折扣因子
- `training.total_timesteps`: 训练总步数
- `training.eval_freq`: 评估频率
- `training.n_eval_episodes`: 每次评估的回合数
- `training.save_freq`: 模型保存频率
- `training.model_dir`: 模型保存目录
- `training.log_dir`: 日志保存目录

#### 算法特定参数

##### PPO (Proximal Policy Optimization)
- `training.n_steps`: 每次更新的步数
- `training.n_epochs`: 每次更新的轮数
- `training.gae_lambda`: GAE参数
- `training.clip_range`: 裁剪范围
- `training.ent_coef`: 熵系数
- `training.vf_coef`: 价值函数系数

##### SAC (Soft Actor-Critic)
- `training.buffer_size`: 经验回放缓冲区大小
- `training.learning_starts`: 开始学习前的步数
- `training.tau`: 目标网络更新率
- `training.ent_coef`: 熵系数

##### DQN (Deep Q-Network)
- `training.buffer_size`: 经验回放缓冲区大小
- `training.learning_starts`: 开始学习前的步数
- `training.train_freq`: 训练频率
- `training.gradient_steps`: 每次更新的梯度步数
- `training.target_update_interval`: 目标网络更新间隔
- `training.exploration_initial_eps`: 初始探索率
- `training.exploration_final_eps`: 最终探索率
- `training.exploration_fraction`: 探索率衰减比例

## 强化学习算法原理

### PPO (Proximal Policy Optimization)
PPO是一种策略梯度方法，通过裁剪机制来确保策略更新的稳定性。它在每次更新时限制新策略与旧策略的差异，避免过大的策略更新导致性能下降。PPO通过以下方式优化策略：
1. 使用优势函数评估动作的好坏
2. 通过裁剪机制限制策略更新幅度
3. 同时优化策略函数和价值函数

### SAC (Soft Actor-Critic)
SAC是一种基于最大熵强化学习的算法，它不仅最大化预期回报，还最大化策略的熵。这种方法鼓励探索，通常能获得更稳定和更好的性能。SAC的主要特点包括：
1. 使用Actor-Critic架构
2. 最大化预期回报和策略熵的加权和
3. 使用双Q网络减少过估计偏差
4. 自动调整熵系数

### DQN (Deep Q-Network)
DQN是首个成功将深度学习与Q学习结合的算法。它通过使用经验回放和固定Q目标来稳定训练过程。DQN的关键技术包括：
1. 使用深度神经网络近似Q函数
2. 经验回放机制打破数据相关性
3. 固定Q目标减少训练不稳定性
4. ε-贪婪策略平衡探索与利用

## 项目目标

通过复现和扩展原论文的研究成果，结合我国电力系统的峰谷电价政策，本项目旨在：
1. 建立适用于我国国情的换热与储热系统模型
2. 应用深度强化学习算法优化系统运行策略
3. 量化分析峰谷电价机制带来的节能和经济效益