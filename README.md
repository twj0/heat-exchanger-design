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

## 项目目标

通过复现和扩展原论文的研究成果，结合我国电力系统的峰谷电价政策，本项目旨在：
1. 建立适用于我国国情的换热与储热系统模型
2. 应用深度强化学习算法优化系统运行策略
3. 量化分析峰谷电价机制带来的节能和经济效益
