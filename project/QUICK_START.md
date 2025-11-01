# Quick Start Guide

快速开始指南 - TES-HeatEx 优化系统

## 安装 (Installation)

### 1. 环境要求

- Python 3.8 或更高版本
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
cd project
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import gymnasium; import stable_baselines3; print('✓ Installation successful!')"
```

## 快速演示 (Quick Demo)

运行完整的演示程序（约3-5分钟）：

```bash
python demo.py
```

这个演示会：
- 展示环境基本功能
- 运行基线控制器
- 训练一个简单的RL智能体
- 比较两种方法的性能

## 基本使用 (Basic Usage)

### 1. 运行基线控制器

```bash
python baselines/rule_based.py --config configs/default.yaml
```

预期输出：
```
Evaluating baseline controller...
Results:
  Mean reward: -XXX.XX
  Total cost: XXX.XX CNY
  Temperature violations: X steps
```

### 2. 训练RL智能体

#### PPO算法（推荐用于离散动作）

```bash
python rl_algorithms/train.py \
    --config configs/default.yaml \
    --algo PPO \
    --timesteps 200000 \
    --save-path models
```

#### SAC算法（推荐用于连续动作）

```bash
python rl_algorithms/train.py \
    --config configs/sac_continuous.yaml \
    --algo SAC \
    --timesteps 200000 \
    --save-path models
```

跨目录运行示例（路径自动解析）：

```bash
python D:/学习/化能任选/换热器设计/workspace/project/rl_algorithms/train.py \
    --config D:/学习/化能任选/换热器设计/workspace/project/configs/sac_continuous.yaml \
    --algo SAC \
    --timesteps 200000 \
    --save-path D:/tmp/models
```

训练时间：
- CPU: ~2-3小时
- GPU: ~30-60分钟

### 3. 评估训练好的模型

```bash
python rl_algorithms/train.py --eval \
    --config configs/sac_continuous.yaml \
    --algo SAC \
    --model-path models/SAC_YYYYMMDD_HHMMSS_final.zip
```

### 4. 完整对比评估

```bash
python simulate/run_eval.py \
    --config configs/sac_continuous.yaml \
    --baseline simple_tou \
    --rl-model models/SAC_YYYYMMDD_HHMMSS_final.zip \
    --algo SAC \
    --episodes 10 \
    --output results
```

结果保存在 `results/` 目录：
- `comparison_report.md`: 详细对比报告
- `comparison_plots.png`: 可视化对比图
- `summary.csv`: 指标汇总表

## 配置文件 (Configuration)

主配置文件位于 `configs/default.yaml`

### 关键参数

#### 仿真参数
```yaml
simulation:
  timestep: 3600      # 时间步长（秒）
  duration: 8760      # 仿真时长（小时，1年）
  seed: 42            # 随机种子
```

#### 热储能参数
```yaml
tes:
  type: "sensible"    # 热储能类型：sensible 或 pcm
  mass: 5000          # 储能材料质量（kg）
  min_temperature: 40.0  # 最低温度（°C）
  max_temperature: 50.0  # 最高温度（°C）
```

#### 分时电价
```yaml
tou_pricing:
  peak_price: 1.2      # 峰时电价（元/kWh）
  shoulder_price: 0.7  # 平时电价
  offpeak_price: 0.3   # 谷时电价
```

#### RL训练参数
```yaml
training:
  algorithm: "SAC"           # PPO, SAC, 或 DQN
  total_timesteps: 200000    # 训练步数
  learning_rate: 0.0003      # 学习率
```

## 常见问题 (Troubleshooting)

### Q1: 导入错误 "No module named 'gymnasium'"

```bash
pip install gymnasium
```

### Q2: 训练过程中出现数值不稳定

检查配置文件中的奖励权重：
```yaml
rl_env:
  reward:
    cost_weight: -1.0
    temperature_violation_penalty: 10.0
    demand_violation_penalty: 20.0
```

建议先从较小的惩罚值开始。

### Q3: 基线控制器成本比RL更低

这可能是因为：
1. 训练步数不够（增加到 500,000+）
2. 奖励函数设计不合理
3. 观测空间不完整

### Q4: GPU训练不起作用

确认PyTorch GPU版本：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，需要重新安装CUDA版本的PyTorch。

## 下一步 (Next Steps)

### 1. 调整系统参数

编辑 `configs/default.yaml` 来：
- 改变储能容量
- 修改电价结构
- 调整热负荷模式

### 2. 尝试不同的RL算法

```bash
# 对比三种算法
python rl_algorithms/train.py --algo PPO --timesteps 200000
python rl_algorithms/train.py --algo SAC --timesteps 200000
python rl_algorithms/train.py --algo DQN --timesteps 200000
```

### 3. 进行场景分析

修改配置文件创建不同场景：
- 高需求场景（增加 `base_load` 和 `peak_load`）
- 价格波动场景（增加峰谷价差）
- 设备降级场景（降低 `efficiency`）

### 4. 深入学习

阅读技术文档：
```bash
docs/technical_doc.md      # 技术细节和数学模型
docs/experiment_results.md  # 实验结果模板
```

### 5. 运行测试

```bash
pytest tests/ -v
```

## 项目结构快速参考

```
project/
├── env/              # RL环境实现
├── models/           # 物理模型（储能、换热器、经济）
├── baselines/        # 基线控制器
├── rl_algorithms/    # RL训练脚本
├── simulate/         # 评估脚本
├── metrics/          # 性能指标计算
├── configs/          # 配置文件
├── tests/            # 单元测试
└── docs/             # 文档
```

## 获取帮助

如有问题：
1. 查看 `README.md` 获取详细说明
2. 阅读 `docs/technical_doc.md` 了解技术细节
3. 查看 `tests/test_models.py` 学习使用示例
4. 运行 `python demo.py` 查看完整演示

## 论文写作建议

### 实验设计

1. **对比实验**：基线 vs RL
2. **消融实验**：测试不同奖励权重
3. **鲁棒性实验**：不同场景下的性能
4. **敏感性分析**：参数变化的影响

### 关键指标

- 成本节约率（%）
- 温度违约率（%）
- 需求满足率（%）
- 储能利用率

### 可视化

使用 `metrics/calculator.py` 中的 `plot_comparison()` 生成：
- 温度轨迹对比
- SoC变化曲线
- 累计成本对比
- 充放电功率模式

祝实验顺利！🚀
