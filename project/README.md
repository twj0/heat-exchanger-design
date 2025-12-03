# Carbon-Aware Building Control with Deep Reinforcement Learning

基于深度强化学习的建筑能源碳感知控制系统

## 创新点

1. **Innovation A**: EnergyPlus + Sinergym 高保真建筑仿真
2. **Innovation B**: 碳-费双目标优化 (Carbon-Cost Dual-Objective)
3. **Innovation C**: Transformer 时序特征提取器

## 项目结构

```
project/
├── data/
│   ├── weather/              # 气象文件 (.epw)
│   ├── schedules/            # 电价和碳排放数据
│   ├── carbon_factors.py     # 上海碳排放因子 (沪环气〔2022〕34号)
│   └── templates/            # IDF模板
├── envs/
│   ├── eplus_env.py          # 主Gymnasium环境 (推荐)
│   ├── energyplus_api.py     # EnergyPlus Python API封装
│   ├── carbon_wrapper.py     # 碳感知环境包装器
│   ├── sinergym_env.py       # Sinergym集成
│   └── builder.py            # IDF模型生成器
├── models/
│   └── transformer_policy.py # Transformer/MLP特征提取器
├── baselines/
│   └── rule_based.py         # 基线控制器 (Rule/AlwaysOn/Random等)
├── configs/
│   └── experiment.yaml       # 实验配置
├── outputs/                  # 训练输出 (logs/models/results)
├── train_rl.py               # 主训练入口 ⭐
└── scripts/                  # 辅助脚本
```

## 环境配置

### 1. 安装 EnergyPlus

下载并安装 [EnergyPlus](https://energyplus.net/) (推荐 v23.1+)

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 下载气象数据

从 [OneBuilding.org](https://climate.onebuilding.org/) 下载中国城市的 `.epw` 文件

## 快捷启动

**Windows PowerShell 用户推荐使用交互式菜单：**

```powershell
# 运行交互式菜单
.\start.ps1
```

菜单选项：
- `[1]` 快速训练测试 (5k steps)
- `[2]` 完整训练 (500k steps)
- `[3]` 评估基线控制器
- `[4]` 运行快速实验
- `[5]` 运行完整实验
- `[6]` 检查 EnergyPlus 设置
- `[7]` 运行 EnergyPlus 仿真
- `[8]` 启动 TensorBoard

---

## 使用方法

### 快速开始 (Mock环境)

```bash
# 安装依赖
pip install -r requirements.txt

# 快速测试训练 (使用模拟环境)
python train_rl.py --timesteps 10000 --eval-freq 2000
```

### 完整训练

```bash
# 使用 Transformer 特征提取器 (默认)
python train_rl.py --timesteps 500000

# 使用 MLP 基线进行消融实验
python train_rl.py --extractor mlp --name mlp_baseline

# 使用增强版 Temporal Transformer
python train_rl.py --extractor temporal_transformer
```

### 基线评估

```bash
# 评估所有基线控制器
python train_rl.py --eval-baselines
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir=outputs/logs
```

## 可用的基线控制器

| 控制器 | 描述 | 用途 |
|--------|------|------|
| `rule_based` | 基于TOU时间表的策略 | 传统方法基准 |
| `always_on` | 紧凑死区, 最大舒适度 | 能耗上限 |
| `carbon_aware_rule` | 考虑碳强度的规则控制 | 强基线 |
| `random` | 随机动作 | 下限检验 |

## 下一步

1. [x] 配置碳排放因子 (沪环气〔2022〕34号)
2. [x] 实现 SAC + Transformer 训练
3. [x] 实现基线控制器
4. [ ] 配置真实 EnergyPlus 环境
5. [ ] 完整实验评估
6. [ ] 论文撰写
