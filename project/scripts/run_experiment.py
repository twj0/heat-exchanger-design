"""
Complete Experiment Runner for Carbon-Aware Building Control.

This script runs the full experimental pipeline:
1. Train RL agent (Transformer-SAC)
2. Train baseline MLP-SAC
3. Evaluate all controllers
4. Generate comparison figures
5. Produce final report

Run from project root:
    python -m scripts.run_experiment --quick  # Quick test (1000 steps)
    python -m scripts.run_experiment          # Full experiment (500k steps)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np

from stable_baselines3 import SAC

# Local imports
from envs.sinergym_env import create_test_env
from models.transformer_policy import create_policy_kwargs
from baselines.rule_based import create_baseline
from scripts.train import train
from scripts.evaluate import compare_controllers, evaluate_controller
from scripts.visualize import generate_all_figures


def run_full_experiment(
    experiment_name: str = "carbon_aware_experiment",
    total_timesteps: int = 500000,
    n_eval_episodes: int = 10,
    seed: int = 42,
    quick_mode: bool = False,
):
    """
    Run complete experimental pipeline.
    
    Args:
        experiment_name: Name for the experiment
        total_timesteps: Training timesteps (reduced in quick mode)
        n_eval_episodes: Evaluation episodes
        seed: Random seed
        quick_mode: If True, run shortened experiment for testing
    """
    print("="*70)
    print("CARBON-AWARE BUILDING CONTROL - FULL EXPERIMENT")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = project_root / f"outputs/experiments/{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust parameters for quick mode
    if quick_mode:
        total_timesteps = 2000
        n_eval_episodes = 3
        print("\n⚡ QUICK MODE: Using reduced parameters for testing")
    
    print(f"\n[Experiment Configuration]")
    print(f"  Name: {experiment_name}")
    print(f"  Output: {experiment_dir}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Eval episodes: {n_eval_episodes}")
    print(f"  Seed: {seed}")
    
    results = {}
    
    # =========================================================================
    # Step 1: Train Transformer-SAC
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Training Transformer-SAC Agent")
    print("="*70)
    
    transformer_model, transformer_dirs = train(
        config_path='configs/experiment.yaml',
        experiment_name=f"{experiment_name}_transformer",
        seed=seed,
        use_transformer=True,
        total_timesteps=total_timesteps,
        verbose=0,
    )
    
    transformer_model_path = Path(transformer_dirs['model_dir']) / 'final_model'
    print(f"  Model saved: {transformer_model_path}")
    
    # =========================================================================
    # Step 2: Train MLP-SAC (Baseline)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Training MLP-SAC Baseline")
    print("="*70)
    
    mlp_model, mlp_dirs = train(
        config_path='configs/experiment.yaml',
        experiment_name=f"{experiment_name}_mlp",
        seed=seed,
        use_transformer=False,
        total_timesteps=total_timesteps,
        verbose=0,
    )
    
    mlp_model_path = Path(mlp_dirs['model_dir']) / 'final_model'
    print(f"  Model saved: {mlp_model_path}")
    
    # =========================================================================
    # Step 3: Evaluate All Controllers
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Evaluating All Controllers")
    print("="*70)
    
    # Environment factory
    def env_factory():
        return create_test_env(use_carbon_wrapper=True)
    
    # Prepare controllers
    env = env_factory()
    controllers = {
        'Transformer-SAC': transformer_model,
        'MLP-SAC': mlp_model,
        'Rule-Based': create_baseline('rule_based', env.action_space),
        'Carbon-Aware-Rule': create_baseline('carbon_aware_rule', env.action_space),
        'Fixed-Setpoint': create_baseline('fixed', env.action_space),
    }
    env.close()
    
    # Run comparison
    eval_dir = experiment_dir / 'evaluation'
    df, eval_results = compare_controllers(
        controllers,
        env_factory,
        n_episodes=n_eval_episodes,
        output_dir=str(eval_dir),
    )
    
    results['evaluation'] = eval_results
    
    # =========================================================================
    # Step 4: Generate Visualizations
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating Visualizations")
    print("="*70)
    
    figures_dir = experiment_dir / 'figures'
    results_file = eval_dir / 'detailed_results.json'
    
    if results_file.exists():
        generate_all_figures(str(results_file), str(figures_dir))
    
    # =========================================================================
    # Step 5: Generate Final Report
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Generating Final Report")
    print("="*70)
    
    report = generate_report(eval_results, experiment_dir)
    
    # Save report
    report_path = experiment_dir / 'EXPERIMENT_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")
    
    # Save experiment config
    config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'total_timesteps': total_timesteps,
        'n_eval_episodes': n_eval_episodes,
        'seed': seed,
        'quick_mode': quick_mode,
        'transformer_model': str(transformer_model_path),
        'mlp_model': str(mlp_model_path),
    }
    
    config_path = experiment_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    print(f"\n📁 All outputs saved to: {experiment_dir}")
    print(f"\n📊 Key Results:")
    
    # Print key comparisons
    if 'Transformer-SAC' in eval_results and 'Rule-Based' in eval_results:
        rl_cost = eval_results['Transformer-SAC']['mean_total_cost_CNY']
        baseline_cost = eval_results['Rule-Based']['mean_total_cost_CNY']
        cost_reduction = (1 - rl_cost / baseline_cost) * 100
        
        rl_carbon = eval_results['Transformer-SAC']['mean_total_carbon_kgCO2']
        baseline_carbon = eval_results['Rule-Based']['mean_total_carbon_kgCO2']
        carbon_reduction = (1 - rl_carbon / baseline_carbon) * 100
        
        print(f"  Cost reduction (vs Rule-Based): {cost_reduction:.1f}%")
        print(f"  Carbon reduction (vs Rule-Based): {carbon_reduction:.1f}%")
        
        if cost_reduction >= 10:
            print("  ✅ Target achieved: RL cost ≤ 90% of baseline")
        else:
            print("  ⚠️ Target not met: Need more training or tuning")
    
    # Compare Transformer vs MLP
    if 'Transformer-SAC' in eval_results and 'MLP-SAC' in eval_results:
        trans_reward = eval_results['Transformer-SAC']['mean_total_reward']
        mlp_reward = eval_results['MLP-SAC']['mean_total_reward']
        
        if trans_reward > mlp_reward:
            print(f"  ✅ Transformer outperforms MLP by {(trans_reward - mlp_reward):.2f} reward")
        else:
            print(f"  ⚠️ MLP performs better than Transformer (need more training)")
    
    return experiment_dir, results


def generate_report(eval_results: Dict, experiment_dir: Path) -> str:
    """Generate markdown experiment report."""
    
    report = f"""# Experiment Report: Carbon-Aware Building Control

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This experiment evaluates the Carbon-Aware Building Control system using 
Deep Reinforcement Learning (SAC) with Transformer-based feature extraction.

## Key Findings

### Performance Comparison

| Controller | Cost (CNY) | Carbon (kgCO2) | Violation Rate |
|------------|-----------|----------------|----------------|
"""
    
    for name, result in eval_results.items():
        cost = result['mean_total_cost_CNY']
        carbon = result['mean_total_carbon_kgCO2']
        violation = result['mean_violation_rate'] * 100
        report += f"| {name} | {cost:.2f} | {carbon:.2f} | {violation:.1f}% |\n"
    
    # Add improvement analysis
    if 'Transformer-SAC' in eval_results and 'Rule-Based' in eval_results:
        rl = eval_results['Transformer-SAC']
        baseline = eval_results['Rule-Based']
        
        cost_improve = (1 - rl['mean_total_cost_CNY'] / baseline['mean_total_cost_CNY']) * 100
        carbon_improve = (1 - rl['mean_total_carbon_kgCO2'] / baseline['mean_total_carbon_kgCO2']) * 100
        
        report += f"""
### Improvement vs Rule-Based Baseline

- **Cost Reduction**: {cost_improve:.1f}%
- **Carbon Reduction**: {carbon_improve:.1f}%
- **Target**: Cost ≤ 90% of baseline (10% reduction)

"""
        if cost_improve >= 10:
            report += "✅ **Target Achieved**\n"
        else:
            report += "⚠️ **Target Not Met** - Consider more training or hyperparameter tuning\n"
    
    report += f"""
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
"""
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete Carbon-Aware Building Control experiment"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="carbon_aware_experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode for testing (reduced timesteps)",
    )
    
    args = parser.parse_args()
    
    run_full_experiment(
        experiment_name=args.name,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.episodes,
        seed=args.seed,
        quick_mode=args.quick,
    )


if __name__ == "__main__":
    main()
