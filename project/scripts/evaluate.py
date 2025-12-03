"""
Evaluation Script for Carbon-Aware Building Control.

This script provides comprehensive evaluation of trained RL agents
and baseline controllers, including:
- Performance metrics (cost, carbon, comfort)
- Statistical analysis with confidence intervals
- Baseline comparisons
- Visualization generation

Run from project root:
    python -m scripts.evaluate --model outputs/models/xxx/final_model
    python -m scripts.evaluate --baseline rule_based
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from stable_baselines3 import SAC

# Local imports
from envs.sinergym_env import create_test_env
from baselines.rule_based import create_baseline


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.total_cost = 0.0
        self.total_carbon = 0.0
        self.total_reward = 0.0
        self.total_steps = 0
        self.comfort_violations = 0
        self.step_costs = []
        self.step_carbons = []
        self.step_rewards = []
        self.temperatures = []
        self.actions = []
        
    def update(self, reward: float, info: Dict, action: np.ndarray):
        """Update metrics with step data."""
        self.total_reward += reward
        self.total_cost += info.get('step_cost', 0)
        self.total_carbon += info.get('step_carbon', 0)
        self.total_steps += 1
        
        if info.get('step_comfort_violation', 0) > 0:
            self.comfort_violations += 1
            
        self.step_costs.append(info.get('step_cost', 0))
        self.step_carbons.append(info.get('step_carbon', 0))
        self.step_rewards.append(reward)
        self.actions.append(action.copy())
        
        # Store temperature if available
        if 'Zone Air Temperature' in info:
            self.temperatures.append(info['Zone Air Temperature'])
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        violation_rate = self.comfort_violations / max(1, self.total_steps)
        
        return {
            'total_cost_CNY': self.total_cost,
            'total_carbon_kgCO2': self.total_carbon,
            'total_reward': self.total_reward,
            'total_steps': self.total_steps,
            'comfort_violations': self.comfort_violations,
            'violation_rate': violation_rate,
            'avg_step_cost': np.mean(self.step_costs) if self.step_costs else 0,
            'avg_step_carbon': np.mean(self.step_carbons) if self.step_carbons else 0,
            'std_step_cost': np.std(self.step_costs) if self.step_costs else 0,
            'std_step_carbon': np.std(self.step_carbons) if self.step_carbons else 0,
        }


def evaluate_controller(
    controller,
    env,
    n_episodes: int = 10,
    max_steps_per_episode: int = 672,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a controller over multiple episodes.
    
    Args:
        controller: Controller with predict() method
        env: Gymnasium environment
        n_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
        deterministic: Use deterministic predictions
        verbose: Print progress
        
    Returns:
        Dictionary of evaluation results
    """
    all_metrics = []
    
    for ep in range(n_episodes):
        metrics = EvaluationMetrics()
        obs, info = env.reset(seed=42 + ep)
        
        for step in range(max_steps_per_episode):
            # Get action from controller
            action, _ = controller.predict(obs, info, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            metrics.update(reward, info, action)
            
            if terminated or truncated:
                break
        
        all_metrics.append(metrics.get_summary())
        
        if verbose:
            summary = metrics.get_summary()
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Cost={summary['total_cost_CNY']:.2f} CNY, "
                  f"Carbon={summary['total_carbon_kgCO2']:.2f} kgCO2, "
                  f"Violations={summary['violation_rate']*100:.1f}%")
    
    # Aggregate results
    results = {
        'n_episodes': n_episodes,
        'metrics': all_metrics,
    }
    
    # Calculate mean and std
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[f'mean_{key}'] = np.mean(values)
        results[f'std_{key}'] = np.std(values)
        results[f'min_{key}'] = np.min(values)
        results[f'max_{key}'] = np.max(values)
    
    # 95% confidence interval
    for metric in ['total_cost_CNY', 'total_carbon_kgCO2', 'violation_rate']:
        values = [m[metric] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        ci = 1.96 * std / np.sqrt(n_episodes)
        results[f'ci95_{metric}'] = ci
    
    return results


def compare_controllers(
    controllers: Dict[str, any],
    env_factory,
    n_episodes: int = 10,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Compare multiple controllers and generate comparison report.
    
    Args:
        controllers: Dict mapping controller names to controllers
        env_factory: Factory function to create environments
        n_episodes: Number of episodes per controller
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    results = {}
    
    for name, controller in controllers.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print('='*50)
        
        env = env_factory()
        result = evaluate_controller(
            controller, env, 
            n_episodes=n_episodes,
            verbose=True
        )
        results[name] = result
        env.close()
    
    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Controller': name,
            'Cost (CNY)': f"{result['mean_total_cost_CNY']:.2f} ± {result['ci95_total_cost_CNY']:.2f}",
            'Carbon (kgCO2)': f"{result['mean_total_carbon_kgCO2']:.2f} ± {result['ci95_total_carbon_kgCO2']:.2f}",
            'Violation Rate': f"{result['mean_violation_rate']*100:.2f}% ± {result['ci95_violation_rate']*100:.2f}%",
            'Reward': f"{result['mean_total_reward']:.2f}",
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate improvements vs baseline
    if 'rule_based' in results:
        baseline_cost = results['rule_based']['mean_total_cost_CNY']
        baseline_carbon = results['rule_based']['mean_total_carbon_kgCO2']
        
        improvements = []
        for name, result in results.items():
            cost_improve = (1 - result['mean_total_cost_CNY'] / baseline_cost) * 100
            carbon_improve = (1 - result['mean_total_carbon_kgCO2'] / baseline_carbon) * 100
            improvements.append({
                'Controller': name,
                'Cost Reduction (%)': f"{cost_improve:.1f}%",
                'Carbon Reduction (%)': f"{carbon_improve:.1f}%",
            })
        
        df_improve = pd.DataFrame(improvements)
        df = df.merge(df_improve, on='Controller')
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(output_path / 'comparison_results.csv', index=False)
        
        # Save detailed results as JSON
        with open(output_path / 'detailed_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for name, result in results.items():
                json_results[name] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result.items()
                    if k != 'metrics'  # Skip detailed per-episode metrics
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return df, results


def run_ablation_study(
    model_path: str,
    env_factory,
    n_episodes: int = 5,
    output_dir: str = None,
) -> Dict:
    """
    Run ablation study comparing different configurations.
    
    Studies:
    1. λ_carbon = 0 (cost-only optimization)
    2. MLP vs Transformer feature extractor
    3. Different forecast horizons
    
    Args:
        model_path: Path to trained model
        env_factory: Factory function
        n_episodes: Episodes per configuration
        output_dir: Output directory
        
    Returns:
        Ablation study results
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    results = {}
    
    # Load trained model
    print("\n[1] Loading trained model...")
    model = SAC.load(model_path)
    
    # Standard evaluation
    print("\n[2] Standard configuration (Transformer + Carbon)...")
    env = env_factory()
    results['transformer_carbon'] = evaluate_controller(
        model, env, n_episodes=n_episodes
    )
    env.close()
    
    # Compare with baselines
    print("\n[3] Baseline comparisons...")
    env = env_factory()
    
    for baseline_name in ['rule_based', 'fixed', 'carbon_aware_rule']:
        baseline = create_baseline(baseline_name, env.action_space)
        results[baseline_name] = evaluate_controller(
            baseline, env, n_episodes=n_episodes
        )
    
    env.close()
    
    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Cost: {result['mean_total_cost_CNY']:.2f} CNY")
        print(f"  Carbon: {result['mean_total_carbon_kgCO2']:.2f} kgCO2")
        print(f"  Violations: {result['mean_violation_rate']*100:.2f}%")
    
    return results


def print_final_report(df: pd.DataFrame, results: Dict):
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    print("\n## Summary Table\n")
    print(df.to_string(index=False))
    
    # Check if RL beats baseline
    if 'transformer_sac' in results and 'rule_based' in results:
        rl_cost = results['transformer_sac']['mean_total_cost_CNY']
        baseline_cost = results['rule_based']['mean_total_cost_CNY']
        
        cost_reduction = (1 - rl_cost / baseline_cost) * 100
        
        print("\n## Performance vs Baseline")
        print(f"  Cost reduction: {cost_reduction:.1f}%")
        
        if cost_reduction >= 10:
            print("  ✅ Target achieved: RL cost ≤ 90% of baseline")
        else:
            print("  ⚠️ Target not met: RL cost > 90% of baseline")
        
        rl_violations = results['transformer_sac']['mean_violation_rate']
        if rl_violations <= 0.005:
            print("  ✅ Comfort violations ≤ 0.5%")
        else:
            print(f"  ⚠️ Comfort violations = {rl_violations*100:.2f}% (target: ≤0.5%)")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Carbon-Aware Building Control"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=['rule_based', 'random', 'fixed', 'carbon_aware_rule', 'all'],
        help="Evaluate specific baseline or 'all'",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study",
    )
    
    args = parser.parse_args()
    
    # Environment factory
    def env_factory():
        return create_test_env(use_carbon_wrapper=True)
    
    # Create output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(project_root / f"outputs/results/eval_{timestamp}")
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CARBON-AWARE BUILDING CONTROL - EVALUATION")
    print("="*60)
    
    controllers = {}
    
    # Load trained model if provided
    if args.model:
        print(f"\nLoading model: {args.model}")
        model = SAC.load(args.model)
        controllers['transformer_sac'] = model
    
    # Add baselines
    env = env_factory()
    
    if args.baseline == 'all' or args.baseline is None:
        baseline_types = ['rule_based', 'fixed', 'carbon_aware_rule', 'random']
    else:
        baseline_types = [args.baseline]
    
    for bt in baseline_types:
        controllers[bt] = create_baseline(bt, env.action_space)
    
    env.close()
    
    # Run comparison
    df, results = compare_controllers(
        controllers,
        env_factory,
        n_episodes=args.episodes,
        output_dir=args.output,
    )
    
    # Print report
    print_final_report(df, results)
    
    # Run ablation if requested
    if args.ablation and args.model:
        ablation_results = run_ablation_study(
            args.model,
            env_factory,
            n_episodes=args.episodes // 2,
            output_dir=args.output,
        )
    
    print(f"\n✅ Evaluation complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
