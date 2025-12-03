#!/usr/bin/env python
"""
Complete Evaluation Script for Carbon-Aware Building Control.

This script provides comprehensive evaluation including:
1. RL model vs baseline comparison
2. Multi-scenario testing (summer/winter/transition)
3. Pareto front analysis for cost-carbon trade-off
4. Visualization and result export

Usage:
    # Evaluate trained model vs baselines
    python evaluate_rl.py --model outputs/models/xxx/best_model.zip
    
    # Run Pareto analysis with different lambda values
    python evaluate_rl.py --pareto --n-points 10
    
    # Generate all visualizations
    python evaluate_rl.py --model outputs/models/xxx/best_model.zip --plot

Author: Auto-generated for Applied Energy publication
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
from envs.eplus_env import EnergyPlusGymEnv, EnvConfig
from baselines.rule_based import create_baseline
from models.transformer_policy import create_policy_kwargs


# =============================================================================
# Evaluation Scenarios
# =============================================================================

SCENARIOS = {
    "summer_peak": {
        "name": "Summer Peak (July)",
        "description": "Hot summer with high cooling demand",
        "start_day": 180,  # July 1
        "duration_days": 7,
        "outdoor_temp_offset": 10.0,  # Hotter
    },
    "winter_peak": {
        "name": "Winter Peak (January)",
        "description": "Cold winter with high heating demand",
        "start_day": 15,
        "duration_days": 7,
        "outdoor_temp_offset": -10.0,  # Colder
    },
    "transition_spring": {
        "name": "Spring Transition (April)",
        "description": "Mild weather, minimal HVAC",
        "start_day": 100,
        "duration_days": 7,
        "outdoor_temp_offset": 0.0,
    },
    "annual": {
        "name": "Annual Average",
        "description": "Full year simulation",
        "start_day": 1,
        "duration_days": 30,  # Sample 30 days
        "outdoor_temp_offset": 0.0,
    },
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def create_eval_env(
    episode_length: int = 672,
    cost_weight: float = 0.5,
    carbon_weight: float = 0.5,
) -> EnergyPlusGymEnv:
    """Create evaluation environment."""
    config = EnvConfig(
        episode_length=episode_length,
        include_forecasts=True,
        forecast_horizon=4,
        cost_weight=cost_weight,
        carbon_weight=carbon_weight,
        comfort_penalty=10.0,
    )
    return EnergyPlusGymEnv(config=config, use_mock=True)


def evaluate_controller(
    controller,
    env: EnergyPlusGymEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a controller on the environment.
    
    Args:
        controller: RL model or baseline controller
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        verbose: Print progress
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_costs = []
    episode_carbons = []
    episode_comfort_violations = []
    episode_lengths = []
    
    step_data = []  # For detailed analysis
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 100)
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            # Get action
            if hasattr(controller, 'predict'):
                if hasattr(controller, 'policy'):
                    # RL model
                    action, _ = controller.predict(obs, deterministic=deterministic)
                else:
                    # Baseline controller
                    action, _ = controller.predict(obs, info, deterministic=deterministic)
            else:
                action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_steps += 1
            
            # Record step data
            step_data.append({
                'episode': ep,
                'step': ep_steps,
                'reward': reward,
                'cost': info.get('step_cost', 0),
                'carbon': info.get('step_carbon', 0),
                'comfort_violation': info.get('step_comfort_violation', 0),
                'avg_temp': info.get('avg_zone_temp', 22),
                'outdoor_temp': info.get('outdoor_temp', 20),
                'price': info.get('price', 0.6),
                'tou_period': info.get('tou_period', 'mid'),
            })
        
        # Episode metrics
        metrics = env.get_episode_metrics()
        episode_rewards.append(ep_reward)
        episode_costs.append(metrics['total_cost'])
        episode_carbons.append(metrics['total_carbon'])
        episode_comfort_violations.append(metrics['total_comfort_violation'])
        episode_lengths.append(ep_steps)
        
        if verbose:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"R={ep_reward:.1f}, Cost=¥{metrics['total_cost']:.2f}, "
                  f"CO2={metrics['total_carbon']:.2f}kg")
    
    # Aggregate results
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_carbon': np.mean(episode_carbons),
        'std_carbon': np.std(episode_carbons),
        'mean_comfort_violation': np.mean(episode_comfort_violations),
        'std_comfort_violation': np.std(episode_comfort_violations),
        'mean_episode_length': np.mean(episode_lengths),
        'n_episodes': n_episodes,
        'step_data': pd.DataFrame(step_data),
    }
    
    return results


def compare_controllers(
    rl_model_path: Optional[str],
    baseline_names: List[str] = ['rule_based', 'always_on', 'carbon_aware_rule', 'random'],
    n_episodes: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare RL model against baseline controllers.
    
    Args:
        rl_model_path: Path to trained RL model
        baseline_names: List of baseline controller names
        n_episodes: Episodes per controller
        verbose: Print results
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    env = create_eval_env()
    
    # Evaluate RL model if provided
    if rl_model_path and Path(rl_model_path).exists():
        if verbose:
            print(f"\n[Evaluating RL Model: {rl_model_path}]")
        
        model = SAC.load(rl_model_path)
        
        # Check for VecNormalize
        vec_norm_path = Path(rl_model_path).parent / "vec_normalize.pkl"
        if vec_norm_path.exists():
            from stable_baselines3.common.vec_env import VecNormalize
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            
            # Use vectorized env for evaluation
            eval_results = evaluate_controller(
                model, vec_env.envs[0], n_episodes, verbose=verbose
            )
        else:
            eval_results = evaluate_controller(model, env, n_episodes, verbose=verbose)
        
        results.append({
            'controller': 'SAC-Transformer',
            'type': 'RL',
            **{k: v for k, v in eval_results.items() if k != 'step_data'}
        })
    
    # Evaluate baselines
    for name in baseline_names:
        if verbose:
            print(f"\n[Evaluating Baseline: {name}]")
        
        controller = create_baseline(name, env.action_space)
        eval_results = evaluate_controller(controller, env, n_episodes, verbose=verbose)
        
        results.append({
            'controller': name,
            'type': 'Baseline',
            **{k: v for k, v in eval_results.items() if k != 'step_data'}
        })
    
    env.close()
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        print(df.to_string(index=False))
    
    return df


# =============================================================================
# Pareto Analysis
# =============================================================================

def pareto_analysis(
    n_points: int = 10,
    timesteps_per_point: int = 50000,
    n_eval_episodes: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate Pareto front by training with different cost/carbon weights.
    
    Args:
        n_points: Number of Pareto points
        timesteps_per_point: Training timesteps per point
        n_eval_episodes: Evaluation episodes per point
        verbose: Print progress
        
    Returns:
        DataFrame with Pareto front data
    """
    if verbose:
        print("=" * 70)
        print("PARETO FRONT ANALYSIS")
        print("=" * 70)
    
    # Generate weight combinations
    lambdas = np.linspace(0.0, 1.0, n_points)
    
    pareto_points = []
    
    for i, lambda_carbon in enumerate(lambdas):
        lambda_cost = 1.0 - lambda_carbon
        
        if verbose:
            print(f"\n[Point {i+1}/{n_points}] λ_cost={lambda_cost:.2f}, λ_carbon={lambda_carbon:.2f}")
        
        # Create environment with specific weights
        env = create_eval_env(
            cost_weight=lambda_cost,
            carbon_weight=lambda_carbon,
        )
        
        # Train a quick model
        from stable_baselines3 import SAC
        
        vec_env = DummyVecEnv([lambda: env])
        
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            learning_starts=1000,
            verbose=0,
        )
        
        model.learn(total_timesteps=timesteps_per_point, progress_bar=False)
        
        # Evaluate
        eval_env = create_eval_env(cost_weight=0.5, carbon_weight=0.5)  # Neutral eval
        results = evaluate_controller(model, eval_env, n_eval_episodes, verbose=False)
        
        pareto_points.append({
            'lambda_cost': lambda_cost,
            'lambda_carbon': lambda_carbon,
            'mean_cost': results['mean_cost'],
            'std_cost': results['std_cost'],
            'mean_carbon': results['mean_carbon'],
            'std_carbon': results['std_carbon'],
            'mean_reward': results['mean_reward'],
        })
        
        if verbose:
            print(f"  → Cost: ¥{results['mean_cost']:.2f}, Carbon: {results['mean_carbon']:.2f} kg")
        
        vec_env.close()
        eval_env.close()
    
    df = pd.DataFrame(pareto_points)
    
    if verbose:
        print("\n" + "=" * 70)
        print("PARETO FRONT DATA")
        print("=" * 70)
        print(df.to_string(index=False))
    
    return df


def quick_pareto_analysis(
    n_eval_episodes: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Quick Pareto analysis using baseline evaluation (no retraining).
    
    Varies evaluation reward weights to show cost-carbon trade-off.
    """
    if verbose:
        print("=" * 70)
        print("QUICK PARETO ANALYSIS (Baseline Controllers)")
        print("=" * 70)
    
    baselines = ['rule_based', 'always_on', 'carbon_aware_rule', 'random']
    all_results = []
    
    env = create_eval_env()
    
    for name in baselines:
        if verbose:
            print(f"\n[Evaluating {name}]")
        
        controller = create_baseline(name, env.action_space)
        results = evaluate_controller(controller, env, n_eval_episodes, verbose=False)
        
        all_results.append({
            'controller': name,
            'cost': results['mean_cost'],
            'carbon': results['mean_carbon'],
            'comfort': results['mean_comfort_violation'],
        })
        
        if verbose:
            print(f"  Cost: ¥{results['mean_cost']:.2f}, Carbon: {results['mean_carbon']:.2f} kg")
    
    env.close()
    
    df = pd.DataFrame(all_results)
    return df


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot comparison bar charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    controllers = comparison_df['controller'].tolist()
    x = np.arange(len(controllers))
    
    # Cost comparison
    ax1 = axes[0]
    costs = comparison_df['mean_cost'].tolist()
    cost_errs = comparison_df['std_cost'].tolist()
    bars1 = ax1.bar(x, costs, yerr=cost_errs, capsize=5, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Controller')
    ax1.set_ylabel('Cost (¥)')
    ax1.set_title('Electricity Cost Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(controllers, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Carbon comparison
    ax2 = axes[1]
    carbons = comparison_df['mean_carbon'].tolist()
    carbon_errs = comparison_df['std_carbon'].tolist()
    bars2 = ax2.bar(x, carbons, yerr=carbon_errs, capsize=5, color='forestgreen', alpha=0.8)
    ax2.set_xlabel('Controller')
    ax2.set_ylabel('Carbon (kg CO₂)')
    ax2.set_title('Carbon Emission Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(controllers, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Reward comparison
    ax3 = axes[2]
    rewards = comparison_df['mean_reward'].tolist()
    reward_errs = comparison_df['std_reward'].tolist()
    colors = ['coral' if r < 0 else 'lightgreen' for r in rewards]
    bars3 = ax3.bar(x, rewards, yerr=reward_errs, capsize=5, color=colors, alpha=0.8)
    ax3.set_xlabel('Controller')
    ax3.set_ylabel('Reward')
    ax3.set_title('Total Reward Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(controllers, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    else:
        plt.savefig('outputs/results/comparison.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_pareto_front(
    pareto_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot Pareto front."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        pareto_df['cost'],
        pareto_df['carbon'],
        c=np.arange(len(pareto_df)),
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
    )
    
    # Annotate points
    for i, row in pareto_df.iterrows():
        ax.annotate(
            row['controller'],
            (row['cost'], row['carbon']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    
    ax.set_xlabel('Electricity Cost (¥)', fontsize=12)
    ax.set_ylabel('Carbon Emission (kg CO₂)', fontsize=12)
    ax.set_title('Cost-Carbon Trade-off (Pareto Front)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add ideal point
    min_cost = pareto_df['cost'].min()
    min_carbon = pareto_df['carbon'].min()
    ax.scatter([min_cost], [min_carbon], c='red', s=200, marker='*', 
               label='Ideal Point', zorder=5)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pareto plot saved: {save_path}")
    else:
        plt.savefig('outputs/results/pareto_front.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_time_series(
    step_data: pd.DataFrame,
    controller_name: str,
    save_path: Optional[str] = None,
):
    """Plot time series of a single episode."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    # Get first episode
    ep_data = step_data[step_data['episode'] == 0].copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    steps = ep_data['step'].values
    
    # Temperature
    ax1 = axes[0]
    ax1.plot(steps, ep_data['avg_temp'], label='Indoor Temp', color='orange')
    ax1.plot(steps, ep_data['outdoor_temp'], label='Outdoor Temp', color='blue', alpha=0.5)
    ax1.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Comfort Min')
    ax1.axhline(y=26, color='red', linestyle='--', alpha=0.5, label='Comfort Max')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'{controller_name} - Episode Analysis')
    ax1.grid(True, alpha=0.3)
    
    # Cost and Carbon
    ax2 = axes[1]
    ax2.plot(steps, ep_data['cost'].cumsum(), label='Cumulative Cost', color='steelblue')
    ax2.set_ylabel('Cost (¥)')
    ax2.legend(loc='upper left')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, ep_data['carbon'].cumsum(), label='Cumulative Carbon', 
                  color='forestgreen', linestyle='--')
    ax2_twin.set_ylabel('Carbon (kg)')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Price
    ax3 = axes[2]
    ax3.fill_between(steps, ep_data['price'], alpha=0.5, color='purple')
    ax3.set_ylabel('Price (¥/kWh)')
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, alpha=0.3)
    
    # Reward
    ax4 = axes[3]
    ax4.plot(steps, ep_data['reward'].cumsum(), color='coral')
    ax4.set_ylabel('Cumulative Reward')
    ax4.set_xlabel('Timestep')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Time series plot saved: {save_path}")
    else:
        plt.savefig(f'outputs/results/timeseries_{controller_name}.png', dpi=150, bbox_inches='tight')
    
    plt.close()


# =============================================================================
# Results Export
# =============================================================================

def export_results(
    comparison_df: pd.DataFrame,
    pareto_df: Optional[pd.DataFrame] = None,
    output_dir: str = "outputs/results",
):
    """Export results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison
    comparison_path = output_path / f"comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved: {comparison_path}")
    
    # Save Pareto
    if pareto_df is not None:
        pareto_path = output_path / f"pareto_{timestamp}.csv"
        pareto_df.to_csv(pareto_path, index=False)
        print(f"Pareto data saved: {pareto_path}")
    
    # Summary JSON
    summary = {
        'timestamp': timestamp,
        'best_controller': comparison_df.loc[comparison_df['mean_reward'].idxmax(), 'controller'],
        'lowest_cost': comparison_df.loc[comparison_df['mean_cost'].idxmin(), 'controller'],
        'lowest_carbon': comparison_df.loc[comparison_df['mean_carbon'].idxmin(), 'controller'],
        'results': comparison_df.to_dict('records'),
    }
    
    summary_path = output_path / f"summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Carbon-Aware Building Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained RL model"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--pareto", action="store_true",
        help="Run Pareto front analysis"
    )
    parser.add_argument(
        "--pareto-points", type=int, default=5,
        help="Number of Pareto points"
    )
    parser.add_argument(
        "--pareto-timesteps", type=int, default=20000,
        help="Training timesteps per Pareto point"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/results",
        help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick evaluation (baselines only)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CARBON-AWARE BUILDING CONTROL - EVALUATION")
    print("=" * 70)
    
    # Quick baseline comparison
    if args.quick:
        pareto_df = quick_pareto_analysis(n_eval_episodes=args.episodes)
        
        if args.plot:
            plot_pareto_front(pareto_df, f"{args.output}/pareto_baselines.png")
        
        pareto_df.to_csv(f"{args.output}/baseline_results.csv", index=False)
        print(f"\nResults saved to {args.output}/")
        return
    
    # Full comparison
    comparison_df = compare_controllers(
        rl_model_path=args.model,
        n_episodes=args.episodes,
        verbose=True,
    )
    
    # Pareto analysis
    pareto_df = None
    if args.pareto:
        pareto_df = pareto_analysis(
            n_points=args.pareto_points,
            timesteps_per_point=args.pareto_timesteps,
            n_eval_episodes=args.episodes,
            verbose=True,
        )
    
    # Generate plots
    if args.plot:
        print("\n[Generating plots...]")
        plot_comparison(comparison_df, f"{args.output}/comparison.png")
        
        if pareto_df is not None:
            plot_pareto_front(pareto_df, f"{args.output}/pareto_front.png")
    
    # Export results
    export_results(comparison_df, pareto_df, args.output)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
