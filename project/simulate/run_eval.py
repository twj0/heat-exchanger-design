"""
Unified evaluation script for comparing baseline and RL controllers.

This script provides comprehensive evaluation and comparison functionality.
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.append('..')

from env.tes_heatex_env import TESHeatExEnv
from baselines.rule_based import SimpleTOUController, run_baseline_evaluation
from rl_algorithms.train import evaluate_rl_agent
from metrics.calculator import (
    MetricsCalculator,
    compare_controllers,
    plot_comparison,
)


def resolve_config_path(config_arg: Optional[str]) -> str:
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = []
    if config_arg:
        if os.path.isabs(config_arg):
            candidates.append(config_arg)
        else:
            # relative to current working directory
            candidates.append(os.path.abspath(config_arg))
            # relative to script location
            candidates.append(os.path.abspath(os.path.join(script_dir, config_arg)))
    # fallback: default config in project root
    candidates.append(os.path.join(project_root, "configs", "default.yaml"))
    # fallback: default config relative to CWD
    candidates.append(os.path.abspath(os.path.join("configs", "default.yaml")))
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not resolve configuration file. Tried: {candidates}"
    )


def run_evaluation(
    config_path: str,
    baseline_type: str = "simple_tou",
    rl_model_path: Optional[str] = None,
    n_episodes: int = 5,
    output_dir: str = "results",
    seed: int = 42,
    algo: Optional[str] = None,
) -> Dict:
    """
    Run comprehensive evaluation of controllers.
    
    Args:
        config_path: Path to configuration file
        baseline_type: Type of baseline controller
        rl_model_path: Path to trained RL model (None to skip RL evaluation)
        n_episodes: Number of evaluation episodes
        output_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary with all evaluation results
    """
    print("=" * 80)
    print("TES-HeatEx System Evaluation")
    print("=" * 80)
    
    # Load configuration (robust path resolution)
    cfg_path = resolve_config_path(config_path)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("\nCreating evaluation environment...")
    env = TESHeatExEnv(config)
    
    # Results storage
    results = {}
    
    # Evaluate baseline controller
    print(f"\n{'='*80}")
    print(f"Evaluating Baseline Controller: {baseline_type}")
    print(f"{'='*80}")
    
    if baseline_type == "simple_tou":
        baseline_controller = SimpleTOUController(
            charge_in_offpeak=config["baseline"]["charge_in_offpeak"],
            discharge_in_peak=config["baseline"]["discharge_in_peak"],
            temperature_hysteresis=config["baseline"]["temperature_hysteresis"],
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    baseline_results = run_baseline_evaluation(
        env, baseline_controller, n_episodes=n_episodes
    )
    results["baseline"] = baseline_results
    
    # Get episode data for detailed analysis
    obs, info = env.reset(seed=seed)
    baseline_controller.reset()
    baseline_episode_data = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = baseline_controller.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
    
    baseline_episode_data = env.get_episode_data()
    
    print(f"\nBaseline Results:")
    print(f"  Mean Cost: {baseline_results['mean_cost']:.2f} CNY")
    print(f"  Mean Violations: {baseline_results['mean_violations']:.0f} steps")
    
    # Evaluate RL controller if model provided
    if rl_model_path:
        print(f"\n{'='*80}")
        print(f"Evaluating RL Controller")
        print(f"{'='*80}")
        
        rl_results = evaluate_rl_agent(
            model_path=rl_model_path,
            config=config,
            algo=algo,
            n_episodes=n_episodes,
            seed=seed,
            render=False,
        )
        results["rl"] = rl_results
        
        # Get RL episode data
        from stable_baselines3 import PPO, SAC, DQN
        
        model = None
        if algo is not None:
            algo_u = algo.upper()
            if algo_u == "PPO":
                model = PPO.load(rl_model_path)
            elif algo_u == "SAC":
                model = SAC.load(rl_model_path)
            elif algo_u == "DQN":
                model = DQN.load(rl_model_path)
            else:
                raise ValueError(f"Unsupported algorithm specified: {algo}")
        else:
            # Fallback to heuristic based on path
            path_u = rl_model_path.upper()
            if "PPO" in path_u:
                model = PPO.load(rl_model_path)
            elif "SAC" in path_u:
                model = SAC.load(rl_model_path)
            elif "DQN" in path_u:
                model = DQN.load(rl_model_path)
            else:
                print("Warning: Algorithm not specified and not inferred from path; defaulting to PPO.")
                model = PPO.load(rl_model_path)
        
        obs, info = env.reset(seed=seed)
        rl_episode_data = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        
        rl_episode_data = env.get_episode_data()
        
        # Compare controllers
        print(f"\n{'='*80}")
        print("Comparison Analysis")
        print(f"{'='*80}")
        
        # Create TOU pricing array for metrics
        tou_prices = np.array([config["tou_pricing"]["peak_price"] 
                              if h in [10, 11, 18, 19, 20] 
                              else config["tou_pricing"]["offpeak_price"]
                              for h in range(24)])
        
        comparison = compare_controllers(
            baseline_data=baseline_episode_data,
            rl_data=rl_episode_data,
            electricity_prices=tou_prices,
            save_path=os.path.join(output_dir, "comparison_report.md"),
        )
        results["comparison"] = comparison
        
        # Print comparison summary
        improvements = comparison["improvements"]
        print(f"\nCost Savings:")
        print(f"  Absolute: {improvements['cost_savings_cny']:.2f} CNY")
        print(f"  Percentage: {improvements['cost_savings_percent']:.1f}%")
        print(f"\nViolation Rate Change: {improvements['violation_improvement']:.2%}")
        
        # Create plots
        print(f"\nGenerating comparison plots...")
        plot_comparison(
            baseline_data=baseline_episode_data,
            rl_data=rl_episode_data,
            save_path=os.path.join(output_dir, "comparison_plots.png"),
        )
        
        # Save detailed results to CSV
        save_results_to_csv(results, output_dir)
        
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return results


def save_results_to_csv(results: Dict, output_dir: str) -> None:
    """
    Save results to CSV files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    # Create summary DataFrame
    summary_data = {
        "Metric": [],
        "Baseline": [],
        "RL": [],
        "Improvement": [],
    }
    
    if "rl" in results and "comparison" in results:
        baseline = results["comparison"]["baseline"]
        rl = results["comparison"]["rl"]
        
        # Cost metrics
        summary_data["Metric"].append("Total Cost (CNY)")
        summary_data["Baseline"].append(f"{baseline['cost']['total_cost']:.2f}")
        summary_data["RL"].append(f"{rl['cost']['total_cost']:.2f}")
        summary_data["Improvement"].append(
            f"{results['comparison']['improvements']['cost_savings_percent']:.1f}%"
        )
        
        # Demand satisfaction
        summary_data["Metric"].append("Demand Satisfaction")
        summary_data["Baseline"].append(
            f"{baseline['energy']['demand_satisfaction_rate']:.2%}"
        )
        summary_data["RL"].append(
            f"{rl['energy']['demand_satisfaction_rate']:.2%}"
        )
        summary_data["Improvement"].append("-")
        
        # Violation rate
        summary_data["Metric"].append("Violation Rate")
        summary_data["Baseline"].append(
            f"{baseline['temperature']['violation_rate']:.2%}"
        )
        summary_data["RL"].append(
            f"{rl['temperature']['violation_rate']:.2%}"
        )
        summary_data["Improvement"].append(
            f"{results['comparison']['improvements']['violation_improvement']:.2%}"
        )
        
        # Storage utilization
        summary_data["Metric"].append("Storage Utilization")
        summary_data["Baseline"].append(
            f"{baseline['storage']['soc_utilization']:.2f}"
        )
        summary_data["RL"].append(
            f"{rl['storage']['soc_utilization']:.2f}"
        )
        summary_data["Improvement"].append("-")
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")


def compare_all_controllers(
    config_path: str,
    rl_model_paths: Dict[str, str],
    output_dir: str = "results/comparison",
    n_episodes: int = 5,
) -> None:
    """
    Compare multiple RL models against baseline.
    
    Args:
        config_path: Path to configuration file
        rl_model_paths: Dictionary of {name: model_path}
        output_dir: Output directory
        n_episodes: Number of evaluation episodes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Evaluate baseline once
    print("Evaluating baseline...")
    baseline_results = run_evaluation(
        config_path=config_path,
        baseline_type="simple_tou",
        rl_model_path=None,
        n_episodes=n_episodes,
        output_dir=os.path.join(output_dir, "baseline"),
    )
    all_results["baseline"] = baseline_results
    
    # Evaluate each RL model
    for name, model_path in rl_model_paths.items():
        print(f"\nEvaluating {name}...")
        results = run_evaluation(
            config_path=config_path,
            baseline_type="simple_tou",
            rl_model_path=model_path,
            n_episodes=n_episodes,
            output_dir=os.path.join(output_dir, name),
        )
        all_results[name] = results
    
    print("\nAll evaluations complete!")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare TES-HeatEx controllers"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="simple_tou",
        help="Baseline controller type",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to trained RL model",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["PPO", "SAC", "DQN"],
        help="Algorithm type for the RL model (optional)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        config_path=args.config,
        baseline_type=args.baseline,
        rl_model_path=args.rl_model,
        n_episodes=args.episodes,
        output_dir=args.output,
        seed=args.seed,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
