#!/usr/bin/env python
"""
Complete Experiment Runner for Carbon-Aware Building Control Paper.

This script automates the full experimental workflow:
1. Train models with different configurations
2. Evaluate against baselines
3. Generate Pareto front analysis
4. Create publication-ready figures
5. Export results for paper

Usage:
    # Run all experiments
    python run_experiments.py --all
    
    # Run specific experiments
    python run_experiments.py --train --evaluate
    
    # Quick test mode
    python run_experiments.py --quick

Author: Auto-generated for Applied Energy publication
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, Optional


# =============================================================================
# Experiment Configurations
# =============================================================================

EXPERIMENTS = {
    "main_transformer": {
        "name": "SAC-Transformer",
        "extractor": "transformer",
        "timesteps": 500000,
        "description": "Main model with Transformer feature extractor",
    },
    "ablation_mlp": {
        "name": "SAC-MLP",
        "extractor": "mlp",
        "timesteps": 500000,
        "description": "Ablation: MLP feature extractor (no temporal attention)",
    },
    "ablation_temporal": {
        "name": "SAC-TemporalTransformer",
        "extractor": "temporal_transformer",
        "timesteps": 500000,
        "description": "Enhanced Transformer with cross-attention",
    },
}

PARETO_CONFIGS = [
    {"lambda_cost": 1.0, "lambda_carbon": 0.0, "name": "cost_only"},
    {"lambda_cost": 0.8, "lambda_carbon": 0.2, "name": "cost_heavy"},
    {"lambda_cost": 0.5, "lambda_carbon": 0.5, "name": "balanced"},
    {"lambda_cost": 0.2, "lambda_carbon": 0.8, "name": "carbon_heavy"},
    {"lambda_cost": 0.0, "lambda_carbon": 1.0, "name": "carbon_only"},
]

BASELINES = ['rule_based', 'always_on', 'carbon_aware_rule', 'random', 'night_setback']


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    experiment_key: str,
    timesteps: int = None,
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """Train a single model configuration."""
    from train_rl import train
    
    config = EXPERIMENTS[experiment_key]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
    
    model = train(
        experiment_name=experiment_key,
        extractor_type=config['extractor'],
        total_timesteps=timesteps or config['timesteps'],
        seed=seed,
        verbose=1 if verbose else 0,
    )
    
    # Return model path
    model_dir = project_root / "outputs" / "models"
    latest = sorted(model_dir.glob(f"{experiment_key}_*"))[-1]
    return str(latest / "best_model.zip")


def train_pareto_models(
    timesteps_per_point: int = 100000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Train models for Pareto front analysis."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.eplus_env import EnergyPlusGymEnv, EnvConfig
    
    results = {}
    
    for config in PARETO_CONFIGS:
        name = config['name']
        
        if verbose:
            print(f"\n[Pareto Point: {name}]")
            print(f"  λ_cost={config['lambda_cost']:.1f}, λ_carbon={config['lambda_carbon']:.1f}")
        
        # Create environment with specific weights
        env_config = EnvConfig(
            episode_length=672,
            include_forecasts=True,
            cost_weight=config['lambda_cost'],
            carbon_weight=config['lambda_carbon'],
        )
        env = EnergyPlusGymEnv(config=env_config, use_mock=True)
        vec_env = DummyVecEnv([lambda: env])
        
        # Train
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
        
        # Save
        save_dir = project_root / "outputs" / "models" / "pareto" / name
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir / "model.zip"))
        
        results[name] = {
            'model_path': str(save_dir / "model.zip"),
            **config,
        }
        
        vec_env.close()
        
        if verbose:
            print(f"  Model saved: {save_dir}")
    
    return results


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_all(
    model_paths: Dict[str, str] = None,
    n_episodes: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate all models and baselines."""
    from evaluate_rl import evaluate_controller, create_eval_env
    from baselines.rule_based import create_baseline
    from stable_baselines3 import SAC
    
    all_results = []
    env = create_eval_env()
    
    # Evaluate RL models
    if model_paths:
        for name, path in model_paths.items():
            if verbose:
                print(f"\n[Evaluating RL: {name}]")
            
            if Path(path).exists():
                model = SAC.load(path)
                results = evaluate_controller(model, env, n_episodes, verbose=False)
                
                all_results.append({
                    'controller': name,
                    'type': 'RL',
                    'mean_cost': results['mean_cost'],
                    'std_cost': results['std_cost'],
                    'mean_carbon': results['mean_carbon'],
                    'std_carbon': results['std_carbon'],
                    'mean_reward': results['mean_reward'],
                    'std_reward': results['std_reward'],
                    'mean_comfort': results['mean_comfort_violation'],
                })
                
                if verbose:
                    print(f"  Cost: ¥{results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
                    print(f"  Carbon: {results['mean_carbon']:.2f} ± {results['std_carbon']:.2f} kg")
    
    # Evaluate baselines
    for baseline in BASELINES:
        if verbose:
            print(f"\n[Evaluating Baseline: {baseline}]")
        
        controller = create_baseline(baseline, env.action_space)
        results = evaluate_controller(controller, env, n_episodes, verbose=False)
        
        all_results.append({
            'controller': baseline,
            'type': 'Baseline',
            'mean_cost': results['mean_cost'],
            'std_cost': results['std_cost'],
            'mean_carbon': results['mean_carbon'],
            'std_carbon': results['std_carbon'],
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'mean_comfort': results['mean_comfort_violation'],
        })
        
        if verbose:
            print(f"  Cost: ¥{results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
            print(f"  Carbon: {results['mean_carbon']:.2f} ± {results['std_carbon']:.2f} kg")
    
    env.close()
    return pd.DataFrame(all_results)


def evaluate_pareto_models(
    pareto_results: Dict,
    n_episodes: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate Pareto front models."""
    from evaluate_rl import evaluate_controller, create_eval_env
    from stable_baselines3 import SAC
    
    eval_results = []
    env = create_eval_env(cost_weight=0.5, carbon_weight=0.5)  # Neutral evaluation
    
    for name, config in pareto_results.items():
        if verbose:
            print(f"\n[Evaluating Pareto: {name}]")
        
        model = SAC.load(config['model_path'])
        results = evaluate_controller(model, env, n_episodes, verbose=False)
        
        eval_results.append({
            'name': name,
            'lambda_cost': config['lambda_cost'],
            'lambda_carbon': config['lambda_carbon'],
            'cost': results['mean_cost'],
            'carbon': results['mean_carbon'],
            'reward': results['mean_reward'],
            'comfort': results['mean_comfort_violation'],
        })
        
        if verbose:
            print(f"  Cost: ¥{results['mean_cost']:.2f}, Carbon: {results['mean_carbon']:.2f} kg")
    
    env.close()
    return pd.DataFrame(eval_results)


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_paper_figures(
    comparison_df: pd.DataFrame,
    pareto_df: pd.DataFrame = None,
    output_dir: str = "outputs/figures",
):
    """Generate publication-ready figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
    except ImportError:
        print("matplotlib not installed")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Main comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort by reward
    df_sorted = comparison_df.sort_values('mean_reward', ascending=False)
    
    # Cost comparison
    ax1 = axes[0]
    colors = ['#2E86AB' if t == 'RL' else '#A23B72' for t in df_sorted['type']]
    bars = ax1.barh(df_sorted['controller'], df_sorted['mean_cost'], 
                    xerr=df_sorted['std_cost'], color=colors, alpha=0.8, capsize=3)
    ax1.set_xlabel('Electricity Cost (¥/week)')
    ax1.set_title('(a) Cost Comparison')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Carbon comparison
    ax2 = axes[1]
    bars = ax2.barh(df_sorted['controller'], df_sorted['mean_carbon'],
                    xerr=df_sorted['std_carbon'], color=colors, alpha=0.8, capsize=3)
    ax2.set_xlabel('Carbon Emission (kg CO₂/week)')
    ax2.set_title('(b) Carbon Comparison')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', label='RL'),
                       Patch(facecolor='#A23B72', label='Baseline')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(output_path / "fig1_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "fig1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/fig1_comparison.pdf")
    
    # Figure 2: Pareto front
    if pareto_df is not None and len(pareto_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Pareto points with color gradient
        scatter = ax.scatter(
            pareto_df['cost'], pareto_df['carbon'],
            c=pareto_df['lambda_carbon'], cmap='RdYlGn',
            s=200, edgecolors='black', linewidth=1.5, zorder=5
        )
        
        # Connect points
        pareto_sorted = pareto_df.sort_values('cost')
        ax.plot(pareto_sorted['cost'], pareto_sorted['carbon'], 
                'k--', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Annotate
        for _, row in pareto_df.iterrows():
            ax.annotate(
                f"λ={row['lambda_carbon']:.1f}",
                (row['cost'], row['carbon']),
                textcoords="offset points", xytext=(8, 8),
                fontsize=10, alpha=0.8
            )
        
        # Ideal point
        min_cost, min_carbon = pareto_df['cost'].min(), pareto_df['carbon'].min()
        ax.scatter([min_cost], [min_carbon], c='gold', s=300, marker='*',
                   edgecolors='black', linewidth=1.5, zorder=10, label='Ideal Point')
        
        ax.set_xlabel('Electricity Cost (¥/week)')
        ax.set_ylabel('Carbon Emission (kg CO₂/week)')
        ax.set_title('Cost-Carbon Pareto Front')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('λ_carbon (Carbon Weight)')
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / "fig2_pareto.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(output_path / "fig2_pareto.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}/fig2_pareto.pdf")
    
    # Figure 3: Improvement summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement vs rule_based baseline
    baseline_cost = comparison_df[comparison_df['controller'] == 'rule_based']['mean_cost'].values[0]
    baseline_carbon = comparison_df[comparison_df['controller'] == 'rule_based']['mean_carbon'].values[0]
    
    df_imp = comparison_df.copy()
    df_imp['cost_improvement'] = (baseline_cost - df_imp['mean_cost']) / baseline_cost * 100
    df_imp['carbon_improvement'] = (baseline_carbon - df_imp['mean_carbon']) / baseline_carbon * 100
    df_imp = df_imp[df_imp['controller'] != 'rule_based']
    df_imp = df_imp.sort_values('cost_improvement', ascending=True)
    
    x = np.arange(len(df_imp))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, df_imp['cost_improvement'], width, 
                    label='Cost Reduction', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(x + width/2, df_imp['carbon_improvement'], width,
                    label='Carbon Reduction', color='#57A773', alpha=0.8)
    
    ax.set_xlabel('Improvement vs Rule-Based Baseline (%)')
    ax.set_ylabel('Controller')
    ax.set_title('Performance Improvement over Baseline')
    ax.set_yticks(x)
    ax.set_yticklabels(df_imp['controller'])
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "fig3_improvement.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "fig3_improvement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}/fig3_improvement.pdf")


# =============================================================================
# Result Export
# =============================================================================

def export_latex_tables(
    comparison_df: pd.DataFrame,
    pareto_df: pd.DataFrame = None,
    output_dir: str = "outputs/tables",
):
    """Export results as LaTeX tables."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Main comparison
    table1 = comparison_df[['controller', 'type', 'mean_cost', 'std_cost', 
                            'mean_carbon', 'std_carbon']].copy()
    table1['Cost'] = table1.apply(lambda r: f"{r['mean_cost']:.2f} ± {r['std_cost']:.2f}", axis=1)
    table1['Carbon'] = table1.apply(lambda r: f"{r['mean_carbon']:.2f} ± {r['std_carbon']:.2f}", axis=1)
    table1 = table1[['controller', 'type', 'Cost', 'Carbon']]
    table1.columns = ['Controller', 'Type', 'Cost (¥)', 'Carbon (kg)']
    
    latex1 = table1.to_latex(index=False, escape=False, column_format='lccc')
    with open(output_path / "table1_comparison.tex", 'w') as f:
        f.write(latex1)
    print(f"Saved: {output_path}/table1_comparison.tex")
    
    # Table 2: Pareto analysis
    if pareto_df is not None:
        table2 = pareto_df[['name', 'lambda_cost', 'lambda_carbon', 'cost', 'carbon']].copy()
        table2.columns = ['Configuration', 'λ_cost', 'λ_carbon', 'Cost (¥)', 'Carbon (kg)']
        
        latex2 = table2.to_latex(index=False, escape=False, column_format='lcccc')
        with open(output_path / "table2_pareto.tex", 'w') as f:
            f.write(latex2)
        print(f"Saved: {output_path}/table2_pareto.tex")


# =============================================================================
# Main Runner
# =============================================================================

def run_quick_experiment(verbose: bool = True):
    """Run quick experiment for testing."""
    if verbose:
        print("=" * 70)
        print("QUICK EXPERIMENT MODE")
        print("=" * 70)
    
    # Quick training
    from train_rl import train
    train(
        experiment_name="quick_test",
        extractor_type="transformer",
        total_timesteps=5000,
        eval_freq=2500,
        verbose=1 if verbose else 0,
    )
    
    # Quick evaluation
    comparison_df = evaluate_all(n_episodes=5, verbose=verbose)
    
    # Quick Pareto (baselines only)
    from analysis.pareto import ParetoAnalyzer, plot_pareto_analysis
    analyzer = ParetoAnalyzer()
    for _, row in comparison_df.iterrows():
        analyzer.add_point(row['controller'], row['mean_cost'], row['mean_carbon'])
    
    pareto_df = analyzer.to_dataframe()
    
    # Generate outputs
    output_dir = project_root / "outputs" / "quick_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    pareto_df.to_csv(output_dir / "pareto.csv", index=False)
    
    plot_pareto_analysis(analyzer, str(output_dir / "pareto_analysis.png"))
    
    if verbose:
        print("\n" + "=" * 70)
        print("QUICK EXPERIMENT COMPLETED")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)


def run_full_experiment(
    train_models: bool = True,
    train_pareto: bool = True,
    timesteps: int = 500000,
    pareto_timesteps: int = 100000,
    n_episodes: int = 20,
    verbose: bool = True,
):
    """Run full experimental workflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "outputs" / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("FULL EXPERIMENT RUN")
        print(f"Output: {output_dir}")
        print("=" * 70)
    
    model_paths = {}
    
    # Step 1: Train main models
    if train_models:
        if verbose:
            print("\n[Step 1/5] Training main models...")
        
        for exp_key in EXPERIMENTS:
            try:
                path = train_model(exp_key, timesteps=timesteps, verbose=verbose)
                model_paths[EXPERIMENTS[exp_key]['name']] = path
            except Exception as e:
                print(f"Error training {exp_key}: {e}")
    
    # Step 2: Train Pareto models
    pareto_results = None
    if train_pareto:
        if verbose:
            print("\n[Step 2/5] Training Pareto models...")
        
        pareto_results = train_pareto_models(
            timesteps_per_point=pareto_timesteps,
            verbose=verbose
        )
    
    # Step 3: Evaluate all
    if verbose:
        print("\n[Step 3/5] Evaluating all controllers...")
    
    comparison_df = evaluate_all(model_paths, n_episodes=n_episodes, verbose=verbose)
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)
    
    # Step 4: Evaluate Pareto
    pareto_df = None
    if pareto_results:
        if verbose:
            print("\n[Step 4/5] Evaluating Pareto models...")
        
        pareto_df = evaluate_pareto_models(pareto_results, n_episodes=n_episodes, verbose=verbose)
        pareto_df.to_csv(output_dir / "pareto.csv", index=False)
    
    # Step 5: Generate outputs
    if verbose:
        print("\n[Step 5/5] Generating figures and tables...")
    
    generate_paper_figures(comparison_df, pareto_df, str(output_dir / "figures"))
    export_latex_tables(comparison_df, pareto_df, str(output_dir / "tables"))
    
    # Save experiment summary
    summary = {
        'timestamp': timestamp,
        'config': {
            'timesteps': timesteps,
            'pareto_timesteps': pareto_timesteps,
            'n_episodes': n_episodes,
        },
        'models': model_paths,
        'results_summary': {
            'best_by_reward': comparison_df.loc[comparison_df['mean_reward'].idxmax(), 'controller'],
            'best_by_cost': comparison_df.loc[comparison_df['mean_cost'].idxmin(), 'controller'],
            'best_by_carbon': comparison_df.loc[comparison_df['mean_carbon'].idxmin(), 'controller'],
        }
    }
    
    with open(output_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"\nResults saved to: {output_dir}")
        print(f"  - comparison.csv")
        print(f"  - pareto.csv")
        print(f"  - figures/")
        print(f"  - tables/")
        print(f"  - experiment_summary.json")
    
    return comparison_df, pareto_df


def main():
    parser = argparse.ArgumentParser(
        description="Run Carbon-Aware Building Control Experiments"
    )
    
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--train", action="store_true", help="Train main models")
    parser.add_argument("--pareto", action="store_true", help="Train Pareto models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument("--figures", action="store_true", help="Generate figures only")
    parser.add_argument("--timesteps", type=int, default=500000, help="Training timesteps")
    parser.add_argument("--pareto-timesteps", type=int, default=100000, help="Pareto training timesteps")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_experiment()
    elif args.all:
        run_full_experiment(
            train_models=True,
            train_pareto=True,
            timesteps=args.timesteps,
            pareto_timesteps=args.pareto_timesteps,
            n_episodes=args.episodes,
        )
    elif args.evaluate or args.figures:
        # Load existing results if available
        latest_exp = sorted((project_root / "outputs").glob("experiment_*"))
        if latest_exp:
            exp_dir = latest_exp[-1]
            comparison_df = pd.read_csv(exp_dir / "comparison.csv")
            pareto_path = exp_dir / "pareto.csv"
            pareto_df = pd.read_csv(pareto_path) if pareto_path.exists() else None
            
            if args.figures:
                generate_paper_figures(comparison_df, pareto_df, str(exp_dir / "figures"))
            else:
                print(comparison_df.to_string())
        else:
            print("No existing experiment results found. Run with --quick or --all first.")
    else:
        run_full_experiment(
            train_models=args.train,
            train_pareto=args.pareto,
            timesteps=args.timesteps,
            pareto_timesteps=args.pareto_timesteps,
            n_episodes=args.episodes,
        )


if __name__ == "__main__":
    main()
