"""
Visualization Tools for Carbon-Aware Building Control.

Generates publication-quality figures for:
- Training curves
- Performance comparisons
- Control strategies
- Ablation studies

Run from project root:
    python -m scripts.visualize --results outputs/results/eval_xxx
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Matplotlib setup for publication quality
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_comparison_bar(
    results: Dict,
    output_path: str,
    metric: str = 'total_cost_CNY',
    title: str = 'Cost Comparison',
    ylabel: str = 'Cost (CNY)',
):
    """
    Create bar chart comparing controllers.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save figure
        metric: Metric to plot
        title: Figure title
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    means = [results[n][f'mean_{metric}'] for n in names]
    stds = [results[n][f'std_{metric}'] for n in names]
    
    # Color based on controller type
    colors = []
    for name in names:
        if 'sac' in name.lower() or 'transformer' in name.lower():
            colors.append('#2ecc71')  # Green for RL
        elif 'rule' in name.lower():
            colors.append('#3498db')  # Blue for rule-based
        else:
            colors.append('#95a5a6')  # Gray for others
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_multi_metric_comparison(
    results: Dict,
    output_path: str,
):
    """
    Create grouped bar chart comparing multiple metrics.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    
    metrics = [
        ('total_cost_CNY', 'Cost (CNY)', 'Cost Comparison'),
        ('total_carbon_kgCO2', 'Carbon (kgCO2)', 'Carbon Emissions'),
        ('violation_rate', 'Violation Rate (%)', 'Comfort Violations'),
    ]
    
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        means = [results[n][f'mean_{metric}'] for n in names]
        stds = [results[n][f'std_{metric}'] for n in names]
        
        # Scale violation rate to percentage
        if 'rate' in metric:
            means = [m * 100 for m in means]
            stds = [s * 100 for s in stds]
        
        colors = ['#2ecc71' if 'sac' in n.lower() else '#3498db' for n in names]
        
        bars = ax.bar(names, means, yerr=stds, capsize=3, color=colors, edgecolor='black')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_training_curve(
    log_path: str,
    output_path: str,
):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_path: Path to TensorBoard log directory
        output_path: Path to save figure
    """
    # Try to read TensorBoard logs
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        
        # Get reward data
        if 'rollout/ep_rew_mean' in ea.scalars.Keys():
            rewards = ea.scalars.Items('rollout/ep_rew_mean')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, values, 'b-', linewidth=1.5, alpha=0.7)
            
            # Smooth curve
            window = min(50, len(values) // 10)
            if window > 1:
                smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                ax.plot(steps, smoothed, 'r-', linewidth=2, label='Smoothed')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Episode Reward')
            ax.set_title('Training Reward Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"  Saved: {output_path}")
        else:
            print(f"  Warning: No reward data found in {log_path}")
            
    except ImportError:
        print("  Warning: tensorboard not available for log parsing")
    except Exception as e:
        print(f"  Warning: Could not parse logs: {e}")


def plot_control_strategy(
    results: Dict,
    output_path: str,
    controller_name: str = 'transformer_sac',
):
    """
    Visualize control strategy over time.
    
    Args:
        results: Results dictionary with action data
        output_path: Path to save figure
        controller_name: Controller to visualize
    """
    if controller_name not in results:
        print(f"  Warning: {controller_name} not in results")
        return
    
    # Check if we have detailed action data
    metrics = results[controller_name].get('metrics', [])
    if not metrics:
        print(f"  Warning: No detailed metrics for {controller_name}")
        return
    
    # For now, create a placeholder visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot step costs
    step_costs = metrics[0].get('step_costs', []) if isinstance(metrics[0], dict) else []
    if step_costs:
        axes[0].plot(step_costs, 'b-', alpha=0.7)
        axes[0].set_ylabel('Step Cost (CNY)')
        axes[0].set_title('Cost over Episode')
    
    # Plot carbon
    step_carbons = metrics[0].get('step_carbons', []) if isinstance(metrics[0], dict) else []
    if step_carbons:
        axes[1].plot(step_carbons, 'g-', alpha=0.7)
        axes[1].set_ylabel('Step Carbon (kgCO2)')
        axes[1].set_xlabel('Step')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_radar_comparison(
    results: Dict,
    output_path: str,
):
    """
    Create radar/spider chart comparing controllers.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save figure
    """
    # Metrics to include (normalized)
    metrics = ['total_cost_CNY', 'total_carbon_kgCO2', 'violation_rate', 'total_reward']
    labels = ['Cost', 'Carbon', 'Violations', 'Reward']
    
    # Normalize metrics (lower is better for cost/carbon/violations, higher for reward)
    normalized_data = {}
    
    for metric in metrics:
        values = [results[n][f'mean_{metric}'] for n in results.keys()]
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            normalized_data[metric] = {n: 0.5 for n in results.keys()}
        else:
            if metric == 'total_reward':
                # Higher is better
                normalized_data[metric] = {
                    n: (results[n][f'mean_{metric}'] - min_val) / (max_val - min_val)
                    for n in results.keys()
                }
            else:
                # Lower is better
                normalized_data[metric] = {
                    n: 1 - (results[n][f'mean_{metric}'] - min_val) / (max_val - min_val)
                    for n in results.keys()
                }
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (name, _), color in zip(results.items(), colors):
        values = [normalized_data[m][name] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Controller Performance Comparison', y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_figures(
    results_path: str,
    output_dir: str,
):
    """
    Generate all figures from evaluation results.
    
    Args:
        results_path: Path to detailed_results.json
        output_dir: Directory to save figures
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION FIGURES")
    print("="*60)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("\n[1] Cost comparison bar chart...")
    plot_comparison_bar(
        results,
        str(output_path / 'fig_cost_comparison.png'),
        metric='total_cost_CNY',
        title='Energy Cost Comparison',
        ylabel='Total Cost (CNY)'
    )
    
    print("\n[2] Carbon comparison bar chart...")
    plot_comparison_bar(
        results,
        str(output_path / 'fig_carbon_comparison.png'),
        metric='total_carbon_kgCO2',
        title='Carbon Emissions Comparison',
        ylabel='Total Carbon (kgCO2)'
    )
    
    print("\n[3] Multi-metric comparison...")
    plot_multi_metric_comparison(
        results,
        str(output_path / 'fig_multi_metric.png')
    )
    
    print("\n[4] Radar comparison chart...")
    plot_radar_comparison(
        results,
        str(output_path / 'fig_radar_comparison.png')
    )
    
    print(f"\n✅ All figures saved to: {output_dir}")


def main():
    """Main visualization entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for evaluation results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory containing detailed_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for figures (default: same as results)",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Path to TensorBoard logs for training curves",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = args.output or str(results_dir / 'figures')
    
    # Check for results file
    results_file = results_dir / 'detailed_results.json'
    if results_file.exists():
        generate_all_figures(str(results_file), output_dir)
    else:
        print(f"Error: Results file not found: {results_file}")
        return
    
    # Generate training curves if logs provided
    if args.logs:
        print("\n[5] Training curves...")
        plot_training_curve(
            args.logs,
            str(Path(output_dir) / 'fig_training_curve.png')
        )


if __name__ == "__main__":
    main()
