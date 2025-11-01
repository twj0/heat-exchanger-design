"""
Metrics calculation and performance evaluation.

Provides tools for calculating various performance indicators and
comparing different control strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class MetricsCalculator:
    """Calculate performance metrics for TES-HeatEx system."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = {}
    
    def calculate_cost_metrics(
        self,
        episode_data: List[Dict],
        electricity_price_data: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate cost-related metrics.
        
        Args:
            episode_data: List of step data from episode
            electricity_price_data: Array of electricity prices
            
        Returns:
            Dictionary with cost metrics
        """
        total_cost = sum(step["cost"] for step in episode_data)
        
        # Peak vs off-peak cost breakdown
        peak_cost = 0.0
        offpeak_cost = 0.0
        shoulder_cost = 0.0
        
        for step in episode_data:
            hour = step["hour"]
            cost = step["cost"]
            price = electricity_price_data[hour]
            
            if price > 1.0:  # Peak
                peak_cost += cost
            elif price < 0.5:  # Off-peak
                offpeak_cost += cost
            else:  # Shoulder
                shoulder_cost += cost
        
        return {
            "total_cost": total_cost,
            "peak_cost": peak_cost,
            "offpeak_cost": offpeak_cost,
            "shoulder_cost": shoulder_cost,
            "average_cost_per_hour": total_cost / len(episode_data),
        }
    
    def calculate_energy_metrics(
        self,
        episode_data: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate energy efficiency metrics.
        
        Args:
            episode_data: List of step data
            
        Returns:
            Dictionary with energy metrics
        """
        # Total energy charged and discharged
        total_charged = 0.0
        total_discharged = 0.0
        total_demand = 0.0
        total_delivered = 0.0
        
        for step in episode_data:
            power = step["power_command"]
            demand = step["heat_demand"]
            delivered = step["heat_delivered"]
            
            if power > 0:
                total_charged += power
            elif power < 0:
                total_discharged += abs(power)
            
            total_demand += demand
            total_delivered += delivered
        
        # Calculate efficiency
        storage_efficiency = (
            total_discharged / total_charged if total_charged > 0 else 0.0
        )
        
        demand_satisfaction = (
            total_delivered / total_demand if total_demand > 0 else 0.0
        )
        
        return {
            "total_energy_charged": total_charged,
            "total_energy_discharged": total_discharged,
            "storage_efficiency": storage_efficiency,
            "demand_satisfaction_rate": demand_satisfaction,
            "total_heat_demand": total_demand,
            "total_heat_delivered": total_delivered,
        }
    
    def calculate_temperature_metrics(
        self,
        episode_data: List[Dict],
        min_temp: float = 40.0,
        max_temp: float = 50.0,
    ) -> Dict[str, float]:
        """
        Calculate temperature-related metrics.
        
        Args:
            episode_data: List of step data
            min_temp: Minimum allowed temperature
            max_temp: Maximum allowed temperature
            
        Returns:
            Dictionary with temperature metrics
        """
        temperatures = [step["temperature"] for step in episode_data]
        
        violations = sum(
            1 for t in temperatures if t < min_temp or t > max_temp
        )
        violation_rate = violations / len(temperatures)
        
        # Temperature statistics
        mean_temp = np.mean(temperatures)
        std_temp = np.std(temperatures)
        min_recorded = np.min(temperatures)
        max_recorded = np.max(temperatures)
        
        return {
            "violation_count": violations,
            "violation_rate": violation_rate,
            "mean_temperature": mean_temp,
            "std_temperature": std_temp,
            "min_temperature": min_recorded,
            "max_temperature": max_recorded,
        }
    
    def calculate_storage_metrics(
        self,
        episode_data: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate storage utilization metrics.
        
        Args:
            episode_data: List of step data
            
        Returns:
            Dictionary with storage metrics
        """
        socs = [step["soc"] for step in episode_data]
        
        mean_soc = np.mean(socs)
        utilization = (np.max(socs) - np.min(socs))  # Range of SoC used
        
        # Count cycles (full charge-discharge cycles)
        cycles = 0
        last_peak_soc = socs[0]
        for soc in socs:
            if soc > last_peak_soc:
                last_peak_soc = soc
            elif soc < last_peak_soc - 0.5:  # Significant discharge
                cycles += 0.5
                last_peak_soc = soc
        
        return {
            "mean_soc": mean_soc,
            "min_soc": np.min(socs),
            "max_soc": np.max(socs),
            "soc_utilization": utilization,
            "estimated_cycles": cycles,
        }
    
    def calculate_all_metrics(
        self,
        episode_data: List[Dict],
        electricity_prices: np.ndarray,
    ) -> Dict[str, any]:
        """
        Calculate all available metrics.
        
        Args:
            episode_data: List of step data
            electricity_prices: Array of electricity prices
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        metrics["cost"] = self.calculate_cost_metrics(
            episode_data, electricity_prices
        )
        metrics["energy"] = self.calculate_energy_metrics(episode_data)
        metrics["temperature"] = self.calculate_temperature_metrics(episode_data)
        metrics["storage"] = self.calculate_storage_metrics(episode_data)
        
        # Overall performance score (weighted combination)
        metrics["overall_score"] = (
            -metrics["cost"]["total_cost"] / 1000.0  # Normalize cost
            + metrics["energy"]["demand_satisfaction_rate"] * 100
            - metrics["temperature"]["violation_rate"] * 1000
        )
        
        return metrics


def compare_controllers(
    baseline_data: List[Dict],
    rl_data: List[Dict],
    electricity_prices: np.ndarray,
    save_path: Optional[str] = None,
) -> Dict[str, any]:
    """
    Compare baseline and RL controller performance.
    
    Args:
        baseline_data: Episode data from baseline controller
        rl_data: Episode data from RL controller
        electricity_prices: Electricity price array
        save_path: Path to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    calc = MetricsCalculator()
    
    # Calculate metrics for both controllers
    baseline_metrics = calc.calculate_all_metrics(baseline_data, electricity_prices)
    rl_metrics = calc.calculate_all_metrics(rl_data, electricity_prices)
    
    # Calculate improvements
    cost_savings = (
        baseline_metrics["cost"]["total_cost"] - rl_metrics["cost"]["total_cost"]
    )
    cost_savings_percent = (
        cost_savings / baseline_metrics["cost"]["total_cost"] * 100
        if baseline_metrics["cost"]["total_cost"] > 0 else 0.0
    )
    
    violation_improvement = (
        baseline_metrics["temperature"]["violation_rate"]
        - rl_metrics["temperature"]["violation_rate"]
    )
    
    comparison = {
        "baseline": baseline_metrics,
        "rl": rl_metrics,
        "improvements": {
            "cost_savings_cny": cost_savings,
            "cost_savings_percent": cost_savings_percent,
            "violation_improvement": violation_improvement,
        },
    }
    
    # Save to file if requested
    if save_path:
        save_comparison_report(comparison, save_path)
    
    return comparison


def save_comparison_report(
    comparison: Dict,
    save_path: str,
) -> None:
    """
    Save comparison results to file.
    
    Args:
        comparison: Comparison dictionary
        save_path: Path to save report
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Controller Comparison Report\n\n")
        
        f.write("## Cost Metrics\n")
        f.write(f"Baseline Total Cost: {comparison['baseline']['cost']['total_cost']:.2f} CNY\n")
        f.write(f"RL Total Cost: {comparison['rl']['cost']['total_cost']:.2f} CNY\n")
        f.write(f"Cost Savings: {comparison['improvements']['cost_savings_cny']:.2f} CNY "
                f"({comparison['improvements']['cost_savings_percent']:.1f}%)\n\n")
        
        f.write("## Energy Metrics\n")
        f.write(f"Baseline Demand Satisfaction: "
                f"{comparison['baseline']['energy']['demand_satisfaction_rate']:.2%}\n")
        f.write(f"RL Demand Satisfaction: "
                f"{comparison['rl']['energy']['demand_satisfaction_rate']:.2%}\n\n")
        
        f.write("## Temperature Metrics\n")
        f.write(f"Baseline Violation Rate: "
                f"{comparison['baseline']['temperature']['violation_rate']:.2%}\n")
        f.write(f"RL Violation Rate: "
                f"{comparison['rl']['temperature']['violation_rate']:.2%}\n")
        f.write(f"Improvement: {comparison['improvements']['violation_improvement']:.2%}\n\n")
        
        f.write("## Storage Utilization\n")
        f.write(f"Baseline SoC Utilization: "
                f"{comparison['baseline']['storage']['soc_utilization']:.2f}\n")
        f.write(f"RL SoC Utilization: "
                f"{comparison['rl']['storage']['soc_utilization']:.2f}\n")


def plot_comparison(
    baseline_data: List[Dict],
    rl_data: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Create comparison plots.
    
    Args:
        baseline_data: Baseline controller episode data
        rl_data: RL controller episode data
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Baseline vs RL Controller Comparison", fontsize=16)
    
    hours = range(len(baseline_data))
    
    # Temperature
    axes[0, 0].plot(hours, [d["temperature"] for d in baseline_data], 
                    label="Baseline", alpha=0.7)
    axes[0, 0].plot(hours, [d["temperature"] for d in rl_data], 
                    label="RL", alpha=0.7)
    axes[0, 0].axhline(40, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(50, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].set_title("Storage Temperature")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SoC
    axes[0, 1].plot(hours, [d["soc"] for d in baseline_data], 
                    label="Baseline", alpha=0.7)
    axes[0, 1].plot(hours, [d["soc"] for d in rl_data], 
                    label="RL", alpha=0.7)
    axes[0, 1].set_ylabel("State of Charge")
    axes[0, 1].set_title("Storage SoC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power commands
    axes[1, 0].plot(hours, [d["power_command"] for d in baseline_data], 
                    label="Baseline", alpha=0.7)
    axes[1, 0].plot(hours, [d["power_command"] for d in rl_data], 
                    label="RL", alpha=0.7)
    axes[1, 0].set_ylabel("Power (kW)")
    axes[1, 0].set_title("Charging/Discharging Power")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative cost
    baseline_cumcost = np.cumsum([d["cost"] for d in baseline_data])
    rl_cumcost = np.cumsum([d["cost"] for d in rl_data])
    axes[1, 1].plot(hours, baseline_cumcost, label="Baseline", alpha=0.7)
    axes[1, 1].plot(hours, rl_cumcost, label="RL", alpha=0.7)
    axes[1, 1].set_ylabel("Cumulative Cost (CNY)")
    axes[1, 1].set_title("Cumulative Operational Cost")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Heat demand vs delivered
    axes[2, 0].plot(hours, [d["heat_demand"] for d in baseline_data], 
                    label="Demand", color='black', alpha=0.5)
    axes[2, 0].plot(hours, [d["heat_delivered"] for d in baseline_data], 
                    label="Baseline Delivered", alpha=0.7)
    axes[2, 0].plot(hours, [d["heat_delivered"] for d in rl_data], 
                    label="RL Delivered", alpha=0.7)
    axes[2, 0].set_ylabel("Heat (kW)")
    axes[2, 0].set_xlabel("Hour")
    axes[2, 0].set_title("Heat Demand vs Delivered")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Electricity price overlay
    axes[2, 1].plot(hours, [d["electricity_price"] for d in baseline_data], 
                    color='orange', linewidth=2)
    axes[2, 1].set_ylabel("Price (CNY/kWh)")
    axes[2, 1].set_xlabel("Hour")
    axes[2, 1].set_title("Electricity Price Profile")
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """Example usage of metrics calculator."""
    print("Metrics calculator module loaded successfully!")
    print("Use MetricsCalculator class to calculate performance metrics.")
