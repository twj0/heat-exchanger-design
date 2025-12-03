"""
Pareto Front Analysis Module for Multi-Objective Optimization.

This module provides tools for analyzing the cost-carbon trade-off
in building energy control systems.

Features:
    - Pareto dominance checking
    - Non-dominated sorting
    - Hypervolume calculation
    - Interactive visualization

Mathematical Background:
    A solution x dominates y if:
    - f_cost(x) ≤ f_cost(y) AND f_carbon(x) ≤ f_carbon(y)
    - At least one inequality is strict

Usage:
    from analysis.pareto import ParetoAnalyzer
    
    analyzer = ParetoAnalyzer()
    analyzer.add_point('RL', cost=100, carbon=50)
    analyzer.add_point('Baseline', cost=120, carbon=45)
    pareto_front = analyzer.get_pareto_front()

Author: Auto-generated for Applied Energy publication
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ParetoPoint:
    """A single point in objective space."""
    name: str
    cost: float
    carbon: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominates(self, other: 'ParetoPoint') -> bool:
        """Check if this point dominates another."""
        better_or_equal = (self.cost <= other.cost) and (self.carbon <= other.carbon)
        strictly_better = (self.cost < other.cost) or (self.carbon < other.carbon)
        return better_or_equal and strictly_better
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [cost, carbon]."""
        return np.array([self.cost, self.carbon])


class ParetoAnalyzer:
    """
    Analyzer for Pareto front computation and visualization.
    
    Attributes:
        points: List of ParetoPoint objects
        reference_point: Reference point for hypervolume calculation
    """
    
    def __init__(self, reference_point: Optional[Tuple[float, float]] = None):
        """
        Initialize analyzer.
        
        Args:
            reference_point: Reference point for hypervolume (max_cost, max_carbon)
        """
        self.points: List[ParetoPoint] = []
        self.reference_point = reference_point
    
    def add_point(
        self,
        name: str,
        cost: float,
        carbon: float,
        **metadata
    ) -> None:
        """Add a point to the analysis."""
        point = ParetoPoint(name=name, cost=cost, carbon=carbon, metadata=metadata)
        self.points.append(point)
    
    def add_points_from_df(self, df: pd.DataFrame) -> None:
        """
        Add points from DataFrame.
        
        DataFrame should have columns: name/controller, cost/mean_cost, carbon/mean_carbon
        """
        for _, row in df.iterrows():
            name = row.get('name') or row.get('controller', 'Unknown')
            cost = row.get('cost') or row.get('mean_cost', 0)
            carbon = row.get('carbon') or row.get('mean_carbon', 0)
            
            metadata = {k: v for k, v in row.items() 
                       if k not in ['name', 'controller', 'cost', 'mean_cost', 'carbon', 'mean_carbon']}
            
            self.add_point(name, cost, carbon, **metadata)
    
    def get_pareto_front(self) -> List[ParetoPoint]:
        """
        Get the Pareto-optimal (non-dominated) points.
        
        Returns:
            List of non-dominated ParetoPoint objects
        """
        pareto_front = []
        
        for candidate in self.points:
            is_dominated = False
            
            for other in self.points:
                if other != candidate and other.dominates(candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        # Sort by cost for visualization
        pareto_front.sort(key=lambda p: p.cost)
        
        return pareto_front
    
    def non_dominated_sort(self) -> List[List[ParetoPoint]]:
        """
        Perform non-dominated sorting (NSGA-II style).
        
        Returns:
            List of fronts, where fronts[0] is the Pareto front
        """
        remaining = self.points.copy()
        fronts = []
        
        while remaining:
            # Find current front
            current_front = []
            for candidate in remaining:
                is_dominated = False
                for other in remaining:
                    if other != candidate and other.dominates(candidate):
                        is_dominated = True
                        break
                if not is_dominated:
                    current_front.append(candidate)
            
            fronts.append(current_front)
            remaining = [p for p in remaining if p not in current_front]
        
        return fronts
    
    def calculate_hypervolume(self, reference: Optional[Tuple[float, float]] = None) -> float:
        """
        Calculate hypervolume indicator.
        
        The hypervolume is the area dominated by the Pareto front
        and bounded by the reference point.
        
        Args:
            reference: Reference point (max_cost, max_carbon)
            
        Returns:
            Hypervolume value
        """
        ref = reference or self.reference_point
        if ref is None:
            # Auto-determine reference point
            max_cost = max(p.cost for p in self.points) * 1.1
            max_carbon = max(p.carbon for p in self.points) * 1.1
            ref = (max_cost, max_carbon)
        
        pareto_front = self.get_pareto_front()
        if not pareto_front:
            return 0.0
        
        # Sort by cost
        sorted_front = sorted(pareto_front, key=lambda p: p.cost)
        
        # Calculate hypervolume using inclusion-exclusion
        hv = 0.0
        prev_cost = 0.0
        
        for point in sorted_front:
            if point.cost <= ref[0] and point.carbon <= ref[1]:
                width = point.cost - prev_cost
                height = ref[1] - point.carbon
                hv += width * height
                prev_cost = point.cost
        
        # Add final rectangle
        if prev_cost < ref[0]:
            # Find minimum carbon on front
            min_carbon = min(p.carbon for p in sorted_front)
            hv += (ref[0] - prev_cost) * (ref[1] - min_carbon)
        
        return hv
    
    def calculate_improvement(
        self,
        point_name: str,
        baseline_name: str = 'rule_based'
    ) -> Dict[str, float]:
        """
        Calculate improvement of one point over baseline.
        
        Args:
            point_name: Name of point to evaluate
            baseline_name: Name of baseline point
            
        Returns:
            Dictionary with cost_improvement, carbon_improvement percentages
        """
        point = next((p for p in self.points if p.name == point_name), None)
        baseline = next((p for p in self.points if p.name == baseline_name), None)
        
        if point is None or baseline is None:
            return {'cost_improvement': 0.0, 'carbon_improvement': 0.0}
        
        cost_imp = (baseline.cost - point.cost) / baseline.cost * 100
        carbon_imp = (baseline.carbon - point.carbon) / baseline.carbon * 100
        
        return {
            'cost_improvement': cost_imp,
            'carbon_improvement': carbon_imp,
            'cost_saving': baseline.cost - point.cost,
            'carbon_saving': baseline.carbon - point.carbon,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all points to DataFrame."""
        data = []
        pareto_front = self.get_pareto_front()
        pareto_names = {p.name for p in pareto_front}
        
        for point in self.points:
            row = {
                'name': point.name,
                'cost': point.cost,
                'carbon': point.carbon,
                'is_pareto': point.name in pareto_names,
                **point.metadata
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_ideal_point(self) -> Tuple[float, float]:
        """Get ideal (utopia) point - minimum of each objective."""
        min_cost = min(p.cost for p in self.points)
        min_carbon = min(p.carbon for p in self.points)
        return (min_cost, min_carbon)
    
    def get_nadir_point(self) -> Tuple[float, float]:
        """Get nadir point - worst values on Pareto front."""
        pareto_front = self.get_pareto_front()
        if not pareto_front:
            return (0, 0)
        
        max_cost = max(p.cost for p in pareto_front)
        max_carbon = max(p.carbon for p in pareto_front)
        return (max_cost, max_carbon)
    
    def find_knee_point(self) -> Optional[ParetoPoint]:
        """
        Find the knee point of the Pareto front.
        
        The knee point is the point with maximum distance from
        the line connecting ideal and nadir points.
        
        Returns:
            Knee point or None if less than 3 points
        """
        pareto_front = self.get_pareto_front()
        if len(pareto_front) < 3:
            return pareto_front[0] if pareto_front else None
        
        ideal = self.get_ideal_point()
        nadir = self.get_nadir_point()
        
        # Line from ideal to nadir: ax + by + c = 0
        a = nadir[1] - ideal[1]
        b = ideal[0] - nadir[0]
        c = nadir[0] * ideal[1] - ideal[0] * nadir[1]
        
        # Find point with maximum distance
        max_dist = -1
        knee_point = None
        
        for point in pareto_front:
            dist = abs(a * point.cost + b * point.carbon + c) / np.sqrt(a**2 + b**2)
            if dist > max_dist:
                max_dist = dist
                knee_point = point
        
        return knee_point


def plot_pareto_analysis(
    analyzer: ParetoAnalyzer,
    save_path: Optional[str] = None,
    show_improvement: bool = True,
    title: str = "Pareto Front Analysis",
):
    """
    Generate comprehensive Pareto analysis plot.
    
    Args:
        analyzer: ParetoAnalyzer instance
        save_path: Path to save figure
        show_improvement: Show improvement percentages
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df = analyzer.to_dataframe()
    pareto_front = analyzer.get_pareto_front()
    ideal = analyzer.get_ideal_point()
    
    # Left: Scatter plot with Pareto front
    ax1 = axes[0]
    
    # Plot all points
    non_pareto = df[~df['is_pareto']]
    pareto = df[df['is_pareto']]
    
    ax1.scatter(non_pareto['cost'], non_pareto['carbon'], 
                c='gray', s=100, alpha=0.5, label='Dominated')
    ax1.scatter(pareto['cost'], pareto['carbon'],
                c='red', s=150, marker='s', label='Pareto Optimal', zorder=5)
    
    # Connect Pareto front
    pareto_sorted = pareto.sort_values('cost')
    ax1.plot(pareto_sorted['cost'], pareto_sorted['carbon'], 
             'r--', alpha=0.5, linewidth=2)
    
    # Mark ideal point
    ax1.scatter([ideal[0]], [ideal[1]], c='gold', s=300, marker='*',
                label='Ideal Point', zorder=10, edgecolors='black')
    
    # Mark knee point
    knee = analyzer.find_knee_point()
    if knee:
        ax1.scatter([knee.cost], [knee.carbon], c='blue', s=200, marker='D',
                    label=f'Knee: {knee.name}', zorder=8, edgecolors='black')
    
    # Annotate points
    for _, row in df.iterrows():
        ax1.annotate(row['name'], (row['cost'], row['carbon']),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax1.set_xlabel('Electricity Cost (¥)', fontsize=12)
    ax1.set_ylabel('Carbon Emission (kg CO₂)', fontsize=12)
    ax1.set_title('Cost-Carbon Trade-off', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right: Improvement bar chart
    ax2 = axes[1]
    
    # Calculate improvements relative to worst
    worst_cost = df['cost'].max()
    worst_carbon = df['carbon'].max()
    
    names = df['name'].tolist()
    cost_savings = [(worst_cost - c) / worst_cost * 100 for c in df['cost']]
    carbon_savings = [(worst_carbon - c) / worst_carbon * 100 for c in df['carbon']]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, cost_savings, width, label='Cost Reduction %', color='steelblue')
    bars2 = ax2.bar(x + width/2, carbon_savings, width, label='Carbon Reduction %', color='forestgreen')
    
    ax2.set_xlabel('Controller')
    ax2.set_ylabel('Reduction vs Worst (%)')
    ax2.set_title('Performance Improvement', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.close()


def generate_pareto_report(analyzer: ParetoAnalyzer) -> str:
    """
    Generate a text report of Pareto analysis.
    
    Args:
        analyzer: ParetoAnalyzer instance
        
    Returns:
        Formatted report string
    """
    df = analyzer.to_dataframe()
    pareto_front = analyzer.get_pareto_front()
    ideal = analyzer.get_ideal_point()
    nadir = analyzer.get_nadir_point()
    knee = analyzer.find_knee_point()
    hv = analyzer.calculate_hypervolume()
    
    report = []
    report.append("=" * 60)
    report.append("PARETO FRONT ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary
    report.append("[Summary]")
    report.append(f"  Total points: {len(analyzer.points)}")
    report.append(f"  Pareto-optimal points: {len(pareto_front)}")
    report.append(f"  Hypervolume: {hv:.2f}")
    report.append("")
    
    # Key points
    report.append("[Key Points]")
    report.append(f"  Ideal point: Cost=¥{ideal[0]:.2f}, Carbon={ideal[1]:.2f}kg")
    report.append(f"  Nadir point: Cost=¥{nadir[0]:.2f}, Carbon={nadir[1]:.2f}kg")
    if knee:
        report.append(f"  Knee point: {knee.name} (Cost=¥{knee.cost:.2f}, Carbon={knee.carbon:.2f}kg)")
    report.append("")
    
    # Pareto front
    report.append("[Pareto Front]")
    for i, p in enumerate(pareto_front, 1):
        report.append(f"  {i}. {p.name}: Cost=¥{p.cost:.2f}, Carbon={p.carbon:.2f}kg")
    report.append("")
    
    # Rankings
    report.append("[Rankings]")
    by_cost = df.sort_values('cost')
    by_carbon = df.sort_values('carbon')
    
    report.append("  Best Cost:")
    for i, (_, row) in enumerate(by_cost.head(3).iterrows(), 1):
        report.append(f"    {i}. {row['name']}: ¥{row['cost']:.2f}")
    
    report.append("  Best Carbon:")
    for i, (_, row) in enumerate(by_carbon.head(3).iterrows(), 1):
        report.append(f"    {i}. {row['name']}: {row['carbon']:.2f}kg")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


# =============================================================================
# Main: Test the module
# =============================================================================

if __name__ == "__main__":
    print("Pareto Analysis Module Test")
    print("=" * 50)
    
    # Create test analyzer
    analyzer = ParetoAnalyzer()
    
    # Add sample points
    test_points = [
        ("SAC-Transformer", 18.5, 12.3),
        ("rule_based", 55.0, 34.0),
        ("always_on", 22.0, 15.5),
        ("carbon_aware_rule", 21.7, 15.2),
        ("random", 21.7, 15.2),
        ("MLP-SAC", 20.0, 14.0),
    ]
    
    for name, cost, carbon in test_points:
        analyzer.add_point(name, cost, carbon)
    
    # Get Pareto front
    pareto_front = analyzer.get_pareto_front()
    print(f"\nPareto Front ({len(pareto_front)} points):")
    for p in pareto_front:
        print(f"  {p.name}: Cost=¥{p.cost:.2f}, Carbon={p.carbon:.2f}kg")
    
    # Ideal and knee points
    ideal = analyzer.get_ideal_point()
    knee = analyzer.find_knee_point()
    print(f"\nIdeal Point: Cost=¥{ideal[0]:.2f}, Carbon={ideal[1]:.2f}kg")
    if knee:
        print(f"Knee Point: {knee.name}")
    
    # Hypervolume
    hv = analyzer.calculate_hypervolume()
    print(f"\nHypervolume: {hv:.2f}")
    
    # Improvement calculation
    if 'SAC-Transformer' in [p.name for p in analyzer.points]:
        imp = analyzer.calculate_improvement('SAC-Transformer', 'rule_based')
        print(f"\nSAC-Transformer vs rule_based:")
        print(f"  Cost improvement: {imp['cost_improvement']:.1f}%")
        print(f"  Carbon improvement: {imp['carbon_improvement']:.1f}%")
    
    # Generate report
    report = generate_pareto_report(analyzer)
    print("\n" + report)
    
    # Save plot
    plot_pareto_analysis(analyzer, "outputs/results/pareto_test.png")
    
    print("\n✅ Pareto module test completed!")
