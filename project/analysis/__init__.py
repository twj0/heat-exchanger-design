"""
Analysis module for Carbon-Aware Building Control.

Provides tools for:
- Pareto front analysis
- Multi-objective optimization
- Result visualization
"""

from analysis.pareto import (
    ParetoPoint,
    ParetoAnalyzer,
    plot_pareto_analysis,
    generate_pareto_report,
)

__all__ = [
    'ParetoPoint',
    'ParetoAnalyzer',
    'plot_pareto_analysis',
    'generate_pareto_report',
]
