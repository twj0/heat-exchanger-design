"""
Baseline Controllers for Comparison.

This module provides baseline control strategies for comparing 
with the RL-based carbon-aware controller.

Available baselines:
    - RuleBasedController: Fixed setpoint + time-of-use scheduling
    - RandomController: Random actions (lower bound)
"""

from baselines.rule_based import RuleBasedController, RandomController

__all__ = ["RuleBasedController", "RandomController"]
