"""
Rule-based baseline controllers.

These controllers serve as benchmarks for RL-based approaches.
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class RuleBasedController(ABC):
    """Abstract base class for rule-based controllers."""
    
    @abstractmethod
    def select_action(self, observation: Dict, info: Dict) -> int:
        """
        Select action based on current state.
        
        Args:
            observation: Environment observation
            info: Additional information
            
        Returns:
            Action to take
        """
        pass
    
    def reset(self) -> None:
        """Reset controller state."""
        pass


class SimpleTOUController(RuleBasedController):
    """
    Simple Time-of-Use based controller.
    
    Strategy:
    - Charge during off-peak hours
    - Discharge during peak hours
    - Idle during shoulder hours (or based on SoC)
    """
    
    def __init__(
        self,
        charge_in_offpeak: bool = True,
        discharge_in_peak: bool = True,
        soc_high_threshold: float = 0.8,
        soc_low_threshold: float = 0.3,
        temperature_hysteresis: float = 2.0,
    ):
        """
        Initialize simple TOU controller.
        
        Args:
            charge_in_offpeak: Whether to charge during off-peak
            discharge_in_peak: Whether to discharge during peak
            soc_high_threshold: SoC threshold to stop charging
            soc_low_threshold: SoC threshold to stop discharging
            temperature_hysteresis: Temperature buffer for control (°C)
        """
        self.charge_in_offpeak = charge_in_offpeak
        self.discharge_in_peak = discharge_in_peak
        self.soc_high = soc_high_threshold
        self.soc_low = soc_low_threshold
        self.hysteresis = temperature_hysteresis
        
        self.last_action = 0  # Idle
    
    def select_action(self, observation: Dict, info: Dict) -> int:
        """
        Select action based on TOU period and storage state.
        
        Action mapping:
        - 0: Idle
        - 1: Charge
        - 2: Discharge
        
        Args:
            observation: Not directly used (using info instead)
            info: Environment info containing period, SoC, etc.
            
        Returns:
            Action index
        """
        period = info["price_period"]
        soc = info["soc"]
        temperature = info["temperature"]
        
        # Get temperature bounds (if available)
        # For now, use heuristic bounds
        temp_min = 40.0
        temp_max = 50.0
        
        # Default action
        action = 0  # Idle
        
        # Off-peak period: charge if SoC is low
        if period == "offpeak" and self.charge_in_offpeak:
            if soc < self.soc_high and temperature < temp_max - self.hysteresis:
                action = 1  # Charge
        
        # Peak period: discharge if SoC is high
        elif period == "peak" and self.discharge_in_peak:
            if soc > self.soc_low and temperature > temp_min + self.hysteresis:
                action = 2  # Discharge
        
        # Shoulder period: maintain or adjust based on SoC
        else:
            if soc < self.soc_low and temperature < temp_max - self.hysteresis:
                action = 1  # Charge to prevent running out
            elif soc > self.soc_high:
                action = 0  # Idle, let it stabilize
        
        self.last_action = action
        return action
    
    def reset(self) -> None:
        """Reset controller state."""
        self.last_action = 0


class PredictiveController(RuleBasedController):
    """
    Predictive rule-based controller with lookahead.
    
    Uses price forecast to make more informed decisions.
    """
    
    def __init__(
        self,
        lookahead_hours: int = 6,
        soc_target_peak: float = 0.7,
        soc_target_offpeak: float = 0.5,
    ):
        """
        Initialize predictive controller.
        
        Args:
            lookahead_hours: Hours to look ahead in price forecast
            soc_target_peak: Target SoC before peak period
            soc_target_offpeak: Target SoC during off-peak
        """
        self.lookahead_hours = lookahead_hours
        self.soc_target_peak = soc_target_peak
        self.soc_target_offpeak = soc_target_offpeak
    
    def select_action(self, observation: Dict, info: Dict) -> int:
        """
        Select action with predictive strategy.
        
        Args:
            observation: Environment observation (may contain price forecast)
            info: Environment info
            
        Returns:
            Action index
        """
        period = info["price_period"]
        soc = info["soc"]
        hour = info["hour"]
        
        # Simple heuristic: prepare for peak by charging beforehand
        # This is a placeholder for more sophisticated predictive logic
        
        # Hours before typical peak (18-21)
        hours_to_peak = (18 - hour) % 24
        
        if hours_to_peak <= 4 and soc < self.soc_target_peak:
            # Charge in preparation for peak
            return 1
        elif period == "peak" and soc > 0.3:
            # Discharge during peak
            return 2
        elif period == "offpeak" and soc < 0.8:
            # Charge during off-peak
            return 1
        else:
            # Idle
            return 0


def run_baseline_evaluation(env, controller: RuleBasedController, n_episodes: int = 1):
    """
    Evaluate a baseline controller on the environment.
    
    Args:
        env: Gym environment
        controller: Baseline controller instance
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        controller.reset()
        
        episode_reward = 0.0
        episode_cost = 0.0
        temperature_violations = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from controller
            action = controller.select_action(obs, info)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info.get("cost", 0.0)
            
            # Check violations
            if not (40.0 <= info["temperature"] <= 50.0):
                temperature_violations += 1
        
        episode_rewards.append(episode_reward)
        episode_costs.append(env.economic.total_cost)
        episode_violations.append(temperature_violations)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "mean_cost": np.mean(episode_costs),
        "mean_violations": np.mean(episode_violations),
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
    }


if __name__ == "__main__":
    """
    Example usage of baseline controllers.
    """
    import yaml
    import sys
    sys.path.append('..')
    from env.tes_heatex_env import TESHeatExEnv
    
    # Load configuration
    with open("../configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = TESHeatExEnv(config)
    
    # Create baseline controller
    controller = SimpleTOUController(
        charge_in_offpeak=config["baseline"]["charge_in_offpeak"],
        discharge_in_peak=config["baseline"]["discharge_in_peak"],
        temperature_hysteresis=config["baseline"]["temperature_hysteresis"],
    )
    
    # Run evaluation
    print("Evaluating baseline controller...")
    results = run_baseline_evaluation(env, controller, n_episodes=1)
    
    print(f"\nResults:")
    print(f"  Mean reward: {results['mean_reward']:.2f}")
    print(f"  Total cost: {results['mean_cost']:.2f} CNY")
    print(f"  Temperature violations: {results['mean_violations']:.0f} steps")
    
    print("\nBaseline evaluation complete!")
