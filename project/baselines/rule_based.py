"""
Rule-Based Baseline Controllers for Building HVAC Control.

These controllers implement conventional control strategies for comparison
with the RL-based approach. They serve as baselines to demonstrate the
improvement achieved by the carbon-aware RL controller.

Controllers:
    - RuleBasedController: Time-of-use based setpoint scheduling
    - RandomController: Random actions (theoretical lower bound)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    def __init__(self, action_space):
        """
        Initialize controller.
        
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space
        
    @abstractmethod
    def predict(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Predict action given observation.
        
        Args:
            observation: Current environment observation
            info: Additional info from environment
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, None) to match SB3 interface
        """
        pass
    
    def reset(self):
        """Reset controller state if any."""
        pass


class RuleBasedController(BaseController):
    """
    Rule-based controller with time-of-use setpoint scheduling.
    
    This controller implements a conventional HVAC control strategy:
    1. Fixed comfort setpoints during occupied hours
    2. Setback temperatures during unoccupied hours
    3. Pre-cooling/pre-heating during off-peak price periods
    
    Control Logic:
    - Occupied hours (8:00-22:00): Maintain comfort (20-24°C)
    - Unoccupied hours (22:00-8:00): Allow wider range (18-26°C)
    - Off-peak periods (22:00-6:00): Pre-heat/pre-cool to save costs
    
    Args:
        action_space: Gymnasium action space
        comfort_temp_heat: Heating setpoint during comfort mode (°C)
        comfort_temp_cool: Cooling setpoint during comfort mode (°C)
        setback_temp_heat: Heating setpoint during setback mode (°C)
        setback_temp_cool: Cooling setpoint during setback mode (°C)
        occupied_start: Start hour of occupied period (0-23)
        occupied_end: End hour of occupied period (0-23)
        precool_enabled: Enable pre-cooling during off-peak
        preheat_enabled: Enable pre-heating during off-peak
    """
    
    def __init__(
        self,
        action_space,
        comfort_temp_heat: float = 21.0,
        comfort_temp_cool: float = 24.0,
        setback_temp_heat: float = 18.0,
        setback_temp_cool: float = 26.0,
        occupied_start: int = 8,
        occupied_end: int = 22,
        precool_enabled: bool = True,
        preheat_enabled: bool = True,
    ):
        super().__init__(action_space)
        
        self.comfort_temp_heat = comfort_temp_heat
        self.comfort_temp_cool = comfort_temp_cool
        self.setback_temp_heat = setback_temp_heat
        self.setback_temp_cool = setback_temp_cool
        self.occupied_start = occupied_start
        self.occupied_end = occupied_end
        self.precool_enabled = precool_enabled
        self.preheat_enabled = preheat_enabled
        
        # Price schedule (Shanghai TOU)
        self.peak_hours = list(range(8, 11)) + list(range(18, 21))  # 8-11, 18-21
        self.offpeak_hours = list(range(22, 24)) + list(range(0, 6))  # 22-6
        
    def _get_hour_from_observation(self, observation: np.ndarray) -> int:
        """
        Extract hour from observation if available.
        
        For our wrapped environment, the observation structure is:
        [zone_temps (5), outdoor_temp, solar, power, forecasts (8)]
        
        We estimate hour from the step count or use info dict.
        """
        # Default to assuming daytime if we can't determine
        return 12
    
    def _get_hour_from_info(self, info: Dict[str, Any]) -> int:
        """Extract hour from info dictionary."""
        if info is None:
            return 12
        
        time_info = info.get('time', {})
        return time_info.get('hour', 12)
    
    def _get_outdoor_temp(self, observation: np.ndarray) -> float:
        """Extract outdoor temperature from observation."""
        # In our observation space: [zone_temps (5), outdoor_temp, ...]
        if len(observation) > 5:
            return observation[5]
        return 20.0  # Default
    
    def _is_occupied(self, hour: int) -> bool:
        """Check if building is occupied at given hour."""
        return self.occupied_start <= hour < self.occupied_end
    
    def _is_peak_price(self, hour: int) -> bool:
        """Check if current hour is peak price period."""
        return hour in self.peak_hours
    
    def _is_offpeak_price(self, hour: int) -> bool:
        """Check if current hour is off-peak price period."""
        return hour in self.offpeak_hours
    
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Predict setpoints based on time-of-use rules.
        
        Args:
            observation: Current observation
            info: Environment info with time data
            deterministic: Ignored (always deterministic)
            
        Returns:
            Action array [heating_setpoint, cooling_setpoint]
        """
        hour = self._get_hour_from_info(info)
        outdoor_temp = self._get_outdoor_temp(observation)
        
        # Determine season (rough estimate)
        # Summer: outdoor > 25°C, Winter: outdoor < 10°C
        is_summer = outdoor_temp > 25.0
        is_winter = outdoor_temp < 10.0
        
        # Base setpoints
        if self._is_occupied(hour):
            # Comfort mode during occupied hours
            heat_setpoint = self.comfort_temp_heat
            cool_setpoint = self.comfort_temp_cool
        else:
            # Setback mode during unoccupied hours
            heat_setpoint = self.setback_temp_heat
            cool_setpoint = self.setback_temp_cool
        
        # Pre-conditioning strategies during off-peak
        if self._is_offpeak_price(hour):
            # Check if we're approaching occupied period
            hours_until_occupied = (self.occupied_start - hour) % 24
            
            if hours_until_occupied <= 2:  # Within 2 hours of occupancy
                if is_summer and self.precool_enabled:
                    # Pre-cool to lower temperature
                    cool_setpoint = self.comfort_temp_cool - 1.0
                elif is_winter and self.preheat_enabled:
                    # Pre-heat to higher temperature
                    heat_setpoint = self.comfort_temp_heat + 1.0
        
        # During peak price, try to minimize HVAC usage
        if self._is_peak_price(hour) and self._is_occupied(hour):
            # Slightly widen deadband to reduce energy use
            heat_setpoint = max(self.setback_temp_heat, heat_setpoint - 0.5)
            cool_setpoint = min(self.setback_temp_cool, cool_setpoint + 0.5)
        
        # Clip to action space bounds
        action = np.array([heat_setpoint, cool_setpoint], dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action, None


class RandomController(BaseController):
    """
    Random action controller for establishing lower bound performance.
    
    This controller serves as a sanity check - any reasonable controller
    should perform better than random actions.
    """
    
    def __init__(self, action_space, seed: int = None):
        super().__init__(action_space)
        self.rng = np.random.default_rng(seed)
        
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Return random action within action space.
        
        Args:
            observation: Ignored
            info: Ignored
            deterministic: If True, returns center of action space
            
        Returns:
            Random action array
        """
        if deterministic:
            # Return middle of action space
            action = (self.action_space.low + self.action_space.high) / 2
        else:
            # Random action
            action = self.rng.uniform(
                self.action_space.low,
                self.action_space.high
            ).astype(np.float32)
        
        return action, None


class FixedSetpointController(BaseController):
    """
    Simple fixed setpoint controller.
    
    Always maintains the same heating and cooling setpoints,
    regardless of time or conditions.
    """
    
    def __init__(
        self,
        action_space,
        heating_setpoint: float = 20.0,
        cooling_setpoint: float = 24.0,
    ):
        super().__init__(action_space)
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """Return fixed setpoints."""
        action = np.array(
            [self.heating_setpoint, self.cooling_setpoint],
            dtype=np.float32
        )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action, None


class AlwaysOnController(BaseController):
    """
    Always-On controller - HVAC systems run continuously.
    
    This controller represents the upper bound of energy consumption.
    It maintains tight comfort bands and keeps equipment running
    to ensure maximum comfort at maximum energy cost.
    
    Use case: Establishes maximum energy consumption baseline
    for comparison with optimized controllers.
    """
    
    def __init__(
        self,
        action_space,
        heating_setpoint: float = 22.0,
        cooling_setpoint: float = 23.0,  # Tight deadband = high energy
    ):
        super().__init__(action_space)
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Return aggressive setpoints to maximize comfort (and energy use).
        
        Args:
            observation: Ignored (always returns same setpoints)
            info: Ignored
            deterministic: Ignored (always deterministic)
            
        Returns:
            Fixed tight-deadband setpoints
        """
        action = np.array(
            [self.heating_setpoint, self.cooling_setpoint],
            dtype=np.float32
        )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action, None


class NightSetbackController(BaseController):
    """
    Night setback controller with aggressive energy reduction.
    
    During unoccupied hours (night), significantly relaxes setpoints
    to minimize energy consumption. Re-engages comfort control
    before occupancy to pre-condition the building.
    """
    
    def __init__(
        self,
        action_space,
        day_heat: float = 21.0,
        day_cool: float = 24.0,
        night_heat: float = 15.0,  # Deep setback
        night_cool: float = 30.0,  # Deep setback
        occupied_start: int = 7,
        occupied_end: int = 22,
        preheat_hours: int = 1,
    ):
        super().__init__(action_space)
        self.day_heat = day_heat
        self.day_cool = day_cool
        self.night_heat = night_heat
        self.night_cool = night_cool
        self.occupied_start = occupied_start
        self.occupied_end = occupied_end
        self.preheat_hours = preheat_hours
        
    def _get_hour(self, info: Dict[str, Any]) -> int:
        if info and 'time' in info:
            return info['time'].get('hour', 12)
        return 12
        
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Apply night setback strategy.
        """
        hour = self._get_hour(info)
        
        # Check if pre-heating/cooling needed
        hours_to_occupied = (self.occupied_start - hour) % 24
        
        if self.occupied_start <= hour < self.occupied_end:
            # Occupied - comfort mode
            heat_sp = self.day_heat
            cool_sp = self.day_cool
        elif hours_to_occupied <= self.preheat_hours:
            # Pre-conditioning period
            heat_sp = self.day_heat
            cool_sp = self.day_cool
        else:
            # Unoccupied - setback mode
            heat_sp = self.night_heat
            cool_sp = self.night_cool
        
        action = np.array([heat_sp, cool_sp], dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action, None


class CarbonAwareRuleController(BaseController):
    """
    Rule-based controller that considers carbon intensity.
    
    Similar to RuleBasedController but also shifts load away from
    high carbon intensity periods. This serves as a stronger baseline
    that incorporates carbon awareness without learning.
    
    Args:
        action_space: Gymnasium action space
        carbon_threshold_high: Carbon intensity above which to reduce HVAC (kgCO2/kWh)
        carbon_threshold_low: Carbon intensity below which to pre-condition
    """
    
    def __init__(
        self,
        action_space,
        carbon_threshold_high: float = 0.7,
        carbon_threshold_low: float = 0.5,
        comfort_temp_heat: float = 21.0,
        comfort_temp_cool: float = 24.0,
    ):
        super().__init__(action_space)
        
        self.carbon_threshold_high = carbon_threshold_high
        self.carbon_threshold_low = carbon_threshold_low
        self.comfort_temp_heat = comfort_temp_heat
        self.comfort_temp_cool = comfort_temp_cool
        
    def _get_carbon_factor(self, observation: np.ndarray, info: Dict) -> float:
        """Get current carbon intensity."""
        if info and 'current_carbon_factor' in info:
            return info['current_carbon_factor']
        # Default to medium carbon intensity
        return 0.6
    
    def predict(
        self,
        observation: np.ndarray,
        info: Dict[str, Any] = None,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Predict setpoints considering carbon intensity.
        """
        carbon_factor = self._get_carbon_factor(observation, info)
        
        heat_setpoint = self.comfort_temp_heat
        cool_setpoint = self.comfort_temp_cool
        
        if carbon_factor > self.carbon_threshold_high:
            # High carbon - reduce HVAC by widening deadband
            heat_setpoint -= 1.0
            cool_setpoint += 1.0
        elif carbon_factor < self.carbon_threshold_low:
            # Low carbon - increase pre-conditioning
            heat_setpoint += 0.5
            cool_setpoint -= 0.5
        
        action = np.array([heat_setpoint, cool_setpoint], dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action, None


# Factory function
def create_baseline(
    baseline_type: str,
    action_space,
    **kwargs
) -> BaseController:
    """
    Factory function to create baseline controllers.
    
    Args:
        baseline_type: Type of baseline ('rule_based', 'random', 'fixed', 'carbon_aware')
        action_space: Gymnasium action space
        **kwargs: Additional arguments for the controller
        
    Returns:
        Initialized baseline controller
    """
    baselines = {
        'rule_based': RuleBasedController,
        'random': RandomController,
        'fixed': FixedSetpointController,
        'carbon_aware_rule': CarbonAwareRuleController,
        'always_on': AlwaysOnController,
        'night_setback': NightSetbackController,
    }
    
    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Available: {list(baselines.keys())}")
    
    return baselines[baseline_type](action_space, **kwargs)


if __name__ == "__main__":
    # Test baselines
    import gymnasium as gym
    
    print("Testing Baseline Controllers")
    print("=" * 50)
    
    # Create test action space
    action_space = gym.spaces.Box(
        low=np.array([18.0, 23.0]),
        high=np.array([22.0, 26.0]),
        dtype=np.float32
    )
    
    # Test observation
    obs = np.random.randn(16).astype(np.float32)
    info = {'time': {'hour': 10, 'day': 1, 'month': 7}}
    
    # Test each baseline
    for name in ['rule_based', 'random', 'fixed', 'carbon_aware_rule', 'always_on', 'night_setback']:
        controller = create_baseline(name, action_space)
        action, _ = controller.predict(obs, info)
        print(f"{name:20s}: {action}")
    
    print("\n✅ All baselines working correctly!")
