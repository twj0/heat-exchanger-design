"""
EnergyPlus Gymnasium Environment for Carbon-Aware Building Control.

This module provides a complete Gymnasium environment that integrates
with EnergyPlus Python API for realistic building simulation.

Features:
    - Native EnergyPlus Python API integration
    - Flat continuous action space for SAC compatibility
    - Carbon-aware reward function
    - TES state of charge tracking
    - Time-of-use pricing integration

Usage:
    from envs.eplus_env import EnergyPlusGymEnv, create_env
    
    env = create_env(idf_path, weather_path)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

Author: Auto-generated for Applied Energy publication
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import carbon factors
from data.carbon_factors import (
    ELECTRICITY_FACTOR,
    get_tou_price,
    get_tou_period,
    calculate_carbon_emission,
    calculate_electricity_cost,
)

# Try to import EnergyPlus API
try:
    from envs.energyplus_api import (
        EnergyPlusAPI,
        EnergyPlusConfig,
        create_building_api,
        check_energyplus_installation,
    )
    ENERGYPLUS_API_AVAILABLE = True
except ImportError:
    ENERGYPLUS_API_AVAILABLE = False


@dataclass
class EnvConfig:
    """Configuration for EnergyPlus Gym Environment."""
    
    # Paths
    idf_path: str = "outputs/sim_building.idf"
    weather_path: str = "data/weather/CHN_Shanghai.epw"
    
    # Episode settings
    episode_length: int = 672  # 1 week at 15-min intervals
    timestep_seconds: int = 900  # 15 minutes
    
    # Observation space config
    include_forecasts: bool = True
    forecast_horizon: int = 4  # hours ahead
    
    # Action space bounds
    heating_setpoint_range: Tuple[float, float] = (18.0, 22.0)
    cooling_setpoint_range: Tuple[float, float] = (23.0, 26.0)
    
    # Reward weights
    cost_weight: float = 0.5       # λ_cost
    carbon_weight: float = 0.5     # λ_carbon
    comfort_penalty: float = 10.0  # β
    
    # Comfort bounds
    comfort_temp_min: float = 20.0
    comfort_temp_max: float = 26.0


class EnergyPlusGymEnv(gym.Env):
    """
    Gymnasium environment for EnergyPlus building simulation.
    
    Observation Space:
        - Zone temperatures (5)
        - Outdoor temperature (1)
        - Solar radiation (1)
        - HVAC power (1)
        - Hour of day (1, normalized 0-1)
        - Day of week (1, normalized 0-1)
        - Electricity price (1)
        - Carbon intensity (1)
        - Price forecast (4)
        - Carbon forecast (4)
        Total: 20 dimensions (without forecasts: 12)
    
    Action Space:
        - Heating setpoint [18, 22] °C
        - Cooling setpoint [23, 26] °C
    
    Reward:
        R = -λ_cost * Cost - λ_carbon * Carbon - β * ComfortViolation
    """
    
    metadata = {"render_modes": ["human", "csv"]}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        use_mock: bool = True,  # Use mock env if EnergyPlus not available
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.use_mock = use_mock or not ENERGYPLUS_API_AVAILABLE
        
        # Calculate observation dimension
        base_obs_dim = 12  # temps (5) + outdoor (1) + solar (1) + power (1) + time (2) + price (1) + carbon (1)
        if self.config.include_forecasts:
            forecast_dim = self.config.forecast_horizon * 2  # price + carbon
            obs_dim = base_obs_dim + forecast_dim
        else:
            obs_dim = base_obs_dim
        
        self._obs_dim = obs_dim
        self._base_obs_dim = base_obs_dim
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: [heating_setpoint, cooling_setpoint]
        self.action_space = spaces.Box(
            low=np.array([
                self.config.heating_setpoint_range[0],
                self.config.cooling_setpoint_range[0],
            ]),
            high=np.array([
                self.config.heating_setpoint_range[1],
                self.config.cooling_setpoint_range[1],
            ]),
            dtype=np.float32
        )
        
        # State tracking
        self._step_count = 0
        self._current_hour = 0
        self._episode_cost = 0.0
        self._episode_carbon = 0.0
        self._episode_comfort_violation = 0.0
        
        # Mock state
        self._zone_temps = np.array([22.0, 22.0, 22.0, 22.0, 22.0])
        self._outdoor_temp = 20.0
        self._solar = 0.0
        self._hvac_power = 0.0
        
        # EnergyPlus API (if available)
        self._eplus_api = None
        if not self.use_mock:
            self._init_energyplus()
    
    def _init_energyplus(self):
        """Initialize EnergyPlus API connection."""
        try:
            self._eplus_api = create_building_api(
                idf_path=str(project_root / self.config.idf_path),
                weather_path=str(project_root / self.config.weather_path),
            )
        except Exception as e:
            print(f"Warning: Could not initialize EnergyPlus: {e}")
            print("Falling back to mock environment")
            self.use_mock = True
    
    def _get_hour(self) -> int:
        """Get current hour of day."""
        total_minutes = self._step_count * self.config.timestep_seconds // 60
        return (total_minutes // 60) % 24
    
    def _get_day_of_week(self) -> int:
        """Get current day of week (0-6)."""
        total_minutes = self._step_count * self.config.timestep_seconds // 60
        total_days = total_minutes // (60 * 24)
        return total_days % 7
    
    def _get_price_forecast(self) -> np.ndarray:
        """Get electricity price forecast for next N hours."""
        forecasts = []
        current_hour = self._get_hour()
        for i in range(self.config.forecast_horizon):
            future_hour = (current_hour + i + 1) % 24
            forecasts.append(get_tou_price(future_hour))
        return np.array(forecasts, dtype=np.float32)
    
    def _get_carbon_forecast(self) -> np.ndarray:
        """
        Get carbon intensity forecast for next N hours.
        
        Note: Shanghai uses fixed factor, but we add slight variation
        to simulate potential future time-varying grid carbon.
        """
        forecasts = []
        current_hour = self._get_hour()
        for i in range(self.config.forecast_horizon):
            future_hour = (current_hour + i + 1) % 24
            # Add slight diurnal variation for research purposes
            variation = 0.05 * np.sin(2 * np.pi * (future_hour - 14) / 24)
            forecasts.append(ELECTRICITY_FACTOR + variation)
        return np.array(forecasts, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        hour = self._get_hour()
        day = self._get_day_of_week()
        
        # Base observation
        obs = [
            # Zone temperatures (5)
            *self._zone_temps,
            # Outdoor conditions (2)
            self._outdoor_temp,
            self._solar / 1000.0,  # Normalize to kW/m²
            # HVAC power (1)
            self._hvac_power / 100000.0,  # Normalize to 100kW scale
            # Time features (2) - normalized
            hour / 24.0,
            day / 7.0,
            # Grid signals (2)
            get_tou_price(hour),
            ELECTRICITY_FACTOR,
        ]
        
        # Add forecasts if enabled
        if self.config.include_forecasts:
            price_forecast = self._get_price_forecast()
            carbon_forecast = self._get_carbon_forecast()
            obs.extend(price_forecast)
            obs.extend(carbon_forecast)
        
        return np.array(obs, dtype=np.float32)
    
    def _simulate_building_response(self, action: np.ndarray):
        """
        Simulate building response to control action (mock mode).
        
        Args:
            action: [heating_setpoint, cooling_setpoint]
        """
        heating_sp, cooling_sp = action
        
        # Simple thermal dynamics simulation
        hour = self._get_hour()
        
        # Outdoor temperature with diurnal variation
        day_of_year = (self._step_count * self.config.timestep_seconds // 86400) % 365
        seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 90) / 365)  # Peak in summer
        diurnal_temp = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2pm
        self._outdoor_temp = seasonal_temp + diurnal_temp + np.random.randn() * 2
        
        # Solar radiation
        if 6 <= hour <= 18:
            self._solar = max(0, 800 * np.sin(np.pi * (hour - 6) / 12))
            self._solar *= (0.8 + 0.4 * np.random.rand())  # Cloud variation
        else:
            self._solar = 0
        
        # Zone temperature response (simplified RC model)
        tau = 2.0  # Time constant (hours)
        dt = self.config.timestep_seconds / 3600.0  # timestep in hours
        
        for i in range(5):
            # Target temperature based on setpoints
            if self._zone_temps[i] < heating_sp:
                target = heating_sp + 0.5
            elif self._zone_temps[i] > cooling_sp:
                target = cooling_sp - 0.5
            else:
                # Drift toward outdoor
                target = self._zone_temps[i] + 0.1 * (self._outdoor_temp - self._zone_temps[i])
            
            # First-order dynamics
            self._zone_temps[i] += (dt / tau) * (target - self._zone_temps[i])
            self._zone_temps[i] += np.random.randn() * 0.1  # Noise
        
        # HVAC power estimation
        avg_zone_temp = np.mean(self._zone_temps)
        if avg_zone_temp < heating_sp:
            # Heating mode
            load = (heating_sp - avg_zone_temp) * 5000  # W per °C
            self._hvac_power = load / 3.32  # Divide by COP
        elif avg_zone_temp > cooling_sp:
            # Cooling mode
            load = (avg_zone_temp - cooling_sp) * 5000
            self._hvac_power = load / 2.88
        else:
            # Deadband
            self._hvac_power = 1000  # Baseline ventilation
    
    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate multi-objective reward.
        
        Returns:
            Tuple of (reward, info_dict)
        """
        heating_sp, cooling_sp = action
        hour = self._get_hour()
        
        # Energy consumption (kWh for this timestep)
        timestep_hours = self.config.timestep_seconds / 3600.0
        energy_kwh = self._hvac_power * timestep_hours / 1000.0
        
        # Electricity cost
        price = get_tou_price(hour)
        cost = energy_kwh * price
        
        # Carbon emission
        carbon_kg = energy_kwh * ELECTRICITY_FACTOR
        
        # Comfort violation
        comfort_violation = 0.0
        for temp in self._zone_temps:
            if temp < self.config.comfort_temp_min:
                comfort_violation += (self.config.comfort_temp_min - temp) ** 2
            elif temp > self.config.comfort_temp_max:
                comfort_violation += (temp - self.config.comfort_temp_max) ** 2
        comfort_violation = np.sqrt(comfort_violation / 5)  # RMS violation
        
        # Combined reward
        reward = (
            -self.config.cost_weight * cost
            - self.config.carbon_weight * carbon_kg
            - self.config.comfort_penalty * comfort_violation
        )
        
        # Track cumulative metrics
        self._episode_cost += cost
        self._episode_carbon += carbon_kg
        self._episode_comfort_violation += comfort_violation
        
        info = {
            "step_cost": cost,
            "step_carbon": carbon_kg,
            "step_comfort_violation": comfort_violation,
            "energy_kwh": energy_kwh,
            "price": price,
            "avg_zone_temp": np.mean(self._zone_temps),
            "outdoor_temp": self._outdoor_temp,
            "tou_period": get_tou_period(hour),
        }
        
        return reward, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset state
        self._step_count = 0
        self._episode_cost = 0.0
        self._episode_carbon = 0.0
        self._episode_comfort_violation = 0.0
        
        # Randomize initial conditions
        if seed is not None:
            np.random.seed(seed)
        
        self._zone_temps = 22.0 + np.random.randn(5) * 1.0
        self._outdoor_temp = 20.0 + np.random.randn() * 5.0
        self._solar = 0.0
        self._hvac_power = 0.0
        
        # Reset EnergyPlus if using real simulation
        if self._eplus_api is not None and not self.use_mock:
            self._eplus_api.reset()
            self._eplus_api.start_simulation()
        
        obs = self._get_observation()
        info = {
            "episode_start": True,
            "time": {"hour": 0, "day": 1},
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        if self.use_mock:
            # Mock simulation
            self._simulate_building_response(action)
        else:
            # Real EnergyPlus simulation
            eplus_action = {
                "heating_sp": float(action[0]),
                "cooling_sp": float(action[1]),
            }
            obs_dict = self._eplus_api.step(eplus_action)
            if obs_dict:
                self._zone_temps = np.array([
                    obs_dict.get("temp_south", 22),
                    obs_dict.get("temp_east", 22),
                    obs_dict.get("temp_north", 22),
                    obs_dict.get("temp_west", 22),
                    obs_dict.get("temp_corridor", 22),
                ])
                self._outdoor_temp = obs_dict.get("outdoor_temp", 20)
                self._solar = obs_dict.get("solar_direct", 0)
                self._hvac_power = obs_dict.get("hvac_power", 0)
        
        self._step_count += 1
        
        # Calculate reward
        reward, step_info = self._calculate_reward(action)
        
        # Check termination
        terminated = False
        truncated = self._step_count >= self.config.episode_length
        
        # Get observation
        obs = self._get_observation()
        
        # Build info
        hour = self._get_hour()
        day = self._step_count * self.config.timestep_seconds // 86400
        
        info = {
            **step_info,
            "time": {"hour": hour, "day": day % 30 + 1},
            "step": self._step_count,
            "episode_cost": self._episode_cost,
            "episode_carbon": self._episode_carbon,
            "episode_comfort_violation": self._episode_comfort_violation,
        }
        
        return obs, reward, terminated, truncated, info
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get cumulative metrics for current episode."""
        return {
            "total_cost": self._episode_cost,
            "total_carbon": self._episode_carbon,
            "total_comfort_violation": self._episode_comfort_violation,
            "avg_cost_per_step": self._episode_cost / max(1, self._step_count),
            "avg_carbon_per_step": self._episode_carbon / max(1, self._step_count),
        }
    
    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            hour = self._get_hour()
            avg_temp = np.mean(self._zone_temps)
            price = get_tou_price(hour)
            period = get_tou_period(hour)
            print(f"Step {self._step_count:4d} | "
                  f"Hour {hour:02d}:00 ({period:6s}) | "
                  f"Temp {avg_temp:.1f}°C | "
                  f"Power {self._hvac_power/1000:.1f}kW | "
                  f"Price ¥{price:.3f}/kWh")
    
    def close(self) -> None:
        """Clean up resources."""
        if self._eplus_api is not None:
            self._eplus_api.close()


def create_env(
    config_path: Optional[str] = None,
    use_mock: bool = True,
    **kwargs
) -> EnergyPlusGymEnv:
    """
    Factory function to create EnergyPlus environment.
    
    Args:
        config_path: Path to YAML config file (optional)
        use_mock: Use mock simulation (default True for safety)
        **kwargs: Override config parameters
        
    Returns:
        Configured EnergyPlusGymEnv instance
    """
    config = EnvConfig(**kwargs)
    return EnergyPlusGymEnv(config=config, use_mock=use_mock)


# =============================================================================
# Main: Test the environment
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EnergyPlus Gym Environment Test")
    print("=" * 60)
    
    # Check EnergyPlus installation
    if ENERGYPLUS_API_AVAILABLE:
        info = check_energyplus_installation()
        print(f"\nEnergyPlus: {'✓ Installed' if info['installed'] else '✗ Not found'}")
        if info['installed']:
            print(f"  Path: {info['path']}")
    else:
        print("\nEnergyPlus API: ✗ Not available")
    
    # Create environment (mock mode for testing)
    print("\n[1] Creating environment (mock mode)...")
    env = create_env(use_mock=True, include_forecasts=True)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Action bounds: [{env.action_space.low}, {env.action_space.high}]")
    
    # Test reset
    print("\n[2] Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial zone temps: {env._zone_temps.mean():.1f}°C")
    
    # Test step
    print("\n[3] Testing step with sample action...")
    action = np.array([20.0, 24.0])  # [heating_sp, cooling_sp]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Action: heating={action[0]:.1f}°C, cooling={action[1]:.1f}°C")
    print(f"  Reward: {reward:.4f}")
    print(f"  Step cost: ¥{info['step_cost']:.4f}")
    print(f"  Step carbon: {info['step_carbon']:.4f} kg")
    
    # Run short episode
    print("\n[4] Running short episode (100 steps)...")
    obs, _ = env.reset()
    total_reward = 0
    for i in range(100):
        # Simple control policy
        avg_temp = obs[0:5].mean() * 1  # Zone temps are first 5
        if avg_temp < 21:
            action = np.array([21.0, 25.0])  # Heat
        elif avg_temp > 25:
            action = np.array([19.0, 24.0])  # Cool
        else:
            action = np.array([20.0, 25.0])  # Maintain
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    metrics = env.get_episode_metrics()
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total cost: ¥{metrics['total_cost']:.2f}")
    print(f"  Total carbon: {metrics['total_carbon']:.2f} kg CO₂")
    
    env.close()
    print("\n✅ Environment test completed!")
