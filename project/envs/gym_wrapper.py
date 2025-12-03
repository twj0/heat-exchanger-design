"""
Gymnasium Environment Wrapper for EnergyPlus Building Control.

This wrapper connects the IDF model built by builder.py to a Gymnasium-compatible
RL environment for training Transformer-based carbon-aware controllers.

Architecture:
    EnergyPlus IDF <-> EMS Actuators <-> Gym Wrapper <-> RL Agent (Transformer)

Key Features:
    1. Multi-objective reward: Cost + Carbon with configurable weights
    2. Rich observation space for Transformer attention mechanism
    3. Action masking for seasonal constraints
    4. Future prediction integration (weather, price, carbon intensity)

References:
    - Gymnasium: https://gymnasium.farama.org/
    - EnergyPlus Python API: https://energyplus.readthedocs.io/
    - Sinergym (alternative): https://github.com/ugr-sail/sinergym

Author: Auto-generated for Applied Energy publication
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import json
import sys

# Import official Shanghai carbon factors
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.carbon_factors import (
    ELECTRICITY_FACTOR,
    GREEN_ELECTRICITY_FACTOR,
    get_tou_price,
    get_tou_period,
    calculate_carbon_emission,
    calculate_electricity_cost,
    CARBON_PRICE_REFERENCE,
)


@dataclass
class BuildingConfig:
    """Configuration for the building simulation environment."""
    
    # Paths
    idf_path: str = "outputs/sim_building.idf"
    weather_path: str = "data/weather/Shanghai_2024.epw"
    carbon_intensity_path: str = "data/schedules/carbon_intensity.csv"
    electricity_price_path: str = "data/schedules/electricity_price.csv"
    
    # Simulation settings
    timestep_minutes: int = 15  # Must match IDF Timestep
    episode_days: int = 7       # Episode length for training
    
    # TES configuration (must match builder.py)
    tes_num_nodes: int = 10     # Hot water TES stratification layers
    chilled_tes_nodes: int = 6  # Chilled water TES layers
    
    # Action space bounds
    hw_setpoint_range: Tuple[float, float] = (30.0, 55.0)  # °C
    cooling_setpoint_range: Tuple[float, float] = (22.0, 28.0)  # °C
    heating_setpoint_range: Tuple[float, float] = (18.0, 24.0)  # °C
    
    # Reward weights (α in paper: R = α*Cost + (1-α)*Carbon)
    cost_weight: float = 0.5
    carbon_weight: float = 0.5
    comfort_penalty: float = 10.0  # Penalty per °C deviation from setpoint
    
    # Transformer-specific: prediction horizon for state augmentation
    prediction_horizon_hours: int = 24


class BuildingEnv(gym.Env):
    """
    Gymnasium environment for EnergyPlus building control.
    
    Observation Space (designed for Transformer attention):
        - Zone temperatures (5 zones)
        - TES stratified temperatures (10 hot + 6 cold nodes)
        - Outdoor conditions (temp, humidity, solar radiation)
        - Equipment states (HP COP, Chiller COP, PV power)
        - Time features (hour, day_of_week, month)
        - Future predictions (optional, for Transformer context)
    
    Action Space:
        - HW Supply Setpoint: Continuous [30, 55] °C
        - HP Enable: Discrete {0, 1}
        - Chiller Enable: Discrete {0, 1}
        - Zone Heating Setpoint: Continuous [18, 24] °C
        - Zone Cooling Setpoint: Continuous [22, 28] °C
    
    Reward:
        R = -α * ElectricityCost - (1-α) * CarbonEmission - β * ComfortViolation
    """
    
    metadata = {"render_modes": ["human", "csv"]}
    
    def __init__(
        self,
        config: Optional[BuildingConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config or BuildingConfig()
        self.render_mode = render_mode
        
        # =====================================================================
        # OBSERVATION SPACE
        # Designed for Transformer: rich, multi-dimensional state
        # =====================================================================
        obs_dim = self._calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # =====================================================================
        # ACTION SPACE
        # Hybrid: Continuous setpoints + Discrete equipment switches
        # =====================================================================
        self.action_space = spaces.Dict({
            # Continuous actions
            "hw_setpoint": spaces.Box(
                low=self.config.hw_setpoint_range[0],
                high=self.config.hw_setpoint_range[1],
                shape=(1,),
                dtype=np.float32
            ),
            "heating_setpoint": spaces.Box(
                low=self.config.heating_setpoint_range[0],
                high=self.config.heating_setpoint_range[1],
                shape=(1,),
                dtype=np.float32
            ),
            "cooling_setpoint": spaces.Box(
                low=self.config.cooling_setpoint_range[0],
                high=self.config.cooling_setpoint_range[1],
                shape=(1,),
                dtype=np.float32
            ),
            # Discrete actions (equipment on/off)
            "hp_enable": spaces.Discrete(2),       # 0=Off, 1=On
            "chiller_enable": spaces.Discrete(2),  # 0=Off, 1=On
        })
        
        # Alternative: Flattened continuous action space for standard RL algorithms
        # self.action_space = spaces.Box(
        #     low=np.array([30.0, 18.0, 22.0, 0.0, 0.0]),
        #     high=np.array([55.0, 24.0, 28.0, 1.0, 1.0]),
        #     dtype=np.float32
        # )
        
        # =====================================================================
        # EXTERNAL DATA (Carbon intensity, Electricity price)
        # =====================================================================
        self.carbon_intensity = None  # Will be loaded from CSV
        self.electricity_price = None
        self.weather_data = None
        
        # =====================================================================
        # STATE TRACKING
        # =====================================================================
        self.current_step = 0
        self.episode_length = (
            self.config.episode_days * 24 * 60 // self.config.timestep_minutes
        )
        self.cumulative_cost = 0.0
        self.cumulative_carbon = 0.0
        self.cumulative_comfort_violation = 0.0
        
        # EnergyPlus connection (placeholder - implement with pyenergyplus)
        self.eplus_api = None
        self.eplus_state = None
        
        print(f"BuildingEnv initialized:")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Episode length: {self.episode_length} steps ({self.config.episode_days} days)")
    
    def _calculate_obs_dim(self) -> int:
        """Calculate observation space dimension."""
        dim = 0
        
        # Zone temperatures (5 zones)
        dim += 5
        
        # TES stratified temperatures
        dim += self.config.tes_num_nodes      # Hot water TES
        dim += self.config.chilled_tes_nodes  # Chilled water TES
        
        # Outdoor conditions
        dim += 4  # OAT, humidity, solar_direct, solar_diffuse
        
        # Equipment states
        dim += 4  # HP_power, HP_COP, Chiller_power, PV_power
        
        # Time features (cyclical encoding)
        dim += 6  # sin/cos for hour, day_of_week, month
        
        # Grid signals
        dim += 2  # electricity_price, carbon_intensity
        
        # Future predictions (optional, for Transformer context window)
        if self.config.prediction_horizon_hours > 0:
            # Future weather (temp only), price, carbon for next 24 hours
            prediction_steps = self.config.prediction_horizon_hours * 60 // self.config.timestep_minutes
            dim += prediction_steps * 3  # (temp, price, carbon) for each future step
        
        return dim
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector from EnergyPlus outputs.
        
        This method should be called after each EnergyPlus timestep.
        For now, returns placeholder data for testing.
        """
        obs = []
        
        # =====================================================================
        # 1. Zone Temperatures (from Output:Variable)
        # =====================================================================
        zone_temps = self._get_zone_temperatures()
        obs.extend(zone_temps)
        
        # =====================================================================
        # 2. TES Stratified Temperatures (CRITICAL for Transformer)
        # These capture the thermocline state - key for storage management
        # =====================================================================
        hot_tes_temps = self._get_tes_node_temperatures("hot")
        cold_tes_temps = self._get_tes_node_temperatures("cold")
        obs.extend(hot_tes_temps)
        obs.extend(cold_tes_temps)
        
        # =====================================================================
        # 3. Outdoor Conditions
        # =====================================================================
        outdoor = self._get_outdoor_conditions()
        obs.extend(outdoor)
        
        # =====================================================================
        # 4. Equipment States
        # =====================================================================
        equipment = self._get_equipment_states()
        obs.extend(equipment)
        
        # =====================================================================
        # 5. Time Features (cyclical encoding for Transformer)
        # =====================================================================
        time_features = self._get_time_features()
        obs.extend(time_features)
        
        # =====================================================================
        # 6. Grid Signals (current)
        # =====================================================================
        price = self._get_electricity_price()
        carbon = self._get_carbon_intensity()
        obs.extend([price, carbon])
        
        # =====================================================================
        # 7. Future Predictions (for Transformer context)
        # =====================================================================
        if self.config.prediction_horizon_hours > 0:
            future = self._get_future_predictions()
            obs.extend(future)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_zone_temperatures(self) -> List[float]:
        """Get zone air temperatures from EnergyPlus."""
        # Placeholder - implement with EMS sensor reading
        # Variable: "Zone Mean Air Temperature"
        # Keys: Classroom_South, Classroom_East, Classroom_North, Classroom_West, Corridor
        return [22.0, 22.5, 21.8, 22.2, 20.0]  # Example values
    
    def _get_tes_node_temperatures(self, tes_type: str) -> List[float]:
        """
        Get stratified TES node temperatures.
        
        CRITICAL for Transformer: These vectors capture thermocline position,
        enabling the model to learn optimal charge/discharge timing.
        
        Args:
            tes_type: "hot" for hot water TES, "cold" for chilled water TES
        
        Returns:
            List of temperatures from top to bottom of tank
        """
        if tes_type == "hot":
            # Variable: "Water Heater Temperature Node N" for N = 1..10
            # Key: TES_Stratified
            num_nodes = self.config.tes_num_nodes
            # Placeholder: typical stratified profile (hot on top)
            return [50.0 - i * 1.0 for i in range(num_nodes)]
        else:
            # Variable: same for Chilled_TES_Stratified
            num_nodes = self.config.chilled_tes_nodes
            # Cold on bottom for chilled water
            return [7.0 + i * 1.5 for i in range(num_nodes)]
    
    def _get_outdoor_conditions(self) -> List[float]:
        """Get outdoor weather conditions."""
        # Variables from EnergyPlus:
        # - Site Outdoor Air Drybulb Temperature
        # - Site Outdoor Air Relative Humidity
        # - Site Direct Solar Radiation Rate per Area
        # - Site Diffuse Solar Radiation Rate per Area
        return [15.0, 60.0, 300.0, 100.0]  # Placeholder
    
    def _get_equipment_states(self) -> List[float]:
        """Get equipment power and efficiency states."""
        # Variables:
        # - Heat Pump Electricity Rate (W)
        # - Heat Pump Part Load Ratio
        # - Chiller Electricity Rate (W)
        # - Generator Produced DC Electricity Rate (W)
        return [5000.0, 0.5, 0.0, 10000.0]  # Placeholder
    
    def _get_pv_generation(self) -> float:
        """
        Get current PV generation (kWh for this timestep).
        
        Green electricity has zero carbon emission per Shanghai regulation.
        Source: 沪环气〔2022〕34号 - 外购绿电排放因子为0
        
        Returns:
            PV generation in kWh for current timestep
        """
        # Variable: Generator Produced DC Electricity Rate (W)
        # Key: Rooftop_PV_Array
        # Placeholder - implement with EMS sensor reading
        step = self.current_step
        hour = (step * self.config.timestep_minutes // 60) % 24
        
        # Simple solar profile (peak at noon)
        if 6 <= hour <= 18:
            # Peak generation at noon (~45 kW), scaled by time
            peak_kw = 45.0  # System capacity
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
            power_kw = peak_kw * solar_factor * 0.8  # 80% typical efficiency
        else:
            power_kw = 0.0
        
        # Convert W to kWh for timestep
        timestep_hours = self.config.timestep_minutes / 60.0
        return power_kw * timestep_hours
    
    def _get_time_features(self) -> List[float]:
        """
        Get cyclical time features for Transformer position encoding.
        
        Returns sin/cos encoding for:
        - Hour of day (period = 24)
        - Day of week (period = 7)
        - Month of year (period = 12)
        """
        step = self.current_step
        minutes_per_step = self.config.timestep_minutes
        
        # Calculate current time
        total_minutes = step * minutes_per_step
        hour = (total_minutes // 60) % 24
        day = (total_minutes // (60 * 24)) % 7
        month = (total_minutes // (60 * 24 * 30)) % 12
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        return [hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]
    
    def _get_electricity_price(self) -> float:
        """
        Get current electricity price (¥/kWh).
        
        Uses official Shanghai TOU pricing from carbon_factors module.
        """
        step = self.current_step
        hour = (step * self.config.timestep_minutes // 60) % 24
        return get_tou_price(hour)
    
    def _get_carbon_intensity(self) -> float:
        """
        Get current grid carbon intensity (kgCO2/kWh).
        
        Official Shanghai factor: 0.42 tCO2/MWh = 0.42 kgCO2/kWh
        Source: 沪环气〔2022〕34号
        
        Note: Shanghai currently uses a fixed emission factor.
        For future work, time-varying factors based on grid mix
        could be implemented when real-time data becomes available.
        """
        # Fixed factor per Shanghai regulation
        # 0.42 tCO2/MWh = 0.42 kgCO2/kWh = 0.00042 tCO2/kWh
        return ELECTRICITY_FACTOR  # 0.42 kgCO2/kWh
    
    def _get_future_predictions(self) -> List[float]:
        """
        Get future predictions for Transformer context window.
        
        Returns flattened array of (temp, price, carbon) for each future timestep.
        This enables the Transformer to attend to future conditions.
        """
        predictions = []
        steps = self.config.prediction_horizon_hours * 60 // self.config.timestep_minutes
        
        for i in range(steps):
            future_step = self.current_step + i + 1
            
            # Future outdoor temperature (from weather file)
            future_temp = 15.0 + 5.0 * np.sin(2 * np.pi * (future_step * self.config.timestep_minutes / 60 - 6) / 24)
            
            # Future price
            future_hour = (future_step * self.config.timestep_minutes // 60) % 24
            if future_hour in [8, 9, 10, 18, 19, 20]:
                future_price = 1.0074
            elif future_hour >= 22 or future_hour < 6:
                future_price = 0.3128
            else:
                future_price = 0.6177
            
            # Future carbon intensity
            future_carbon = 0.55 + 0.15 * np.sin(2 * np.pi * (future_hour - 14) / 24)
            
            predictions.extend([future_temp, future_price, future_carbon])
        
        return predictions
    
    def _calculate_reward(
        self,
        electricity_consumption: float,  # kWh
        zone_temps: List[float],
        setpoints: Tuple[float, float],  # (heating, cooling)
    ) -> float:
        """
        Calculate multi-objective reward.
        
        R = -α * Cost - (1-α) * Carbon - β * ComfortViolation
        
        Args:
            electricity_consumption: Total electricity use this timestep (kWh)
            zone_temps: Current zone temperatures
            setpoints: (heating_setpoint, cooling_setpoint)
        
        Returns:
            Reward value (negative = cost/penalty)
        """
        # 1. Electricity cost
        price = self._get_electricity_price()
        cost = electricity_consumption * price
        
        # 2. Carbon emission (using official Shanghai factor)
        # Source: 沪环气〔2022〕34号 - 0.42 tCO2/MWh
        carbon_intensity = self._get_carbon_intensity()  # kgCO2/kWh
        
        # Get PV generation (reduces net grid purchase)
        pv_generation = self._get_pv_generation()  # kWh
        net_grid_consumption = max(0, electricity_consumption - pv_generation)
        
        # Carbon = net grid consumption × factor (PV is zero-carbon)
        carbon = net_grid_consumption * carbon_intensity  # kgCO2
        
        # 3. Comfort violation (penalty for temperature deviation)
        heating_sp, cooling_sp = setpoints
        comfort_violation = 0.0
        for temp in zone_temps:
            if temp < heating_sp:
                comfort_violation += (heating_sp - temp) ** 2
            elif temp > cooling_sp:
                comfort_violation += (temp - cooling_sp) ** 2
        comfort_violation = np.sqrt(comfort_violation / len(zone_temps))
        
        # Combined reward
        reward = (
            -self.config.cost_weight * cost
            -self.config.carbon_weight * carbon
            -self.config.comfort_penalty * comfort_violation
        )
        
        # Track cumulative metrics
        self.cumulative_cost += cost
        self.cumulative_carbon += carbon
        self.cumulative_comfort_violation += comfort_violation
        
        return reward
    
    def _apply_action(self, action: Dict[str, Any]) -> None:
        """
        Apply RL action to EnergyPlus via EMS actuators.
        
        Maps Gym action dict to EMS actuator values:
        - HW_Setpoint_Actuator <- action["hw_setpoint"]
        - HP_Enable_Actuator <- action["hp_enable"]
        - Chiller_Enable_Actuator <- action["chiller_enable"]
        - Heating_Setpoint_Actuator <- action["heating_setpoint"]
        - Cooling_Setpoint_Actuator <- action["cooling_setpoint"]
        """
        # Placeholder - implement with EnergyPlus Python API
        # Example using pyenergyplus:
        # self.eplus_api.exchange.set_actuator_value(
        #     self.eplus_state,
        #     actuator_handle,
        #     action_value
        # )
        pass
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_carbon = 0.0
        self.cumulative_comfort_violation = 0.0
        
        # Reset EnergyPlus simulation
        # Placeholder - implement with EnergyPlus Python API
        
        obs = self._get_obs()
        info = {
            "episode_start": True,
            "episode_length": self.episode_length,
        }
        
        return obs, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep in the environment.
        
        Args:
            action: Dict with keys: hw_setpoint, heating_setpoint, cooling_setpoint,
                    hp_enable, chiller_enable
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action to EnergyPlus
        self._apply_action(action)
        
        # Advance EnergyPlus simulation by one timestep
        # Placeholder - implement with EnergyPlus callback
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        zone_temps = self._get_zone_temperatures()
        electricity = 5.0  # Placeholder kWh
        setpoints = (
            float(action.get("heating_setpoint", [21.0])[0]),
            float(action.get("cooling_setpoint", [24.0])[0]),
        )
        reward = self._calculate_reward(electricity, zone_temps, setpoints)
        
        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Info dict for logging
        info = {
            "step": self.current_step,
            "cumulative_cost": self.cumulative_cost,
            "cumulative_carbon": self.cumulative_carbon,
            "cumulative_comfort": self.cumulative_comfort_violation,
            "electricity_price": self._get_electricity_price(),
            "carbon_intensity": self._get_carbon_intensity(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            step = self.current_step
            price = self._get_electricity_price()
            carbon = self._get_carbon_intensity()
            print(f"Step {step}: Price={price:.4f} ¥/kWh, Carbon={carbon:.4f} kgCO2/kWh")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.eplus_api is not None:
            # Terminate EnergyPlus simulation
            pass


def calculate_tes_soc(
    node_temperatures: List[float],
    temp_min: float,
    temp_max: float,
) -> float:
    """
    Calculate State of Charge (SOC) from stratified TES temperatures.
    
    SOC = (avg_temp - temp_min) / (temp_max - temp_min)
    
    For more accurate SOC, consider energy-based calculation:
    SOC = Σ(m_node * cp * (T_node - T_min)) / (M_total * cp * (T_max - T_min))
    
    Args:
        node_temperatures: List of temperatures from top to bottom
        temp_min: Minimum useful temperature (discharge limit)
        temp_max: Maximum storage temperature (charge limit)
    
    Returns:
        SOC value in range [0, 1]
    """
    avg_temp = np.mean(node_temperatures)
    soc = (avg_temp - temp_min) / (temp_max - temp_min)
    return np.clip(soc, 0.0, 1.0)


def normalize_observation(
    obs: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> np.ndarray:
    """
    Normalize observation for Transformer input.
    
    Transformer models are sensitive to input scale.
    Use running mean/std from training data.
    
    Args:
        obs: Raw observation vector
        obs_mean: Running mean from training
        obs_std: Running std from training
    
    Returns:
        Normalized observation
    """
    return (obs - obs_mean) / (obs_std + 1e-8)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Create environment
    config = BuildingConfig(
        idf_path="outputs/sim_building.idf",
        weather_path="data/weather/Shanghai_2024.epw",
        cost_weight=0.5,
        carbon_weight=0.5,
        prediction_horizon_hours=24,
    )
    
    env = BuildingEnv(config=config, render_mode="human")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation sample (first 20): {obs[:20]}")
    
    # Test step with random action
    action = {
        "hw_setpoint": np.array([45.0]),
        "heating_setpoint": np.array([21.0]),
        "cooling_setpoint": np.array([24.0]),
        "hp_enable": 1,
        "chiller_enable": 0,
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nAfter step:")
    print(f"  Reward: {reward:.4f}")
    print(f"  Info: {info}")
    
    # Test SOC calculation
    hot_tes_temps = [50.0, 48.0, 46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 34.0, 32.0]
    soc = calculate_tes_soc(hot_tes_temps, temp_min=35.0, temp_max=55.0)
    print(f"\nHot TES SOC: {soc:.2%}")
    
    env.close()
    print("\n✅ Gym wrapper test completed!")
