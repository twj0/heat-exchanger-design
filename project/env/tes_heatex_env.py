"""
Thermal Energy Storage and Heat Exchanger Gym Environment.

This environment simulates a TES system with heat exchanger under TOU pricing,
suitable for training reinforcement learning agents for optimal energy management.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import deque

import sys
sys.path.append('..')

from models.thermal_storage import SensibleHeatStorage, PCMStorage
from models.heat_exchanger import create_heat_exchanger
from models.economic_model import TOUPricing, EconomicModel, create_tou_pricing
from env.utils import (
    generate_demand_profile,
    generate_weather_data,
    calculate_time_features,
)


class TESHeatExEnv(gym.Env):
    """
    Gymnasium environment for TES-HeatEx system optimization.
    
    State Space:
        - Storage temperature / SoC
        - Electricity price (current + forecast)
        - Heat demand (current)
        - Time features (hour, day)
        - Weather conditions (optional)
    
    Action Space:
        - Discrete: [idle, charge, discharge]
        - Continuous: charging/discharging power [-max_power, +max_power]
    
    Reward:
        - Primary: Negative operational cost
        - Penalties: Temperature violations, unmet demand, cycling
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, config: Dict):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        super().__init__()
        
        self.config = config
        self.timestep = config["simulation"]["timestep"]  # seconds
        self.max_steps = config["simulation"]["duration"]  # hours
        self.seed_value = config["simulation"]["seed"]
        
        # Initialize models
        self._init_storage(config["tes"])
        self._init_heat_exchanger(config["heat_exchanger"])
        self._init_economics(config["tou_pricing"])
        self._init_demand_and_weather(config["heat_demand"])
        
        # RL parameters
        self._init_action_space(config["rl_env"]["action_space"])
        self._init_observation_space(config["rl_env"]["observation_space"])
        self.reward_config = config["rl_env"]["reward"]
        
        # State tracking
        self.current_step = 0
        self.episode_data = []
        
        # Electric heater
        self.heater_max_power = config["electric_heater"]["max_power"]
        self.heater_efficiency = config["electric_heater"]["efficiency"]
        
    def _init_storage(self, tes_config: Dict) -> None:
        """Initialize thermal storage model."""
        storage_type = tes_config["type"]
        
        if storage_type == "sensible":
            self.storage = SensibleHeatStorage(
                mass=tes_config["mass"],
                specific_heat=tes_config["specific_heat"],
                initial_temperature=tes_config["initial_temperature"],
                min_temperature=tes_config["min_temperature"],
                max_temperature=tes_config["max_temperature"],
                loss_coefficient=tes_config["loss_coefficient"],
                ambient_temperature=tes_config["ambient_temperature"],
            )
        elif storage_type == "pcm":
            self.storage = PCMStorage(
                mass=tes_config["mass"],
                specific_heat_solid=tes_config.get("specific_heat", 2.0),
                specific_heat_liquid=tes_config.get("specific_heat", 4.18),
                latent_heat=tes_config["latent_heat"],
                melting_point=tes_config["melting_point"],
                initial_temperature=tes_config["initial_temperature"],
                min_temperature=tes_config["min_temperature"],
                max_temperature=tes_config["max_temperature"],
                loss_coefficient=tes_config["loss_coefficient"],
                ambient_temperature=tes_config["ambient_temperature"],
            )
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    def _init_heat_exchanger(self, hx_config: Dict) -> None:
        """Initialize heat exchanger model."""
        self.heat_exchanger = create_heat_exchanger(hx_config)
    
    def _init_economics(self, tou_config: Dict) -> None:
        """Initialize economic model and TOU pricing."""
        self.tou = create_tou_pricing(tou_config)
        self.economic = EconomicModel(self.tou)
    
    def _init_demand_and_weather(self, demand_config: Dict) -> None:
        """Initialize heat demand and weather profiles."""
        demand_type = demand_config["type"]
        
        if demand_type == "synthetic":
            self.heat_demand = generate_demand_profile(
                hours=self.max_steps,
                base_load=demand_config["base_load"],
                peak_load=demand_config["peak_load"],
                seed=self.seed_value,
            )
        else:
            # Load from file (to be implemented)
            raise NotImplementedError("Loading demand from file not yet implemented")
        
        # Generate weather data
        self.weather_data = generate_weather_data(
            hours=self.max_steps,
            seed=self.seed_value,
        )
    
    def _init_action_space(self, action_config: Dict) -> None:
        """Initialize action space."""
        self.action_type = action_config["type"]
        
        if self.action_type == "discrete":
            # 0: idle, 1: charge, 2: discharge
            self.action_space = gym.spaces.Discrete(action_config["n_actions"])
            self.n_actions = action_config["n_actions"]
        else:  # continuous
            # Single continuous action: power [-max_power, +max_power]
            self.action_space = gym.spaces.Box(
                low=action_config["min_power"],
                high=action_config["max_power"],
                shape=(1,),
                dtype=np.float32,
            )
    
    def _init_observation_space(self, obs_config: Dict) -> None:
        """Initialize observation space."""
        self.obs_config = obs_config
        
        # Calculate observation dimension
        obs_dim = 0
        
        if obs_config["include_temperature"]:
            obs_dim += 1
        if obs_config["include_soc"]:
            obs_dim += 1
        if obs_config["include_price_forecast"]:
            obs_dim += obs_config["price_forecast_hours"]
        if obs_config["include_demand"]:
            obs_dim += 1
        if obs_config["include_time_features"]:
            obs_dim += 4  # hour_sin, hour_cos, day_sin, day_cos
        
        # All observations normalized to [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        
        # Reset models
        self.storage.reset()
        self.economic.reset()
        
        # Reset state
        self.current_step = 0
        self.episode_data = []
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int or np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: Agent's action
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to power
        power_command = self._action_to_power(action)
        
        # Get current conditions
        hour = self.current_step % 24
        current_demand = self.heat_demand[self.current_step]
        current_temp = self.weather_data["temperature"][self.current_step]
        
        # Determine charging/discharging power
        if power_command > 0:
            # Charging mode
            power_in = min(power_command, self.heater_max_power)
            power_out = 0.0
            grid_power = power_in / self.heater_efficiency
        elif power_command < 0:
            # Discharging mode
            power_in = 0.0
            power_out = min(abs(power_command), current_demand * 1.2)  # Limit to demand
            grid_power = 0.0  # Assuming no direct grid usage during discharge
        else:
            # Idle mode
            power_in = 0.0
            power_out = 0.0
            grid_power = 0.0
        
        # Update storage
        storage_state = self.storage.step(
            power_in=power_in,
            power_out=power_out,
            timestep=self.timestep,
        )
        
        # Calculate heat delivered to load via heat exchanger
        # Simplified: assume storage directly provides heat
        heat_delivered = power_out
        
        # Check demand satisfaction
        demand_deficit = max(0, current_demand - heat_delivered)
        if demand_deficit > 0.1:  # Small tolerance
            # Need to use backup (e.g., gas boiler or direct electric)
            backup_power = demand_deficit
            grid_power += backup_power  # Simplified: use electric backup
        else:
            backup_power = 0.0
        
        # Calculate cost
        cost_info = self.economic.calculate_step_cost(
            hour=hour,
            electricity_from_grid=grid_power,
            electricity_to_grid=0.0,  # No feed-in in this simplified model
            gas_consumption=0.0,  # No gas in this simplified model
            timestep=self.timestep,
        )
        
        # Calculate reward
        reward = self._calculate_reward(
            cost=cost_info["net_cost"],
            temperature_violation=storage_state["temperature_violation"],
            demand_deficit=demand_deficit,
            power_command=power_command,
        )
        
        # Store episode data
        self.episode_data.append({
            "step": self.current_step,
            "hour": hour,
            "action": action,
            "power_command": power_command,
            "temperature": storage_state["temperature"],
            "soc": storage_state["soc"],
            "heat_demand": current_demand,
            "heat_delivered": heat_delivered,
            "cost": cost_info["net_cost"],
            "reward": reward,
            "electricity_price": cost_info["electricity_price"],
        })
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _action_to_power(self, action: int or np.ndarray) -> float:
        """
        Convert action to power command.
        
        Args:
            action: Raw action from agent
            
        Returns:
            Power command (kW), positive for charging, negative for discharging
        """
        if self.action_type == "discrete":
            if self.n_actions == 3:
                # 0: idle, 1: charge, 2: discharge
                if action == 0:
                    return 0.0
                elif action == 1:
                    return self.heater_max_power
                else:  # action == 2
                    return -self.heater_max_power
            else:
                # More discrete levels
                power_levels = np.linspace(
                    -self.heater_max_power,
                    self.heater_max_power,
                    self.n_actions,
                )
                return power_levels[action]
        else:
            # Continuous action
            return float(action[0])
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            Normalized observation array
        """
        obs = []
        
        # Clamp index to avoid out-of-bounds access at terminal step
        idx = min(self.current_step, self.max_steps - 1)
        
        # Temperature (normalized)
        if self.obs_config["include_temperature"]:
            temp_norm = (self.storage.temperature - self.storage.min_temperature) / (
                self.storage.max_temperature - self.storage.min_temperature
            )
            obs.append(np.clip(temp_norm, 0.0, 1.0))
        
        # State of charge
        if self.obs_config["include_soc"]:
            obs.append(self.storage.get_state_of_charge())
        
        # Price forecast
        if self.obs_config["include_price_forecast"]:
            hour = idx % 24
            forecast_hours = self.obs_config["price_forecast_hours"]
            prices = self.tou.get_price_forecast(hour, forecast_hours)
            # Normalize prices to [0, 1]
            price_max = max(self.tou.peak_price, 1.0)
            prices_norm = prices / price_max
            obs.extend(prices_norm)
        
        # Heat demand (normalized)
        if self.obs_config["include_demand"]:
            max_demand = max(self.heat_demand)
            current_demand = self.heat_demand[idx]
            demand_norm = current_demand / max_demand if max_demand > 0 else 0.0
            obs.append(demand_norm)
        
        # Time features
        if self.obs_config["include_time_features"]:
            hour = idx % 24
            day = (idx // 24) % 365
            time_features = calculate_time_features(hour, day)
            # Map [-1, 1] to [0, 1]
            obs.append((time_features["hour_sin"] + 1) / 2)
            obs.append((time_features["hour_cos"] + 1) / 2)
            obs.append((time_features["day_sin"] + 1) / 2)
            obs.append((time_features["day_cos"] + 1) / 2)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self,
        cost: float,
        temperature_violation: float,
        demand_deficit: float,
        power_command: float,
    ) -> float:
        """
        Calculate reward for current step.
        
        Reward = cost_weight * cost 
                 - temp_penalty * temperature_violation
                 - demand_penalty * demand_deficit
                 - cycling_penalty * |power_change|
        
        Args:
            cost: Operational cost (CNY)
            temperature_violation: Temperature constraint violation (°C)
            demand_deficit: Unmet demand (kW)
            power_command: Power command (kW)
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Cost component (main objective)
        cost_weight = self.reward_config["cost_weight"]
        reward += cost_weight * cost
        
        # Temperature violation penalty
        if temperature_violation > 0:
            temp_penalty = self.reward_config["temperature_violation_penalty"]
            reward -= temp_penalty * temperature_violation
        
        # Demand deficit penalty
        if demand_deficit > 0:
            demand_penalty = self.reward_config["demand_violation_penalty"]
            reward -= demand_penalty * demand_deficit
        
        # Cycling penalty (optional, to reduce frequent switching)
        if len(self.episode_data) > 0:
            prev_power = self.episode_data[-1]["power_command"]
            power_change = abs(power_command - prev_power)
            cycling_penalty = self.reward_config["cycling_penalty"]
            reward -= cycling_penalty * power_change
        
        return reward
    
    def _get_info(self) -> Dict:
        """
        Get additional information about current state.
        
        Returns:
            Info dictionary
        """
        # Clamp index to avoid out-of-bounds access at terminal step
        idx = min(self.current_step, self.max_steps - 1)
        hour = idx % 24
        
        info = {
            "step": self.current_step,
            "hour": hour,
            "temperature": self.storage.temperature,
            "soc": self.storage.get_state_of_charge(),
            "electricity_price": self.tou.get_price(hour),
            "price_period": self.tou.get_period(hour),
            "heat_demand": self.heat_demand[idx],
            "total_cost": self.economic.total_cost,
        }
        
        return info
    
    def render(self, mode: str = "human") -> None:
        """
        Render the environment (optional, for debugging).
        
        Args:
            mode: Render mode
        """
        if len(self.episode_data) == 0:
            return
        
        latest = self.episode_data[-1]
        print(f"Step {latest['step']:4d} | "
              f"Hour {latest['hour']:2d} | "
              f"T={latest['temperature']:.1f}°C | "
              f"SoC={latest['soc']:.2f} | "
              f"Demand={latest['heat_demand']:.1f}kW | "
              f"Cost={latest['cost']:.2f}CNY | "
              f"Reward={latest['reward']:.2f}")
    
    def get_episode_data(self) -> list:
        """
        Get data from current episode.
        
        Returns:
            List of step data dictionaries
        """
        return self.episode_data
    
    def close(self) -> None:
        """Clean up resources."""
        pass
