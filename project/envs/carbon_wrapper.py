"""
Carbon-Aware Gymnasium Wrapper for Sinergym environments.

This wrapper implements Innovation B: Carbon-Cost dual-objective optimization
by injecting real-time carbon intensity data into the reward function.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


class CarbonAwareWrapper(gym.Wrapper):
    """
    Wrapper that adds carbon emission tracking and carbon-aware rewards.
    
    The reward function is modified to:
        R = -(cost + λ_carbon * carbon_emission + λ_comfort * comfort_violation)
    
    Args:
        env: Base Sinergym environment
        carbon_file: Path to CSV with [timestamp, carbon_factor_kgCO2_per_kWh]
        price_file: Path to CSV with [timestamp, price_CNY_per_kWh]
        lambda_carbon: Weight for carbon emission penalty
        lambda_comfort: Weight for comfort violation penalty
        forecast_horizon: Number of future timesteps to include in observation
    """
    
    def __init__(  # 初始化方法
        self,  # 实例引用
        env: gym.Env,  # 基础Sinergym环境，将被包装
        carbon_file: str,  # 包含碳强度数据的CSV文件路径
        price_file: str,  # 包含电价数据的CSV文件路径
        lambda_carbon: float = 0.5,  # 奖励函数中碳排放惩罚的权重系数
        lambda_comfort: float = 10.0,  # 奖励函数中舒适度违规惩罚的权重系数
        forecast_horizon: int = 4,  # 观测中包含的未来时间步数
    ):
        """
        Initialize the CarbonAwareWrapper.
        
        Args:
            env: Base Sinergym environment to wrap
            carbon_file: Path to the CSV file containing carbon intensity data
            price_file: Path to the CSV file containing electricity price data
            lambda_carbon: Weight coefficient for carbon emission penalty in reward function
            lambda_comfort: Weight coefficient for comfort violation penalty in reward function
            forecast_horizon: Number of future time steps to include in observations
        """
        super().__init__(env)
        
        self.lambda_carbon = lambda_carbon
        self.lambda_comfort = lambda_comfort
        self.forecast_horizon = forecast_horizon
        
        # Load external data
        self.carbon_df = self._load_schedule(carbon_file, 'carbon_factor')
        self.price_df = self._load_schedule(price_file, 'price')
        
        # Extend observation space for forecast data
        base_obs_dim = env.observation_space.shape[0]
        # Add: future prices (N) + future carbon factors (N)
        new_obs_dim = base_obs_dim + 2 * forecast_horizon
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
        
        # Tracking metrics
        self.episode_cost = 0.0
        self.episode_carbon = 0.0
        self.episode_comfort_violations = 0
        
    def _load_schedule(self, filepath: str, value_col: str) -> pd.Series:
        """
        Load and preprocess schedule CSV file.
        
        Args:
            filepath: Path to the CSV file to load
            value_col: Name of the column containing the values to extract
            
        Returns:
            Series with hour as index and values from value_col
        """
        # Read CSV, skip comment lines starting with #
        df = pd.read_csv(filepath, comment='#')
        
        # Check if hour column exists
        if 'hour' not in df.columns:
            df['hour'] = range(len(df))
        
        # Expand 24-hour schedule to full year (8760 hours) if needed
        if len(df) == 24:
            # Repeat daily pattern for entire year
            hours = np.arange(8760)
            values = df[value_col].values
            full_data = pd.Series(
                [values[h % 24] for h in hours],
                index=hours
            )
            return full_data
        
        return df.set_index('hour')[value_col]
    
    def _get_current_hour(self, info: Dict) -> int:
        """
        Extract current simulation hour from info dict.
        
        Args:
            info: Dictionary containing environment information
            
        Returns:
            Current hour of the year (0-8759)
        """
        # Sinergym provides time info - adjust based on actual output format
        time_info = info.get('time', {})
        month = time_info.get('month', 1)
        day = time_info.get('day', 1)
        hour = time_info.get('hour', 0)
        
        # Convert to hour of year (0-8759)
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_of_year = sum(days_in_month[:month]) + day - 1
        hour_of_year = day_of_year * 24 + hour
        
        return hour_of_year % 8760
    
    def _get_forecast(self, current_hour: int) -> np.ndarray:
        """
        Get forecast data for future timesteps.
        
        Args:
            current_hour: Current hour of the year (0-8759)
            
        Returns:
            Array containing forecasted prices and carbon factors for future time steps
        """
        forecast = []
        
        for h in range(1, self.forecast_horizon + 1):
            future_hour = (current_hour + h) % 8760
            
            # Get price forecast
            price = self.price_df.get(future_hour, self.price_df.mean())
            forecast.append(price)
        
        for h in range(1, self.forecast_horizon + 1):
            future_hour = (current_hour + h) % 8760
            
            # Get carbon factor forecast
            carbon = self.carbon_df.get(future_hour, self.carbon_df.mean())
            forecast.append(carbon)
        
        return np.array(forecast, dtype=np.float32)
    
    def reset(  # 定义reset方法，用于重置环境和跟踪指标
        self,  # 实例方法，通过类实例调用
        seed: Optional[int] = None,  # 可选的随机种子参数，用于环境重置
        options: Optional[Dict] = None,  # 可选的字典参数，包含环境重置的额外选项
    ) -> Tuple[np.ndarray, Dict]:  # 返回类型为元组，包含numpy数组和字典
        """
        Reset environment and tracking metrics.
        
        Args:
            seed: Random seed for environment reset
            options: Additional options for environment reset
            
        Returns:
            Tuple of augmented observation and info dictionary
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset episode metrics
        self.episode_cost = 0.0
        self.episode_carbon = 0.0
        self.episode_comfort_violations = 0
        
        # Augment observation with forecast
        current_hour = self._get_current_hour(info)
        forecast = self._get_forecast(current_hour)
        augmented_obs = np.concatenate([obs, forecast])
        
        return augmented_obs.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step with carbon-aware reward calculation.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple containing:
                - Augmented observation with forecast data
                - Carbon-aware reward value
                - Termination flag
                - Truncation flag
                - Info dictionary with additional metrics
        """
        # Execute base environment step
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Get current time and external data
        current_hour = self._get_current_hour(info)
        current_price = self.price_df.get(current_hour, self.price_df.mean())
        current_carbon = self.carbon_df.get(current_hour, self.carbon_df.mean())
        
        # Calculate energy consumption (kWh)
        # Adjust based on actual Sinergym output variables
        power_kw = info.get('total_power_demand', 0) / 1000  # W to kW
        timestep_hours = self.env.timestep / 3600 if hasattr(self.env, 'timestep') else 1.0
        energy_kwh = power_kw * timestep_hours
        
        # Calculate costs and emissions
        cost = energy_kwh * current_price
        carbon_emission = energy_kwh * current_carbon
        
        # Calculate comfort violation
        comfort_violation = self._calculate_comfort_violation(info)
        
        # Compute new reward (Innovation B: dual-objective)
        reward = -(
            cost + 
            self.lambda_carbon * carbon_emission + 
            self.lambda_comfort * comfort_violation
        )
        
        # Update tracking metrics
        self.episode_cost += cost
        self.episode_carbon += carbon_emission
        if comfort_violation > 0:
            self.episode_comfort_violations += 1
        
        # Augment info with detailed metrics
        info['step_cost'] = cost
        info['step_carbon'] = carbon_emission
        info['step_comfort_violation'] = comfort_violation
        info['current_price'] = current_price
        info['current_carbon_factor'] = current_carbon
        info['episode_cost'] = self.episode_cost
        info['episode_carbon'] = self.episode_carbon
        
        # Augment observation with forecast
        forecast = self._get_forecast(current_hour)
        augmented_obs = np.concatenate([obs, forecast])
        
        return augmented_obs.astype(np.float32), reward, terminated, truncated, info
    
    def _calculate_comfort_violation(self, info: Dict) -> float:
        """
        Calculate thermal comfort violation penalty.
        
        Args:
            info: Dictionary containing environment information including zone temperatures
            
        Returns:
            Comfort violation value (quadratic penalty)
        """
        # Get zone temperature and setpoints from info
        # Adjust variable names based on actual Sinergym output
        T_zone = info.get('Zone Air Temperature', 22.0)
        T_set_heat = info.get('Zone Thermostat Heating Setpoint', 20.0)
        T_set_cool = info.get('Zone Thermostat Cooling Setpoint', 26.0)
        
        violation = 0.0
        if T_zone < T_set_heat:
            violation = (T_set_heat - T_zone) ** 2
        elif T_zone > T_set_cool:
            violation = (T_zone - T_set_cool) ** 2
        
        return violation
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Get accumulated episode metrics for logging.
        
        Returns:
            Dictionary containing total cost, carbon emissions, and comfort violations for the episode
        """
        return {
            'total_cost_CNY': self.episode_cost,
            'total_carbon_kgCO2': self.episode_carbon,
            'comfort_violations': self.episode_comfort_violations,
        }