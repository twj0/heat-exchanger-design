"""
Sinergym Environment Configuration for Carbon-Aware Building Control.

This module creates a custom Sinergym environment using our generated IDF model
and integrates it with the CarbonAwareWrapper for dual-objective optimization.

Usage:
    from envs.sinergym_env import create_env, create_wrapped_env
    
    # Basic environment
    env = create_env(config)
    
    # With carbon-aware wrapper
    env = create_wrapped_env(config)
"""

import os
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

# Sinergym imports
try:
    import sinergym
    from sinergym.utils.wrappers import (
        NormalizeObservation,
        LoggerWrapper,
        MultiObsWrapper,
    )
    SINERGYM_AVAILABLE = True
except ImportError:
    SINERGYM_AVAILABLE = False
    print("Warning: sinergym not installed. Run: pip install sinergym")

# Local imports
from envs.carbon_wrapper import CarbonAwareWrapper


# =============================================================================
# Environment Configuration
# =============================================================================

# Define observation variables (what the agent sees)
OBSERVATION_VARIABLES = [
    # Zone temperatures
    'Zone Air Temperature(Classroom_South)',
    'Zone Air Temperature(Classroom_East)',
    'Zone Air Temperature(Classroom_North)',
    'Zone Air Temperature(Classroom_West)',
    'Zone Air Temperature(Corridor)',
    
    # Outdoor conditions
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Site Direct Solar Radiation Rate per Area(Environment)',
    
    # HVAC power
    'Facility Total HVAC Electricity Demand Rate(Whole Building)',
    
    # Time features (for schedule awareness)
    'hour',
    'day_of_week',
    'month',
]

# Define action variables (what the agent controls)
ACTION_VARIABLES = [
    # Zone thermostat setpoints
    'Zone Thermostat Heating Setpoint Temperature(Classroom_South)',
    'Zone Thermostat Cooling Setpoint Temperature(Classroom_South)',
]

# Action space bounds
ACTION_SPACE_LOW = np.array([18.0, 23.0])  # [heating_min, cooling_min]
ACTION_SPACE_HIGH = np.array([22.0, 26.0])  # [heating_max, cooling_max]

# Define output variables for meters
OUTPUT_VARIABLES = {
    'zone_air_temp': 'Zone Air Temperature',
    'outdoor_temp': 'Site Outdoor Air Drybulb Temperature',
    'hvac_power': 'Facility Total HVAC Electricity Demand Rate',
    'solar_radiation': 'Site Direct Solar Radiation Rate per Area',
}


# =============================================================================
# Sinergym Environment Configuration Dictionary
# =============================================================================

def get_env_config(
    idf_path: str = 'outputs/sim_building.idf',
    weather_path: str = 'data/weather/Shanghai_2024.epw',
    timestep: int = 4,  # timesteps per hour (15 min)
    run_period: Tuple[int, int, int, int] = (1, 1, 12, 31),
) -> Dict[str, Any]:
    """
    Generate Sinergym environment configuration dictionary.
    
    Args:
        idf_path: Path to IDF building model
        weather_path: Path to EPW weather file
        timestep: Timesteps per hour (4 = 15 min intervals)
        run_period: (start_month, start_day, end_month, end_day)
        
    Returns:
        Configuration dictionary for sinergym.make()
    """
    # Convert to absolute paths
    project_root = Path(__file__).parent.parent
    idf_abs = str(project_root / idf_path)
    weather_abs = str(project_root / weather_path)
    
    config = {
        # Building model
        'idf_file': idf_abs,
        'weather_file': weather_abs,
        
        # Simulation settings
        'timestep': timestep,
        'runperiod': run_period,
        
        # Observation space definition
        'observation_variables': [
            ('Zone Air Temperature', 'Classroom_South'),
            ('Zone Air Temperature', 'Classroom_East'),
            ('Zone Air Temperature', 'Classroom_North'),
            ('Zone Air Temperature', 'Classroom_West'),
            ('Zone Air Temperature', 'Corridor'),
            ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            ('Site Direct Solar Radiation Rate per Area', 'Environment'),
            ('Facility Total HVAC Electricity Demand Rate', 'Whole Building'),
        ],
        
        # Action space definition
        'action_variables': [
            ('Zone Thermostat Heating Setpoint Temperature', 'Classroom_South'),
            ('Zone Thermostat Cooling Setpoint Temperature', 'Classroom_South'),
        ],
        
        # Action bounds
        'action_space': gym.spaces.Box(
            low=ACTION_SPACE_LOW,
            high=ACTION_SPACE_HIGH,
            dtype=np.float32
        ),
        
        # Reward settings (base reward, will be overridden by wrapper)
        'reward_function': {
            'type': 'LinearReward',
            'energy_weight': 1.0,
            'comfort_weight': 1.0,
        },
    }
    
    return config


# =============================================================================
# Environment Creation Functions
# =============================================================================

def create_env(
    config_path: str = 'configs/experiment.yaml',
    env_name: str = 'CarbonAware-5Zone-v1',
) -> gym.Env:
    """
    Create base Sinergym environment without wrappers.
    
    Args:
        config_path: Path to experiment configuration YAML
        env_name: Name for the environment
        
    Returns:
        Sinergym Gymnasium environment
    """
    if not SINERGYM_AVAILABLE:
        raise ImportError("Sinergym not available. Please install: pip install sinergym")
    
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_abs = project_root / config_path
    
    with open(config_abs, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get environment config
    env_config = get_env_config(
        idf_path=config.get('building', {}).get('idf_path', 'outputs/sim_building.idf'),
        weather_path=config.get('environment', {}).get('weather_file', 'data/weather/Shanghai_2024.epw'),
        timestep=config.get('environment', {}).get('timestep', 900) // 900 * 4,  # convert to timesteps/hour
    )
    
    # Create environment using sinergym
    # Note: Sinergym 3.x uses a different API than earlier versions
    env = sinergym.make(
        env_name,
        **env_config
    )
    
    return env


def create_wrapped_env(
    config_path: str = 'configs/experiment.yaml',
    normalize: bool = True,
    use_carbon_wrapper: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Create Sinergym environment with all wrappers for training.
    
    This creates a fully configured environment with:
    1. Base Sinergym environment
    2. Observation normalization
    3. Carbon-aware reward wrapper
    
    Args:
        config_path: Path to experiment configuration YAML
        normalize: Whether to normalize observations
        use_carbon_wrapper: Whether to use carbon-aware reward
        seed: Random seed for reproducibility
        
    Returns:
        Wrapped Gymnasium environment
    """
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_abs = project_root / config_path
    
    with open(config_abs, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create base environment
    env = create_env(config_path)
    
    # Apply normalization wrapper
    if normalize:
        env = NormalizeObservation(env)
    
    # Apply carbon-aware wrapper
    if use_carbon_wrapper:
        carbon_config = config.get('carbon_wrapper', {})
        carbon_file = str(project_root / carbon_config.get('carbon_file', 'data/schedules/carbon_intensity.csv'))
        price_file = str(project_root / carbon_config.get('price_file', 'data/schedules/electricity_price.csv'))
        
        env = CarbonAwareWrapper(
            env,
            carbon_file=carbon_file,
            price_file=price_file,
            lambda_carbon=carbon_config.get('lambda_carbon', 0.5),
            lambda_comfort=carbon_config.get('lambda_comfort', 10.0),
            forecast_horizon=carbon_config.get('forecast_horizon', 4),
        )
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    return env


# =============================================================================
# Alternative: Direct EnergyPlus Environment (without Sinergym)
# =============================================================================

class SimpleEnergyPlusEnv(gym.Env):
    """
    A simplified EnergyPlus environment for testing without full Sinergym setup.
    
    This can be used for initial development and testing before integrating
    with the full Sinergym framework.
    
    Note: This is a mock environment for testing the wrapper logic.
    For actual training, use the Sinergym-based environment.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        n_zones: int = 5,
        timestep: int = 900,  # seconds
        episode_length: int = 672,  # 1 week at 15-min intervals
    ):
        """Initialize the mock environment."""
        super().__init__()
        
        self.n_zones = n_zones
        self.timestep = timestep
        self.episode_length = episode_length
        
        # Observation space: zone temps + outdoor temp + solar + power
        self.observation_space = gym.spaces.Box(
            low=-50.0,
            high=100.0,
            shape=(n_zones + 3,),
            dtype=np.float32
        )
        
        # Action space: heating and cooling setpoints
        self.action_space = gym.spaces.Box(
            low=np.array([18.0, 23.0]),
            high=np.array([22.0, 26.0]),
            dtype=np.float32
        )
        
        self._step_count = 0
        self._current_hour = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self._step_count = 0
        self._current_hour = 0
        
        # Generate initial observation
        obs = self._generate_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self._step_count += 1
        self._current_hour = (self._step_count * self.timestep // 3600) % 8760
        
        # Generate observation
        obs = self._generate_observation()
        
        # Simple reward (will be overridden by wrapper)
        reward = -np.sum(np.abs(obs[:self.n_zones] - 22.0))  # Comfort-based
        
        # Episode termination
        terminated = False
        truncated = self._step_count >= self.episode_length
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _generate_observation(self) -> np.ndarray:
        """Generate a realistic-looking observation."""
        # Zone temperatures (20-26°C with some noise)
        zone_temps = 22.0 + np.random.randn(self.n_zones) * 2.0
        
        # Outdoor temperature (seasonal variation)
        day = self._current_hour // 24
        hour = self._current_hour % 24
        outdoor_temp = 15.0 + 10.0 * np.sin(2 * np.pi * day / 365) + \
                       5.0 * np.sin(2 * np.pi * hour / 24) + np.random.randn() * 2.0
        
        # Solar radiation (0 at night, peak at noon)
        solar = max(0, 600 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        solar += np.random.randn() * 50
        solar = max(0, solar)
        
        # HVAC power (10-100 kW)
        power = 30000 + 20000 * np.random.rand()
        
        obs = np.concatenate([
            zone_temps,
            [outdoor_temp, solar, power]
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get current environment info."""
        hour = self._current_hour % 24
        day = self._current_hour // 24
        month = day // 30 + 1
        
        return {
            'time': {
                'hour': hour,
                'day': day % 30 + 1,
                'month': min(month, 12),
            },
            'total_power_demand': 30000 + 20000 * np.random.rand(),
            'Zone Air Temperature': 22.0 + np.random.randn() * 2.0,
            'Zone Thermostat Heating Setpoint': 20.0,
            'Zone Thermostat Cooling Setpoint': 26.0,
        }


def create_test_env(
    config_path: str = 'configs/experiment.yaml',
    use_carbon_wrapper: bool = True,
) -> gym.Env:
    """
    Create a test environment for development without EnergyPlus.
    
    Args:
        config_path: Path to configuration file
        use_carbon_wrapper: Whether to apply carbon wrapper
        
    Returns:
        Test environment (with or without carbon wrapper)
    """
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_abs = project_root / config_path
    
    with open(config_abs, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create mock environment
    env = SimpleEnergyPlusEnv(
        n_zones=5,
        timestep=config.get('environment', {}).get('timestep', 900),
        episode_length=config.get('environment', {}).get('episode_length', 672),
    )
    
    # Apply carbon wrapper if requested
    if use_carbon_wrapper:
        carbon_config = config.get('carbon_wrapper', {})
        carbon_file = str(project_root / carbon_config.get('carbon_file', 'data/schedules/carbon_intensity.csv'))
        price_file = str(project_root / carbon_config.get('price_file', 'data/schedules/electricity_price.csv'))
        
        env = CarbonAwareWrapper(
            env,
            carbon_file=carbon_file,
            price_file=price_file,
            lambda_carbon=carbon_config.get('lambda_carbon', 0.5),
            lambda_comfort=carbon_config.get('lambda_comfort', 10.0),
            forecast_horizon=carbon_config.get('forecast_horizon', 4),
        )
    
    return env


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sinergym Environment Test")
    print("=" * 60)
    
    # Test with mock environment first
    print("\n[1] Creating test environment...")
    try:
        env = create_test_env(use_carbon_wrapper=True)
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        
        print("\n[2] Testing reset...")
        obs, info = env.reset(seed=42)
        print(f"  Observation shape: {obs.shape}")
        print(f"  Info keys: {list(info.keys())}")
        
        print("\n[3] Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Step cost: {info.get('step_cost', 'N/A')}")
        print(f"  Step carbon: {info.get('step_carbon', 'N/A')}")
        
        print("\n[4] Running short episode...")
        obs, info = env.reset()
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode metrics: {metrics}")
        
        env.close()
        print("\n✅ Test environment working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try full Sinergym environment if available
    if SINERGYM_AVAILABLE:
        print("\n" + "=" * 60)
        print("[5] Testing full Sinergym environment...")
        try:
            # Note: This requires EnergyPlus to be properly configured
            env = create_wrapped_env()
            print("  Sinergym environment created successfully!")
            env.close()
        except Exception as e:
            print(f"  Sinergym setup issue: {e}")
            print("  (This is expected if EnergyPlus paths are not configured)")
