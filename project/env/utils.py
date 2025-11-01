"""Utility functions for the RL environment."""

import numpy as np
from typing import Dict, Tuple, Optional


def generate_demand_profile(
    hours: int = 8760,
    base_load: float = 30.0,
    peak_load: float = 80.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic heat demand profile.
    
    Creates a realistic daily pattern with:
    - Higher demand in morning (6-9) and evening (18-22)
    - Lower demand at night
    - Weekly patterns (weekday vs weekend)
    - Random variations
    
    Args:
        hours: Number of hours to generate
        base_load: Baseline heat demand (kW)
        peak_load: Peak heat demand (kW)
        seed: Random seed for reproducibility
        
    Returns:
        Array of heat demand values (kW)
    """
    if seed is not None:
        np.random.seed(seed)
    
    demand = np.zeros(hours)
    
    for h in range(hours):
        hour_of_day = h % 24
        day_of_week = (h // 24) % 7
        
        # Daily pattern
        if 6 <= hour_of_day < 9:  # Morning peak
            daily_factor = 0.8 + 0.2 * (hour_of_day - 6) / 3
        elif 18 <= hour_of_day < 22:  # Evening peak
            daily_factor = 0.9 + 0.1 * (21 - hour_of_day) / 4
        elif 0 <= hour_of_day < 6:  # Night
            daily_factor = 0.3
        else:  # Day time
            daily_factor = 0.6
        
        # Weekly pattern (reduced demand on weekends)
        if day_of_week >= 5:  # Weekend
            weekly_factor = 0.7
        else:  # Weekday
            weekly_factor = 1.0
        
        # Seasonal pattern (simplified)
        day_of_year = (h // 24) % 365
        seasonal_factor = 0.8 + 0.4 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        
        # Combine factors
        combined_factor = daily_factor * weekly_factor * seasonal_factor
        
        # Add random noise
        noise = np.random.normal(0, 0.05)
        combined_factor = max(0.2, combined_factor + noise)
        
        # Calculate demand
        demand[h] = base_load + (peak_load - base_load) * combined_factor
    
    return demand


def generate_weather_data(
    hours: int = 8760,
    mean_temp: float = 15.0,
    temp_range: float = 15.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic weather data.
    
    Args:
        hours: Number of hours to generate
        mean_temp: Mean annual temperature (°C)
        temp_range: Temperature variation range (°C)
        seed: Random seed
        
    Returns:
        Dictionary with 'temperature', 'solar_irradiance'
    """
    if seed is not None:
        np.random.seed(seed)
    
    temperature = np.zeros(hours)
    solar = np.zeros(hours)
    
    for h in range(hours):
        hour_of_day = h % 24
        day_of_year = (h // 24) % 365
        
        # Seasonal temperature variation
        seasonal_temp = mean_temp + temp_range * np.cos(
            2 * np.pi * (day_of_year - 200) / 365
        )
        
        # Daily temperature variation
        daily_variation = 5.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
        
        # Add noise
        noise = np.random.normal(0, 1.0)
        
        temperature[h] = seasonal_temp + daily_variation + noise
        
        # Solar irradiance (simplified)
        if 6 <= hour_of_day <= 18:
            solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
            seasonal_solar = 1.0 - 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            solar[h] = max(0, 800 * solar_factor * seasonal_solar * (1 + np.random.normal(0, 0.1)))
        else:
            solar[h] = 0.0
    
    return {
        "temperature": temperature,
        "solar_irradiance": solar,
    }


def normalize_observation(
    obs: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Normalize observation to [0, 1] range.
    
    Args:
        obs: Observation dictionary
        bounds: Dictionary of (min, max) bounds for each feature
        
    Returns:
        Normalized observation array
    """
    normalized = []
    
    for key, value in obs.items():
        if key in bounds:
            min_val, max_val = bounds[key]
            if max_val > min_val:
                norm_val = (value - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            normalized.append(np.clip(norm_val, 0.0, 1.0))
        else:
            # If bounds not specified, assume already normalized
            normalized.append(np.clip(value, 0.0, 1.0))
    
    return np.array(normalized, dtype=np.float32)


def denormalize_action(
    action: np.ndarray,
    action_space_type: str,
    bounds: Tuple[float, float],
) -> float:
    """
    Convert normalized action to actual power value.
    
    Args:
        action: Normalized action from agent
        action_space_type: 'discrete' or 'continuous'
        bounds: (min_power, max_power) in kW
        
    Returns:
        Actual power value (kW)
    """
    if action_space_type == "discrete":
        # action is an integer index
        return action  # Will be interpreted by environment
    else:
        # action is continuous in [-1, 1], map to [min_power, max_power]
        min_power, max_power = bounds
        power = (action + 1.0) / 2.0 * (max_power - min_power) + min_power
        return np.clip(power, min_power, max_power)


def calculate_time_features(hour: int, day: int) -> Dict[str, float]:
    """
    Calculate cyclical time features.
    
    Uses sine/cosine encoding for periodic features to help RL agent
    understand time patterns.
    
    Args:
        hour: Hour of day (0-23)
        day: Day of year (0-364)
        
    Returns:
        Dictionary with time features
    """
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 365)
    day_cos = np.cos(2 * np.pi * day / 365)
    
    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
    }


def check_energy_conservation(
    energy_in: float,
    energy_out: float,
    energy_stored_before: float,
    energy_stored_after: float,
    losses: float,
    tolerance: float = 0.01,
) -> bool:
    """
    Check energy conservation (for testing/validation).
    
    Energy balance: E_stored_after = E_stored_before + E_in - E_out - E_losses
    
    Args:
        energy_in: Energy input (kJ)
        energy_out: Energy output (kJ)
        energy_stored_before: Initial stored energy (kJ)
        energy_stored_after: Final stored energy (kJ)
        losses: Energy losses (kJ)
        tolerance: Relative tolerance for check
        
    Returns:
        True if energy is conserved within tolerance
    """
    expected = energy_stored_before + energy_in - energy_out - losses
    error = abs(energy_stored_after - expected)
    relative_error = error / max(abs(expected), 1.0)
    
    return relative_error < tolerance


def create_synthetic_dataset(
    n_episodes: int = 100,
    episode_length: int = 168,  # 1 week
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic dataset for training/testing.
    
    Args:
        n_episodes: Number of episodes to generate
        episode_length: Length of each episode (hours)
        save_path: Path to save dataset (optional)
        seed: Random seed
        
    Returns:
        Dictionary with demand, weather, and pricing data
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_hours = n_episodes * episode_length
    
    # Generate data
    demand = generate_demand_profile(total_hours, seed=seed)
    weather = generate_weather_data(total_hours, seed=seed)
    
    dataset = {
        "demand": demand,
        "temperature": weather["temperature"],
        "solar_irradiance": weather["solar_irradiance"],
        "n_episodes": n_episodes,
        "episode_length": episode_length,
    }
    
    if save_path:
        np.savez(save_path, **dataset)
    
    return dataset
