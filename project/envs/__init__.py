"""
Custom Gymnasium environments for EnergyPlus-based building simulation.

Modules:
    - carbon_wrapper: Wrapper for carbon-aware reward calculation
    - builder: IDF model generation utilities (physics-based HP, stratified TES)
    - sinergym_env: Sinergym environment configuration and creation
    - gym_wrapper: Native Gymnasium wrapper for Transformer-based RL
"""

from envs.sinergym_env import (
    SimpleEnergyPlusEnv,
    create_test_env,
    create_env,
    create_wrapped_env,
    SINERGYM_AVAILABLE,
)

from envs.carbon_wrapper import CarbonAwareWrapper

from envs.gym_wrapper import (
    BuildingEnv,
    BuildingConfig,
    calculate_tes_soc,
    normalize_observation,
)

from envs.eplus_env import (
    EnergyPlusGymEnv,
    EnvConfig,
    create_env as create_eplus_env,
)

from envs.real_eplus_env import (
    RealEnergyPlusEnv,
    RealEnvConfig,
    check_eplus_setup,
    find_energyplus,
    run_standalone_simulation,
)

# Check EnergyPlus API availability
try:
    from envs.energyplus_api import (
        EnergyPlusAPI,
        EnergyPlusConfig,
        create_building_api,
        check_energyplus_installation,
        STANDARD_SENSORS,
        STANDARD_ACTUATORS,
    )
    ENERGYPLUS_API_AVAILABLE = True
except ImportError:
    ENERGYPLUS_API_AVAILABLE = False

__all__ = [
    # Main environment (recommended)
    'EnergyPlusGymEnv',
    'EnvConfig',
    'create_eplus_env',
    
    # Sinergym-based environments
    'SimpleEnergyPlusEnv',
    'create_test_env',
    'create_env',
    'create_wrapped_env',
    'SINERGYM_AVAILABLE',
    
    # Wrappers
    'CarbonAwareWrapper',
    
    # Direct EnergyPlus wrapper
    'BuildingEnv',
    'BuildingConfig',
    'calculate_tes_soc',
    'normalize_observation',
    
    # EnergyPlus API (if available)
    'ENERGYPLUS_API_AVAILABLE',
]