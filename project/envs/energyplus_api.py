"""
EnergyPlus Python API Integration Module.

This module provides a clean interface to EnergyPlus Python API for
building simulation control in reinforcement learning.

Features:
    - EnergyPlus Runtime API integration
    - EMS actuator and sensor management
    - Callback-based simulation control
    - State exchange for RL environments

Requirements:
    - EnergyPlus 23.1+ with Python API support
    - pyenergyplus (included with EnergyPlus installation)

References:
    - https://energyplus.readthedocs.io/en/latest/api.html
    - EnergyPlus Python API Examples

Author: Auto-generated for Applied Energy publication
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import threading
import queue


@dataclass
class EnergyPlusConfig:
    """Configuration for EnergyPlus simulation."""
    
    # Paths
    idf_path: str = ""
    weather_path: str = ""
    output_path: str = "outputs/eplus_output"
    
    # Simulation settings
    timestep_per_hour: int = 4  # 15-minute intervals
    run_period_start: Tuple[int, int] = (1, 1)   # (month, day)
    run_period_end: Tuple[int, int] = (12, 31)   # (month, day)
    
    # EnergyPlus installation
    energyplus_path: Optional[str] = None  # Auto-detect if None


def find_energyplus_path() -> Optional[str]:
    """
    Auto-detect EnergyPlus installation path.
    
    Searches common installation locations on Windows and Linux.
    
    Returns:
        Path to EnergyPlus installation or None if not found
    """
    # Common installation paths
    common_paths = [
        # Windows paths
        r"C:\EnergyPlusV24-2-0",
        r"C:\EnergyPlusV24-1-0",
        r"C:\EnergyPlusV23-2-0",
        r"C:\EnergyPlusV23-1-0",
        # Linux/Mac paths
        "/usr/local/EnergyPlus-24-2-0",
        "/usr/local/EnergyPlus-24-1-0",
        "/usr/local/EnergyPlus-23-2-0",
        "/opt/EnergyPlus-23-2-0",
    ]
    
    # Check environment variable first
    env_path = os.environ.get("ENERGYPLUS_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Search common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


class EnergyPlusAPI:
    """
    EnergyPlus Python API wrapper for RL integration.
    
    This class manages:
    1. EnergyPlus runtime lifecycle
    2. EMS sensor registration and reading
    3. EMS actuator registration and setting
    4. Callback-based control flow
    
    Usage:
        api = EnergyPlusAPI(config)
        api.register_sensor("Zone Air Temperature", "Zone1")
        api.register_actuator("Schedule:Constant", "Setpoint", "HeatingSP")
        api.start_simulation()
    """
    
    def __init__(self, config: EnergyPlusConfig):
        """Initialize EnergyPlus API connection."""
        self.config = config
        
        # Find EnergyPlus installation
        self.eplus_path = config.energyplus_path or find_energyplus_path()
        if self.eplus_path is None:
            raise RuntimeError(
                "EnergyPlus installation not found. "
                "Set ENERGYPLUS_DIR environment variable or install EnergyPlus."
            )
        
        # Add EnergyPlus to Python path
        api_path = os.path.join(self.eplus_path, "python_api")
        if api_path not in sys.path:
            sys.path.insert(0, api_path)
        
        # Import EnergyPlus API
        try:
            from pyenergyplus.api import EnergyPlusAPI as EPlusAPI
            self._eplus_api = EPlusAPI()
        except ImportError as e:
            raise ImportError(
                f"Could not import pyenergyplus. "
                f"Make sure EnergyPlus is properly installed at {self.eplus_path}. "
                f"Error: {e}"
            )
        
        # State management
        self._state = None
        self._running = False
        self._step_count = 0
        
        # Sensor/Actuator handles
        self._sensors: Dict[str, int] = {}
        self._actuators: Dict[str, int] = {}
        self._sensor_values: Dict[str, float] = {}
        self._actuator_values: Dict[str, float] = {}
        
        # Callback synchronization
        self._action_queue = queue.Queue()
        self._observation_queue = queue.Queue()
        self._step_event = threading.Event()
        
        # Simulation thread
        self._sim_thread: Optional[threading.Thread] = None
    
    @property
    def exchange(self):
        """Get EnergyPlus data exchange interface."""
        return self._eplus_api.exchange
    
    @property
    def runtime(self):
        """Get EnergyPlus runtime interface."""
        return self._eplus_api.runtime
    
    def register_sensor(
        self,
        variable_name: str,
        variable_key: str,
        alias: Optional[str] = None,
    ) -> str:
        """
        Register an output variable as sensor for reading.
        
        Args:
            variable_name: EnergyPlus output variable name
            variable_key: Key (zone name, surface name, etc.)
            alias: Optional alias for accessing the sensor
            
        Returns:
            Sensor identifier (alias or auto-generated)
        """
        sensor_id = alias or f"{variable_name}_{variable_key}"
        self._sensors[sensor_id] = {
            "variable": variable_name,
            "key": variable_key,
            "handle": None,
        }
        return sensor_id
    
    def register_actuator(
        self,
        component_type: str,
        control_type: str,
        actuator_key: str,
        alias: Optional[str] = None,
    ) -> str:
        """
        Register an EMS actuator for control.
        
        Args:
            component_type: Type of component (e.g., "Schedule:Constant")
            control_type: Type of control (e.g., "Schedule Value")
            actuator_key: Key for the actuator
            alias: Optional alias for accessing the actuator
            
        Returns:
            Actuator identifier (alias or auto-generated)
        """
        actuator_id = alias or f"{component_type}_{actuator_key}"
        self._actuators[actuator_id] = {
            "component_type": component_type,
            "control_type": control_type,
            "key": actuator_key,
            "handle": None,
        }
        return actuator_id
    
    def _callback_begin_timestep(self, state) -> int:
        """
        Callback called at beginning of each timestep.
        
        This is where we:
        1. Read sensor values from previous timestep
        2. Wait for RL agent action
        3. Set actuator values for current timestep
        """
        # Initialize handles on first call
        if self._step_count == 0:
            self._initialize_handles(state)
        
        # Skip warmup period
        if not self.exchange.warmup_flag(state):
            # Read all sensors
            self._read_all_sensors(state)
            
            # Send observation to RL agent
            self._observation_queue.put(self._sensor_values.copy())
            
            # Wait for action from RL agent
            self._step_event.set()
            try:
                action = self._action_queue.get(timeout=30.0)
                self._apply_action(state, action)
            except queue.Empty:
                print("Warning: Timeout waiting for action, using default")
            
            self._step_count += 1
        
        return 0
    
    def _initialize_handles(self, state):
        """Initialize sensor and actuator handles."""
        # Get sensor handles
        for sensor_id, sensor_info in self._sensors.items():
            handle = self.exchange.get_variable_handle(
                state,
                sensor_info["variable"],
                sensor_info["key"],
            )
            if handle < 0:
                print(f"Warning: Could not get handle for sensor {sensor_id}")
            sensor_info["handle"] = handle
        
        # Get actuator handles
        for actuator_id, actuator_info in self._actuators.items():
            handle = self.exchange.get_actuator_handle(
                state,
                actuator_info["component_type"],
                actuator_info["control_type"],
                actuator_info["key"],
            )
            if handle < 0:
                print(f"Warning: Could not get handle for actuator {actuator_id}")
            actuator_info["handle"] = handle
    
    def _read_all_sensors(self, state):
        """Read all registered sensor values."""
        for sensor_id, sensor_info in self._sensors.items():
            handle = sensor_info["handle"]
            if handle >= 0:
                value = self.exchange.get_variable_value(state, handle)
                self._sensor_values[sensor_id] = value
    
    def _apply_action(self, state, action: Dict[str, float]):
        """Apply action to all registered actuators."""
        for actuator_id, value in action.items():
            if actuator_id in self._actuators:
                handle = self._actuators[actuator_id]["handle"]
                if handle >= 0:
                    self.exchange.set_actuator_value(state, handle, value)
    
    def _run_simulation(self):
        """Run EnergyPlus simulation in separate thread."""
        # Create state
        self._state = self._eplus_api.state_manager.new_state()
        
        # Register callback
        self.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            self._state,
            self._callback_begin_timestep,
        )
        
        # Create output directory
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run simulation
        self.runtime.run_energyplus(
            self._state,
            [
                "-d", str(output_dir),
                "-w", self.config.weather_path,
                self.config.idf_path,
            ]
        )
        
        # Cleanup
        self._running = False
        self._observation_queue.put(None)  # Signal end
    
    def start_simulation(self):
        """Start EnergyPlus simulation in background thread."""
        if self._running:
            raise RuntimeError("Simulation already running")
        
        self._running = True
        self._step_count = 0
        self._sim_thread = threading.Thread(target=self._run_simulation)
        self._sim_thread.start()
    
    def step(self, action: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Execute one simulation step with given action.
        
        Args:
            action: Dictionary of actuator_id -> value
            
        Returns:
            Dictionary of sensor_id -> value, or None if simulation ended
        """
        if not self._running:
            return None
        
        # Send action to simulation
        self._action_queue.put(action)
        
        # Wait for observation
        try:
            observation = self._observation_queue.get(timeout=60.0)
            return observation
        except queue.Empty:
            return None
    
    def get_observation(self) -> Optional[Dict[str, float]]:
        """Get current observation without stepping."""
        try:
            return self._observation_queue.get(timeout=60.0)
        except queue.Empty:
            return None
    
    def stop_simulation(self):
        """Stop EnergyPlus simulation."""
        if self._running and self._state is not None:
            self.runtime.stop_simulation(self._state)
            self._running = False
        
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=10.0)
    
    def reset(self):
        """Reset simulation for new episode."""
        self.stop_simulation()
        
        # Clear state
        if self._state is not None:
            self._eplus_api.state_manager.delete_state(self._state)
            self._state = None
        
        # Clear queues
        while not self._action_queue.empty():
            self._action_queue.get()
        while not self._observation_queue.empty():
            self._observation_queue.get()
        
        self._step_count = 0
        self._sensor_values.clear()
    
    def close(self):
        """Clean up resources."""
        self.stop_simulation()
        if self._state is not None:
            self._eplus_api.state_manager.delete_state(self._state)


# =============================================================================
# Building-specific sensor/actuator configurations
# =============================================================================

# Standard sensors for 5-zone building
STANDARD_SENSORS = [
    ("Zone Air Temperature", "Classroom_South", "temp_south"),
    ("Zone Air Temperature", "Classroom_East", "temp_east"),
    ("Zone Air Temperature", "Classroom_North", "temp_north"),
    ("Zone Air Temperature", "Classroom_West", "temp_west"),
    ("Zone Air Temperature", "Corridor", "temp_corridor"),
    ("Site Outdoor Air Drybulb Temperature", "Environment", "outdoor_temp"),
    ("Site Outdoor Air Relative Humidity", "Environment", "outdoor_rh"),
    ("Site Direct Solar Radiation Rate per Area", "Environment", "solar_direct"),
    ("Site Diffuse Solar Radiation Rate per Area", "Environment", "solar_diffuse"),
    ("Facility Total HVAC Electricity Demand Rate", "Whole Building", "hvac_power"),
]

# Standard actuators for setpoint control
STANDARD_ACTUATORS = [
    ("Schedule:Constant", "Schedule Value", "Heating_SP_Schedule", "heating_sp"),
    ("Schedule:Constant", "Schedule Value", "Cooling_SP_Schedule", "cooling_sp"),
    ("Schedule:Constant", "Schedule Value", "HP_Enable_Schedule", "hp_enable"),
    ("Schedule:Constant", "Schedule Value", "Chiller_Enable_Schedule", "chiller_enable"),
]


def create_building_api(
    idf_path: str,
    weather_path: str,
    output_path: str = "outputs/eplus_output",
) -> EnergyPlusAPI:
    """
    Create EnergyPlus API with standard building configuration.
    
    Args:
        idf_path: Path to IDF building model
        weather_path: Path to EPW weather file
        output_path: Path for EnergyPlus output files
        
    Returns:
        Configured EnergyPlusAPI instance
    """
    config = EnergyPlusConfig(
        idf_path=idf_path,
        weather_path=weather_path,
        output_path=output_path,
    )
    
    api = EnergyPlusAPI(config)
    
    # Register standard sensors
    for variable, key, alias in STANDARD_SENSORS:
        api.register_sensor(variable, key, alias)
    
    # Register standard actuators
    for comp_type, ctrl_type, key, alias in STANDARD_ACTUATORS:
        api.register_actuator(comp_type, ctrl_type, key, alias)
    
    return api


# =============================================================================
# Utility functions
# =============================================================================

def check_energyplus_installation() -> Dict[str, Any]:
    """
    Check EnergyPlus installation status.
    
    Returns:
        Dictionary with installation info and status
    """
    result = {
        "installed": False,
        "path": None,
        "version": None,
        "python_api": False,
    }
    
    eplus_path = find_energyplus_path()
    if eplus_path:
        result["installed"] = True
        result["path"] = eplus_path
        
        # Try to get version
        version_file = os.path.join(eplus_path, "Version.txt")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                result["version"] = f.read().strip()
        
        # Check Python API
        api_path = os.path.join(eplus_path, "python_api")
        if os.path.exists(api_path):
            result["python_api"] = True
    
    return result


if __name__ == "__main__":
    print("EnergyPlus API Module Test")
    print("=" * 50)
    
    # Check installation
    info = check_energyplus_installation()
    print(f"EnergyPlus installed: {info['installed']}")
    if info['installed']:
        print(f"  Path: {info['path']}")
        print(f"  Version: {info['version']}")
        print(f"  Python API: {info['python_api']}")
    else:
        print("\nEnergyPlus not found. Please install EnergyPlus 23.1+")
        print("Download: https://energyplus.net/downloads")
        print("Set ENERGYPLUS_DIR environment variable after installation")
