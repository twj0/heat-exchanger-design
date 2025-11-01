"""Thermal Energy Storage (TES) models.

This module implements both sensible heat storage and phase change material (PCM)
storage models with proper energy balance equations and physical constraints.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class ThermalStorage(ABC):
    """Abstract base class for thermal energy storage systems."""
    
    def __init__(
        self,
        mass: float,
        initial_temperature: float,
        min_temperature: float,
        max_temperature: float,
        loss_coefficient: float,
        ambient_temperature: float,
    ):
        """
        Initialize thermal storage system.
        
        Args:
            mass: Mass of storage material (kg)
            initial_temperature: Initial temperature (°C)
            min_temperature: Minimum allowed temperature (°C)
            max_temperature: Maximum allowed temperature (°C)
            loss_coefficient: Heat loss coefficient (W/K)
            ambient_temperature: Ambient temperature (°C)
        """
        self.mass = mass
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.loss_coefficient = loss_coefficient
        self.ambient_temperature = ambient_temperature
        
    @abstractmethod
    def step(
        self, 
        power_in: float, 
        power_out: float, 
        timestep: float,
        efficiency_charge: float = 0.98,
        efficiency_discharge: float = 0.95,
    ) -> Dict[str, float]:
        """
        Update storage state for one timestep.
        
        Args:
            power_in: Charging power (kW)
            power_out: Discharging power (kW)
            timestep: Time step duration (seconds)
            efficiency_charge: Charging efficiency
            efficiency_discharge: Discharging efficiency
            
        Returns:
            Dictionary with updated state information
        """
        pass
    
    @abstractmethod
    def get_state_of_charge(self) -> float:
        """Get current state of charge (0-1)."""
        pass
    
    def get_heat_losses(self) -> float:
        """
        Calculate heat losses to ambient.
        
        Returns:
            Heat loss rate (kW)
        """
        delta_t = self.temperature - self.ambient_temperature
        q_loss = self.loss_coefficient * delta_t / 1000.0  # Convert W to kW
        return max(0, q_loss)
    
    def reset(self) -> None:
        """Reset storage to initial state."""
        self.temperature = self.initial_temperature
    
    def is_temperature_valid(self) -> bool:
        """Check if current temperature is within limits."""
        return self.min_temperature <= self.temperature <= self.max_temperature


class SensibleHeatStorage(ThermalStorage):
    """Sensible heat thermal energy storage model."""
    
    def __init__(
        self,
        mass: float,
        specific_heat: float,
        initial_temperature: float,
        min_temperature: float,
        max_temperature: float,
        loss_coefficient: float,
        ambient_temperature: float,
    ):
        """
        Initialize sensible heat storage.
        
        Args:
            mass: Mass of storage material (kg)
            specific_heat: Specific heat capacity (kJ/kg·K)
            initial_temperature: Initial temperature (°C)
            min_temperature: Minimum temperature (°C)
            max_temperature: Maximum temperature (°C)
            loss_coefficient: Heat loss coefficient (W/K)
            ambient_temperature: Ambient temperature (°C)
        """
        super().__init__(
            mass, initial_temperature, min_temperature, 
            max_temperature, loss_coefficient, ambient_temperature
        )
        self.specific_heat = specific_heat
        
    def step(
        self,
        power_in: float,
        power_out: float,
        timestep: float,
        efficiency_charge: float = 0.98,
        efficiency_discharge: float = 0.95,
    ) -> Dict[str, float]:
        """
        Update temperature based on energy balance.
        
        Energy balance equation:
        dE/dt = η_charge * P_in - P_out/η_discharge - Q_loss
        
        For sensible heat:
        E = m * c_p * T
        dE/dt = m * c_p * dT/dt
        
        Therefore:
        dT/dt = (η_charge * P_in - P_out/η_discharge - Q_loss) / (m * c_p)
        
        Args:
            power_in: Charging power (kW)
            power_out: Discharging power (kW)
            timestep: Time step (seconds)
            efficiency_charge: Charging efficiency
            efficiency_discharge: Discharging efficiency
            
        Returns:
            State dictionary with temperature, SoC, losses, etc.
        """
        # Get heat losses
        q_loss = self.get_heat_losses()
        
        # Calculate net power (kW)
        p_charge = efficiency_charge * power_in
        p_discharge = power_out / efficiency_discharge
        p_net = p_charge - p_discharge - q_loss
        
        # Calculate temperature change
        # Convert: kW * s = kJ, divide by (kg * kJ/kg·K) = K
        energy_change = p_net * timestep  # kJ
        heat_capacity = self.mass * self.specific_heat  # kJ/K
        delta_t = energy_change / heat_capacity  # K or °C
        
        # Update temperature
        new_temperature = self.temperature + delta_t
        
        # Apply constraints
        temperature_violation = 0.0
        if new_temperature < self.min_temperature:
            temperature_violation = self.min_temperature - new_temperature
            new_temperature = self.min_temperature
        elif new_temperature > self.max_temperature:
            temperature_violation = new_temperature - self.max_temperature
            new_temperature = self.max_temperature
            
        self.temperature = new_temperature
        
        return {
            "temperature": self.temperature,
            "soc": self.get_state_of_charge(),
            "power_charge": p_charge,
            "power_discharge": p_discharge,
            "heat_loss": q_loss,
            "temperature_violation": temperature_violation,
            "energy_stored": self.get_stored_energy(),
        }
    
    def get_state_of_charge(self) -> float:
        """
        Calculate state of charge based on temperature.
        
        SoC = (T - T_min) / (T_max - T_min)
        
        Returns:
            State of charge (0-1)
        """
        temp_range = self.max_temperature - self.min_temperature
        if temp_range == 0:
            return 0.5
        soc = (self.temperature - self.min_temperature) / temp_range
        return np.clip(soc, 0.0, 1.0)
    
    def get_stored_energy(self) -> float:
        """
        Get total stored energy relative to minimum temperature.
        
        Returns:
            Stored energy (kJ)
        """
        return self.mass * self.specific_heat * (
            self.temperature - self.min_temperature
        )
    
    def get_max_capacity(self) -> float:
        """
        Get maximum storage capacity.
        
        Returns:
            Maximum energy capacity (kJ)
        """
        return self.mass * self.specific_heat * (
            self.max_temperature - self.min_temperature
        )


class PCMStorage(ThermalStorage):
    """Phase Change Material (PCM) thermal energy storage model."""
    
    def __init__(
        self,
        mass: float,
        specific_heat_solid: float,
        specific_heat_liquid: float,
        latent_heat: float,
        melting_point: float,
        initial_temperature: float,
        min_temperature: float,
        max_temperature: float,
        loss_coefficient: float,
        ambient_temperature: float,
    ):
        """
        Initialize PCM storage with three-zone model.
        
        Args:
            mass: Mass of PCM (kg)
            specific_heat_solid: Specific heat in solid phase (kJ/kg·K)
            specific_heat_liquid: Specific heat in liquid phase (kJ/kg·K)
            latent_heat: Latent heat of fusion (kJ/kg)
            melting_point: Melting/freezing temperature (°C)
            initial_temperature: Initial temperature (°C)
            min_temperature: Minimum temperature (°C)
            max_temperature: Maximum temperature (°C)
            loss_coefficient: Heat loss coefficient (W/K)
            ambient_temperature: Ambient temperature (°C)
        """
        super().__init__(
            mass, initial_temperature, min_temperature,
            max_temperature, loss_coefficient, ambient_temperature
        )
        self.specific_heat_solid = specific_heat_solid
        self.specific_heat_liquid = specific_heat_liquid
        self.latent_heat = latent_heat
        self.melting_point = melting_point
        
        # Initialize liquid fraction
        self.liquid_fraction = self._calculate_initial_liquid_fraction()
        
    def _calculate_initial_liquid_fraction(self) -> float:
        """Calculate initial liquid fraction based on temperature."""
        if self.temperature < self.melting_point:
            return 0.0
        elif self.temperature > self.melting_point:
            return 1.0
        else:
            return 0.5  # Assume mid-transition
    
    def step(
        self,
        power_in: float,
        power_out: float,
        timestep: float,
        efficiency_charge: float = 0.98,
        efficiency_discharge: float = 0.95,
    ) -> Dict[str, float]:
        """
        Update PCM state using enthalpy method.
        
        The model handles three zones:
        1. Solid phase (T < T_melt)
        2. Phase change zone (T = T_melt)
        3. Liquid phase (T > T_melt)
        
        Args:
            power_in: Charging power (kW)
            power_out: Discharging power (kW)
            timestep: Time step (seconds)
            efficiency_charge: Charging efficiency
            efficiency_discharge: Discharging efficiency
            
        Returns:
            State dictionary
        """
        # Get heat losses
        q_loss = self.get_heat_losses()
        
        # Calculate net power
        p_charge = efficiency_charge * power_in
        p_discharge = power_out / efficiency_discharge
        p_net = p_charge - p_discharge - q_loss
        
        # Energy change (kJ)
        energy_change = p_net * timestep
        
        # Determine current phase and update
        temperature_violation = 0.0
        
        if self.temperature < self.melting_point:
            # Solid phase
            delta_t = energy_change / (self.mass * self.specific_heat_solid)
            new_temp = self.temperature + delta_t
            
            if new_temp >= self.melting_point and energy_change > 0:
                # Transition to melting
                energy_to_melt = self.mass * self.specific_heat_solid * (
                    self.melting_point - self.temperature
                )
                remaining_energy = energy_change - energy_to_melt
                self.temperature = self.melting_point
                
                # Use remaining energy for phase change
                max_latent = self.mass * self.latent_heat
                if remaining_energy < max_latent:
                    self.liquid_fraction = remaining_energy / max_latent
                else:
                    self.liquid_fraction = 1.0
                    excess_energy = remaining_energy - max_latent
                    self.temperature += excess_energy / (
                        self.mass * self.specific_heat_liquid
                    )
            else:
                self.temperature = new_temp
                
        elif self.temperature > self.melting_point:
            # Liquid phase
            delta_t = energy_change / (self.mass * self.specific_heat_liquid)
            new_temp = self.temperature + delta_t
            
            if new_temp <= self.melting_point and energy_change < 0:
                # Transition to freezing
                energy_to_cool = self.mass * self.specific_heat_liquid * (
                    self.temperature - self.melting_point
                )
                remaining_energy = abs(energy_change) - energy_to_cool
                self.temperature = self.melting_point
                
                # Use remaining energy for phase change
                max_latent = self.mass * self.latent_heat
                if remaining_energy < max_latent:
                    self.liquid_fraction = 1.0 - (remaining_energy / max_latent)
                else:
                    self.liquid_fraction = 0.0
                    excess_energy = remaining_energy - max_latent
                    self.temperature -= excess_energy / (
                        self.mass * self.specific_heat_solid
                    )
            else:
                self.temperature = new_temp
                
        else:
            # At melting point - phase change
            max_latent = self.mass * self.latent_heat
            delta_fraction = energy_change / max_latent
            self.liquid_fraction = np.clip(
                self.liquid_fraction + delta_fraction, 0.0, 1.0
            )
            
        # Apply temperature constraints
        if self.temperature < self.min_temperature:
            temperature_violation = self.min_temperature - self.temperature
            self.temperature = self.min_temperature
            self.liquid_fraction = 0.0
        elif self.temperature > self.max_temperature:
            temperature_violation = self.temperature - self.max_temperature
            self.temperature = self.max_temperature
            self.liquid_fraction = 1.0
            
        return {
            "temperature": self.temperature,
            "soc": self.get_state_of_charge(),
            "liquid_fraction": self.liquid_fraction,
            "power_charge": p_charge,
            "power_discharge": p_discharge,
            "heat_loss": q_loss,
            "temperature_violation": temperature_violation,
            "energy_stored": self.get_stored_energy(),
        }
    
    def get_state_of_charge(self) -> float:
        """
        Calculate SoC considering both sensible and latent heat.
        
        Returns:
            State of charge (0-1)
        """
        # Energy components
        if self.temperature < self.melting_point:
            # Solid phase
            sensible = self.mass * self.specific_heat_solid * (
                self.temperature - self.min_temperature
            )
            latent = 0.0
        elif self.temperature > self.melting_point:
            # Liquid phase
            sensible_solid = self.mass * self.specific_heat_solid * (
                self.melting_point - self.min_temperature
            )
            latent = self.mass * self.latent_heat
            sensible_liquid = self.mass * self.specific_heat_liquid * (
                self.temperature - self.melting_point
            )
            sensible = sensible_solid + latent + sensible_liquid
        else:
            # Phase change
            sensible_solid = self.mass * self.specific_heat_solid * (
                self.melting_point - self.min_temperature
            )
            latent = self.liquid_fraction * self.mass * self.latent_heat
            sensible = sensible_solid + latent
            
        max_capacity = self.get_max_capacity()
        if max_capacity == 0:
            return 0.5
        return np.clip(sensible / max_capacity, 0.0, 1.0)
    
    def get_stored_energy(self) -> float:
        """Get total stored energy (kJ)."""
        return self.get_state_of_charge() * self.get_max_capacity()
    
    def get_max_capacity(self) -> float:
        """Get maximum storage capacity (kJ)."""
        sensible_solid = self.mass * self.specific_heat_solid * (
            self.melting_point - self.min_temperature
        )
        latent = self.mass * self.latent_heat
        sensible_liquid = self.mass * self.specific_heat_liquid * (
            self.max_temperature - self.melting_point
        )
        return sensible_solid + latent + sensible_liquid
    
    def reset(self) -> None:
        """Reset storage to initial state."""
        super().reset()
        self.liquid_fraction = self._calculate_initial_liquid_fraction()
