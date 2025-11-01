"""Heat exchanger models.

This module implements different heat exchanger modeling methods including
Effectiveness-NTU and LMTD (Log Mean Temperature Difference) approaches.
"""

import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod


class HeatExchanger(ABC):
    """Abstract base class for heat exchanger models."""
    
    def __init__(
        self,
        heat_transfer_area: float,
        overall_heat_transfer_coefficient: float,
    ):
        """
        Initialize heat exchanger.
        
        Args:
            heat_transfer_area: Heat transfer area (m²)
            overall_heat_transfer_coefficient: U value (kW/m²·K)
        """
        self.area = heat_transfer_area
        self.u_value = overall_heat_transfer_coefficient
        
    @abstractmethod
    def calculate_heat_transfer(
        self,
        mass_flow_hot: float,
        mass_flow_cold: float,
        temp_hot_in: float,
        temp_cold_in: float,
        cp_hot: float,
        cp_cold: float,
    ) -> Dict[str, float]:
        """
        Calculate heat transfer and outlet temperatures.
        
        Args:
            mass_flow_hot: Hot fluid mass flow rate (kg/s)
            mass_flow_cold: Cold fluid mass flow rate (kg/s)
            temp_hot_in: Hot fluid inlet temperature (°C)
            temp_cold_in: Cold fluid inlet temperature (°C)
            cp_hot: Hot fluid specific heat (kJ/kg·K)
            cp_cold: Cold fluid specific heat (kJ/kg·K)
            
        Returns:
            Dictionary with heat transfer rate and outlet temperatures
        """
        pass


class EffectivenessNTU(HeatExchanger):
    """Effectiveness-NTU method for heat exchanger analysis."""
    
    def __init__(
        self,
        heat_transfer_area: float,
        overall_heat_transfer_coefficient: float,
        flow_arrangement: str = "counterflow",
        effectiveness: Optional[float] = None,
    ):
        """
        Initialize Effectiveness-NTU heat exchanger.
        
        Args:
            heat_transfer_area: Heat transfer area (m²)
            overall_heat_transfer_coefficient: U value (kW/m²·K)
            flow_arrangement: Type of flow ('parallel', 'counterflow', 'crossflow')
            effectiveness: Fixed effectiveness (if None, calculate from NTU)
        """
        super().__init__(heat_transfer_area, overall_heat_transfer_coefficient)
        self.flow_arrangement = flow_arrangement.lower()
        self.fixed_effectiveness = effectiveness
        
    def calculate_heat_transfer(
        self,
        mass_flow_hot: float,
        mass_flow_cold: float,
        temp_hot_in: float,
        temp_cold_in: float,
        cp_hot: float = 4.18,  # Water default
        cp_cold: float = 4.18,
    ) -> Dict[str, float]:
        """
        Calculate heat transfer using Effectiveness-NTU method.
        
        The method follows:
        1. Calculate heat capacity rates: C = m * cp
        2. Find C_min and C_max
        3. Calculate NTU = U*A / C_min
        4. Calculate effectiveness based on flow arrangement
        5. Calculate actual heat transfer: Q = ε * C_min * (T_h,in - T_c,in)
        6. Calculate outlet temperatures
        
        Args:
            mass_flow_hot: Hot fluid mass flow (kg/s)
            mass_flow_cold: Cold fluid mass flow (kg/s)
            temp_hot_in: Hot fluid inlet temp (°C)
            temp_cold_in: Cold fluid inlet temp (°C)
            cp_hot: Hot fluid specific heat (kJ/kg·K)
            cp_cold: Cold fluid specific heat (kJ/kg·K)
            
        Returns:
            Dictionary with Q, T_hot_out, T_cold_out, effectiveness, NTU
        """
        # Handle edge cases
        if mass_flow_hot <= 0 or mass_flow_cold <= 0:
            return {
                "heat_transfer": 0.0,
                "temp_hot_out": temp_hot_in,
                "temp_cold_out": temp_cold_in,
                "effectiveness": 0.0,
                "ntu": 0.0,
            }
            
        if abs(temp_hot_in - temp_cold_in) < 0.01:
            return {
                "heat_transfer": 0.0,
                "temp_hot_out": temp_hot_in,
                "temp_cold_out": temp_cold_in,
                "effectiveness": 0.0,
                "ntu": 0.0,
            }
        
        # Calculate heat capacity rates (kW/K)
        c_hot = mass_flow_hot * cp_hot
        c_cold = mass_flow_cold * cp_cold
        c_min = min(c_hot, c_cold)
        c_max = max(c_hot, c_cold)
        
        # Calculate capacity rate ratio
        c_ratio = c_min / c_max if c_max > 0 else 0.0
        
        # Calculate NTU
        ntu = self.u_value * self.area / c_min
        
        # Calculate effectiveness
        if self.fixed_effectiveness is not None:
            effectiveness = self.fixed_effectiveness
        else:
            effectiveness = self._calculate_effectiveness(ntu, c_ratio)
        
        # Maximum possible heat transfer
        q_max = c_min * abs(temp_hot_in - temp_cold_in)
        
        # Actual heat transfer
        q_actual = effectiveness * q_max
        
        # Calculate outlet temperatures
        temp_hot_out = temp_hot_in - q_actual / c_hot
        temp_cold_out = temp_cold_in + q_actual / c_cold
        
        return {
            "heat_transfer": q_actual,
            "temp_hot_out": temp_hot_out,
            "temp_cold_out": temp_cold_out,
            "effectiveness": effectiveness,
            "ntu": ntu,
            "c_ratio": c_ratio,
        }
    
    def _calculate_effectiveness(self, ntu: float, c_ratio: float) -> float:
        """
        Calculate effectiveness based on NTU and flow arrangement.
        
        Args:
            ntu: Number of transfer units
            c_ratio: Capacity rate ratio (C_min/C_max)
            
        Returns:
            Effectiveness (0-1)
        """
        if self.flow_arrangement == "counterflow":
            if c_ratio < 0.99:
                effectiveness = (1 - np.exp(-ntu * (1 - c_ratio))) / (
                    1 - c_ratio * np.exp(-ntu * (1 - c_ratio))
                )
            else:
                # Special case when C_hot = C_cold
                effectiveness = ntu / (1 + ntu)
                
        elif self.flow_arrangement == "parallel":
            effectiveness = (1 - np.exp(-ntu * (1 + c_ratio))) / (1 + c_ratio)
            
        elif self.flow_arrangement == "crossflow":
            # Simplified crossflow (both fluids unmixed)
            effectiveness = 1 - np.exp(
                (np.exp(-ntu * c_ratio) - 1) * (c_ratio / ntu)
            )
            
        else:
            # Default to counterflow
            if c_ratio < 0.99:
                effectiveness = (1 - np.exp(-ntu * (1 - c_ratio))) / (
                    1 - c_ratio * np.exp(-ntu * (1 - c_ratio))
                )
            else:
                effectiveness = ntu / (1 + ntu)
        
        return np.clip(effectiveness, 0.0, 1.0)


class LMTD(HeatExchanger):
    """Log Mean Temperature Difference (LMTD) method."""
    
    def __init__(
        self,
        heat_transfer_area: float,
        overall_heat_transfer_coefficient: float,
        flow_arrangement: str = "counterflow",
    ):
        """
        Initialize LMTD heat exchanger.
        
        Args:
            heat_transfer_area: Heat transfer area (m²)
            overall_heat_transfer_coefficient: U value (kW/m²·K)
            flow_arrangement: Type of flow ('parallel', 'counterflow')
        """
        super().__init__(heat_transfer_area, overall_heat_transfer_coefficient)
        self.flow_arrangement = flow_arrangement.lower()
        
    def calculate_heat_transfer(
        self,
        mass_flow_hot: float,
        mass_flow_cold: float,
        temp_hot_in: float,
        temp_cold_in: float,
        cp_hot: float = 4.18,
        cp_cold: float = 4.18,
    ) -> Dict[str, float]:
        """
        Calculate heat transfer using LMTD method.
        
        This is an iterative method:
        1. Assume outlet temperatures
        2. Calculate LMTD
        3. Calculate Q = U*A*LMTD
        4. Calculate outlet temperatures from energy balance
        5. Iterate until convergence
        
        Args:
            mass_flow_hot: Hot fluid mass flow (kg/s)
            mass_flow_cold: Cold fluid mass flow (kg/s)
            temp_hot_in: Hot fluid inlet temp (°C)
            temp_cold_in: Cold fluid inlet temp (°C)
            cp_hot: Hot fluid specific heat (kJ/kg·K)
            cp_cold: Cold fluid specific heat (kJ/kg·K)
            
        Returns:
            Dictionary with Q, outlet temperatures, LMTD
        """
        # Handle edge cases
        if mass_flow_hot <= 0 or mass_flow_cold <= 0:
            return {
                "heat_transfer": 0.0,
                "temp_hot_out": temp_hot_in,
                "temp_cold_out": temp_cold_in,
                "lmtd": 0.0,
            }
        
        # Heat capacity rates
        c_hot = mass_flow_hot * cp_hot
        c_cold = mass_flow_cold * cp_cold
        
        # Initial guess for outlet temperatures
        temp_hot_out = temp_hot_in - 5.0
        temp_cold_out = temp_cold_in + 5.0
        
        # Iterative solution
        max_iterations = 50
        tolerance = 0.01
        
        for _ in range(max_iterations):
            # Calculate temperature differences
            if self.flow_arrangement == "counterflow":
                dt1 = temp_hot_in - temp_cold_out
                dt2 = temp_hot_out - temp_cold_in
            else:  # parallel flow
                dt1 = temp_hot_in - temp_cold_in
                dt2 = temp_hot_out - temp_cold_out
            
            # Avoid division by zero and log of negative/zero
            if dt1 <= 0 or dt2 <= 0 or abs(dt1 - dt2) < 1e-6:
                lmtd = (dt1 + dt2) / 2.0 if dt1 > 0 and dt2 > 0 else 0.0
            else:
                lmtd = (dt1 - dt2) / np.log(dt1 / dt2)
            
            # Calculate heat transfer
            q = self.u_value * self.area * abs(lmtd)
            
            # Calculate new outlet temperatures from energy balance
            temp_hot_out_new = temp_hot_in - q / c_hot
            temp_cold_out_new = temp_cold_in + q / c_cold
            
            # Check convergence
            error = max(
                abs(temp_hot_out_new - temp_hot_out),
                abs(temp_cold_out_new - temp_cold_out),
            )
            
            temp_hot_out = temp_hot_out_new
            temp_cold_out = temp_cold_out_new
            
            if error < tolerance:
                break
        
        # Ensure physical constraints
        if temp_hot_out < temp_cold_in:
            temp_hot_out = temp_cold_in
        if temp_cold_out > temp_hot_in:
            temp_cold_out = temp_hot_in
            
        # Recalculate final heat transfer
        q_final = c_hot * (temp_hot_in - temp_hot_out)
        
        return {
            "heat_transfer": q_final,
            "temp_hot_out": temp_hot_out,
            "temp_cold_out": temp_cold_out,
            "lmtd": lmtd,
        }


def create_heat_exchanger(config: Dict) -> HeatExchanger:
    """
    Factory function to create heat exchanger from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HeatExchanger instance
    """
    hx_type = config.get("type", "effectiveness_ntu")
    area = config.get("heat_transfer_area", 50.0)
    u_value = config.get("overall_heat_transfer_coefficient", 0.5)
    flow = config.get("flow_arrangement", "counterflow")
    
    if hx_type == "effectiveness_ntu":
        effectiveness = config.get("effectiveness", None)
        return EffectivenessNTU(area, u_value, flow, effectiveness)
    elif hx_type == "lmtd":
        return LMTD(area, u_value, flow)
    else:
        raise ValueError(f"Unknown heat exchanger type: {hx_type}")
