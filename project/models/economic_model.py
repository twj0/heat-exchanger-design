"""Economic models for cost calculation and TOU pricing.

This module handles time-of-use electricity pricing, gas costs,
and overall economic calculations for the system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time


class TOUPricing:
    """Time-of-Use electricity pricing model."""
    
    def __init__(
        self,
        peak_price: float,
        shoulder_price: float,
        offpeak_price: float,
        peak_hours: List[List[int]],
        shoulder_hours: List[List[int]],
        offpeak_hours: List[List[int]],
    ):
        """
        Initialize TOU pricing structure.
        
        Args:
            peak_price: Peak period price (CNY/kWh)
            shoulder_price: Shoulder period price (CNY/kWh)
            offpeak_price: Off-peak period price (CNY/kWh)
            peak_hours: List of [start_hour, end_hour] for peak periods
            shoulder_hours: List of [start_hour, end_hour] for shoulder periods
            offpeak_hours: List of [start_hour, end_hour] for off-peak periods
        """
        self.peak_price = peak_price
        self.shoulder_price = shoulder_price
        self.offpeak_price = offpeak_price
        
        self.peak_hours = peak_hours
        self.shoulder_hours = shoulder_hours
        self.offpeak_hours = offpeak_hours
        
        # Create lookup table for 24 hours
        self.hourly_prices = self._create_hourly_lookup()
        
    def _create_hourly_lookup(self) -> np.ndarray:
        """Create 24-hour price lookup table."""
        prices = np.zeros(24)
        
        # Default to off-peak
        prices[:] = self.offpeak_price
        
        # Set peak hours
        for start, end in self.peak_hours:
            if end > start:
                prices[start:end] = self.peak_price
            else:  # Crosses midnight
                prices[start:] = self.peak_price
                prices[:end] = self.peak_price
        
        # Set shoulder hours
        for start, end in self.shoulder_hours:
            if end > start:
                prices[start:end] = self.shoulder_price
            else:
                prices[start:] = self.shoulder_price
                prices[:end] = self.shoulder_price
        
        return prices
    
    def get_price(self, hour: int) -> float:
        """
        Get electricity price for a given hour.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Price (CNY/kWh)
        """
        return self.hourly_prices[hour % 24]
    
    def get_period(self, hour: int) -> str:
        """
        Get pricing period name for a given hour.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Period name: 'peak', 'shoulder', or 'offpeak'
        """
        price = self.get_price(hour)
        
        if abs(price - self.peak_price) < 1e-6:
            return "peak"
        elif abs(price - self.shoulder_price) < 1e-6:
            return "shoulder"
        else:
            return "offpeak"
    
    def get_price_forecast(self, current_hour: int, forecast_hours: int) -> np.ndarray:
        """
        Get price forecast for next N hours.
        
        Args:
            current_hour: Current hour of day (0-23)
            forecast_hours: Number of hours to forecast
            
        Returns:
            Array of prices
        """
        forecast = np.zeros(forecast_hours)
        for i in range(forecast_hours):
            hour = (current_hour + i) % 24
            forecast[i] = self.get_price(hour)
        return forecast
    
    def is_offpeak(self, hour: int) -> bool:
        """Check if hour is in off-peak period."""
        return self.get_period(hour) == "offpeak"
    
    def is_peak(self, hour: int) -> bool:
        """Check if hour is in peak period."""
        return self.get_period(hour) == "peak"


class EconomicModel:
    """Economic model for calculating operational costs and savings."""
    
    def __init__(
        self,
        tou_pricing: TOUPricing,
        gas_price: float = 3.5,  # CNY/m³
        feed_in_tariff: float = 0.4,  # CNY/kWh for excess electricity
    ):
        """
        Initialize economic model.
        
        Args:
            tou_pricing: TOU pricing instance
            gas_price: Natural gas price (CNY/m³)
            feed_in_tariff: Feed-in tariff for excess electricity (CNY/kWh)
        """
        self.tou = tou_pricing
        self.gas_price = gas_price
        self.feed_in_tariff = feed_in_tariff
        
        # Tracking variables
        self.total_cost = 0.0
        self.total_electricity_cost = 0.0
        self.total_gas_cost = 0.0
        self.total_revenue = 0.0
        
        self.electricity_consumed = 0.0  # kWh
        self.electricity_sold = 0.0  # kWh
        self.gas_consumed = 0.0  # m³
        
    def calculate_step_cost(
        self,
        hour: int,
        electricity_from_grid: float,
        electricity_to_grid: float,
        gas_consumption: float,
        timestep: float,
    ) -> Dict[str, float]:
        """
        Calculate cost for one timestep.
        
        Cost = Price_buy * P_grid+ * Δt - FeedIn * P_grid- * Δt + GasPrice * Gas * Δt
        
        Args:
            hour: Current hour of day (0-23)
            electricity_from_grid: Power drawn from grid (kW)
            electricity_to_grid: Power sent to grid (kW)
            gas_consumption: Gas consumption (m³/h)
            timestep: Time step duration (seconds)
            
        Returns:
            Dictionary with cost breakdown
        """
        # Convert to energy (kWh or m³)
        hours = timestep / 3600.0
        energy_from_grid = electricity_from_grid * hours
        energy_to_grid = electricity_to_grid * hours
        gas_used = gas_consumption * hours
        
        # Get electricity price
        price = self.tou.get_price(hour)
        
        # Calculate costs
        electricity_cost = price * energy_from_grid
        electricity_revenue = self.feed_in_tariff * energy_to_grid
        gas_cost = self.gas_price * gas_used
        
        net_cost = electricity_cost - electricity_revenue + gas_cost
        
        # Update tracking
        self.total_electricity_cost += electricity_cost
        self.total_revenue += electricity_revenue
        self.total_gas_cost += gas_cost
        self.total_cost += net_cost
        
        self.electricity_consumed += energy_from_grid
        self.electricity_sold += energy_to_grid
        self.gas_consumed += gas_used
        
        return {
            "net_cost": net_cost,
            "electricity_cost": electricity_cost,
            "electricity_revenue": electricity_revenue,
            "gas_cost": gas_cost,
            "electricity_price": price,
            "period": self.tou.get_period(hour),
        }
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of cumulative costs and consumption.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "total_cost": self.total_cost,
            "total_electricity_cost": self.total_electricity_cost,
            "total_gas_cost": self.total_gas_cost,
            "total_revenue": self.total_revenue,
            "net_cost": self.total_cost,
            "electricity_consumed_kwh": self.electricity_consumed,
            "electricity_sold_kwh": self.electricity_sold,
            "gas_consumed_m3": self.gas_consumed,
        }
    
    def reset(self) -> None:
        """Reset all tracking variables."""
        self.total_cost = 0.0
        self.total_electricity_cost = 0.0
        self.total_gas_cost = 0.0
        self.total_revenue = 0.0
        self.electricity_consumed = 0.0
        self.electricity_sold = 0.0
        self.gas_consumed = 0.0
    
    def calculate_savings(
        self,
        baseline_cost: float,
        optimized_cost: float,
    ) -> Dict[str, float]:
        """
        Calculate cost savings and percentage.
        
        Args:
            baseline_cost: Cost with baseline controller (CNY)
            optimized_cost: Cost with optimized controller (CNY)
            
        Returns:
            Dictionary with savings metrics
        """
        savings = baseline_cost - optimized_cost
        savings_percent = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        
        return {
            "absolute_savings": savings,
            "percent_savings": savings_percent,
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
        }
    
    def calculate_payback_period(
        self,
        capital_cost: float,
        annual_savings: float,
    ) -> float:
        """
        Calculate simple payback period.
        
        Args:
            capital_cost: Initial capital investment (CNY)
            annual_savings: Annual operational savings (CNY/year)
            
        Returns:
            Payback period (years)
        """
        if annual_savings <= 0:
            return float('inf')
        return capital_cost / annual_savings


def create_tou_pricing(config: Dict) -> TOUPricing:
    """
    Factory function to create TOU pricing from configuration.
    
    Args:
        config: Configuration dictionary with TOU pricing parameters
        
    Returns:
        TOUPricing instance
    """
    return TOUPricing(
        peak_price=config.get("peak_price", 1.2),
        shoulder_price=config.get("shoulder_price", 0.7),
        offpeak_price=config.get("offpeak_price", 0.3),
        peak_hours=config.get("peak_hours", [[10, 12], [18, 21]]),
        shoulder_hours=config.get("shoulder_hours", [[8, 10], [12, 18]]),
        offpeak_hours=config.get("offpeak_hours", [[21, 24], [0, 8]]),
    )


def create_economic_model(config: Dict, tou_pricing: TOUPricing) -> EconomicModel:
    """
    Factory function to create economic model from configuration.
    
    Args:
        config: Configuration dictionary
        tou_pricing: TOUPricing instance
        
    Returns:
        EconomicModel instance
    """
    return EconomicModel(
        tou_pricing=tou_pricing,
        gas_price=config.get("gas_price", 3.5),
        feed_in_tariff=config.get("feed_in_tariff", 0.4),
    )
