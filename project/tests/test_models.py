"""
Unit tests for physical models.

Tests verify energy conservation, physical constraints, and numerical stability.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from models.thermal_storage import SensibleHeatStorage, PCMStorage
from models.heat_exchanger import EffectivenessNTU, LMTD
from models.economic_model import TOUPricing, EconomicModel


class TestSensibleHeatStorage:
    """Tests for sensible heat storage model."""
    
    def test_initialization(self):
        """Test storage initialization."""
        storage = SensibleHeatStorage(
            mass=1000.0,
            specific_heat=4.18,
            initial_temperature=45.0,
            min_temperature=40.0,
            max_temperature=50.0,
            loss_coefficient=10.0,
            ambient_temperature=20.0,
        )
        
        assert storage.temperature == 45.0
        assert 0.0 <= storage.get_state_of_charge() <= 1.0
    
    def test_charging(self):
        """Test charging behavior."""
        storage = SensibleHeatStorage(
            mass=1000.0,
            specific_heat=4.18,
            initial_temperature=45.0,
            min_temperature=40.0,
            max_temperature=50.0,
            loss_coefficient=10.0,
            ambient_temperature=20.0,
        )
        
        initial_temp = storage.temperature
        
        # Charge for 1 hour
        result = storage.step(
            power_in=50.0,  # kW
            power_out=0.0,
            timestep=3600,  # seconds
        )
        
        # Temperature should increase
        assert storage.temperature > initial_temp
        assert storage.temperature <= 50.0  # Respect max limit
    
    def test_discharging(self):
        """Test discharging behavior."""
        storage = SensibleHeatStorage(
            mass=1000.0,
            specific_heat=4.18,
            initial_temperature=45.0,
            min_temperature=40.0,
            max_temperature=50.0,
            loss_coefficient=10.0,
            ambient_temperature=20.0,
        )
        
        initial_temp = storage.temperature
        
        # Discharge for 1 hour
        result = storage.step(
            power_in=0.0,
            power_out=30.0,  # kW
            timestep=3600,
        )
        
        # Temperature should decrease
        assert storage.temperature < initial_temp
        assert storage.temperature >= 40.0  # Respect min limit
    
    def test_temperature_limits(self):
        """Test temperature constraint enforcement."""
        storage = SensibleHeatStorage(
            mass=1000.0,
            specific_heat=4.18,
            initial_temperature=49.0,
            min_temperature=40.0,
            max_temperature=50.0,
            loss_coefficient=10.0,
            ambient_temperature=20.0,
        )
        
        # Try to overcharge
        result = storage.step(
            power_in=200.0,  # Large power
            power_out=0.0,
            timestep=3600,
        )
        
        # Should be clamped to max
        assert storage.temperature <= 50.0
        assert result["temperature_violation"] >= 0.0


class TestHeatExchanger:
    """Tests for heat exchanger models."""
    
    def test_effectiveness_ntu(self):
        """Test Effectiveness-NTU method."""
        hx = EffectivenessNTU(
            heat_transfer_area=50.0,
            overall_heat_transfer_coefficient=0.5,
            flow_arrangement="counterflow",
        )
        
        result = hx.calculate_heat_transfer(
            mass_flow_hot=1.0,  # kg/s
            mass_flow_cold=1.0,
            temp_hot_in=60.0,   # °C
            temp_cold_in=20.0,
            cp_hot=4.18,
            cp_cold=4.18,
        )
        
        # Heat transfer should be positive
        assert result["heat_transfer"] > 0
        
        # Outlet temperatures should be between inlet values
        assert 20.0 < result["temp_cold_out"] < 60.0
        assert 20.0 < result["temp_hot_out"] < 60.0
        
        # Hot fluid should cool down
        assert result["temp_hot_out"] < 60.0
        
        # Cold fluid should heat up
        assert result["temp_cold_out"] > 20.0
    
    def test_lmtd_method(self):
        """Test LMTD method."""
        hx = LMTD(
            heat_transfer_area=50.0,
            overall_heat_transfer_coefficient=0.5,
            flow_arrangement="counterflow",
        )
        
        result = hx.calculate_heat_transfer(
            mass_flow_hot=1.0,
            mass_flow_cold=1.0,
            temp_hot_in=60.0,
            temp_cold_in=20.0,
            cp_hot=4.18,
            cp_cold=4.18,
        )
        
        assert result["heat_transfer"] > 0
        assert result["lmtd"] > 0


class TestTOUPricing:
    """Tests for TOU pricing model."""
    
    def test_price_periods(self):
        """Test price period classification."""
        tou = TOUPricing(
            peak_price=1.2,
            shoulder_price=0.7,
            offpeak_price=0.3,
            peak_hours=[[10, 12], [18, 21]],
            shoulder_hours=[[8, 10], [12, 18]],
            offpeak_hours=[[21, 24], [0, 8]],
        )
        
        # Test peak hours
        assert tou.is_peak(10)
        assert tou.is_peak(19)
        
        # Test off-peak hours
        assert tou.is_offpeak(2)
        assert tou.is_offpeak(23)
        
        # Test prices
        assert tou.get_price(10) == 1.2
        assert tou.get_price(2) == 0.3
    
    def test_price_forecast(self):
        """Test price forecasting."""
        tou = TOUPricing(
            peak_price=1.2,
            shoulder_price=0.7,
            offpeak_price=0.3,
            peak_hours=[[10, 12], [18, 21]],
            shoulder_hours=[[8, 10], [12, 18]],
            offpeak_hours=[[21, 24], [0, 8]],
        )
        
        forecast = tou.get_price_forecast(8, 6)
        
        assert len(forecast) == 6
        assert all(price >= 0 for price in forecast)


class TestEconomicModel:
    """Tests for economic model."""
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        tou = TOUPricing(
            peak_price=1.2,
            shoulder_price=0.7,
            offpeak_price=0.3,
            peak_hours=[[10, 12], [18, 21]],
            shoulder_hours=[[8, 10], [12, 18]],
            offpeak_hours=[[21, 24], [0, 8]],
        )
        
        economic = EconomicModel(tou)
        
        # Test off-peak charging cost
        cost = economic.calculate_step_cost(
            hour=2,  # Off-peak
            electricity_from_grid=50.0,  # kW
            electricity_to_grid=0.0,
            gas_consumption=0.0,
            timestep=3600,  # 1 hour
        )
        
        # Cost should be positive
        assert cost["net_cost"] > 0
        
        # Should use off-peak price
        assert cost["electricity_price"] == 0.3
        
        # Check total tracking
        assert economic.total_cost > 0


def test_energy_conservation():
    """Test energy conservation in storage."""
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=45.0,
        min_temperature=40.0,
        max_temperature=50.0,
        loss_coefficient=0.0,  # No losses for this test
        ambient_temperature=20.0,
    )
    
    initial_energy = storage.get_stored_energy()
    
    # Charge
    power_in = 50.0  # kW
    timestep = 3600  # seconds
    energy_added = power_in * timestep * 0.98  # kJ (with efficiency)
    
    storage.step(power_in=power_in, power_out=0.0, timestep=timestep)
    
    final_energy = storage.get_stored_energy()
    
    # Check energy balance (with small tolerance for numerical errors)
    expected_energy = initial_energy + energy_added
    relative_error = abs(final_energy - expected_energy) / max(expected_energy, 1.0)
    
    assert relative_error < 0.01, f"Energy not conserved: {relative_error:.4f}"


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v"])
