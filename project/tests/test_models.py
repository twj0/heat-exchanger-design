"""
Unit tests for physical models.

Tests verify energy conservation, physical constraints, and numerical stability.
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from models.thermal_storage import SensibleHeatStorage, PCMStorage
from models.heat_exchanger import EffectivenessNTU, LMTD, create_heat_exchanger
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


def test_heat_losses_positive():
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=60.0,
        min_temperature=40.0,
        max_temperature=80.0,
        loss_coefficient=15.0,
        ambient_temperature=20.0,
    )
    assert storage.get_heat_losses() > 0.0


def test_heat_losses_zero_when_cooler():
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=35.0,
        min_temperature=20.0,
        max_temperature=80.0,
        loss_coefficient=15.0,
        ambient_temperature=40.0,
    )
    assert storage.get_heat_losses() == 0.0


def test_state_of_charge_clamped_high():
    storage = SensibleHeatStorage(
        mass=500.0,
        specific_heat=4.18,
        initial_temperature=45.0,
        min_temperature=40.0,
        max_temperature=50.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=500.0, power_out=0.0, timestep=3600)
    assert 0.99 <= storage.get_state_of_charge() <= 1.0


def test_state_of_charge_clamped_low():
    storage = SensibleHeatStorage(
        mass=500.0,
        specific_heat=4.18,
        initial_temperature=45.0,
        min_temperature=40.0,
        max_temperature=50.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=0.0, power_out=400.0, timestep=3600)
    assert 0.0 <= storage.get_state_of_charge() <= 0.01


def test_storage_reset_restores_temperature():
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=45.0,
        min_temperature=40.0,
        max_temperature=50.0,
        loss_coefficient=10.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=200.0, power_out=0.0, timestep=1800)
    storage.reset()
    assert storage.temperature == pytest.approx(45.0)


def test_is_temperature_valid_detects_violation():
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=45.0,
        min_temperature=40.0,
        max_temperature=50.0,
        loss_coefficient=10.0,
        ambient_temperature=20.0,
    )
    storage.temperature = 55.0
    assert not storage.is_temperature_valid()


def test_temperature_violation_on_cooling():
    storage = SensibleHeatStorage(
        mass=1000.0,
        specific_heat=4.18,
        initial_temperature=42.0,
        min_temperature=40.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    result = storage.step(power_in=0.0, power_out=500.0, timestep=3600)
    assert result["temperature_violation"] > 0.0
    assert storage.temperature == storage.min_temperature


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


def test_pcm_initial_liquid_fraction_below_melting():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=20.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    assert storage.liquid_fraction == 0.0


def test_pcm_initial_liquid_fraction_above_melting():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=35.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    assert storage.liquid_fraction == 1.0


def test_pcm_charging_increases_liquid_fraction():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=25.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=80.0, power_out=0.0, timestep=3600)
    assert storage.liquid_fraction == 1.0


def test_pcm_discharging_decreases_liquid_fraction():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=35.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=0.0, power_out=60.0, timestep=3600)
    assert storage.liquid_fraction == 0.0


def test_pcm_temperature_constraint_enforced():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=35.0,
        min_temperature=10.0,
        max_temperature=50.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    result = storage.step(power_in=200.0, power_out=0.0, timestep=3600)
    assert result["temperature_violation"] > 0.0
    assert storage.temperature == storage.max_temperature


def test_pcm_state_of_charge_at_phase_change():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=25.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.liquid_fraction = 0.3
    soc = storage.get_state_of_charge()
    assert soc == pytest.approx(0.292, rel=1e-2)


def test_pcm_reset_restores_fraction():
    storage = PCMStorage(
        mass=500.0,
        specific_heat_solid=2.1,
        specific_heat_liquid=2.3,
        latent_heat=150.0,
        melting_point=25.0,
        initial_temperature=35.0,
        min_temperature=10.0,
        max_temperature=60.0,
        loss_coefficient=0.0,
        ambient_temperature=20.0,
    )
    storage.step(power_in=0.0, power_out=70.0, timestep=3600)
    storage.reset()
    assert storage.temperature == pytest.approx(35.0)
    assert storage.liquid_fraction == 1.0


def test_effectiveness_ntu_zero_flow_returns_zero():
    hx = EffectivenessNTU(heat_transfer_area=30.0, overall_heat_transfer_coefficient=0.5)
    result = hx.calculate_heat_transfer(
        mass_flow_hot=0.0,
        mass_flow_cold=1.0,
        temp_hot_in=60.0,
        temp_cold_in=20.0,
    )
    assert result["heat_transfer"] == 0.0
    assert result["effectiveness"] == 0.0


def test_effectiveness_ntu_uses_fixed_effectiveness():
    hx = EffectivenessNTU(
        heat_transfer_area=30.0,
        overall_heat_transfer_coefficient=0.5,
        effectiveness=0.7,
    )
    result = hx.calculate_heat_transfer(
        mass_flow_hot=1.2,
        mass_flow_cold=0.8,
        temp_hot_in=80.0,
        temp_cold_in=30.0,
    )
    assert result["effectiveness"] == pytest.approx(0.7)


def test_effectiveness_ntu_crossflow_effectiveness_range():
    hx = EffectivenessNTU(
        heat_transfer_area=40.0,
        overall_heat_transfer_coefficient=0.8,
        flow_arrangement="crossflow",
    )
    result = hx.calculate_heat_transfer(
        mass_flow_hot=1.5,
        mass_flow_cold=1.0,
        temp_hot_in=90.0,
        temp_cold_in=20.0,
    )
    assert 0.0 <= result["effectiveness"] <= 1.0


def test_lmtd_zero_flow_returns_zero():
    hx = LMTD(heat_transfer_area=50.0, overall_heat_transfer_coefficient=0.5)
    result = hx.calculate_heat_transfer(
        mass_flow_hot=1.0,
        mass_flow_cold=0.0,
        temp_hot_in=60.0,
        temp_cold_in=20.0,
    )
    assert result["heat_transfer"] == 0.0
    assert result["lmtd"] == 0.0


def test_lmtd_parallel_flow_limits():
    hx = LMTD(
        heat_transfer_area=50.0,
        overall_heat_transfer_coefficient=0.5,
        flow_arrangement="parallel",
    )
    result = hx.calculate_heat_transfer(
        mass_flow_hot=1.1,
        mass_flow_cold=1.0,
        temp_hot_in=70.0,
        temp_cold_in=25.0,
    )
    assert result["temp_hot_out"] <= 70.0
    assert result["temp_cold_out"] >= 25.0
    assert result["temp_hot_out"] >= result["temp_cold_out"]


def test_create_heat_exchanger_returns_effectiveness_instance():
    config = {
        "type": "effectiveness_ntu",
        "heat_transfer_area": 30.0,
        "overall_heat_transfer_coefficient": 0.6,
        "flow_arrangement": "counterflow",
    }
    hx = create_heat_exchanger(config)
    assert isinstance(hx, EffectivenessNTU)


def test_create_heat_exchanger_invalid_type_raises():
    config = {
        "type": "invalid",
        "heat_transfer_area": 30.0,
        "overall_heat_transfer_coefficient": 0.6,
    }
    with pytest.raises(ValueError):
        create_heat_exchanger(config)


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
