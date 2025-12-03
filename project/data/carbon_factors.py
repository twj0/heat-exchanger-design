"""
Shanghai Carbon Emission Factors

Official source: 沪环气〔2022〕34号
上海市生态环境局关于调整本市温室气体排放核算指南相关排放因子数值的通知
Published: 2022-02-14

This module provides carbon emission factors for Shanghai grid electricity
and related energy sources, based on official government regulations.
"""

# =============================================================================
# Official Carbon Emission Factors (Shanghai)
# =============================================================================

# Electricity emission factor (tCO2/MWh)
# Source: 沪环气〔2022〕34号
# Original: 4.2 tCO2/10^4 kWh = 0.42 tCO2/MWh
ELECTRICITY_FACTOR = 0.42  # tCO2/MWh

# Green electricity (PV, wind) emission factor
# Source: 沪环气〔2022〕34号 - 外购绿电排放因子为0
GREEN_ELECTRICITY_FACTOR = 0.0  # tCO2/MWh

# Heat (district heating) emission factor
# Source: 沪环气〔2022〕34号
# Original: 0.06 tCO2/GJ
HEAT_FACTOR = 0.06  # tCO2/GJ

# Natural gas emission factors (from 2024 配额分配方案)
GAS_BOILER_FACTOR = 0.06233  # tCO2/GJ (燃气锅炉供热)
GAS_CHP_FACTOR = 0.06885     # tCO2/GJ (燃气热电联产)

# =============================================================================
# Time-of-Use Electricity Pricing (Shanghai 2024)
# =============================================================================

TOU_PRICING = {
    "peak": {
        "hours": [(8, 11), (18, 21)],  # 08:00-11:00, 18:00-21:00
        "price": 1.0074,  # CNY/kWh
    },
    "valley": {
        "hours": [(22, 24), (0, 6)],   # 22:00-06:00
        "price": 0.3128,  # CNY/kWh
    },
    "mid": {
        "hours": [(6, 8), (11, 18), (21, 22)],  # Other hours
        "price": 0.6177,  # CNY/kWh
    },
}

# =============================================================================
# Carbon Market Reference (Shanghai Carbon Exchange)
# =============================================================================

# Reference carbon price (CNY/tCO2) - approximate market price
CARBON_PRICE_REFERENCE = 80.0  # CNY/tCO2 (2024 average)

# Maximum offset ratio using CCER/SHCERCIR
MAX_OFFSET_RATIO = 0.05  # 5% as per 2024 policy


def get_tou_price(hour: int) -> float:
    """
    Get time-of-use electricity price for given hour.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Electricity price in CNY/kWh
    """
    for period, info in TOU_PRICING.items():
        for start, end in info["hours"]:
            if start <= hour < end:
                return info["price"]
    return TOU_PRICING["mid"]["price"]


def get_tou_period(hour: int) -> str:
    """
    Get time-of-use period name for given hour.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Period name: 'peak', 'valley', or 'mid'
    """
    for period, info in TOU_PRICING.items():
        for start, end in info["hours"]:
            if start <= hour < end:
                return period
    return "mid"


def calculate_carbon_emission(
    electricity_kwh: float,
    pv_generation_kwh: float = 0.0,
    heat_gj: float = 0.0,
) -> float:
    """
    Calculate carbon emission based on energy consumption.
    
    Following Shanghai's official methodology (沪环气〔2022〕34号):
    - Grid electricity: 0.42 tCO2/MWh
    - PV self-consumption: 0 tCO2/MWh
    - District heat: 0.06 tCO2/GJ
    
    Args:
        electricity_kwh: Grid electricity consumption (kWh)
        pv_generation_kwh: PV generation used on-site (kWh)
        heat_gj: District heat consumption (GJ)
        
    Returns:
        Total carbon emission (tCO2)
    """
    # Net grid electricity (after PV offset)
    net_electricity_kwh = max(0, electricity_kwh - pv_generation_kwh)
    
    # Convert to MWh for calculation
    net_electricity_mwh = net_electricity_kwh / 1000.0
    
    # Calculate emissions
    electricity_emission = net_electricity_mwh * ELECTRICITY_FACTOR
    heat_emission = heat_gj * HEAT_FACTOR
    
    return electricity_emission + heat_emission


def calculate_electricity_cost(
    electricity_kwh: float,
    hour: int,
    pv_generation_kwh: float = 0.0,
) -> float:
    """
    Calculate electricity cost with TOU pricing and PV offset.
    
    Args:
        electricity_kwh: Total electricity demand (kWh)
        hour: Current hour (0-23)
        pv_generation_kwh: PV generation (kWh)
        
    Returns:
        Electricity cost (CNY)
    """
    # Net purchased electricity
    net_electricity = max(0, electricity_kwh - pv_generation_kwh)
    
    # Get TOU price
    price = get_tou_price(hour)
    
    return net_electricity * price


def calculate_carbon_cost(
    carbon_emission_tco2: float,
    carbon_quota_tco2: float = 0.0,
) -> float:
    """
    Calculate carbon cost based on emission vs quota.
    
    Args:
        carbon_emission_tco2: Actual carbon emission (tCO2)
        carbon_quota_tco2: Allocated carbon quota (tCO2)
        
    Returns:
        Carbon cost (CNY), negative if under quota (potential income)
    """
    excess_emission = carbon_emission_tco2 - carbon_quota_tco2
    return excess_emission * CARBON_PRICE_REFERENCE


# =============================================================================
# Hourly Carbon Intensity Profile (Optional - for future use)
# =============================================================================

# Note: Shanghai currently uses a fixed emission factor (0.42 tCO2/MWh).
# However, for research purposes, you may want to use time-varying factors
# based on grid generation mix. This is a placeholder for future enhancement.

HOURLY_CARBON_INTENSITY = None  # To be populated with real-time data if available


if __name__ == "__main__":
    # Test calculations
    print("Shanghai Carbon Factors Test")
    print("=" * 50)
    print(f"Electricity Factor: {ELECTRICITY_FACTOR} tCO2/MWh")
    print(f"Heat Factor: {HEAT_FACTOR} tCO2/GJ")
    print()
    
    # Test TOU pricing
    print("Time-of-Use Pricing:")
    for hour in [3, 9, 14, 19, 23]:
        price = get_tou_price(hour)
        period = get_tou_period(hour)
        print(f"  {hour:02d}:00 -> {period:6s} @ ¥{price:.4f}/kWh")
    print()
    
    # Test carbon calculation
    test_elec = 1000  # kWh
    test_pv = 300     # kWh
    emission = calculate_carbon_emission(test_elec, test_pv)
    print(f"Carbon Emission Test:")
    print(f"  Grid demand: {test_elec} kWh")
    print(f"  PV offset: {test_pv} kWh")
    print(f"  Net emission: {emission:.4f} tCO2")
