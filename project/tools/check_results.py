#!/usr/bin/env python3
"""
Simulation Results Health Check Script

Verifies that EnergyPlus simulation outputs are physically reasonable
for DRL training. Checks:
1. Zone temperatures (comfort range)
2. TES stratification (temperature gradient)
3. Heat pump operation (power consumption)
4. PV production (solar generation)

Usage:
    python tools/check_results.py [--csv PATH] [--plot]
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Default paths
DEFAULT_CSV = "outputs/sizing_run/eplusout.csv"
DEFAULT_ESO = "outputs/sizing_run/eplusout.eso"


def find_column(df: pd.DataFrame, patterns: list) -> str:
    """Find column matching any of the patterns (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        if all(p.lower() in col_lower for p in patterns):
            return col
    return None


def check_zone_temperatures(df: pd.DataFrame) -> dict:
    """Check if zone temperatures are in reasonable range."""
    print("\n" + "="*60)
    print("1. Zone Temperature Check")
    print("="*60)
    
    results = {"status": "OK", "issues": []}
    
    zones = ["CLASSROOM_SOUTH", "CLASSROOM_EAST", "CLASSROOM_NORTH", "CLASSROOM_WEST"]
    
    for zone in zones:
        col = find_column(df, [zone, "Zone Mean Air Temperature"])
        if col:
            temp = df[col]
            t_min, t_max, t_mean = temp.min(), temp.max(), temp.mean()
            
            status = "✅"
            if t_min < 10:
                status = "⚠️"
                results["issues"].append(f"{zone}: Too cold (min={t_min:.1f}°C)")
            if t_max > 35:
                status = "⚠️"
                results["issues"].append(f"{zone}: Too hot (max={t_max:.1f}°C)")
            
            print(f"  {status} {zone}: {t_min:.1f}°C - {t_max:.1f}°C (avg={t_mean:.1f}°C)")
        else:
            print(f"  ❓ {zone}: Column not found")
    
    if results["issues"]:
        results["status"] = "WARNING"
    
    return results


def check_tes_stratification(df: pd.DataFrame, num_nodes: int = 10) -> dict:
    """Check TES tank stratification (temperature gradient)."""
    print("\n" + "="*60)
    print("2. TES Stratification Check (Hot Water)")
    print("="*60)
    
    results = {"status": "OK", "issues": [], "data": {}}
    
    # Find TES node temperatures
    node_temps = {}
    for i in range(1, num_nodes + 1):
        col = find_column(df, ["TES_STRATIFIED", f"Node {i}"])
        if col:
            node_temps[i] = df[col]
    
    if not node_temps:
        print("  ❓ TES temperature data not found")
        results["status"] = "MISSING"
        return results
    
    # Check temperature gradient (top should be hotter than bottom)
    if 1 in node_temps and num_nodes in node_temps:
        t_top = node_temps[1].mean()
        t_bottom = node_temps[num_nodes].mean()
        delta_t = t_top - t_bottom
        
        print(f"  Top (Node 1):    {t_top:.2f}°C (avg)")
        print(f"  Bottom (Node {num_nodes}): {t_bottom:.2f}°C (avg)")
        print(f"  ΔT Gradient:     {delta_t:.2f}°C")
        
        if delta_t > 2:
            print("  ✅ Stratification is working properly")
        elif delta_t > 0:
            print("  ⚠️ Weak stratification - may need tuning")
            results["issues"].append("Weak TES stratification")
        else:
            print("  ❌ Inverted gradient - check inlet/outlet heights!")
            results["issues"].append("Inverted TES gradient")
            results["status"] = "ERROR"
        
        results["data"] = {
            "t_top": t_top,
            "t_bottom": t_bottom,
            "delta_t": delta_t,
            "node_temps": node_temps
        }
    
    return results


def check_heat_pump(df: pd.DataFrame) -> dict:
    """Check heat pump operation."""
    print("\n" + "="*60)
    print("3. Heat Pump Operation Check")
    print("="*60)
    
    results = {"status": "OK", "issues": []}
    
    # Check electricity consumption
    elec_col = find_column(df, ["CENTRAL_ASHP", "Electricity Rate"])
    if elec_col:
        power = df[elec_col]
        p_max = power.max()
        p_mean = power.mean()
        runtime_pct = (power > 0).sum() / len(power) * 100
        
        print(f"  Max Power:       {p_max/1000:.2f} kW")
        print(f"  Avg Power:       {p_mean/1000:.2f} kW")
        print(f"  Runtime:         {runtime_pct:.1f}%")
        
        if p_max == 0:
            print("  ❌ Heat pump never operated!")
            results["issues"].append("HP never turned on")
            results["status"] = "ERROR"
        elif runtime_pct < 1:
            print("  ⚠️ Very low runtime - check setpoints")
            results["issues"].append("Very low HP runtime")
        else:
            print("  ✅ Heat pump is operating")
    else:
        print("  ❓ HP electricity data not found")
    
    # Check heat transfer
    heat_col = find_column(df, ["CENTRAL_ASHP", "Load Side Heat Transfer"])
    if heat_col:
        heat = df[heat_col]
        print(f"  Max Heat Output: {heat.max()/1000:.2f} kW")
    
    # Check PLR
    plr_col = find_column(df, ["CENTRAL_ASHP", "Part Load Ratio"])
    if plr_col:
        plr = df[plr_col]
        print(f"  Avg PLR:         {plr.mean()*100:.1f}%")
    
    return results


def check_chiller(df: pd.DataFrame) -> dict:
    """Check chiller operation."""
    print("\n" + "="*60)
    print("4. Chiller Operation Check")
    print("="*60)
    
    results = {"status": "OK", "issues": []}
    
    elec_col = find_column(df, ["CENTRAL CHILLER", "Electricity Rate"])
    if elec_col:
        power = df[elec_col]
        p_max = power.max()
        runtime_pct = (power > 0).sum() / len(power) * 100
        
        print(f"  Max Power:       {p_max/1000:.2f} kW")
        print(f"  Runtime:         {runtime_pct:.1f}%")
        
        if p_max == 0:
            print("  ⚠️ Chiller never operated (may be OK if winter-only sim)")
        else:
            print("  ✅ Chiller is operating")
    else:
        print("  ❓ Chiller data not found")
    
    return results


def check_pv_system(df: pd.DataFrame) -> dict:
    """Check PV system output."""
    print("\n" + "="*60)
    print("5. PV System Check")
    print("="*60)
    
    results = {"status": "OK", "issues": []}
    
    pv_col = find_column(df, ["Rooftop_PV_Array", "DC Electricity Rate"])
    if pv_col:
        power = df[pv_col]
        p_max = power.max()
        p_total = power.sum() * 0.25  # Wh (assuming 15-min timestep)
        
        print(f"  Peak Power:      {p_max/1000:.2f} kW")
        print(f"  Total Energy:    {p_total/1e6:.2f} MWh")
        
        if p_max > 0:
            print("  ✅ PV is generating power")
        else:
            print("  ⚠️ No PV generation detected")
            results["issues"].append("No PV output")
    else:
        print("  ❓ PV data not found")
    
    return results


def check_facility_energy(df: pd.DataFrame) -> dict:
    """Check overall facility energy consumption."""
    print("\n" + "="*60)
    print("6. Facility Energy Summary")
    print("="*60)
    
    results = {"status": "OK", "data": {}}
    
    # Total demand
    demand_col = find_column(df, ["Facility Total Electricity Demand"])
    if demand_col:
        demand = df[demand_col]
        total_demand = demand.sum() * 0.25 / 1e6  # MWh
        print(f"  Total Demand:    {total_demand:.2f} MWh")
        results["data"]["total_demand_mwh"] = total_demand
    
    # Net purchased
    net_col = find_column(df, ["Facility Net Purchased"])
    if net_col:
        net = df[net_col]
        total_net = net.sum() * 0.25 / 1e6  # MWh
        print(f"  Net Purchased:   {total_net:.2f} MWh")
        results["data"]["net_purchased_mwh"] = total_net
    
    # Self-consumption ratio
    if "total_demand_mwh" in results["data"] and "net_purchased_mwh" in results["data"]:
        if results["data"]["total_demand_mwh"] > 0:
            self_consumption = 1 - results["data"]["net_purchased_mwh"] / results["data"]["total_demand_mwh"]
            print(f"  Self-Consumption: {self_consumption*100:.1f}%")
    
    return results


def plot_results(df: pd.DataFrame, tes_data: dict, output_dir: str = "outputs"):
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB:
        print("\n⚠️ Matplotlib not installed, skipping plots")
        return
    
    print("\n" + "="*60)
    print("Generating Diagnostic Plots...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Zone temperatures
    ax = axes[0, 0]
    for zone in ["CLASSROOM_SOUTH", "CLASSROOM_NORTH"]:
        col = find_column(df, [zone, "Zone Mean Air Temperature"])
        if col:
            ax.plot(df[col].iloc[:500], label=zone.replace("CLASSROOM_", ""), alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Zone Air Temperature (First 5 Days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. TES stratification
    ax = axes[0, 1]
    if tes_data.get("data", {}).get("node_temps"):
        node_temps = tes_data["data"]["node_temps"]
        for i in [1, 5, 10]:
            if i in node_temps:
                ax.plot(node_temps[i].iloc[:500], label=f"Node {i}", alpha=0.8)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("TES Stratification (First 5 Days)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Heat pump power
    ax = axes[1, 0]
    elec_col = find_column(df, ["CENTRAL_ASHP", "Electricity Rate"])
    if elec_col:
        ax.plot(df[elec_col].iloc[:500] / 1000, color='red', alpha=0.7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Heat Pump Electricity (First 5 Days)")
        ax.grid(True, alpha=0.3)
    
    # 4. Facility energy balance
    ax = axes[1, 1]
    demand_col = find_column(df, ["Facility Total Electricity Demand"])
    net_col = find_column(df, ["Facility Net Purchased"])
    if demand_col and net_col:
        ax.plot(df[demand_col].iloc[:500] / 1000, label="Total Demand", alpha=0.7)
        ax.plot(df[net_col].iloc[:500] / 1000, label="Net Purchased", alpha=0.7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Facility Energy Balance (First 5 Days)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "simulation_health_check.png"
    plt.savefig(output_path, dpi=150)
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Check EnergyPlus simulation results")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to eplusout.csv")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--tes-nodes", type=int, default=10, help="Number of TES nodes")
    args = parser.parse_args()
    
    print("="*60)
    print("EnergyPlus Simulation Health Check")
    print("="*60)
    
    # Find and load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\n❌ Error: CSV file not found: {csv_path}")
        print("\nPlease run EnergyPlus simulation first:")
        print("  python tools/freeze_model.py --input outputs/sim_building.idf")
        sys.exit(1)
    
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Run checks
    all_results = {}
    all_results["zones"] = check_zone_temperatures(df)
    all_results["tes"] = check_tes_stratification(df, args.tes_nodes)
    all_results["hp"] = check_heat_pump(df)
    all_results["chiller"] = check_chiller(df)
    all_results["pv"] = check_pv_system(df)
    all_results["facility"] = check_facility_energy(df)
    
    # Generate plots if requested
    if args.plot:
        plot_results(df, all_results["tes"])
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues = []
    for name, result in all_results.items():
        if result.get("issues"):
            issues.extend(result["issues"])
    
    if not issues:
        print("✅ All checks passed - Model is ready for RL training!")
    else:
        print(f"⚠️ Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
