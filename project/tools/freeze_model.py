"""
Freeze Model Tool - Extract autosized values and hardcode them into IDF.

This tool runs EnergyPlus once in sizing mode, parses the .eio output file
to extract calculated capacities and flow rates, then creates a "frozen" 
IDF with all autosize values replaced by concrete numbers.

Purpose:
- Eliminates sizing uncertainty for reproducible DRL training
- Speeds up subsequent simulations (no sizing calculations)
- Required for publication-grade results in Applied Energy, etc.

Usage:
    python tools/freeze_model.py [--input outputs/sim_building.idf] [--weather data/weather/Shanghai_2024.epw]
"""

import sys
import os
import shutil
import argparse
import re
from pathlib import Path
from typing import Dict, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_sizing_results(eio_path: str) -> Dict[str, str]:
    """
    Parse EnergyPlus .eio file to extract calculated capacities and flow rates.
    
    The .eio file contains component sizing information in CSV-like format:
    - Component Sizing, Type, Name, Description, Value, Units
    
    Args:
        eio_path: Path to the eplusout.eio file
        
    Returns:
        Dictionary mapping component_name_field to calculated value
    """
    results = {}
    
    if not os.path.exists(eio_path):
        print(f"Warning: EIO file not found: {eio_path}")
        return results
    
    print(f"\n[Parsing] Reading sizing results from: {eio_path}")
    
    try:
        with open(eio_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                # Parse HVAC Component Sizing lines
                if line.startswith('Component Sizing'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        comp_type = parts[1]
                        comp_name = parts[2]
                        description = parts[3]
                        value = parts[4]
                        
                        # Create key for lookup
                        key = f"{comp_name}_{description.replace(' ', '_')}"
                        results[key] = value
                        
                        # Debug: Print extracted values
                        if 'Capacity' in description or 'Flow Rate' in description:
                            print(f"  Found: {comp_name} | {description} = {value}")
                
                # Parse Coil Sizing
                if 'Coil:Heating:Water' in line and 'Component Sizing' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        name = parts[2]
                        desc = parts[3]
                        val = parts[4]
                        
                        if 'Design Water Flow Rate' in desc:
                            results[f"{name}_Flow"] = val
                        elif 'Rated Capacity' in desc or 'Design Capacity' in desc:
                            results[f"{name}_Cap"] = val
                
                # Parse Plant Loop Sizing (Pump, etc)
                if 'Pump:VariableSpeed' in line and 'Component Sizing' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        name = parts[2]
                        desc = parts[3]
                        val = parts[4]
                        
                        if 'Design Maximum Flow Rate' in desc or 'Design Flow Rate' in desc:
                            results[f"{name}_Flow"] = val
                            
                # Parse Heat Pump sizing
                if 'HeatPump' in line and 'Component Sizing' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        name = parts[2]
                        desc = parts[3]
                        val = parts[4]
                        results[f"{name}_{desc.replace(' ', '_')}"] = val
                        
    except Exception as e:
        print(f"Error parsing EIO file: {e}")
    
    print(f"  Total sizing values extracted: {len(results)}")
    return results


def apply_hard_sizing(idf_path: str, sizing_data: Dict[str, str], output_suffix: str = "_fixed") -> str:
    """
    Open IDF and replace autosize with concrete values.
    
    Args:
        idf_path: Path to the IDF file to modify
        sizing_data: Dictionary of extracted sizing values
        output_suffix: Suffix to add to output filename
        
    Returns:
        Path to the frozen IDF file
    """
    try:
        from eppy.modeleditor import IDF
    except ImportError:
        print("Error: eppy not installed. Run: pip install eppy")
        return None
    
    # Find IDD path
    idd_paths = [
        "D:/energyplus/2320/Energy+.idd",
        "C:/EnergyPlusV24-1-0/Energy+.idd",
        "C:/EnergyPlusV23-2-0/Energy+.idd",
        "C:/EnergyPlusV23-1-0/Energy+.idd",
    ]
    
    idd_path = None
    for p in idd_paths:
        if os.path.exists(p):
            idd_path = p
            break
    
    if idd_path is None:
        print("Error: Could not find Energy+.idd file")
        return None
    
    # Set IDD if not already set
    if not IDF.iddname:
        IDF.setiddname(idd_path)
    
    print(f"\n[Applying] Hard-coding sizes into: {idf_path}")
    idf = IDF(idf_path)
    
    changes_made = 0
    
    # 1. Fix Heating Coils
    for coil in idf.idfobjects['COIL:HEATING:WATER']:
        cap_key = f"{coil.Name}_Cap"
        flow_key = f"{coil.Name}_Flow"
        
        if cap_key in sizing_data:
            try:
                val = float(sizing_data[cap_key])
                if str(coil.Rated_Capacity).lower() == 'autosize':
                    coil.Rated_Capacity = val
                    print(f"  Fixed {coil.Name} Rated_Capacity = {val:.2f}")
                    changes_made += 1
            except:
                pass
                
        if flow_key in sizing_data:
            try:
                val = float(sizing_data[flow_key])
                if str(coil.Maximum_Water_Flow_Rate).lower() == 'autosize':
                    coil.Maximum_Water_Flow_Rate = val
                    print(f"  Fixed {coil.Name} Maximum_Water_Flow_Rate = {val:.6f}")
                    changes_made += 1
            except:
                pass
    
    # 2. Fix Cooling Coils
    for coil in idf.idfobjects['COIL:COOLING:WATER']:
        for key, val_str in sizing_data.items():
            if coil.Name in key:
                try:
                    val = float(val_str)
                    if 'Capacity' in key and str(getattr(coil, 'Design_Coil_Load', 'autosize')).lower() == 'autosize':
                        coil.Design_Coil_Load = val
                        changes_made += 1
                    if 'Water_Flow' in key and str(getattr(coil, 'Design_Water_Flow_Rate', 'autosize')).lower() == 'autosize':
                        coil.Design_Water_Flow_Rate = val
                        changes_made += 1
                except:
                    pass
    
    # 3. Fix Pumps
    for pump in idf.idfobjects['PUMP:VARIABLESPEED']:
        flow_key = f"{pump.Name}_Flow"
        if flow_key in sizing_data:
            try:
                val = float(sizing_data[flow_key])
                if str(pump.Design_Maximum_Flow_Rate).lower() == 'autosize':
                    pump.Design_Maximum_Flow_Rate = val
                    print(f"  Fixed {pump.Name} Design_Maximum_Flow_Rate = {val:.6f}")
                    changes_made += 1
            except:
                pass
    
    # 4. Fix Constant Speed Pumps
    for pump in idf.idfobjects['PUMP:CONSTANTSPEED']:
        flow_key = f"{pump.Name}_Flow"
        if flow_key in sizing_data:
            try:
                val = float(sizing_data[flow_key])
                if str(pump.Design_Flow_Rate).lower() == 'autosize':
                    pump.Design_Flow_Rate = val
                    print(f"  Fixed {pump.Name} Design_Flow_Rate = {val:.6f}")
                    changes_made += 1
            except:
                pass
    
    # 5. Fix Heat Pumps (EIR)
    for hp in idf.idfobjects['HEATPUMP:PLANTLOOP:EIR:HEATING']:
        for key, val_str in sizing_data.items():
            if hp.Name in key:
                try:
                    val = float(val_str)
                    if 'Capacity' in key and str(hp.Reference_Capacity).lower() == 'autosize':
                        hp.Reference_Capacity = val
                        print(f"  Fixed {hp.Name} Reference_Capacity = {val:.2f}")
                        changes_made += 1
                except:
                    pass
    
    # 6. Fix Boilers (if still present)
    for boiler in idf.idfobjects['BOILER:HOTWATER']:
        for key, val_str in sizing_data.items():
            if boiler.Name in key:
                try:
                    val = float(val_str)
                    if 'Capacity' in key and str(boiler.Nominal_Capacity).lower() == 'autosize':
                        boiler.Nominal_Capacity = val
                        changes_made += 1
                    if 'Flow' in key and str(boiler.Design_Water_Flow_Rate).lower() == 'autosize':
                        boiler.Design_Water_Flow_Rate = val
                        changes_made += 1
                except:
                    pass
    
    # 7. Fix Chillers
    for chiller in idf.idfobjects['CHILLER:ELECTRIC:EIR']:
        for key, val_str in sizing_data.items():
            if chiller.Name in key:
                try:
                    val = float(val_str)
                    if 'Capacity' in key and str(chiller.Reference_Capacity).lower() == 'autosize':
                        chiller.Reference_Capacity = val
                        changes_made += 1
                except:
                    pass
    
    # 8. Fix WaterHeater (TES)
    for tank in idf.idfobjects['WATERHEATER:STRATIFIED']:
        for key, val_str in sizing_data.items():
            if tank.Name in key and 'Flow' in key:
                try:
                    val = float(val_str)
                    # These are typically not autosized but check anyway
                except:
                    pass
    
    # Save to new file
    output_path = idf_path.replace(".idf", f"{output_suffix}.idf")
    idf.saveas(output_path)
    
    print(f"\n[Complete] Total changes made: {changes_made}")
    print(f"[Saved] Frozen model: {output_path}")
    
    return output_path


def run_sizing_simulation(idf_path: str, weather_path: str, output_dir: str = "outputs/sizing_run") -> str:
    """
    Run EnergyPlus simulation for sizing calculations only.
    
    Args:
        idf_path: Path to IDF file
        weather_path: Path to EPW weather file
        output_dir: Directory for simulation outputs
        
    Returns:
        Path to the .eio file
    """
    import subprocess
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find EnergyPlus executable
    ep_paths = [
        "D:/energyplus/2320/energyplus.exe",
        "C:/EnergyPlusV24-1-0/energyplus.exe",
        "C:/EnergyPlusV23-2-0/energyplus.exe",
        "C:/EnergyPlusV23-1-0/energyplus.exe",
        "energyplus",  # If in PATH
    ]
    
    ep_exe = None
    for p in ep_paths:
        if os.path.exists(p) or shutil.which(p):
            ep_exe = p
            break
    
    if ep_exe is None:
        print("Error: Could not find EnergyPlus executable")
        print("Please ensure EnergyPlus is installed and accessible")
        return None
    
    print(f"\n[Simulation] Running sizing simulation...")
    print(f"  IDF: {idf_path}")
    print(f"  Weather: {weather_path}")
    print(f"  Output: {output_dir}")
    print(f"  EnergyPlus: {ep_exe}")
    
    # Run EnergyPlus
    cmd = [
        ep_exe,
        "-w", weather_path,
        "-d", output_dir,
        "-r",  # Readvars
        idf_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Warning: EnergyPlus returned code {result.returncode}")
            if result.stderr:
                print(f"Errors: {result.stderr[:500]}")
        else:
            print("  Simulation completed successfully")
            
    except subprocess.TimeoutExpired:
        print("Error: Simulation timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"Error running simulation: {e}")
        return None
    
    eio_file = os.path.join(output_dir, "eplusout.eio")
    return eio_file if os.path.exists(eio_file) else None


def freeze_model(
    input_idf: str = "outputs/sim_building.idf",
    weather_file: str = "data/weather/Shanghai_2024.epw",
    output_suffix: str = "_frozen",
    skip_simulation: bool = False,
    eio_path: Optional[str] = None,
) -> str:
    """
    Main function to freeze an IDF model with hard-coded sizes.
    
    Args:
        input_idf: Path to input IDF file
        weather_file: Path to weather file for simulation
        output_suffix: Suffix for output filename
        skip_simulation: If True, skip simulation and use existing eio_path
        eio_path: Path to existing .eio file (if skip_simulation=True)
        
    Returns:
        Path to frozen IDF file
    """
    print("=" * 60)
    print("Freeze Model Tool - Hard-code Autosized Values")
    print("=" * 60)
    
    # Resolve paths
    input_idf = str(Path(input_idf).resolve())
    weather_file = str(Path(weather_file).resolve())
    
    if not os.path.exists(input_idf):
        print(f"Error: Input IDF not found: {input_idf}")
        return None
    
    if not skip_simulation:
        if not os.path.exists(weather_file):
            print(f"Error: Weather file not found: {weather_file}")
            return None
        
        # Run sizing simulation
        eio_path = run_sizing_simulation(
            input_idf, 
            weather_file,
            output_dir="outputs/sizing_run"
        )
        
        if eio_path is None:
            print("Error: Failed to run sizing simulation")
            return None
    else:
        if eio_path is None or not os.path.exists(eio_path):
            print("Error: No valid EIO path provided for skip_simulation mode")
            return None
    
    # Parse sizing results
    sizing_data = get_sizing_results(eio_path)
    
    if not sizing_data:
        print("Warning: No sizing data extracted. Model may already be fixed.")
    
    # Apply hard sizing
    frozen_idf = apply_hard_sizing(input_idf, sizing_data, output_suffix)
    
    return frozen_idf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Freeze autosize values in EnergyPlus IDF model"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="outputs/sim_building.idf",
        help="Input IDF file path"
    )
    parser.add_argument(
        "--weather", "-w",
        type=str,
        default="data/weather/Shanghai_2024.epw",
        help="Weather file (EPW) path"
    )
    parser.add_argument(
        "--suffix", "-s",
        type=str,
        default="_frozen",
        help="Suffix for output filename (default: _frozen)"
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Skip simulation and use existing .eio file"
    )
    parser.add_argument(
        "--eio",
        type=str,
        default=None,
        help="Path to existing .eio file (use with --skip-sim)"
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    os.chdir(PROJECT_ROOT)
    
    result = freeze_model(
        input_idf=args.input,
        weather_file=args.weather,
        output_suffix=args.suffix,
        skip_simulation=args.skip_sim,
        eio_path=args.eio,
    )
    
    if result:
        print(f"\n✅ Model frozen successfully: {result}")
    else:
        print("\n❌ Failed to freeze model")
        sys.exit(1)
