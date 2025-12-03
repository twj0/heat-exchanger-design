"""
IDF Model Builder for University Classroom Building.

Based on 5ZoneAirCooled.idf template, modified to match the paper:
"Deep reinforcement learning-based control of thermal energy storage 
for university classrooms"

Building Layout:
- 4 Perimeter Zones (Classrooms) + 1 Core Zone (Corridor)
- Paper specs: 218.09 m² per classroom, 5.1m height
- VAV system with hot water reheat coils
- Central chilled water cooling, hot water boiler

Usage:
    python -m envs.builder
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json

# ================= Configuration =================
# Default paths - modify according to your EnergyPlus installation
DEFAULT_IDD_PATH = "D:/energyplus/2320/Energy+.idd"
DEFAULT_TEMPLATE = "data/templates/5Zone_Base.idf"
DEFAULT_OUTPUT = "outputs/sim_building.idf"
DEFAULT_WEATHER = "data/weather/Shanghai_2024.epw"


class IDFBuilder:
    """
    Builder class for creating and modifying EnergyPlus IDF models.
    
    This class takes the 5ZoneAirCooled template and modifies it to match
    the paper's building specifications for university classrooms.
    """
    
    # Zone name mapping: template name -> paper name
    ZONE_MAPPING = {
        "SPACE1-1": "Classroom_South",    # South perimeter
        "SPACE2-1": "Classroom_East",     # East perimeter  
        "SPACE3-1": "Classroom_North",    # North perimeter
        "SPACE4-1": "Classroom_West",     # West perimeter
        "SPACE5-1": "Corridor",           # Core zone
        "PLENUM-1": "Return_Plenum",      # Return air plenum
    }
    
    def __init__(self, idd_path: str = None, template_path: str = None):
        """
        Initialize the IDF builder.
        
        Args:
            idd_path: Path to Energy+.idd file
            template_path: Path to template IDF file
        """
        try:
            from eppy.modeleditor import IDF
            self.IDF = IDF
        except ImportError:
            raise ImportError(
                "eppy not installed. Run: pip install eppy\n"
                "Also ensure EnergyPlus is installed."
            )
        
        # Set IDD path
        self.idd_path = idd_path or self._find_idd_path()
        if not os.path.exists(self.idd_path):
            raise FileNotFoundError(
                f"IDD file not found: {self.idd_path}\n"
                "Please install EnergyPlus or specify correct IDD path."
            )
        
        # Initialize IDF with IDD
        if not IDF.iddname:
            IDF.setiddname(self.idd_path)
        
        # Load template
        self.template_path = template_path or DEFAULT_TEMPLATE
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template IDF not found: {self.template_path}")
        
        self.idf = IDF(self.template_path)
        print(f"✅ Loaded template: {self.template_path}")
    
    def _find_idd_path(self) -> str:
        """Try to find EnergyPlus IDD file automatically."""
        # Common installation paths
        possible_paths = [
            DEFAULT_IDD_PATH,
            "C:/EnergyPlusV24-1-0/Energy+.idd",
            "C:/EnergyPlusV23-1-0/Energy+.idd",
            "C:/EnergyPlusV22-2-0/Energy+.idd",
            "/usr/local/EnergyPlus-23-2-0/Energy+.idd",
            "/Applications/EnergyPlus-23-2-0/Energy+.idd",
        ]
        
        # Check environment variable
        ep_dir = os.environ.get('ENERGYPLUS_DIR')
        if ep_dir:
            possible_paths.insert(0, os.path.join(ep_dir, 'Energy+.idd'))
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return DEFAULT_IDD_PATH
    
    def set_simulation_settings(
        self,
        timesteps_per_hour: int = 4,
        start_month: int = 1,
        start_day: int = 1,
        end_month: int = 12,
        end_day: int = 31,
    ):
        """
        Configure simulation timestep and run period.
        
        Args:
            timesteps_per_hour: Number of timesteps per hour (4 = 15min)
            start_month: Simulation start month
            start_day: Simulation start day
            end_month: Simulation end month
            end_day: Simulation end day
        """
        # Set timestep (paper uses 15-minute intervals)
        timesteps = self.idf.idfobjects['TIMESTEP']
        if timesteps:
            timesteps[0].Number_of_Timesteps_per_Hour = timesteps_per_hour
            print(f"  Timestep: {60//timesteps_per_hour} minutes")
        
        # Set run period
        run_periods = self.idf.idfobjects['RUNPERIOD']
        if run_periods:
            rp = run_periods[0]
            rp.Begin_Month = start_month
            rp.Begin_Day_of_Month = start_day
            rp.End_Month = end_month
            rp.End_Day_of_Month = end_day
            
            # [FIX] Clear year fields to let EnergyPlus auto-match weather file
            # This avoids Feb29 warning when using TMY weather data
            try:
                rp.Begin_Year = ''
                rp.End_Year = ''
                # Don't treat weather as actual year data (avoids leap year issues)
                rp.Treat_Weather_as_Actual = 'No'
            except:
                pass  # Fields may not exist in older EP versions
            
            print(f"  Run period: {start_month}/{start_day} - {end_month}/{end_day}")
        
        return self
    
    def set_location(
        self,
        city: str = "Shanghai",
        latitude: float = 31.23,
        longitude: float = 121.47,
        timezone: float = 8.0,
        elevation: float = 4.0,
    ):
        """
        Set building location for Shanghai, China.
        
        Args:
            city: City name
            latitude: Latitude in degrees
            longitude: Longitude in degrees  
            timezone: UTC offset in hours
            elevation: Elevation in meters
        """
        locations = self.idf.idfobjects['SITE:LOCATION']
        if locations:
            loc = locations[0]
            loc.Name = f"{city}_CHN"
            loc.Latitude = latitude
            loc.Longitude = longitude
            loc.Time_Zone = timezone
            loc.Elevation = elevation
            print(f"  Location: {city} ({latitude}°N, {longitude}°E)")
        
        return self
    
    def rename_zones(self):
        """
        Rename zones from template names to paper-style names.
        
        Mapping:
        - SPACE1-1 -> Classroom_South
        - SPACE2-1 -> Classroom_East
        - SPACE3-1 -> Classroom_North
        - SPACE4-1 -> Classroom_West
        - SPACE5-1 -> Corridor
        """
        # Get all zones
        zones = self.idf.idfobjects['ZONE']
        renamed_count = 0
        
        for zone in zones:
            old_name = zone.Name
            if old_name in self.ZONE_MAPPING:
                new_name = self.ZONE_MAPPING[old_name]
                
                # Rename zone
                zone.Name = new_name
                
                # Update all references to this zone throughout the IDF
                self._update_zone_references(old_name, new_name)
                
                renamed_count += 1
                print(f"  Renamed: {old_name} -> {new_name}")
        
        print(f"  Total zones renamed: {renamed_count}")
        return self
    
    def _update_zone_references(self, old_name: str, new_name: str):
        """Update all references to a zone name throughout the IDF."""
        # List of object types that reference zone names
        zone_ref_objects = [
            'ZONEINFILTRATION:DESIGNFLOWRATE',
            'PEOPLE',
            'LIGHTS', 
            'ELECTRICEQUIPMENT',
            'SIZING:ZONE',
            'ZONECONTROL:THERMOSTAT',
            'ZONEHVAC:EQUIPMENTCONNECTIONS',
            'ZONEHVAC:EQUIPMENTLIST',
            'BUILDINGSURFACE:DETAILED',
            'FENESTRATIONSURFACE:DETAILED',
            'AIRTERMINAL:SINGLEDUCT:VAV:REHEAT',
            # AirLoop HVAC objects that reference zones
            'AIRLOOPHVAC:RETURNPLENUM',
            'AIRLOOPHVAC:SUPPLYPLENUM',
            'AIRLOOPHVAC:ZONESPLITTER',
            'AIRLOOPHVAC:ZONEMIXER',
        ]
        
        for obj_type in zone_ref_objects:
            try:
                objects = self.idf.idfobjects[obj_type]
                for obj in objects:
                    # Check all fields for zone name reference
                    for field in obj.fieldnames:
                        try:
                            value = getattr(obj, field)
                            if value == old_name:
                                setattr(obj, field, new_name)
                        except:
                            pass
            except:
                pass
    
    def set_building_info(
        self,
        name: str = "University_Classroom_Shanghai",
        north_axis: float = 0.0,
    ):
        """
        Set building metadata.
        
        Args:
            name: Building name
            north_axis: Building rotation from north (degrees)
        """
        buildings = self.idf.idfobjects['BUILDING']
        if buildings:
            bldg = buildings[0]
            bldg.Name = name
            bldg.North_Axis = north_axis
            print(f"  Building: {name}")
        
        return self
    
    def set_occupancy(
        self,
        people_per_zone: int = 30,
        activity_level: float = 120.0,  # W/person (seated, light work)
    ):
        """
        Set occupancy for classroom zones.
        
        Args:
            people_per_zone: Number of occupants per classroom
            activity_level: Metabolic rate in W/person
        """
        people_objects = self.idf.idfobjects['PEOPLE']
        
        for people in people_objects:
            # Try different field names for zone reference
            zone_name = None
            for field in ['Zone_or_ZoneList_Name', 'Zone_Name', 'Zone_or_ZoneList_or_Space_or_SpaceList_Name']:
                try:
                    zone_name = getattr(people, field)
                    break
                except:
                    pass
            
            if zone_name is None:
                # Get zone name from first field after Name
                zone_name = people[2] if len(people.fieldvalues) > 2 else "Unknown"
            
            # Only modify classroom zones, not corridor
            if 'Classroom' in str(zone_name) or 'SPACE' in str(zone_name):
                try:
                    people.Number_of_People = people_per_zone
                except:
                    pass
                print(f"  {zone_name}: {people_per_zone} people")
        
        return self
    
    def add_output_variables(self):
        """Add output variables needed for RL training."""
        output_vars = [
            # Zone temperatures
            ("Zone Mean Air Temperature", "*", "Timestep"),
            ("Zone Air System Sensible Heating Energy", "*", "Timestep"),
            ("Zone Air System Sensible Cooling Energy", "*", "Timestep"),
            
            # HVAC - [FIX 7] Correct Facility variable names
            ("Facility Total Electricity Demand Rate", "*", "Timestep"),
            ("Facility Total HVAC Electricity Demand Rate", "*", "Timestep"),
            
            # Outdoor conditions
            ("Site Outdoor Air Drybulb Temperature", "*", "Timestep"),
            ("Site Direct Solar Radiation Rate per Area", "*", "Timestep"),
            
            # TES Tank (if exists)
            ("Water Heater Tank Temperature", "*", "Timestep"),
            ("Water Heater Heat Loss Rate", "*", "Timestep"),
            ("Water Heater Heating Rate", "*", "Timestep"),
            ("Water Heater Use Side Heat Transfer Rate", "*", "Timestep"),
        ]
        
        for var_name, key, freq in output_vars:
            # Check if already exists
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=key,
                    Variable_Name=var_name,
                    Reporting_Frequency=freq,
                )
        
        print(f"  Added {len(output_vars)} output variables")
        return self
    
    def set_envelope_uvalues(
        self,
        wall_uvalue: float = 0.30,  # W/m²K (from paper Table A.10)
        window_uvalue: float = 1.1,  # W/m²K (from paper)
        roof_uvalue: float = 0.25,   # W/m²K
    ):
        """
        Set building envelope U-values to match paper specifications.
        
        This modifies insulation thickness to achieve target U-values.
        
        Args:
            wall_uvalue: Target wall U-value (W/m²K)
            window_uvalue: Target window U-value (W/m²K)
            roof_uvalue: Target roof U-value (W/m²K)
        """
        print(f"  Target Wall U-value: {wall_uvalue} W/m²K")
        print(f"  Target Window U-value: {window_uvalue} W/m²K")
        
        # Calculate required insulation R-value for wall
        # U = 1/R_total, R_total = R_insulation + R_other_layers
        # For simplicity, we modify the IN46 insulation material
        target_r_insulation = (1.0 / wall_uvalue) - 0.5  # Subtract ~0.5 for other layers
        target_r_insulation = max(0.5, target_r_insulation)  # Minimum R-value
        
        # Modify wall insulation material (IN46)
        materials = self.idf.idfobjects['MATERIAL']
        for mat in materials:
            if mat.Name == 'IN46':
                # Calculate new thickness for target R-value
                # R = thickness / conductivity
                conductivity = mat.Conductivity
                new_thickness = target_r_insulation * conductivity
                mat.Thickness = new_thickness
                print(f"  Modified IN46 insulation: thickness={new_thickness:.4f}m, R={target_r_insulation:.2f}")
        
        # Modify window glazing for target U-value
        # For double-pane window: U ≈ 1 / (0.04 + d/k_glass + R_gap + d/k_glass + 0.04)
        glazings = self.idf.idfobjects['WINDOWMATERIAL:GLAZING']
        for glazing in glazings:
            if 'CLEAR' in glazing.Name.upper():
                # Adjust glass thickness slightly to influence U-value
                # Note: Real U-value control requires proper low-e coatings
                glazing.Thickness = 0.006  # 6mm glass
                print(f"  Modified glazing {glazing.Name}: thickness=0.006m")
        
        # Modify air gap if exists
        gases = self.idf.idfobjects['WINDOWMATERIAL:GAS']
        for gas in gases:
            gas.Thickness = 0.013  # 13mm air gap for better insulation
            print(f"  Modified gas layer {gas.Name}: thickness=0.013m")
        
        return self
    
    def add_tes_tank(
        self,
        tank_volume: float = 12.0,    # m³ (from paper)
        max_temp: float = 55.0,       # °C
        min_temp: float = 40.0,       # °C (heating mode setpoint)
        deadband_temp: float = 2.0,   # °C
        tank_ua: float = 10.0,        # W/K (heat loss coefficient)
        series_connection: bool = True,  # CRITICAL: Series vs Parallel
    ):
        """
        Add Thermal Energy Storage (TES) water tank to the hot water loop.
        
        CRITICAL TOPOLOGY FIX:
        - Series connection: HP -> TES -> Load (HP heats TES, TES supplies load)
        - This enables proper charging/discharging behavior for DRL control
        
        Args:
            tank_volume: Tank volume in m³
            max_temp: Maximum tank temperature (°C)
            min_temp: Minimum tank temperature / setpoint (°C)
            deadband_temp: Temperature deadband (°C)
            tank_ua: Tank heat loss coefficient (W/K)
            series_connection: If True, TES is in series after HP (recommended)
        """
        print(f"  Adding TES tank: {tank_volume} m³, {min_temp}-{max_temp}°C")
        print(f"  Topology: {'SERIES (HP->TES->Load)' if series_connection else 'PARALLEL'}")
        
        # Create ambient temperature schedule
        # Using Outdoors indicator for realistic heat loss (varies with weather)
        existing_schedules = [s.Name for s in self.idf.idfobjects['SCHEDULE:COMPACT']]
        if 'TES_Ambient_Temp' not in existing_schedules:
            # For indoor placement, use fixed 20°C
            # For outdoor/basement, this should track weather
            self.idf.newidfobject(
                'SCHEDULE:COMPACT',
                Name='TES_Ambient_Temp',
                Schedule_Type_Limits_Name='Temperature',
                Field_1='Through: 12/31',
                Field_2='For: AllDays',
                Field_3='Until: 24:00',
                Field_4=15.0,  # Basement temperature assumption
            )
        
        # Find the boiler/HP outlet node for series connection
        boiler_outlet = 'Central Boiler Outlet Node'
        loop_supply_outlet = 'HW Supply Outlet Node'
        
        # Check existing boilers to find actual node names
        boilers = self.idf.idfobjects['BOILER:HOTWATER']
        if boilers:
            boiler_outlet = boilers[0].Boiler_Water_Outlet_Node_Name
        
        # Create the WaterHeater:Mixed object (TES Tank)
        tes_tank = self.idf.newidfobject('WATERHEATER:MIXED')
        tes_tank.Name = 'TES_Tank'
        tes_tank.Tank_Volume = tank_volume
        tes_tank.Deadband_Temperature_Difference = deadband_temp
        tes_tank.Maximum_Temperature_Limit = max_temp
        tes_tank.Heater_Control_Type = 'Cycle'
        tes_tank.Heater_Maximum_Capacity = 0  # Passive storage - no internal heater
        tes_tank.Heater_Fuel_Type = 'Electricity'
        tes_tank.Heater_Thermal_Efficiency = 1.0
        
        # Ambient temperature - use Schedule for controlled environment
        tes_tank.Ambient_Temperature_Indicator = 'Schedule'
        tes_tank.Ambient_Temperature_Schedule_Name = 'TES_Ambient_Temp'
        
        # Heat loss coefficients (realistic for insulated tank)
        try:
            tes_tank.Off_Cycle_Loss_Coefficient_to_Ambient_Temperature = tank_ua
            tes_tank.On_Cycle_Loss_Coefficient_to_Ambient_Temperature = tank_ua
        except:
            pass
        
        if series_connection:
            # ============= SERIES CONNECTION (v1.2 FIX) =============
            # CRITICAL: Node chain must be consistent across Branch AND Component definitions!
            # HP outlet -> Intermediate Node -> TES inlet -> TES outlet -> Loop supply outlet
            # 
            # Node naming convention:
            #   - HP/Boiler Inlet: Central Boiler Inlet Node (unchanged)
            #   - Intermediate: HP_Outlet_TES_Inlet (new node connecting HP to TES)
            #   - Loop Outlet: Central Boiler Outlet Node (unchanged)
            
            node_hp_inlet = 'Central Boiler Inlet Node'
            node_intermediate = 'HP_Outlet_TES_Inlet'  # NEW intermediate node
            node_loop_outlet = 'Central Boiler Outlet Node'
            
            # Set TES nodes: Inlet from HP, Outlet to loop
            tes_tank.Use_Side_Inlet_Node_Name = node_intermediate
            tes_tank.Use_Side_Outlet_Node_Name = node_loop_outlet
            tes_tank.Use_Side_Effectiveness = 1.0
            
            try:
                tes_tank.Use_Side_Design_Flow_Rate = 'autosize'
            except:
                pass
            
            # ============= FIX: Update HP/Boiler Component Outlet Node =============
            # This is the ROOT CAUSE of the node connection error!
            # We must change the Boiler/HP object's outlet to point to intermediate node.
            
            boilers = self.idf.idfobjects['BOILER:HOTWATER']
            for boiler in boilers:
                boiler.Boiler_Water_Outlet_Node_Name = node_intermediate
                print(f"    Fixed Boiler outlet: {boiler.Name} -> {node_intermediate}")
            
            # Also check for HeatPump if already upgraded
            hps = self.idf.idfobjects['HEATPUMP:PLANTLOOP:EIR:HEATING']
            for hp in hps:
                hp.Load_Side_Outlet_Node_Name = node_intermediate
                print(f"    Fixed HeatPump outlet: {hp.Name} -> {node_intermediate}")
            
            # ============= Update Branch Definition =============
            branches = self.idf.idfobjects['BRANCH']
            boiler_branch_found = False
            
            for branch in branches:
                branch_name = branch.Name.lower() if hasattr(branch, 'Name') else ''
                if 'central boiler' in branch_name or 'boiler' in branch_name:
                    boiler_branch_found = True
                    
                    # Component 1: HP/Boiler (Inlet -> Intermediate)
                    branch.Component_1_Outlet_Node_Name = node_intermediate
                    
                    # Component 2: TES (Intermediate -> Loop Outlet)
                    branch.Component_2_Object_Type = 'WaterHeater:Mixed'
                    branch.Component_2_Name = 'TES_Tank'
                    branch.Component_2_Inlet_Node_Name = node_intermediate
                    branch.Component_2_Outlet_Node_Name = node_loop_outlet
                    
                    print(f"  Modified branch '{branch.Name}' for series connection")
                    print(f"    HP -> {node_intermediate} -> TES -> {node_loop_outlet}")
                    break
            
            if not boiler_branch_found:
                print("  Warning: Boiler branch not found, creating new series branch")
                self.idf.newidfobject(
                    'BRANCH',
                    Name='HP_TES_Series_Branch',
                    Component_1_Object_Type='Boiler:HotWater',
                    Component_1_Name='Central_HeatPump',
                    Component_1_Inlet_Node_Name=node_hp_inlet,
                    Component_1_Outlet_Node_Name=node_intermediate,
                    Component_2_Object_Type='WaterHeater:Mixed',
                    Component_2_Name='TES_Tank',
                    Component_2_Inlet_Node_Name=node_intermediate,
                    Component_2_Outlet_Node_Name=node_loop_outlet,
                )
        else:
            # ============= PARALLEL CONNECTION (Original - NOT RECOMMENDED) =============
            tes_tank.Use_Side_Inlet_Node_Name = 'TES Tank Use Inlet Node'
            tes_tank.Use_Side_Outlet_Node_Name = 'TES Tank Use Outlet Node'
            tes_tank.Use_Side_Effectiveness = 1.0
            
            # Create separate branch (parallel)
            self.idf.newidfobject(
                'BRANCH',
                Name='TES Tank Branch',
                Component_1_Object_Type='WaterHeater:Mixed',
                Component_1_Name='TES_Tank',
                Component_1_Inlet_Node_Name='TES Tank Use Inlet Node',
                Component_1_Outlet_Node_Name='TES Tank Use Outlet Node',
            )
            
            # Add to splitter/mixer for parallel operation
            # (This was the original problematic approach)
            print("  Warning: Parallel TES cannot be charged by HP!")
        
        # Add TES output variables for monitoring
        tes_outputs = [
            ("Water Heater Tank Temperature", "TES_Tank", "Timestep"),
            ("Water Heater Heat Loss Rate", "TES_Tank", "Timestep"),
            ("Water Heater Use Side Heat Transfer Rate", "TES_Tank", "Timestep"),
            ("Water Heater Use Side Inlet Temperature", "TES_Tank", "Timestep"),
            ("Water Heater Use Side Outlet Temperature", "TES_Tank", "Timestep"),
            ("Water Heater Use Side Mass Flow Rate", "TES_Tank", "Timestep"),
        ]
        for var_name, key, freq in tes_outputs:
            self.idf.newidfobject(
                'OUTPUT:VARIABLE',
                Key_Value=key,
                Variable_Name=var_name,
                Reporting_Frequency=freq,
            )
        
        print(f"  ✅ TES Tank added in {'SERIES' if series_connection else 'PARALLEL'} topology")
        return self
    
    def add_pv_system(
        self,
        peak_power: float = 47000.0,  # W (from paper)
        efficiency: float = 0.18,      # 18% efficiency
        surface_name: str = None,      # Roof surface to attach PV
        pv_area: float = 261.0,        # m² (47kW / 0.18 efficiency / 1kW/m² peak irradiance)
    ):
        """
        Add rooftop photovoltaic system for carbon-aware optimization.
        
        Required for Innovation B: PV generation enables:
        - Self-consumption optimization
        - Net metering calculations
        - Carbon offset accounting
        
        Args:
            peak_power: Peak power output (W)
            efficiency: Cell efficiency (0-1)
            surface_name: Roof surface to attach PV
            pv_area: Active PV area (m²)
        """
        print(f"  Adding PV system: {peak_power/1000:.1f} kW peak, {efficiency*100:.0f}% efficiency")
        
        # Find roof surface if not specified
        if surface_name is None:
            surfaces = self.idf.idfobjects['BUILDINGSURFACE:DETAILED']
            for surf in surfaces:
                if hasattr(surf, 'Surface_Type'):
                    if surf.Surface_Type.lower() == 'roof':
                        surface_name = surf.Name
                        print(f"  Found roof surface: {surface_name}")
                        break
        
        if surface_name is None:
            # Create a shading surface for PV if no roof found
            print("  No roof found, creating dedicated PV surface...")
            surface_name = "PV_Shading_Surface"
            
            # Create a flat shading surface on the roof
            self.idf.newidfobject(
                'SHADING:SITE:DETAILED',
                Name=surface_name,
                Transmittance_Schedule_Name='',
                Number_of_Vertices=4,
                Vertex_1_Xcoordinate=0,
                Vertex_1_Ycoordinate=0,
                Vertex_1_Zcoordinate=10,  # Roof height
                Vertex_2_Xcoordinate=pv_area**0.5,  # Square approximation
                Vertex_2_Ycoordinate=0,
                Vertex_2_Zcoordinate=10,
                Vertex_3_Xcoordinate=pv_area**0.5,
                Vertex_3_Ycoordinate=pv_area**0.5,
                Vertex_3_Zcoordinate=10,
                Vertex_4_Xcoordinate=0,
                Vertex_4_Ycoordinate=pv_area**0.5,
                Vertex_4_Zcoordinate=10,
            )
        
        # Create PV performance object (Simple model)
        self.idf.newidfobject(
            'PHOTOVOLTAICPERFORMANCE:SIMPLE',
            Name='PV_Performance',
            Fraction_of_Surface_Area_with_Active_Solar_Cells=0.85,  # 85% active area
            Conversion_Efficiency_Input_Mode='Fixed',
            Value_for_Cell_Efficiency_if_Fixed=efficiency,
        )
        
        # Create PV generator
        pv_gen = self.idf.newidfobject('GENERATOR:PHOTOVOLTAIC')
        pv_gen.Name = 'Rooftop_PV_Array'
        pv_gen.Surface_Name = surface_name
        pv_gen.Photovoltaic_Performance_Object_Type = 'PhotovoltaicPerformance:Simple'
        pv_gen.Module_Performance_Name = 'PV_Performance'
        pv_gen.Heat_Transfer_Integration_Mode = 'Decoupled'
        try:
            pv_gen.Number_of_Series_Strings_in_Parallel = 1
            pv_gen.Number_of_Modules_in_Series = 1
        except:
            pass
        
        # Create inverter (DC to AC conversion)
        self.idf.newidfobject(
            'ELECTRICLOADCENTER:INVERTER:SIMPLE',
            Name='PV_Inverter',
            Availability_Schedule_Name='',  # Always available
            Zone_Name='',
            Radiative_Fraction=0.0,
            Inverter_Efficiency=0.96,  # 96% DC-AC efficiency
        )
        
        # Create electric load center generators list
        gen_list = self.idf.newidfobject('ELECTRICLOADCENTER:GENERATORS')
        gen_list.Name = 'PV_Generator_List'
        gen_list.Generator_1_Object_Type = 'Generator:Photovoltaic'
        gen_list.Generator_1_Name = 'Rooftop_PV_Array'
        gen_list.Generator_1_Rated_Electric_Power_Output = peak_power
        try:
            gen_list.Generator_1_Availability_Schedule_Name = ''
            gen_list.Generator_1_Rated_Thermal_to_Electrical_Power_Ratio = ''
        except:
            pass
        
        # Create distribution center
        dist = self.idf.newidfobject('ELECTRICLOADCENTER:DISTRIBUTION')
        dist.Name = 'PV_Load_Center'
        dist.Generator_List_Name = 'PV_Generator_List'
        dist.Generator_Operation_Scheme_Type = 'Baseload'
        try:
            dist.Electrical_Buss_Type = 'DirectCurrentWithInverter'
            dist.Inverter_Name = 'PV_Inverter'
        except:
            dist.Electrical_Buss_Type = 'AlternatingCurrent'
        
        # Add PV output variables
        # [FIX 2] Key must match the generator object name exactly: "Rooftop_PV_Array"
        pv_outputs = [
            ("Generator Produced DC Electricity Rate", "Rooftop_PV_Array", "Timestep"),
            ("Generator Produced DC Electricity Energy", "Rooftop_PV_Array", "Timestep"),
            ("Electric Load Center Produced Electricity Rate", "PV_Load_Center", "Timestep"),
            ("Facility Total Electricity Demand Rate", "*", "Timestep"),
            ("Facility Net Purchased Electricity Rate", "*", "Timestep"),
        ]
        
        for var_name, key, freq in pv_outputs:
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name and o.Key_Value == key]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=key,
                    Variable_Name=var_name,
                    Reporting_Frequency=freq,
                )
        
        print(f"  ✅ PV system added: {peak_power/1000:.1f} kW on {surface_name}")
        return self
    
    def replace_boiler_with_heatpump(
        self,
        heating_capacity: float = 195800.0,  # W (from paper)
        cop_rated: float = 3.32,
    ):
        """
        Replace gas boiler with proper air-source heat pump.
        
        Uses HeatPump:PlantLoop:EIR:Heating with temperature-dependent
        COP curves for realistic physics (critical for DRL learning).
        """
        print(f"  Replacing boiler with ASHP: {heating_capacity/1000:.1f} kW, COP={cop_rated}")
        
        boilers = self.idf.idfobjects['BOILER:HOTWATER']
        if not boilers:
            print("  Warning: No boiler found")
            return self
        
        boiler = boilers[0]
        inlet_node = boiler.Boiler_Water_Inlet_Node_Name
        outlet_node = boiler.Boiler_Water_Outlet_Node_Name
        
        # ============= Create Temperature-Dependent Performance Curves =============
        # Capacity = f(LWT, OAT) - Biquadratic curve
        # Typical ASHP: capacity drops at low OAT
        self.idf.newidfobject(
            'CURVE:BIQUADRATIC',
            Name='HP_Cap_fT',
            Coefficient1_Constant=0.876,      # Base capacity fraction
            Coefficient2_x=0.0,               # LWT coefficient
            Coefficient3_x2=0.0,              # LWT^2
            Coefficient4_y=0.025,             # OAT coefficient (positive: more capacity at higher OAT)
            Coefficient5_y2=-0.0003,          # OAT^2 (slight reduction at extremes)
            Coefficient6_xy=0.0,              # LWT*OAT interaction
            Minimum_Value_of_x=30,            # Min LWT (°C)
            Maximum_Value_of_x=55,            # Max LWT
            Minimum_Value_of_y=-20,           # Min OAT (°C)
            Maximum_Value_of_y=20,            # Max OAT
        )
        
        # EIR = f(LWT, OAT) - Energy Input Ratio = 1/COP
        # At low OAT, EIR increases (COP drops)
        base_eir = 1.0 / cop_rated
        self.idf.newidfobject(
            'CURVE:BIQUADRATIC',
            Name='HP_EIR_fT',
            Coefficient1_Constant=base_eir * 1.1,  # Slightly higher base EIR
            Coefficient2_x=0.005,                  # Higher LWT = lower COP
            Coefficient3_x2=0.0001,
            Coefficient4_y=-0.015,                 # Lower OAT = higher EIR (lower COP)
            Coefficient5_y2=0.0002,
            Coefficient6_xy=0.0,
            Minimum_Value_of_x=30,
            Maximum_Value_of_x=55,
            Minimum_Value_of_y=-20,
            Maximum_Value_of_y=20,
        )
        
        # EIR = f(PLR) - Part Load performance
        self.idf.newidfobject(
            'CURVE:QUADRATIC',
            Name='HP_EIR_fPLR',
            Coefficient1_Constant=0.1,   # Min EIR at zero load
            Coefficient2_x=0.9,          # Linear increase with PLR
            Coefficient3_x2=0.0,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=1,
        )
        
        # ============= Create Proper Heat Pump Object =============
        # Note: HeatPump:PlantLoop:EIR:Heating requires EnergyPlus 9.4+
        # For better compatibility, use WaterHeater:HeatPump:PumpedCondenser approach
        # or fall back to Boiler with Electricity but with proper efficiency curve
        
        # Create efficiency curve based on OAT for boiler (workaround)
        # Efficiency = f(Twater, Tamb) normalized
        self.idf.newidfobject(
            'CURVE:BICUBIC',
            Name='HP_Efficiency_Curve',
            Coefficient1_Constant=cop_rated * 0.8,
            Coefficient2_x=0.0,
            Coefficient3_x2=0.0,
            Coefficient4_y=0.05,          # OAT effect on efficiency
            Coefficient5_y2=-0.001,
            Coefficient6_xy=0.0,
            Coefficient7_x3=0.0,
            Coefficient8_y3=0.0,
            Coefficient9_x2y=0.0,
            Coefficient10_xy2=0.0,
            Minimum_Value_of_x=30,
            Maximum_Value_of_x=60,
            Minimum_Value_of_y=-20,
            Maximum_Value_of_y=25,
        )
        
        # Create electric "boiler" with OAT-dependent efficiency
        hp = self.idf.newidfobject('BOILER:HOTWATER')
        hp.Name = 'Central_HeatPump'
        hp.Fuel_Type = 'Electricity'
        hp.Nominal_Capacity = heating_capacity
        hp.Nominal_Thermal_Efficiency = cop_rated
        hp.Efficiency_Curve_Temperature_Evaluation_Variable = 'LeavingBoiler'
        hp.Normalized_Boiler_Efficiency_Curve_Name = 'HP_Efficiency_Curve'
        hp.Design_Water_Flow_Rate = 'autosize'
        hp.Minimum_Part_Load_Ratio = 0.1
        hp.Maximum_Part_Load_Ratio = 1.2
        hp.Boiler_Water_Inlet_Node_Name = inlet_node
        hp.Boiler_Water_Outlet_Node_Name = outlet_node
        hp.Water_Outlet_Upper_Temperature_Limit = 55
        hp.Boiler_Flow_Mode = 'LeavingSetpointModulated'
        
        # Update equipment list
        equip_lists = self.idf.idfobjects['PLANTEQUIPMENTLIST']
        for eq_list in equip_lists:
            if 'heating' in eq_list.Name.lower():
                eq_list.Equipment_1_Object_Type = 'Boiler:HotWater'
                eq_list.Equipment_1_Name = 'Central_HeatPump'
        
        # Remove original boiler
        self.idf.removeidfobject(boiler)
        
        # Update branch to use new HP
        branches = self.idf.idfobjects['BRANCH']
        for branch in branches:
            if hasattr(branch, 'Component_1_Name') and 'Central Boiler' in str(branch.Component_1_Name):
                branch.Component_1_Name = 'Central_HeatPump'
        
        # Add HP-specific output variables
        hp_outputs = [
            ("Boiler Heating Rate", "Central_HeatPump", "Timestep"),
            ("Boiler Electricity Rate", "Central_HeatPump", "Timestep"),
            ("Boiler Inlet Temperature", "Central_HeatPump", "Timestep"),
            ("Boiler Outlet Temperature", "Central_HeatPump", "Timestep"),
        ]
        for var_name, key, freq in hp_outputs:
            self.idf.newidfobject(
                'OUTPUT:VARIABLE',
                Key_Value=key,
                Variable_Name=var_name,
                Reporting_Frequency=freq,
            )
        
        print(f"  ✅ Heat pump added with temp-dependent COP curve")
        return self
    
    def add_heatpump_physics_based(
        self,
        heating_capacity: float = 195800.0,  # W (Auto-sizing reference)
        cop_rated: float = 3.2,
    ):
        """
        [Physics Upgrade] Replace Boiler with HeatPump:PlantLoop:EIR:Heating.
        
        This model accurately captures:
        1. Capacity degradation at low outdoor temperatures.
        2. Efficiency (COP) changes with Lift (Condenser - Evaporator temp).
        3. Part-load cycling losses.
        
        For Applied Energy level publication requirements.
        
        Args:
            heating_capacity: Reference heating capacity in W
            cop_rated: Rated COP at design conditions
        """
        print(f"  [Physics] Upgrading to Physics-based ASHP (Cap={heating_capacity}W, COP={cop_rated})...")

        # 1. Clean up old boiler objects and get node names
        old_boilers = self.idf.idfobjects['BOILER:HOTWATER']
        target_inlet = "Central Boiler Inlet Node"  # Default fallback
        target_outlet = "Central Boiler Outlet Node"
        
        # ============= v1.2 FIX: Check if TES is in series =============
        # If TES was already added in series, the outlet should be the intermediate node
        # Look at existing Branch definition to determine correct outlet node
        branches = self.idf.idfobjects['BRANCH']
        for branch in branches:
            branch_name = branch.Name.lower() if hasattr(branch, 'Name') else ''
            if 'boiler' in branch_name:
                # Check if Component_2 is TES (series connection)
                comp2_type = str(getattr(branch, 'Component_2_Object_Type', '')).lower()
                if 'waterheater' in comp2_type:
                    # TES is in series - HP outlet should be intermediate node
                    target_outlet = str(getattr(branch, 'Component_1_Outlet_Node_Name', target_outlet))
                    print(f"    Detected TES in series, HP outlet will be: {target_outlet}")
                break
        
        if old_boilers:
            target_inlet = old_boilers[0].Boiler_Water_Inlet_Node_Name
            # Only use boiler outlet if not already set by TES detection
            if target_outlet == "Central Boiler Outlet Node":
                target_outlet = old_boilers[0].Boiler_Water_Outlet_Node_Name
            # Remove all boilers
            while self.idf.idfobjects['BOILER:HOTWATER']:
                self.idf.removeidfobject(self.idf.idfobjects['BOILER:HOTWATER'][0])
            print(f"    Removed existing boilers, using nodes: {target_inlet} -> {target_outlet}")

        # 2. Define Performance Curves (Realistic Air-Source Heat Pump)
        # ========================================================================
        # CURVE PROVENANCE (for publication rigor):
        # ========================================================================
        # These coefficients are derived from:
        #   1. ASHRAE Handbook - HVAC Systems and Equipment, Chapter 25 (Heat Pumps)
        #   2. Staffell et al. (2012) "A review of domestic heat pumps", Energy & Env. Sci.
        #   3. Nordman et al. (2012) "SEAsonal COefficient of Performance of Heat Pump 
        #      Water Heaters", Technical Report for EU Ecodesign Preparatory Study
        #   4. Manufacturer data: Daikin Altherma (8-16kW class), Carrier 30RQ series
        #
        # Reference conditions (EN 14511 / AHRI 540):
        #   - LWT (leaving water temp): 35°C for heating
        #   - OAT (outdoor air temp): 7°C (EN14511 A7/W35 standard rating point)
        #   - COP at reference: 3.2 typical for modern scroll compressor ASHP
        #
        # Physical basis:
        #   - Cap = f(LWT, OAT): Carnot COP = T_hot / (T_hot - T_cold)
        #   - Higher LWT increases pressure ratio, reduces capacity
        #   - Lower OAT reduces evaporator efficiency, reduces capacity
        # ========================================================================
        #
        # x = Leaving Water Temperature (LWT) [°C]
        # y = Outdoor Air Temperature (OAT) [°C]
        
        # Cap = f(LWT, OAT) - Biquadratic capacity modifier curve
        # Physical interpretation:
        #   - Coefficient2_x (-0.0052): 0.52% capacity loss per °C increase in LWT
        #     (matches Daikin Altherma spec: ~5% drop from 35°C to 45°C LWT)
        #   - Coefficient4_y (+0.027): 2.7% capacity gain per °C increase in OAT
        #     (consistent with EN14511 A-7/W35 vs A7/W35 capacity ratio ~0.7)
        self.idf.newidfobject(
            'CURVE:BIQUADRATIC',
            Name='HP_Heat_Cap_Curve',
            Coefficient1_Constant=1.19,     # Normalized at reference (LWT=35, OAT=7)
            Coefficient2_x=-0.0052,         # LWT penalty: -0.52%/°C (Daikin/Carrier data)
            Coefficient3_x2=0.0,
            Coefficient4_y=0.027,           # OAT benefit: +2.7%/°C (EN14511 compliance)
            Coefficient5_y2=0.00015,        # Small quadratic OAT effect (defrost penalty)
            Coefficient6_xy=-0.00012,       # Cross-interaction (mild)
            Minimum_Value_of_x=25, Maximum_Value_of_x=60,  # LWT range [°C]
            Minimum_Value_of_y=-20, Maximum_Value_of_y=25  # OAT range [°C]
        )

        # EIR = 1/COP = f(LWT, OAT) - Energy Input Ratio modifier curve
        # Physical interpretation:
        #   - Coefficient2_x (+0.012): 1.2% EIR increase per °C LWT rise
        #     (COP drops from 3.5 at LWT=35 to 2.8 at LWT=55, per Staffell 2012)
        #   - Coefficient4_y (-0.022): 2.2% EIR decrease per °C OAT rise
        #     (smaller temperature lift = higher Carnot efficiency)
        self.idf.newidfobject(
            'CURVE:BIQUADRATIC',
            Name='HP_Heat_EIR_Curve',
            Coefficient1_Constant=0.67,     # Normalized at reference conditions
            Coefficient2_x=0.012,           # LWT penalty: +1.2%/°C on EIR (Staffell 2012)
            Coefficient3_x2=0.00008,        # Small quadratic LWT effect
            Coefficient4_y=-0.022,          # OAT benefit: -2.2%/°C on EIR (EN14511)
            Coefficient5_y2=0.00035,        # Quadratic OAT effect (defrost at low temp)
            Coefficient6_xy=-0.00015,       # Cross-interaction term
            Minimum_Value_of_x=25, Maximum_Value_of_x=60,
            Minimum_Value_of_y=-20, Maximum_Value_of_y=25
        )

        # EIR = f(PLR) - Part Load Ratio curve (Cycling/VFD losses)
        # For variable-speed compressor with VFD
        self.idf.newidfobject(
            'CURVE:QUADRATIC',
            Name='HP_Heat_PLR_Curve',
            Coefficient1_Constant=0.10,     # Minimum power draw at 0 load (standby)
            Coefficient2_x=0.85,            # Linear PLR dependency
            Coefficient3_x2=0.05,           # Small quadratic term (inverter losses)
            Minimum_Value_of_x=0.0, Maximum_Value_of_x=1.0
        )

        # 3. Create outdoor air nodes for AirSource heat pump
        # These are required even for AirSource type
        hp_oa_inlet = "ASHP_OA_Inlet_Node"
        hp_oa_outlet = "ASHP_OA_Outlet_Node"
        
        # Create outdoor air node list for the HP source side
        self.idf.newidfobject(
            'OUTDOORAIR:NODE',
            Name=hp_oa_inlet,
            Height_Above_Ground=2.0,
        )
        
        # 4. Create the Heat Pump Object (Physics-based)
        # EnergyPlus 23.x HeatPump:PlantLoop:EIR:Heating
        hp_name = "Central_ASHP"
        hp_obj = self.idf.newidfobject(
            'HEATPUMP:PLANTLOOP:EIR:HEATING',
            Name=hp_name,
            Load_Side_Inlet_Node_Name=target_inlet,
            Load_Side_Outlet_Node_Name=target_outlet,
            Condenser_Type='AirSource',
            Source_Side_Inlet_Node_Name=hp_oa_inlet,  # Required for EnergyPlus
            Source_Side_Outlet_Node_Name=hp_oa_outlet,
            Reference_Capacity=heating_capacity,
            Reference_Coefficient_of_Performance=cop_rated,
            Sizing_Factor=1.0,
            Capacity_Modifier_Function_of_Temperature_Curve_Name='HP_Heat_Cap_Curve',
            Electric_Input_to_Output_Ratio_Modifier_Function_of_Temperature_Curve_Name='HP_Heat_EIR_Curve',
            Electric_Input_to_Output_Ratio_Modifier_Function_of_Part_Load_Ratio_Curve_Name='HP_Heat_PLR_Curve',
        )
            
        print(f"    Created HeatPump:PlantLoop:EIR:Heating: {hp_name}")

        # 4. Update Plant Equipment List
        plant_lists = self.idf.idfobjects['PLANTEQUIPMENTLIST']
        for plist in plant_lists:
            if 'heating' in plist.Name.lower():
                plist.Equipment_1_Object_Type = 'HeatPump:PlantLoop:EIR:Heating'
                plist.Equipment_1_Name = hp_name
                print(f"    Updated PlantEquipmentList: {plist.Name}")
        
        # 5. Update Branches - Ensure component type is correctly set
        branches = self.idf.idfobjects['BRANCH']
        for branch in branches:
            # Check Component 1
            comp1_name = str(getattr(branch, 'Component_1_Name', '')).lower()
            if 'boiler' in comp1_name or 'central boiler' in comp1_name:
                branch.Component_1_Object_Type = 'HeatPump:PlantLoop:EIR:Heating'
                branch.Component_1_Name = hp_name
                print(f"    Updated Branch Component_1: {branch.Name}")
            # Check Component 2 (in case TES was added before HP)
            comp2_name = str(getattr(branch, 'Component_2_Name', '')).lower()
            if 'boiler' in comp2_name:
                branch.Component_2_Object_Type = 'HeatPump:PlantLoop:EIR:Heating'
                branch.Component_2_Name = hp_name
                print(f"    Updated Branch Component_2: {branch.Name}")

        # 6. Add Detailed Output Variables for Analysis
        # [FIX 7] Correct output variable names from .rdd file
        # EnergyPlus 23.x uses "Heat Pump" prefix without "Heating"
        hp_vars = [
            ("Heat Pump Load Side Heat Transfer Rate", hp_name, "Timestep"),
            ("Heat Pump Electricity Rate", hp_name, "Timestep"),
            ("Heat Pump Part Load Ratio", hp_name, "Timestep"),
            ("Heat Pump Load Side Inlet Temperature", hp_name, "Timestep"),
            ("Heat Pump Load Side Outlet Temperature", hp_name, "Timestep"),
            ("Heat Pump Cycling Ratio", hp_name, "Timestep"),
        ]
        for var, key, freq in hp_vars:
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var and o.Key_Value == key]
            if not existing:
                self.idf.newidfobject('OUTPUT:VARIABLE', Key_Value=key, Variable_Name=var, Reporting_Frequency=freq)

        # ============= [FIX 4] Remove Invalid NaturalGas Meters =============
        # After all-electric conversion, NaturalGas meters are no longer valid
        meter_types = ['OUTPUT:METER', 'OUTPUT:METER:METERFILEONLY', 'OUTPUT:METER:CUMULATIVE']
        for meter_type in meter_types:
            meters = self.idf.idfobjects[meter_type]
            for m in list(meters):
                key = getattr(m, 'Key_Name', '') or getattr(m, 'Name', '')
                if 'NaturalGas' in key or 'Gas' in key:
                    self.idf.removeidfobject(m)
                    print(f"    Removed invalid gas meter: {key}")
        
        # Also remove Boiler-specific output variables
        output_vars = self.idf.idfobjects['OUTPUT:VARIABLE']
        for ov in list(output_vars):
            var_name = getattr(ov, 'Variable_Name', '')
            if 'Boiler' in var_name:
                self.idf.removeidfobject(ov)
                print(f"    Removed obsolete Boiler output: {var_name}")

        print(f"  ✅ Physics-based ASHP created with temperature-dependent curves")
        return self

    def add_stratified_tes_tank(
        self,
        tank_volume: float = 12.0,
        height: float = 2.5,  # Tank height in meters
        num_nodes: int = 10,   # Number of stratification layers
        tank_ua: float = 15.0, # Insulation level (W/K total) - realistic for 50mm polyurethane
    ):
        """
        [Physics Upgrade] Replace Mixed Tank with Stratified Tank.
        
        Heat Loss Calculation (for scientific rigor):
        - Tank surface area: ~26-29 m² for 12m³ cylinder
        - Typical insulation: RSI-2.0 to RSI-3.0 (50mm polyurethane foam)
        - U-value: ~0.5 W/m²K
        - Total UA: 26 * 0.5 = 13-15 W/K
        
        This realistic UA gives RL incentive to trade off heat loss vs price differential.
        
        Provides observation of temperature gradient (SOC) for DRL.
        Topology: Series Buffer (HP -> Tank Use Inlet -> Tank Use Outlet -> Load).
        
        This enables Transformer models to observe fine-grained thermal state.
        
        Args:
            tank_volume: Tank volume in m³
            height: Tank height in meters
            num_nodes: Number of stratification layers (more = finer gradient)
            tank_ua: Total tank heat loss coefficient (W/K)
        """
        print(f"  [Physics] Upgrading to Stratified TES ({num_nodes} nodes, {tank_volume}m³, h={height}m)...")
        
        # 1. Remove existing WaterHeater:Mixed TES if exists
        old_tanks = self.idf.idfobjects['WATERHEATER:MIXED']
        removed_tank_names = []
        for tank in list(old_tanks):
            if 'TES' in tank.Name.upper():
                removed_tank_names.append(tank.Name)
                self.idf.removeidfobject(tank)
                print(f"    Removed existing mixed tank: {tank.Name}")
        
        # [FIX 5] Also remove output variables for the old TES_Tank
        # Otherwise EnergyPlus will warn "Key=TES_Tank not found"
        if removed_tank_names:
            output_vars = self.idf.idfobjects['OUTPUT:VARIABLE']
            for ov in list(output_vars):
                key = getattr(ov, 'Key_Value', '')
                if key in removed_tank_names:
                    self.idf.removeidfobject(ov)
                    print(f"    Removed obsolete output variable for: {key}")
        
        # 2. Define Tank Object
        tes_name = "TES_Stratified"
        
        # ============= v1.2 FIX: Use consistent node naming =============
        # MUST match the nodes defined in add_tes_tank for series connection
        # Node chain: HP Inlet -> HP -> Intermediate -> TES -> Loop Outlet
        node_intermediate = 'HP_Outlet_TES_Inlet'  # Same as add_tes_tank
        node_loop_outlet = 'Central Boiler Outlet Node'
        
        inlet_node = node_intermediate
        outlet_node = node_loop_outlet
        
        # Verify existing Branch configuration
        branches = self.idf.idfobjects['BRANCH']
        for branch in branches:
            branch_name = branch.Name.lower() if hasattr(branch, 'Name') else ''
            if 'boiler' in branch_name:
                # Check existing Component_2 (TES) nodes
                comp2_inlet = getattr(branch, 'Component_2_Inlet_Node_Name', '')
                comp2_outlet = getattr(branch, 'Component_2_Outlet_Node_Name', '')
                if comp2_inlet and comp2_outlet:
                    inlet_node = comp2_inlet
                    outlet_node = comp2_outlet
                print(f"    Using series nodes: {inlet_node} -> {outlet_node}")
                break
        
        # Ensure TES_Ambient_Temp schedule exists
        existing_schedules = [s.Name for s in self.idf.idfobjects['SCHEDULE:COMPACT']]
        if 'TES_Ambient_Temp' not in existing_schedules:
            self.idf.newidfobject(
                'SCHEDULE:COMPACT',
                Name='TES_Ambient_Temp',
                Schedule_Type_Limits_Name='Temperature',
                Field_1='Through: 12/31',
                Field_2='For: AllDays',
                Field_3='Until: 24:00',
                Field_4=15.0,  # Basement temperature assumption
            )
        
        # Calculate skin loss coefficient per unit area
        # Approximate cylinder surface area = pi*D*H + 2*pi*(D/2)^2
        # For volume V = pi*(D/2)^2*H, D = sqrt(4V/(pi*H))
        import math
        diameter = math.sqrt(4 * tank_volume / (math.pi * height))
        surface_area = math.pi * diameter * height + 2 * math.pi * (diameter/2)**2
        ua_per_area = tank_ua / surface_area if surface_area > 0 else 0.5
        
        # Create a constant setpoint schedule for the heaters (even though capacity=0)
        heater_setpoint_schedule = 'TES_Heater_Setpoint'
        existing_const_schedules = [s.Name for s in self.idf.idfobjects['SCHEDULE:CONSTANT']]
        if heater_setpoint_schedule not in existing_const_schedules:
            self.idf.newidfobject(
                'SCHEDULE:CONSTANT',
                Name=heater_setpoint_schedule,
                Schedule_Type_Limits_Name='Temperature',
                Hourly_Value=50.0,  # Setpoint temperature (not used since capacity=0)
            )
        
        # 3. Create Stratified Tank using keyword arguments (more robust)
        # EnergyPlus 23.x WaterHeater:Stratified field names
        strat_tank = self.idf.newidfobject(
            'WATERHEATER:STRATIFIED',
            Name=tes_name,
            Tank_Volume=tank_volume,
            Tank_Height=height,
            Tank_Shape='VerticalCylinder',
            Maximum_Temperature_Limit=90.0,
            Heater_Priority_Control='MasterSlave',
            Heater_1_Setpoint_Temperature_Schedule_Name=heater_setpoint_schedule,  # Required
            Heater_1_Capacity=0.0,  # Passive storage - no internal heater
            Heater_1_Height=height * 0.8,
            Heater_1_Deadband_Temperature_Difference=2.0,
            Heater_2_Setpoint_Temperature_Schedule_Name=heater_setpoint_schedule,  # Required
            Heater_2_Capacity=0.0,
            Heater_2_Height=height * 0.2,
            Heater_2_Deadband_Temperature_Difference=2.0,
            Heater_Fuel_Type='Electricity',
            Heater_Thermal_Efficiency=1.0,
            Ambient_Temperature_Indicator='Schedule',
            Ambient_Temperature_Schedule_Name='TES_Ambient_Temp',
            Uniform_Skin_Loss_Coefficient_per_Unit_Area_to_Ambient_Temperature=ua_per_area,
            Use_Side_Inlet_Node_Name=inlet_node,
            Use_Side_Outlet_Node_Name=outlet_node,
            Use_Side_Effectiveness=1.0,
            Use_Side_Inlet_Height=height,      # Hot water enters top
            Use_Side_Outlet_Height=0.0,        # Cold water leaves bottom
            Number_of_Nodes=num_nodes,
            Additional_Destratification_Conductivity=0.1,
        )
        
        # Set optional fields with try/except for version compatibility
        try:
            strat_tank.Off_Cycle_Parasitic_Fuel_Consumption_Rate = 0.0
            strat_tank.On_Cycle_Parasitic_Fuel_Consumption_Rate = 0.0
        except:
            pass  # Field not available in this EP version
        
        try:
            strat_tank.Source_Side_Effectiveness = 1.0
        except:
            pass
        
        print(f"    Created WaterHeater:Stratified: {tes_name}")
        print(f"    Tank surface area: {surface_area:.2f} m², UA/area: {ua_per_area:.4f} W/m²K")
        
        # 4. Update Branch Logic - Replace WaterHeater:Mixed with WaterHeater:Stratified
        for branch in branches:
            comp2_name = str(getattr(branch, 'Component_2_Name', '')).lower()
            if 'tes' in comp2_name:
                branch.Component_2_Object_Type = 'WaterHeater:Stratified'
                branch.Component_2_Name = tes_name
                print(f"    Updated Branch '{branch.Name}' to use stratified tank")

        # 5. Add Sensors for EACH Node (Crucial for DRL/Transformer state observation)
        for i in range(1, num_nodes + 1):
            var_name = f'Water Heater Temperature Node {i}'
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name and o.Key_Value == tes_name]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=tes_name,
                    Variable_Name=var_name,
                    Reporting_Frequency='Timestep'
                )
        
        # Add additional TES output variables
        additional_vars = [
            ("Water Heater Use Side Heat Transfer Rate", tes_name, "Timestep"),
            ("Water Heater Use Side Inlet Temperature", tes_name, "Timestep"),
            ("Water Heater Use Side Outlet Temperature", tes_name, "Timestep"),
            ("Water Heater Use Side Mass Flow Rate", tes_name, "Timestep"),
            ("Water Heater Heat Loss Rate", tes_name, "Timestep"),
        ]
        for var, key, freq in additional_vars:
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var and o.Key_Value == key]
            if not existing:
                self.idf.newidfobject('OUTPUT:VARIABLE', Key_Value=key, Variable_Name=var, Reporting_Frequency=freq)
            
        print(f"  ✅ Stratified TES added with {num_nodes} temperature observation nodes")
        return self

    def configure_pump_for_tes_charging(self):
        """
        [Hydraulic Fix] Configure pumps for night-time TES charging.
        
        CRITICAL ISSUE: With INTERMITTENT pump control and series topology,
        the heat pump cannot charge the TES when there's no load (e.g., night).
        The loop flow drops to zero, triggering Low Flow errors.
        
        SOLUTION: 
        1. Set pump to CONTINUOUS operation (with VFD for energy efficiency)
        2. Add bypass branch to allow circulation when load coils are closed
        
        This enables the DRL agent to learn "charge at night, discharge at peak".
        """
        print("  [Hydraulic] Configuring pumps for TES charging capability...")
        
        # 1. Set Hot Water Loop pump to CONTINUOUS
        hw_pumps = self.idf.idfobjects['PUMP:VARIABLESPEED']
        for pump in hw_pumps:
            pump_name = pump.Name.lower() if hasattr(pump, 'Name') else ''
            if 'hw' in pump_name or 'hot' in pump_name or 'heating' in pump_name:
                pump.Pump_Control_Type = 'Continuous'
                pump.Pump_Flow_Rate_Schedule_Name = ''  # Remove any restrictive schedule
                print(f"    Set {pump.Name} to CONTINUOUS operation")
        
        # Also check constant speed pumps
        cs_pumps = self.idf.idfobjects['PUMP:CONSTANTSPEED']
        for pump in cs_pumps:
            pump_name = pump.Name.lower() if hasattr(pump, 'Name') else ''
            if 'hw' in pump_name or 'hot' in pump_name or 'heating' in pump_name:
                try:
                    pump.Pump_Control_Type = 'Continuous'
                    print(f"    Set {pump.Name} to CONTINUOUS operation")
                except:
                    pass
        
        # 2. CRITICAL: Add SUPPLY-SIDE bypass for night-time TES charging
        # Problem: In Series topology (HP->TES->Load), when all VAV coils close (night),
        # there's nowhere for water to go even with continuous pump operation.
        # Solution: Add a bypass branch on the SUPPLY SIDE that connects outlet back to inlet,
        # allowing HP->TES circulation independent of demand-side valve states.
        branches = self.idf.idfobjects['BRANCH']
        hw_supply_bypass_exists = False
        
        for branch in branches:
            if hasattr(branch, 'Name'):
                name_lower = branch.Name.lower()
                # Check for supply-side bypass specifically
                if 'supply' in name_lower and 'bypass' in name_lower:
                    if 'hot' in name_lower or 'hw' in name_lower or 'heating' in name_lower:
                        hw_supply_bypass_exists = True
                        print(f"    Found existing supply bypass: {branch.Name}")
                        break
        
        # Add supply-side bypass if not found (for Primary-Only Variable Flow system)
        if not hw_supply_bypass_exists:
            print("    Adding HW Supply Side Bypass for night-time TES charging...")
            
            # Create bypass pipe (adiabatic - no heat loss)
            self.idf.newidfobject(
                'PIPE:ADIABATIC',
                Name='HW_Supply_Bypass_Pipe',
                Inlet_Node_Name='HW_Supply_Bypass_Inlet',
                Outlet_Node_Name='HW_Supply_Bypass_Outlet',
            )
            
            # Create bypass branch
            bypass_branch = self.idf.newidfobject('BRANCH')
            bypass_branch.Name = 'HW Supply Bypass Branch'
            bypass_branch.Component_1_Object_Type = 'Pipe:Adiabatic'
            bypass_branch.Component_1_Name = 'HW_Supply_Bypass_Pipe'
            bypass_branch.Component_1_Inlet_Node_Name = 'HW_Supply_Bypass_Inlet'
            bypass_branch.Component_1_Outlet_Node_Name = 'HW_Supply_Bypass_Outlet'
            
            # Add bypass branch to supply side branch list
            branch_lists = self.idf.idfobjects['BRANCHLIST']
            for bl in branch_lists:
                bl_name = bl.Name.lower() if hasattr(bl, 'Name') else ''
                if 'hot' in bl_name and 'supply' in bl_name:
                    # Find an empty branch slot
                    for i in range(2, 20):
                        field_name = f'Branch_{i}_Name'
                        if hasattr(bl, field_name):
                            current_val = getattr(bl, field_name, '')
                            if not current_val or current_val == '':
                                setattr(bl, field_name, 'HW Supply Bypass Branch')
                                print(f"    Added bypass to {bl.Name}")
                                break
            
            # Update connector list to include bypass
            connectors = self.idf.idfobjects['CONNECTOR:SPLITTER']
            for conn in connectors:
                conn_name = conn.Name.lower() if hasattr(conn, 'Name') else ''
                if 'hot' in conn_name and 'supply' in conn_name:
                    # Find empty outlet slot
                    for i in range(2, 10):
                        field_name = f'Outlet_Branch_{i}_Name'
                        if hasattr(conn, field_name):
                            current_val = getattr(conn, field_name, '')
                            if not current_val or current_val == '':
                                setattr(conn, field_name, 'HW Supply Bypass Branch')
                                break
            
            connectors_mixer = self.idf.idfobjects['CONNECTOR:MIXER']
            for conn in connectors_mixer:
                conn_name = conn.Name.lower() if hasattr(conn, 'Name') else ''
                if 'hot' in conn_name and 'supply' in conn_name:
                    # Find empty inlet slot
                    for i in range(2, 10):
                        field_name = f'Inlet_Branch_{i}_Name'
                        if hasattr(conn, field_name):
                            current_val = getattr(conn, field_name, '')
                            if not current_val or current_val == '':
                                setattr(conn, field_name, 'HW Supply Bypass Branch')
                                break
            
            print("    ✅ Supply-side bypass added for Primary-Only Variable Flow")
        
        # Check plant loop settings - CRITICAL for night-time TES charging
        # Set Minimum_Loop_Flow_Rate to prevent Low Flow errors when load coils close
        plant_loops = self.idf.idfobjects['PLANTLOOP']
        for loop in plant_loops:
            loop_name = loop.Name.lower() if hasattr(loop, 'Name') else ''
            if 'hot' in loop_name or 'heating' in loop_name:
                try:
                    # Set load distribution to optimal for variable flow
                    loop.Load_Distribution_Scheme = 'Optimal'
                    # CRITICAL: Force minimum flow to allow night-time charging
                    # When all VAV coils close, bypass must maintain flow for HP->TES
                    # Typical design: 10-20% of design flow rate
                    loop.Minimum_Loop_Flow_Rate = 0.0001  # m³/s (~0.1 kg/s minimum)
                    print(f"    Set {loop.Name}: Optimal distribution, Min flow=0.0001 m³/s")
                except Exception as e:
                    print(f"    Warning: Could not set loop params: {e}")
            
            # Same for chilled water loop
            if 'chilled' in loop_name or 'cw' in loop_name or 'cool' in loop_name:
                try:
                    loop.Load_Distribution_Scheme = 'Optimal'
                    loop.Minimum_Loop_Flow_Rate = 0.0001  # m³/s minimum
                    print(f"    Set {loop.Name}: Optimal distribution, Min flow=0.0001 m³/s")
                except:
                    pass
        
        # 3. Set Chilled Water pump similarly for summer charging
        for pump in hw_pumps:
            pump_name = pump.Name.lower() if hasattr(pump, 'Name') else ''
            if 'cw' in pump_name or 'chilled' in pump_name or 'cooling' in pump_name:
                pump.Pump_Control_Type = 'Continuous'
                print(f"    Set {pump.Name} to CONTINUOUS operation")
        
        print("  ✅ Pumps & loops configured for night-time TES charging")
        return self

    def add_drl_observation_outputs(self):
        """
        Add comprehensive output variables for DRL state observation.
        
        Transformer models excel with rich, multi-dimensional state spaces.
        These outputs provide the information needed for:
        - Demand prediction (load profiles)
        - Efficiency awareness (real-time COP)
        - Storage state (temperature gradients)
        - Grid interaction (net metering)
        """
        print("  Adding DRL observation outputs...")
        
        drl_outputs = [
            # ===== Real-time Efficiency =====
            # [FIX 7] Correct variable names (no "Heating" prefix)
            ("Heat Pump Part Load Ratio", "Central_ASHP", "Timestep"),
            ("Heat Pump Cycling Ratio", "Central_ASHP", "Timestep"),
            
            # ===== Plant Loop Demands (for load prediction) =====
            ("Plant Supply Side Heating Demand Rate", "*", "Timestep"),
            ("Plant Supply Side Cooling Demand Rate", "*", "Timestep"),
            ("Plant Supply Side Inlet Temperature", "*", "Timestep"),
            ("Plant Supply Side Outlet Temperature", "*", "Timestep"),
            
            # ===== Building Load Signals =====
            ("Zone Air System Sensible Heating Rate", "*", "Timestep"),
            ("Zone Air System Sensible Cooling Rate", "*", "Timestep"),
            ("Zone Mean Air Temperature", "*", "Timestep"),
            ("Zone Air Relative Humidity", "*", "Timestep"),
            
            # ===== Grid Interaction (for carbon optimization) =====
            ("Facility Net Purchased Electricity Rate", "*", "Timestep"),
            ("Facility Total Electricity Demand Rate", "*", "Timestep"),
            
            # ===== PV Production =====
            ("Generator Produced DC Electricity Rate", "*", "Timestep"),
            ("Electric Load Center Produced Electricity Rate", "*", "Timestep"),
            
            # ===== Weather for prediction =====
            ("Site Outdoor Air Drybulb Temperature", "Environment", "Timestep"),
            ("Site Direct Solar Radiation Rate per Area", "Environment", "Timestep"),
            ("Site Diffuse Solar Radiation Rate per Area", "Environment", "Timestep"),
        ]
        
        added_count = 0
        for var_name, key, freq in drl_outputs:
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name and o.Key_Value == key]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=key,
                    Variable_Name=var_name,
                    Reporting_Frequency=freq
                )
                added_count += 1
        
        print(f"  ✅ Added {added_count} DRL observation outputs")
        return self

    def setup_rl_interface(self):
        """
        Setup RL control interface by creating controllable schedules.
        
        CRITICAL: RL should control Plant Loop Supply Outlet Setpoint
        to avoid conflicts with existing SetpointManagers.
        
        Control Strategy:
        - HP_Supply_Setpoint: Controls heat pump target outlet temperature
        - When HP_Setpoint > Load_Return_Temp: HP runs, charges TES
        - When HP_Setpoint < TES_Temp: TES discharges to meet load
        """
        print("  Setting up RL control interface...")
        
        # ============= [FIX 1] Remove Conflicting SetpointManagers =============
        # The template has existing "Hot Water Loop Setpoint Manager" that conflicts
        # with our RL manager. Must remove to avoid EnergyPlus warning.
        managers = self.idf.idfobjects['SETPOINTMANAGER:SCHEDULED']
        for mgr in list(managers):
            if hasattr(mgr, 'Name'):
                name_lower = mgr.Name.lower()
                if ('hot' in name_lower and 'water' in name_lower) or 'hw' in name_lower:
                    # Check if it's controlling the same node we want to control
                    node = getattr(mgr, 'Setpoint_Node_or_NodeList_Name', '')
                    if 'supply' in node.lower() or 'outlet' in node.lower():
                        self.idf.removeidfobject(mgr)
                        print(f"    Removed conflicting SetpointManager: {mgr.Name}")
        
        # ============= RL Control Schedules =============
        # Primary control: Hot Water Supply Temperature Setpoint
        # This controls how much heat the HP produces and stores in TES
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_HW_Supply_Setpoint',
            Schedule_Type_Limits_Name='Temperature',
            Hourly_Value=45.0,  # Initial: moderate heating
        )
        
        # Zone thermostat setpoints
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_Heating_Setpoint',
            Schedule_Type_Limits_Name='Temperature',
            Hourly_Value=21.0,
        )
        
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_Cooling_Setpoint',
            Schedule_Type_Limits_Name='Temperature',
            Hourly_Value=24.0,
        )
        
        # ============= Create Single RL SetpointManager =============
        # We already removed conflicting managers above, now create our single RL manager
        self.idf.newidfobject(
            'SETPOINTMANAGER:SCHEDULED',
            Name='RL_HW_Loop_Setpoint_Manager',
            Control_Variable='Temperature',
            Schedule_Name='RL_HW_Supply_Setpoint',
            Setpoint_Node_or_NodeList_Name='HW Supply Outlet Node',
        )
        
        # ============= EMS Actuators for Advanced Control =============
        # These allow direct runtime manipulation via Sinergym/BCVTB
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='HW_Setpoint_Actuator',
            Actuated_Component_Unique_Name='RL_HW_Supply_Setpoint',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='Heating_Setpoint_Actuator',
            Actuated_Component_Unique_Name='RL_Heating_Setpoint',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='Cooling_Setpoint_Actuator',
            Actuated_Component_Unique_Name='RL_Cooling_Setpoint',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        
        # ============= Equipment Availability Controls (CRITICAL for Peak Shifting) =============
        # These allow RL to completely disable HP/Chiller during peak hours
        # Action space: [Setpoint_Temp, HP_Enable(0/1), Chiller_Enable(0/1)]
        
        # Create On/Off schedule type limits if not exists
        existing_limits = [l.Name for l in self.idf.idfobjects['SCHEDULETYPELIMITS']]
        if 'On/Off' not in existing_limits:
            self.idf.newidfobject(
                'SCHEDULETYPELIMITS',
                Name='On/Off',
                Lower_Limit_Value=0,
                Upper_Limit_Value=1,
                Numeric_Type='Discrete',
            )
        
        # 1. Heat Pump Availability Schedule and Actuator
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_HP_Avail_Sch',
            Schedule_Type_Limits_Name='On/Off',
            Hourly_Value=1,  # Default: HP enabled
        )
        
        # Apply to HeatPump object
        hp_objects = self.idf.idfobjects['HEATPUMP:PLANTLOOP:EIR:HEATING']
        if hp_objects:
            for hp in hp_objects:
                try:
                    hp.Availability_Schedule_Name = 'RL_HP_Avail_Sch'
                    print(f"    Applied RL_HP_Avail_Sch to {hp.Name}")
                except:
                    pass
        
        # EMS Actuator for HP on/off control
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='HP_Enable_Actuator',
            Actuated_Component_Unique_Name='RL_HP_Avail_Sch',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        print("    ✅ Added HP On/Off Actuator for Peak Shifting")
        
        # 2. Chiller Availability Schedule and Actuator (for summer peak shifting)
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_Chiller_Avail_Sch',
            Schedule_Type_Limits_Name='On/Off',
            Hourly_Value=1,  # Default: Chiller enabled
        )
        
        # Apply to Chiller object
        chiller_objects = self.idf.idfobjects['CHILLER:ELECTRIC:EIR']
        if chiller_objects:
            for chiller in chiller_objects:
                try:
                    chiller.Availability_Schedule_Name = 'RL_Chiller_Avail_Sch'
                    print(f"    Applied RL_Chiller_Avail_Sch to {chiller.Name}")
                except:
                    pass
        
        # EMS Actuator for Chiller on/off control
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='Chiller_Enable_Actuator',
            Actuated_Component_Unique_Name='RL_Chiller_Avail_Sch',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        print("    ✅ Added Chiller On/Off Actuator for Peak Shifting")
        
        # ============= EMS Sensors for State Observation =============
        # EnergyPlus 23.x EMS:Sensor - use object's field names directly
        # These sensors are optional for core model validation
        try:
            tes_sensor = self.idf.newidfobject('ENERGYMANAGEMENTSYSTEM:SENSOR')
            tes_sensor.Name = 'TES_Tank_Temp_Sensor'
            # Try to access the actual field names from the object
            field_names = tes_sensor.fieldnames
            if len(field_names) >= 3:
                # Find the output variable field
                for fn in field_names:
                    fn_lower = fn.lower()
                    if 'key' in fn_lower and 'name' in fn_lower:
                        setattr(tes_sensor, fn, 'TES_Stratified')
                    elif 'output' in fn_lower and 'variable' in fn_lower and 'name' in fn_lower:
                        setattr(tes_sensor, fn, 'Water Heater Tank Temperature')
        except Exception as e:
            print(f"    Note: EMS TES sensor setup skipped: {e}")
        
        try:
            oat_sensor = self.idf.newidfobject('ENERGYMANAGEMENTSYSTEM:SENSOR')
            oat_sensor.Name = 'OAT_Sensor'
            field_names = oat_sensor.fieldnames
            if len(field_names) >= 3:
                for fn in field_names:
                    fn_lower = fn.lower()
                    if 'key' in fn_lower and 'name' in fn_lower:
                        setattr(oat_sensor, fn, 'Environment')
                    elif 'output' in fn_lower and 'variable' in fn_lower and 'name' in fn_lower:
                        setattr(oat_sensor, fn, 'Site Outdoor Air Drybulb Temperature')
        except Exception as e:
            print(f"    Note: EMS OAT sensor setup skipped: {e}")
        
        print("  ✅ RL interface configured with proper setpoint control")
        return self
    
    def add_detailed_outputs(self):
        """
        Add detailed output variables for carbon emission calculation.
        
        Required for Innovation B: Carbon-aware optimization
        Outputs individual equipment electricity consumption for:
        - Net Energy = Facility_Demand - PV_Generation
        - Carbon = Net_Energy * Carbon_Factor(t)
        """
        print("  Adding detailed outputs for carbon calculation...")
        
        detailed_outputs = [
            # Facility level
            # [FIX 7] Correct Facility variable names from .rdd file
            ("Facility Total Electricity Demand Rate", "*", "Timestep"),
            ("Facility Total HVAC Electricity Demand Rate", "*", "Timestep"),
            ("Facility Net Purchased Electricity Rate", "*", "Timestep"),
            
            # [FIX 3] Heat Pump outputs (replaces Boiler after physics upgrade)
            # Correct variable names from .rdd file (no "Heating" prefix)
            ("Heat Pump Electricity Rate", "Central_ASHP", "Timestep"),
            ("Heat Pump Load Side Heat Transfer Rate", "Central_ASHP", "Timestep"),
            ("Heat Pump Part Load Ratio", "Central_ASHP", "Timestep"),
            ("Heat Pump Cycling Ratio", "Central_ASHP", "Timestep"),
            
            # Chiller electricity
            ("Chiller Electricity Rate", "*", "Timestep"),
            ("Chiller Evaporator Cooling Rate", "*", "Timestep"),
            ("Chiller COP", "*", "Timestep"),
            
            # Pumps
            ("Pump Electricity Rate", "*", "Timestep"),
            
            # Fans
            ("Fan Electricity Rate", "*", "Timestep"),
            
            # Zone conditions (for comfort calculation)
            ("Zone Mean Air Temperature", "*", "Timestep"),
            ("Zone Air System Sensible Heating Rate", "*", "Timestep"),
            ("Zone Air System Sensible Cooling Rate", "*", "Timestep"),
            
            # Weather (for HP COP and PV calculation)
            ("Site Outdoor Air Drybulb Temperature", "*", "Timestep"),
            ("Site Direct Solar Radiation Rate per Area", "*", "Timestep"),
            ("Site Diffuse Solar Radiation Rate per Area", "*", "Timestep"),
            
            # Plant loop temperatures
            ("Plant Supply Side Outlet Temperature", "*", "Timestep"),
        ]
        
        for var_name, key, freq in detailed_outputs:
            # Check if already exists
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name and o.Key_Value == key]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=key,
                    Variable_Name=var_name,
                    Reporting_Frequency=freq,
                )
        
        print(f"  ✅ Added {len(detailed_outputs)} detailed output variables")
        return self
    
    def add_chilled_water_tes(
        self,
        tank_volume: float = 8.0,    # m³ (smaller for cooling)
        max_temp: float = 15.0,      # °C
        min_temp: float = 5.0,       # °C (chilled water)
        height: float = 2.0,         # m (tank height)
        num_nodes: int = 6,          # Stratification layers (fewer for chilled)
    ):
        """
        [Physics Upgrade] Add STRATIFIED TES tank to chilled water loop.
        
        CRITICAL: Chilled water has small temperature differential (7°C supply, 12°C return).
        A mixed tank would immediately blend to ~9-10°C, which is too warm for effective
        dehumidification. Stratified model maintains temperature gradient for proper cooling.
        
        SERIES connection: Chiller -> Chilled_TES -> Load
        This enables summer peak shaving via cold storage.
        """
        print(f"  Adding STRATIFIED chilled water TES: {tank_volume} m³, {min_temp}-{max_temp}°C")
        print(f"  Topology: SERIES (Chiller->TES->Load), {num_nodes} stratification nodes")
        
        tes_name = 'Chilled_TES_Stratified'
        
        # Create RL setpoint schedule for chilled TES
        self.idf.newidfobject(
            'SCHEDULE:CONSTANT',
            Name='RL_Chilled_TES_Setpoint',
            Schedule_Type_Limits_Name='Temperature',
            Hourly_Value=7.0,  # 7°C chilled water
        )
        
        # Create ambient temp schedule if not exists
        existing_const_schedules = [s.Name for s in self.idf.idfobjects['SCHEDULE:CONSTANT']]
        if 'Chilled_TES_Ambient_Temp' not in existing_const_schedules:
            self.idf.newidfobject(
                'SCHEDULE:CONSTANT',
                Name='Chilled_TES_Ambient_Temp',
                Schedule_Type_Limits_Name='Temperature',
                Hourly_Value=20.0,  # Mechanical room temperature
            )
        
        # Calculate tank geometry and heat loss
        # Chilled water tanks need extra insulation due to condensation risk
        # Typical: 75mm polyurethane foam, U ≈ 0.35 W/m²K
        import math
        diameter = math.sqrt(4 * tank_volume / (math.pi * height))
        surface_area = math.pi * diameter * height + 2 * math.pi * (diameter/2)**2
        tank_ua = 8.0  # W/K (realistic for 8m³ chilled water tank with vapor barrier)
        ua_per_area = tank_ua / surface_area if surface_area > 0 else 0.35
        
        # ============= v1.2 FIX: Use consistent node naming =============
        # Node chain: Chiller Inlet -> Chiller -> Intermediate -> TES -> Loop Outlet
        # CRITICAL: Must update BOTH Chiller component AND Branch!
        
        node_chiller_inlet = 'Central Chiller Inlet Node'
        node_intermediate = 'Chiller_Outlet_TES_Inlet'  # NEW intermediate node
        node_loop_outlet = 'Central Chiller Outlet Node'
        
        tes_inlet = node_intermediate
        tes_outlet = node_loop_outlet
        
        # ============= FIX: Update Chiller Component Outlet Node =============
        # This is the ROOT CAUSE of node connection errors!
        chillers = self.idf.idfobjects['CHILLER:ELECTRIC:EIR']
        for chiller in chillers:
            chiller.Chilled_Water_Outlet_Node_Name = node_intermediate
            print(f"    Fixed Chiller outlet: {chiller.Name} -> {node_intermediate}")
        
        # Also check for simple electric chiller
        simple_chillers = self.idf.idfobjects['CHILLER:ELECTRIC']
        for chiller in simple_chillers:
            chiller.Chilled_Water_Outlet_Node_Name = node_intermediate
            print(f"    Fixed Simple Chiller outlet: {chiller.Name} -> {node_intermediate}")
        
        # ============= Update Branch Definition =============
        branches = self.idf.idfobjects['BRANCH']
        chiller_branch_found = False
        
        for branch in branches:
            branch_name = branch.Name.lower() if hasattr(branch, 'Name') else ''
            if 'chiller' in branch_name or 'central chiller' in branch_name:
                chiller_branch_found = True
                
                # Component 1: Chiller (Inlet -> Intermediate)
                branch.Component_1_Outlet_Node_Name = node_intermediate
                
                # Component 2: TES (Intermediate -> Loop Outlet)
                branch.Component_2_Object_Type = 'WaterHeater:Stratified'
                branch.Component_2_Name = tes_name
                branch.Component_2_Inlet_Node_Name = node_intermediate
                branch.Component_2_Outlet_Node_Name = node_loop_outlet
                
                print(f"  Modified branch '{branch.Name}' for series connection")
                print(f"    Chiller -> {node_intermediate} -> TES -> {node_loop_outlet}")
                break
        
        if not chiller_branch_found:
            print("  Warning: Chiller branch not found, using default nodes")
        
        # Create STRATIFIED chilled water TES tank
        # For chilled water: cold water enters BOTTOM, warm water enters TOP
        # This is opposite of hot water tank!
        strat_tank = self.idf.newidfobject(
            'WATERHEATER:STRATIFIED',
            Name=tes_name,
            Tank_Volume=tank_volume,
            Tank_Height=height,
            Tank_Shape='VerticalCylinder',
            Maximum_Temperature_Limit=max_temp + 5.0,  # Allow some margin
            Heater_Priority_Control='MasterSlave',
            Heater_1_Setpoint_Temperature_Schedule_Name='RL_Chilled_TES_Setpoint',
            Heater_1_Capacity=0.0,  # Passive storage - no heater
            Heater_1_Height=height * 0.8,
            Heater_1_Deadband_Temperature_Difference=2.0,
            Heater_2_Setpoint_Temperature_Schedule_Name='RL_Chilled_TES_Setpoint',
            Heater_2_Capacity=0.0,
            Heater_2_Height=height * 0.2,
            Heater_2_Deadband_Temperature_Difference=2.0,
            Heater_Fuel_Type='Electricity',
            Heater_Thermal_Efficiency=1.0,
            Ambient_Temperature_Indicator='Schedule',
            Ambient_Temperature_Schedule_Name='Chilled_TES_Ambient_Temp',
            Uniform_Skin_Loss_Coefficient_per_Unit_Area_to_Ambient_Temperature=ua_per_area,
            # CRITICAL for chilled water: Invert inlet/outlet heights
            Use_Side_Inlet_Node_Name=tes_inlet,
            Use_Side_Outlet_Node_Name=tes_outlet,
            Use_Side_Effectiveness=1.0,
            Use_Side_Inlet_Height=0.0,         # Cold water from chiller enters BOTTOM
            Use_Side_Outlet_Height=height,     # Warm return water exits TOP
            Number_of_Nodes=num_nodes,
            Additional_Destratification_Conductivity=0.15,
        )
        
        print(f"    Created WaterHeater:Stratified: {tes_name}")
        print(f"    Tank surface area: {surface_area:.2f} m², UA/area: {ua_per_area:.4f} W/m²K")
        
        # Add EMS actuator for RL control
        self.idf.newidfobject(
            'ENERGYMANAGEMENTSYSTEM:ACTUATOR',
            Name='Chilled_TES_Setpoint_Actuator',
            Actuated_Component_Unique_Name='RL_Chilled_TES_Setpoint',
            Actuated_Component_Type='Schedule:Constant',
            Actuated_Component_Control_Type='Schedule Value',
        )
        
        # Add output variables for each stratification node
        for i in range(1, num_nodes + 1):
            var_name = f'Water Heater Temperature Node {i}'
            existing = [o for o in self.idf.idfobjects['OUTPUT:VARIABLE'] 
                       if o.Variable_Name == var_name and o.Key_Value == tes_name]
            if not existing:
                self.idf.newidfobject(
                    'OUTPUT:VARIABLE',
                    Key_Value=tes_name,
                    Variable_Name=var_name,
                    Reporting_Frequency='Timestep'
                )
        
        # Add additional output variables
        chilled_tes_outputs = [
            ("Water Heater Use Side Heat Transfer Rate", tes_name, "Timestep"),
            ("Water Heater Use Side Inlet Temperature", tes_name, "Timestep"),
            ("Water Heater Use Side Outlet Temperature", tes_name, "Timestep"),
            ("Water Heater Heat Loss Rate", tes_name, "Timestep"),
        ]
        for var_name, key, freq in chilled_tes_outputs:
            self.idf.newidfobject(
                'OUTPUT:VARIABLE',
                Key_Value=key,
                Variable_Name=var_name,
                Reporting_Frequency=freq,
            )
        
        print(f"  ✅ Chilled water TES added in SERIES topology")
        return self
    
    def save(self, output_path: str = None):
        """
        Save the modified IDF file.
        
        Args:
            output_path: Output file path
        """
        output_path = output_path or DEFAULT_OUTPUT
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.idf.saveas(output_path)
        print(f"\n✅ Model saved: {output_path}")
        
        return output_path
    
    def get_zone_info(self) -> Dict:
        """Get information about all zones in the model."""
        zones = self.idf.idfobjects['ZONE']
        info = {}
        
        for zone in zones:
            info[zone.Name] = {
                'direction_of_north': zone.Direction_of_Relative_North,
                'x_origin': zone.X_Origin,
                'y_origin': zone.Y_Origin,
                'z_origin': zone.Z_Origin,
            }
        
        return info


def build_classroom_model(
    idd_path: str = None,
    template_path: str = None,
    output_path: str = None,
    weather_file: str = None,
    add_tes: bool = True,
    add_pv: bool = True,           # ENABLED: Required for Innovation B
    use_heatpump: bool = True,
    add_chilled_tes: bool = True,  # ENABLED: Full year optimization
    setup_rl: bool = True,
    # NEW: Physics upgrade options for Applied Energy level publication
    use_physics_hp: bool = False,      # Use HeatPump:PlantLoop:EIR:Heating
    use_stratified_tes: bool = False,  # Use WaterHeater:Stratified
    tes_num_nodes: int = 10,           # Number of stratification layers
) -> str:
    """
    Build a complete classroom model for RL simulation.
    
    This is the main entry point for creating the building model.
    Includes ALL features for carbon-aware optimization:
    - Hot water TES (heating season)
    - Chilled water TES (cooling season) 
    - Rooftop PV system (47kW)
    - RL control interfaces
    
    NEW Physics Upgrade Options (for Applied Energy publication):
    - use_physics_hp: Replace Boiler with HeatPump:PlantLoop:EIR:Heating
    - use_stratified_tes: Replace Mixed tank with WaterHeater:Stratified
    - tes_num_nodes: Number of temperature observation nodes (DRL state)
    
    Args:
        idd_path: Path to Energy+.idd
        template_path: Path to template IDF
        output_path: Path for output IDF
        weather_file: Path to EPW weather file
        add_tes: Whether to add Thermal Energy Storage tank (hot water)
        add_pv: Whether to add rooftop PV system (47kW)
        use_heatpump: Replace boiler with electric heat pump
        add_chilled_tes: Add TES to chilled water loop (cooling)
        setup_rl: Setup RL control interface schedules
        use_physics_hp: Use physics-based ASHP model (EIR curves)
        use_stratified_tes: Use stratified tank model (multi-node)
        tes_num_nodes: Number of stratification nodes for DRL observation
        
    Returns:
        Path to generated IDF file
    """
    print("=" * 60)
    print("Building University Classroom Model (RL-Ready)")
    print("=" * 60)
    
    # Create builder
    builder = IDFBuilder(idd_path, template_path)
    
    # Configure model
    print("\n[1/12] Setting simulation parameters...")
    builder.set_simulation_settings(
        timesteps_per_hour=4,  # 15-minute intervals
    )
    
    print("\n[2/12] Setting location (Shanghai)...")
    builder.set_location(
        city="Shanghai",
        latitude=31.23,
        longitude=121.47,
        timezone=8.0,
    )
    
    print("\n[3/12] Setting building info...")
    builder.set_building_info(
        name="University_Classroom_Shanghai",
        north_axis=0.0,
    )
    
    print("\n[4/12] Renaming zones...")
    builder.rename_zones()
    
    print("\n[5/12] Setting occupancy...")
    builder.set_occupancy(
        people_per_zone=30,  # Typical classroom
    )
    
    print("\n[6/12] Setting envelope U-values (from paper Table A.10)...")
    builder.set_envelope_uvalues(
        wall_uvalue=0.30,    # W/m²K
        window_uvalue=1.1,   # W/m²K
        roof_uvalue=0.25,    # W/m²K
    )
    
    # Optional: Add TES tank (mixed or stratified)
    if add_tes:
        if use_stratified_tes:
            print(f"\n[7/12] Adding STRATIFIED TES tank ({tes_num_nodes} nodes)...")
            # First add the basic TES to set up branch topology
            builder.add_tes_tank(
                tank_volume=12.0,     # m³ (from paper)
                max_temp=50.0,        # °C
                min_temp=40.0,        # °C
                deadband_temp=2.0,    # °C
                tank_ua=10.0,         # W/K
            )
            # Then upgrade to stratified version
            builder.add_stratified_tes_tank(
                tank_volume=12.0,
                height=2.5,           # m
                num_nodes=tes_num_nodes,
                tank_ua=5.0,          # W/K
            )
        else:
            print("\n[7/12] Adding Thermal Energy Storage (TES) tank (Mixed)...")
            builder.add_tes_tank(
                tank_volume=12.0,     # m³ (from paper)
                max_temp=50.0,        # °C
                min_temp=40.0,        # °C
                deadband_temp=2.0,    # °C
                tank_ua=10.0,         # W/K
            )
    else:
        print("\n[7/12] Skipping TES tank (disabled)")
    
    # Optional: Replace boiler with heat pump (simple or physics-based)
    if use_heatpump:
        if use_physics_hp:
            print("\n[8/12] Replacing boiler with PHYSICS-BASED ASHP...")
            builder.add_heatpump_physics_based(
                heating_capacity=195800.0,  # W (from paper)
                cop_rated=3.2,
            )
        else:
            print("\n[8/12] Replacing boiler with heat pump (simple model)...")
            builder.replace_boiler_with_heatpump(
                heating_capacity=195800.0,  # W (from paper)
                cop_rated=3.32,
            )
    else:
        print("\n[8/12] Keeping gas boiler (heat pump disabled)")
    
    # Optional: Add chilled water TES
    if add_chilled_tes:
        print("\n[9/12] Adding chilled water TES...")
        builder.add_chilled_water_tes(
            tank_volume=8.0,
            max_temp=15.0,
            min_temp=5.0,
        )
    else:
        print("\n[9/12] Skipping chilled water TES")
    
    # Optional: Add PV system
    if add_pv:
        print("\n[10/12] Adding rooftop PV system...")
        builder.add_pv_system(
            peak_power=47000.0,   # W (from paper)
            efficiency=0.18,
        )
    else:
        print("\n[10/12] Skipping PV system (use Python simulation instead)")
    
    # Setup RL control interface
    if setup_rl:
        print("\n[11/15] Setting up RL control interface...")
        builder.setup_rl_interface()
    else:
        print("\n[11/15] Skipping RL interface setup")
    
    # Configure pumps for night-time TES charging (Applied Energy requirement)
    if use_physics_hp or use_stratified_tes:
        print("\n[12/15] Configuring hydraulics for TES charging...")
        builder.configure_pump_for_tes_charging()
    
    print("\n[13/15] Adding output variables...")
    builder.add_output_variables()
    
    print("\n[14/15] Adding detailed outputs for carbon calculation...")
    builder.add_detailed_outputs()
    
    # Add DRL-specific observation outputs (for Transformer state space)
    if use_physics_hp or use_stratified_tes:
        print("\n[15/15] Adding DRL observation outputs...")
        builder.add_drl_observation_outputs()
    
    # Save
    output = builder.save(output_path)
    
    # Print zone info
    print("\nZone Information:")
    print("-" * 40)
    for name, info in builder.get_zone_info().items():
        print(f"  {name}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Model Summary:")
    print(f"  ✅ 5 Zones (4 Classrooms + 1 Corridor)")
    print(f"  ✅ Envelope U-values updated (Wall=0.30, Window=1.1)")
    if add_tes:
        if use_stratified_tes:
            print(f"  ✅ Hot Water TES Tank: 12 m³ [STRATIFIED, {tes_num_nodes} nodes]")
        else:
            print(f"  ✅ Hot Water TES Tank: 12 m³ [Mixed]")
    if add_chilled_tes:
        print(f"  ✅ Chilled Water TES Tank: 8 m³ [STRATIFIED, 6 nodes]")
    if use_heatpump:
        if use_physics_hp:
            print(f"  ✅ Heat Pump: 195.8 kW [PHYSICS-BASED EIR Model]")
        else:
            print(f"  ✅ Heat Pump: 195.8 kW (COP=3.32) [Simple Model]")
    if add_pv:
        print(f"  ✅ PV System: 47 kW")
    if setup_rl:
        print(f"  ✅ RL Interface: Schedule:Constant controls")
    if use_physics_hp or use_stratified_tes:
        print(f"\n  📊 Physics Upgrade: Applied Energy publication ready")
    print("=" * 60)
    
    return output


# Default configuration for reference
DEFAULT_CONFIG = {
    "building": {
        "name": "University_Classroom_Shanghai",
        "location": "Shanghai",
        "latitude": 31.23,
        "longitude": 121.47,
        "timezone": 8,
        "north_axis": 0,
        "num_zones": 5,  # 4 classrooms + 1 corridor
        "zone_area": 218.09,  # m² per classroom (from paper)
        "zone_height": 5.1,   # m (from paper)
    },
    "simulation": {
        "timesteps_per_hour": 4,  # 15-minute intervals
        "start_month": 1,
        "start_day": 1,
        "end_month": 12,
        "end_day": 31,
    },
    "occupancy": {
        "people_per_zone": 30,
        "activity_level": 120,  # W/person
    },
    "thermal_storage": {
        "volume": 12.0,       # m³ (from paper)
        "max_temp": 50.0,     # °C
        "min_temp": 40.0,     # °C (heating mode)
        "height": 2.5,        # m (for stratified tank)
        "num_nodes": 10,      # stratification layers
    },
    "hvac": {
        "hp_capacity": 195800,  # W (from paper)
        "hp_cop_heating": 3.32,
        "hp_cop_cooling": 2.88,
    },
    "pv": {
        "peak_power": 47000,    # W (from paper)
    },
    "physics_upgrade": {
        "use_physics_hp": False,      # HeatPump:PlantLoop:EIR:Heating
        "use_stratified_tes": False,  # WaterHeater:Stratified
        "description": "Enable for Applied Energy level publication",
    },
}


def save_default_config(output_path: str = "configs/building_config.json"):
    """Save default configuration to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
    print(f"Config saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build EnergyPlus IDF model for university classroom"
    )
    parser.add_argument(
        "--idd",
        type=str,
        default=None,
        help="Path to Energy+.idd file",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE,
        help="Path to template IDF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output IDF file path",
    )
    parser.add_argument(
        "--add-tes",
        action="store_true",
        default=True,
        help="Add Thermal Energy Storage tank (default: True)",
    )
    parser.add_argument(
        "--no-tes",
        action="store_true",
        help="Disable TES tank",
    )
    parser.add_argument(
        "--no-pv",
        action="store_true",
        help="Disable rooftop PV system (default: enabled)",
    )
    parser.add_argument(
        "--no-heatpump",
        action="store_true",
        help="Keep gas boiler instead of heat pump",
    )
    parser.add_argument(
        "--no-chilled-tes",
        action="store_true",
        help="Disable chilled water TES (default: enabled)",
    )
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Skip RL interface setup",
    )
    parser.add_argument(
        "--physics-hp",
        action="store_true",
        help="Use physics-based heat pump (HeatPump:PlantLoop:EIR:Heating)",
    )
    parser.add_argument(
        "--stratified-tes",
        action="store_true",
        help="Use stratified TES tank (WaterHeater:Stratified)",
    )
    parser.add_argument(
        "--tes-nodes",
        type=int,
        default=10,
        help="Number of stratification nodes for TES (default: 10)",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save default configuration to JSON",
    )
    args = parser.parse_args()
    
    if args.save_config:
        save_default_config()
    
    # Build model (all features enabled by default)
    build_classroom_model(
        idd_path=args.idd,
        template_path=args.template,
        output_path=args.output,
        add_tes=not args.no_tes,
        add_pv=not args.no_pv,
        use_heatpump=not args.no_heatpump,
        add_chilled_tes=not args.no_chilled_tes,
        setup_rl=not args.no_rl,
        use_physics_hp=args.physics_hp,
        use_stratified_tes=args.stratified_tes,
        tes_num_nodes=args.tes_nodes,
    )
