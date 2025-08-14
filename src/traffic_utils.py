"""
Traffic Utilities for SUMO Integration

This module provides utility functions for SUMO traffic simulation,
including traffic data processing, network generation, and simulation setup.
"""

import os
import sys
import numpy as np
import traci
import sumolib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import SUMO_CONFIG, TRAFFIC_CONFIG, RAW_DATA_DIR


def get_queue_length(direction: str, detection_range: float = 50) -> float:
    """
    Get queue length for a specific direction.
    
    Args:
        direction: Direction ('north', 'south', 'east', 'west')
        detection_range: Detection zone range in meters
        
    Returns:
        Queue length in vehicles
    """
    if not traci.isConnected():
        return 0.0
    
    # Define edge IDs for each direction
    edge_mapping = {
        'north': ['edge_north_in', 'edge_north_out'],
        'south': ['edge_south_in', 'edge_south_out'],
        'east': ['edge_east_in', 'edge_east_out'],
        'west': ['edge_west_in', 'edge_west_out']
    }
    
    if direction not in edge_mapping:
        return 0.0
    
    edge_ids = edge_mapping[direction]
    queue_length = 0
    
    for edge_id in edge_ids:
        try:
            vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
            queue_length += vehicles
        except:
            continue
    
    return float(queue_length)


def get_waiting_time(direction: str) -> float:
    """
    Get average waiting time for vehicles in a specific direction.
    
    Args:
        direction: Direction ('north', 'south', 'east', 'west')
        
    Returns:
        Average waiting time in seconds
    """
    if not traci.isConnected():
        return 0.0
    
    # Define edge IDs for each direction
    edge_mapping = {
        'north': ['edge_north_in'],
        'south': ['edge_south_in'],
        'east': ['edge_east_in'],
        'west': ['edge_west_in']
    }
    
    if direction not in edge_mapping:
        return 0.0
    
    edge_ids = edge_mapping[direction]
    total_wait_time = 0
    total_vehicles = 0
    
    for edge_id in edge_ids:
        try:
            vehicles = traci.edge.getLastStepVehicleIDList(edge_id)
            for vehicle_id in vehicles:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                total_wait_time += wait_time
                total_vehicles += 1
        except:
            continue
    
    return total_wait_time / max(total_vehicles, 1)


def get_vehicle_count(direction: str, detection_range: float = 50) -> int:
    """
    Get number of vehicles in detection zone for a specific direction.
    
    Args:
        direction: Direction ('north', 'south', 'east', 'west')
        detection_range: Detection zone range in meters
        
    Returns:
        Number of vehicles
    """
    if not traci.isConnected():
        return 0
    
    # Define edge IDs for each direction
    edge_mapping = {
        'north': ['edge_north_in'],
        'south': ['edge_south_in'],
        'east': ['edge_east_in'],
        'west': ['edge_west_in']
    }
    
    if direction not in edge_mapping:
        return 0
    
    edge_ids = edge_mapping[direction]
    vehicle_count = 0
    
    for edge_id in edge_ids:
        try:
            vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
            vehicle_count += vehicles
        except:
            continue
    
    return vehicle_count


def calculate_reward(throughput: float, wait_time: float, 
                    fuel_consumption: float, phase_changes: int,
                    weights: Dict[str, float] = None) -> float:
    """
    Calculate reward based on traffic metrics.
    
    Args:
        throughput: Vehicles per hour
        wait_time: Average wait time in seconds
        fuel_consumption: Total fuel consumption in liters
        phase_changes: Number of phase changes
        weights: Reward weights for different components
        
    Returns:
        Calculated reward
    """
    if weights is None:
        weights = {
            'throughput': 0.4,
            'wait_time': 0.3,
            'fuel_consumption': 0.2,
            'phase_changes': 0.1
        }
    
    reward = (
        weights['throughput'] * throughput -
        weights['wait_time'] * wait_time -
        weights['fuel_consumption'] * fuel_consumption -
        weights['phase_changes'] * phase_changes
    )
    
    return reward


def normalize_state(state: List[float], normalization: Dict[str, float]) -> List[float]:
    """
    Normalize state values to [0, 1] range.
    
    Args:
        state: State values to normalize
        normalization: Normalization factors for each component
        
    Returns:
        Normalized state values
    """
    normalized_state = []
    
    # Normalize queue lengths (first 4 values)
    for i in range(4):
        normalized_state.append(min(state[i] / normalization['queue_length'], 1.0))
    
    # Normalize wait times (next 4 values)
    for i in range(4, 8):
        normalized_state.append(min(state[i] / normalization['wait_time'], 1.0))
    
    # Normalize vehicle counts (next 4 values)
    for i in range(8, 12):
        normalized_state.append(min(state[i] / normalization['vehicle_count'], 1.0))
    
    # Traffic light state (already normalized)
    normalized_state.extend(state[12:15])
    
    # Traffic density
    normalized_state.append(min(state[15] / normalization['vehicle_count'], 1.0))
    
    # Emergency vehicles, pedestrians, weather (already normalized)
    normalized_state.extend(state[16:19])
    
    # Time of day (already normalized)
    normalized_state.append(state[19])
    
    return normalized_state


def create_sumo_config(num_seconds: int = 3600, gui: bool = False) -> str:
    """
    Create SUMO configuration files for traffic simulation.
    
    Args:
        num_seconds: Simulation duration in seconds
        gui: Whether to show GUI
        
    Returns:
        Path to SUMO configuration file
    """
    # Create network file
    network_file = create_network_file()
    
    # Create route file
    route_file = create_route_file(num_seconds)
    
    # Create traffic light file
    traffic_light_file = create_traffic_light_file()
    
    # Create SUMO configuration file
    config_file = RAW_DATA_DIR / "simulation.sumocfg"
    
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{network_file.name}"/>
        <route-files value="{route_file.name}"/>
        <additional-files value="{traffic_light_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{num_seconds}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
    </processing>
    <routing>
        <device.rerouting.probability value="0.1"/>
    </routing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>
</configuration>"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return str(config_file)


def create_network_file() -> Path:
    """Create SUMO network file for intersection simulation."""
    network_file = RAW_DATA_DIR / "network.net.xml"
    
    network_content = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    <edge id="edge_north_in" from="intersection_0" to="north_in" priority="1">
        <lane id="edge_north_in_0" index="0" speed="13.89" length="50.00" shape="100.00,0.00 100.00,50.00"/>
    </edge>
    <edge id="edge_north_out" from="north_out" to="intersection_0" priority="1">
        <lane id="edge_north_out_0" index="0" speed="13.89" length="50.00" shape="100.00,150.00 100.00,200.00"/>
    </edge>
    <edge id="edge_south_in" from="intersection_0" to="south_in" priority="1">
        <lane id="edge_south_in_0" index="0" speed="13.89" length="50.00" shape="100.00,150.00 100.00,200.00"/>
    </edge>
    <edge id="edge_south_out" from="south_out" to="intersection_0" priority="1">
        <lane id="edge_south_out_0" index="0" speed="13.89" length="50.00" shape="100.00,0.00 100.00,50.00"/>
    </edge>
    <edge id="edge_east_in" from="intersection_0" to="east_in" priority="1">
        <lane id="edge_east_in_0" index="0" speed="13.89" length="50.00" shape="150.00,100.00 200.00,100.00"/>
    </edge>
    <edge id="edge_east_out" from="east_out" to="intersection_0" priority="1">
        <lane id="edge_east_out_0" index="0" speed="13.89" length="50.00" shape="0.00,100.00 50.00,100.00"/>
    </edge>
    <edge id="edge_west_in" from="intersection_0" to="west_in" priority="1">
        <lane id="edge_west_in_0" index="0" speed="13.89" length="50.00" shape="0.00,100.00 50.00,100.00"/>
    </edge>
    <edge id="edge_west_out" from="west_out" to="intersection_0" priority="1">
        <lane id="edge_west_out_0" index="0" speed="13.89" length="50.00" shape="150.00,100.00 200.00,100.00"/>
    </edge>
    
    <junction id="intersection_0" type="traffic_light" x="100.00" y="100.00" incLanes="edge_north_out_0 edge_south_out_0 edge_east_out_0 edge_west_out_0" intLanes=":intersection_0_0_0 :intersection_0_1_0 :intersection_0_2_0 :intersection_0_3_0" shape="95.00,95.00 105.00,95.00 105.00,105.00 95.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
        <request index="2" response="00" foes="00" cont="0"/>
        <request index="3" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="north_in" type="priority" x="100.00" y="150.00" incLanes="edge_north_in_0" intLanes=":north_in_0_0" shape="95.00,145.00 105.00,145.00 105.00,155.00 95.00,155.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="north_out" type="priority" x="100.00" y="50.00" incLanes="edge_north_out_0" intLanes=":north_out_0_0" shape="95.00,45.00 105.00,45.00 105.00,55.00 95.00,55.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="south_in" type="priority" x="100.00" y="50.00" incLanes="edge_south_in_0" intLanes=":south_in_0_0" shape="95.00,45.00 105.00,45.00 105.00,55.00 95.00,55.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="south_out" type="priority" x="100.00" y="150.00" incLanes="edge_south_out_0" intLanes=":south_out_0_0" shape="95.00,145.00 105.00,145.00 105.00,155.00 95.00,155.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="east_in" type="priority" x="150.00" y="100.00" incLanes="edge_east_in_0" intLanes=":east_in_0_0" shape="145.00,95.00 155.00,95.00 155.00,105.00 145.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="east_out" type="priority" x="50.00" y="100.00" incLanes="edge_east_out_0" intLanes=":east_out_0_0" shape="45.00,95.00 55.00,95.00 55.00,105.00 45.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="west_in" type="priority" x="50.00" y="100.00" incLanes="edge_west_in_0" intLanes=":west_in_0_0" shape="45.00,95.00 55.00,95.00 55.00,105.00 45.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <junction id="west_out" type="priority" x="150.00" y="100.00" incLanes="edge_west_out_0" intLanes=":west_out_0_0" shape="145.00,95.00 155.00,95.00 155.00,105.00 145.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <connection from="edge_north_out" to="edge_south_in" fromLane="0" toLane="0" via=":intersection_0_0_0" dir="s" state="M"/>
    <connection from="edge_south_out" to="edge_north_in" fromLane="0" toLane="0" via=":intersection_0_1_0" dir="s" state="M"/>
    <connection from="edge_east_out" to="edge_west_in" fromLane="0" toLane="0" via=":intersection_0_2_0" dir="s" state="M"/>
    <connection from="edge_west_out" to="edge_east_in" fromLane="0" toLane="0" via=":intersection_0_3_0" dir="s" state="M"/>
</net>"""
    
    with open(network_file, 'w') as f:
        f.write(network_content)
    
    return network_file


def create_route_file(num_seconds: int = 3600) -> Path:
    """Create SUMO route file with vehicle flows."""
    route_file = RAW_DATA_DIR / "routes.rou.xml"
    
    # Calculate vehicle flows based on traffic patterns
    flows = generate_traffic_flows(num_seconds)
    
    route_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
    <vType id="truck" accel="1.3" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="13.89" guiShape="truck"/>
    <vType id="bus" accel="1.2" decel="4.5" sigma="0.5" length="12" minGap="3" maxSpeed="11.11" guiShape="bus"/>
    <vType id="emergency" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="emergency" color="255,0,0"/>
    
    <route id="route_north_south" edges="edge_north_out edge_south_in"/>
    <route id="route_south_north" edges="edge_south_out edge_north_in"/>
    <route id="route_east_west" edges="edge_east_out edge_west_in"/>
    <route id="route_west_east" edges="edge_west_out edge_east_in"/>
    
    {flows}
</routes>"""
    
    with open(route_file, 'w') as f:
        f.write(route_content)
    
    return route_file


def create_traffic_light_file() -> Path:
    """Create SUMO traffic light file."""
    traffic_light_file = RAW_DATA_DIR / "traffic_lights.add.xml"
    
    traffic_light_content = """<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <tlLogic id="traffic_light_0" type="static" programID="0" offset="0">
        <phase duration="30" state="GGrr"/>
        <phase duration="3" state="yyrr"/>
        <phase duration="30" state="rrGG"/>
        <phase duration="3" state="rryy"/>
    </tlLogic>
</additional>"""
    
    with open(traffic_light_file, 'w') as f:
        f.write(traffic_light_content)
    
    return traffic_light_file


def generate_traffic_flows(num_seconds: int) -> str:
    """Generate traffic flows based on time patterns."""
    flows = []
    
    # Peak morning (7-9 AM)
    flows.append(f'<flow id="flow_morning_ns" begin="25200" end="32400" vehsPerHour="800" route="route_north_south" type="passenger"/>')
    flows.append(f'<flow id="flow_morning_sn" begin="25200" end="32400" vehsPerHour="800" route="route_south_north" type="passenger"/>')
    flows.append(f'<flow id="flow_morning_ew" begin="25200" end="32400" vehsPerHour="400" route="route_east_west" type="passenger"/>')
    flows.append(f'<flow id="flow_morning_we" begin="25200" end="32400" vehsPerHour="400" route="route_west_east" type="passenger"/>')
    
    # Off-peak (10 AM - 4 PM)
    flows.append(f'<flow id="flow_offpeak_ns" begin="36000" end="57600" vehsPerHour="300" route="route_north_south" type="passenger"/>')
    flows.append(f'<flow id="flow_offpeak_sn" begin="36000" end="57600" vehsPerHour="300" route="route_south_north" type="passenger"/>')
    flows.append(f'<flow id="flow_offpeak_ew" begin="36000" end="57600" vehsPerHour="200" route="route_east_west" type="passenger"/>')
    flows.append(f'<flow id="flow_offpeak_we" begin="36000" end="57600" vehsPerHour="200" route="route_west_east" type="passenger"/>')
    
    # Peak evening (5-7 PM)
    flows.append(f'<flow id="flow_evening_ns" begin="61200" end="68400" vehsPerHour="700" route="route_north_south" type="passenger"/>')
    flows.append(f'<flow id="flow_evening_sn" begin="61200" end="68400" vehsPerHour="700" route="route_south_north" type="passenger"/>')
    flows.append(f'<flow id="flow_evening_ew" begin="61200" end="68400" vehsPerHour="400" route="route_east_west" type="passenger"/>')
    flows.append(f'<flow id="flow_evening_we" begin="61200" end="68400" vehsPerHour="400" route="route_west_east" type="passenger"/>')
    
    # Night (8 PM - 6 AM)
    flows.append(f'<flow id="flow_night_ns" begin="72000" end="86400" vehsPerHour="100" route="route_north_south" type="passenger"/>')
    flows.append(f'<flow id="flow_night_sn" begin="72000" end="86400" vehsPerHour="100" route="route_south_north" type="passenger"/>')
    flows.append(f'<flow id="flow_night_ew" begin="72000" end="86400" vehsPerHour="100" route="route_east_west" type="passenger"/>')
    flows.append(f'<flow id="flow_night_we" begin="72000" end="86400" vehsPerHour="100" route="route_west_east" type="passenger"/>')
    
    # Add some trucks and buses
    flows.append(f'<flow id="flow_trucks" begin="0" end="{num_seconds}" vehsPerHour="50" route="route_north_south" type="truck"/>')
    flows.append(f'<flow id="flow_buses" begin="0" end="{num_seconds}" vehsPerHour="20" route="route_east_west" type="bus"/>')
    
    # Add emergency vehicles occasionally
    flows.append(f'<flow id="flow_emergency" begin="0" end="{num_seconds}" vehsPerHour="2" route="route_north_south" type="emergency"/>')
    
    return '\n    '.join(flows)


def get_traffic_metrics() -> Dict[str, float]:
    """
    Get comprehensive traffic metrics from SUMO simulation.
    
    Returns:
        Dictionary containing various traffic metrics
    """
    if not traci.isConnected():
        return {}
    
    metrics = {}
    
    # Basic metrics
    metrics['total_vehicles'] = len(traci.vehicle.getIDList())
    metrics['completed_vehicles'] = len(traci.simulation.getArrivedIDList())
    metrics['total_wait_time'] = sum([traci.vehicle.getAccumulatedWaitingTime(v) 
                                    for v in traci.vehicle.getIDList()])
    
    # Fuel consumption and emissions
    metrics['total_fuel_consumption'] = sum([traci.vehicle.getFuelConsumption(v) 
                                           for v in traci.vehicle.getIDList()])
    metrics['total_co2_emissions'] = sum([traci.vehicle.getCO2Emission(v) 
                                        for v in traci.vehicle.getIDList()])
    
    # Average metrics
    if metrics['total_vehicles'] > 0:
        metrics['average_wait_time'] = metrics['total_wait_time'] / metrics['total_vehicles']
        metrics['average_fuel_consumption'] = metrics['total_fuel_consumption'] / metrics['total_vehicles']
        metrics['average_co2_emissions'] = metrics['total_co2_emissions'] / metrics['total_vehicles']
    else:
        metrics['average_wait_time'] = 0
        metrics['average_fuel_consumption'] = 0
        metrics['average_co2_emissions'] = 0
    
    # Throughput
    simulation_time = traci.simulation.getTime()
    if simulation_time > 0:
        metrics['throughput'] = metrics['completed_vehicles'] / (simulation_time / 3600)  # veh/h
    else:
        metrics['throughput'] = 0
    
    return metrics


def cleanup_sumo_files():
    """Clean up temporary SUMO files."""
    files_to_remove = [
        RAW_DATA_DIR / "network.net.xml",
        RAW_DATA_DIR / "routes.rou.xml", 
        RAW_DATA_DIR / "traffic_lights.add.xml",
        RAW_DATA_DIR / "simulation.sumocfg"
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
