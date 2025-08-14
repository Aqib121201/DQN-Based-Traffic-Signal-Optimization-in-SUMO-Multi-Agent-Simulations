"""
SUMO Traffic Environment for DQN Training

This module provides a Gymnasium-compatible environment wrapper for SUMO
traffic simulation, enabling reinforcement learning training for traffic signal control.
"""

import os
import sys
import time
import numpy as np
import traci
import sumolib
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import gymnasium as gym
from gymnasium import spaces

from .config import SUMO_CONFIG, ENV_CONFIG, TRAFFIC_CONFIG
from .traffic_utils import (
    get_queue_length, get_waiting_time, get_vehicle_count,
    calculate_reward, normalize_state, create_sumo_config
)


class TrafficEnvironment(gym.Env):
    """
    SUMO Traffic Environment for DQN-based traffic signal control.
    
    This environment simulates a traffic intersection and provides
    reinforcement learning interface for training traffic signal controllers.
    """
    
    def __init__(self, 
                 sumo_config: Dict = None,
                 env_config: Dict = None,
                 traffic_config: Dict = None,
                 gui: bool = False,
                 num_seconds: int = 3600):
        """
        Initialize the traffic environment.
        
        Args:
            sumo_config: SUMO simulation configuration
            env_config: Environment configuration
            traffic_config: Traffic generation configuration
            gui: Whether to show SUMO GUI
            num_seconds: Simulation duration in seconds
        """
        super().__init__()
        
        # Configuration
        self.sumo_config = sumo_config or SUMO_CONFIG.copy()
        self.env_config = env_config or ENV_CONFIG.copy()
        self.traffic_config = traffic_config or TRAFFIC_CONFIG.copy()
        self.gui = gui
        self.num_seconds = num_seconds
        
        # SUMO setup
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"
        self.sumo_cmd = None
        self.sumo_config_file = None
        
        # Environment state
        self.current_step = 0
        self.episode_reward = 0
        self.episode_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'total_fuel_consumption': 0,
            'phase_changes': 0
        }
        
        # Traffic light state
        self.traffic_light_id = self.sumo_config["traffic_light_id"]
        self.current_phase = 0
        self.phase_duration = 0
        self.last_phase_change = 0
        
        # Detection zones
        self.detection_ranges = {
            'north': 50,
            'south': 50,
            'east': 50,
            'west': 50
        }
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(self.env_config["action_size"])
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.env_config["state_size"],), 
            dtype=np.float32
        )
        
        # Initialize SUMO
        self._setup_sumo()
        
    def _setup_sumo(self):
        """Setup SUMO simulation files and configuration."""
        try:
            # Create SUMO configuration files
            self.sumo_config_file = create_sumo_config(
                num_seconds=self.num_seconds,
                gui=self.gui
            )
            
            # Set SUMO command
            self.sumo_cmd = [
                self.sumo_binary,
                "-c", self.sumo_config_file,
                "--no-step-log",
                "--no-warnings",
                "--random"
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup SUMO: {e}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Close existing SUMO connection
        if traci.isConnected():
            traci.close()
        
        # Start new SUMO simulation
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}")
        
        # Reset environment state
        self.current_step = 0
        self.episode_reward = 0
        self.episode_metrics = {
            'total_wait_time': 0,
            'total_vehicles': 0,
            'total_fuel_consumption': 0,
            'phase_changes': 0
        }
        
        # Reset traffic light state
        self.current_phase = 0
        self.phase_duration = 0
        self.last_phase_change = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'episode': 0,
            'step': self.current_step,
            'total_reward': self.episode_reward
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: Current state observation
            reward: Reward for the action
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Execute action
        self._execute_action(action)
        
        # Simulate one step
        traci.simulationStep()
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Update metrics
        self._update_metrics()
        
        info = {
            'step': self.current_step,
            'total_reward': self.episode_reward,
            'action': action,
            'current_phase': self.current_phase,
            'metrics': self.episode_metrics.copy()
        }
        
        return observation, reward, done, False, info
    
    def _execute_action(self, action: int):
        """Execute the given action on the traffic light."""
        if action == 0:  # Extend current green phase
            if self.phase_duration < self.sumo_config["max_green"]:
                self.phase_duration += 1
        elif action == 1:  # Switch to next phase
            self._switch_to_next_phase()
        elif action == 2:  # Skip to specific phase
            self._switch_to_phase((self.current_phase + 2) % 4)
        elif action == 3:  # Emergency vehicle priority
            self._handle_emergency_vehicle()
    
    def _switch_to_next_phase(self):
        """Switch to the next traffic light phase."""
        next_phase = (self.current_phase + 1) % 4
        self._switch_to_phase(next_phase)
    
    def _switch_to_phase(self, phase: int):
        """Switch to a specific traffic light phase."""
        if phase != self.current_phase:
            # Set yellow light first
            traci.trafficlight.setPhase(self.traffic_light_id, 1)  # Yellow
            traci.simulationStep()
            
            # Switch to new phase
            traci.trafficlight.setPhase(self.traffic_light_id, phase)
            self.current_phase = phase
            self.phase_duration = 0
            self.last_phase_change = self.current_step
            self.episode_metrics['phase_changes'] += 1
    
    def _handle_emergency_vehicle(self):
        """Handle emergency vehicle priority."""
        # Check for emergency vehicles
        vehicles = traci.vehicle.getIDList()
        emergency_vehicles = [v for v in vehicles if traci.vehicle.getTypeID(v) == "emergency"]
        
        if emergency_vehicles:
            # Give priority to emergency vehicles
            self._switch_to_phase(0)  # Force green for emergency route
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Get traffic data for each direction
        directions = ['north', 'south', 'east', 'west']
        state = []
        
        for direction in directions:
            # Queue length
            queue_length = get_queue_length(direction, self.detection_ranges[direction])
            state.append(queue_length)
            
            # Average wait time
            wait_time = get_waiting_time(direction)
            state.append(wait_time)
            
            # Vehicle count
            vehicle_count = get_vehicle_count(direction, self.detection_ranges[direction])
            state.append(vehicle_count)
        
        # Traffic light state
        state.extend([
            self.current_phase,
            self.phase_duration,
            self.current_step - self.last_phase_change
        ])
        
        # Traffic density
        total_vehicles = sum([get_vehicle_count(d, self.detection_ranges[d]) 
                            for d in directions])
        state.append(total_vehicles)
        
        # Emergency vehicle presence
        emergency_vehicles = len([v for v in traci.vehicle.getIDList() 
                                if traci.vehicle.getTypeID(v) == "emergency"])
        state.append(emergency_vehicles)
        
        # Pedestrian waiting
        pedestrians = len(traci.person.getIDList())
        state.append(pedestrians)
        
        # Weather condition (simplified)
        weather = 0  # Normal conditions
        state.append(weather)
        
        # Time of day (normalized)
        time_of_day = (self.current_step % 86400) / 86400  # 24-hour cycle
        state.append(time_of_day)
        
        # Normalize state
        state = normalize_state(state, self.env_config["normalization"])
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current traffic conditions."""
        # Get current metrics
        total_wait_time = sum([get_waiting_time(d) for d in ['north', 'south', 'east', 'west']])
        total_vehicles = len(traci.vehicle.getIDList())
        fuel_consumption = sum([traci.vehicle.getFuelConsumption(v) 
                              for v in traci.vehicle.getIDList()])
        
        # Calculate throughput (vehicles that passed through)
        throughput = self._calculate_throughput()
        
        # Calculate reward using weights from config
        weights = self.env_config["reward_weights"]
        reward = (
            weights["throughput"] * throughput -
            weights["wait_time"] * total_wait_time -
            weights["fuel_consumption"] * fuel_consumption -
            weights["phase_changes"] * self.episode_metrics["phase_changes"]
        )
        
        return reward
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (vehicles per hour)."""
        # Count vehicles that have completed their journey
        completed_vehicles = len(traci.simulation.getArrivedIDList())
        
        # Convert to vehicles per hour
        time_hours = self.current_step / 3600
        throughput = completed_vehicles / time_hours if time_hours > 0 else 0
        
        return throughput
    
    def _update_metrics(self):
        """Update episode metrics."""
        # Update wait time
        total_wait_time = sum([get_waiting_time(d) for d in ['north', 'south', 'east', 'west']])
        self.episode_metrics['total_wait_time'] += total_wait_time
        
        # Update vehicle count
        self.episode_metrics['total_vehicles'] = len(traci.vehicle.getIDList())
        
        # Update fuel consumption
        fuel_consumption = sum([traci.vehicle.getFuelConsumption(v) 
                              for v in traci.vehicle.getIDList()])
        self.episode_metrics['total_fuel_consumption'] += fuel_consumption
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        return self.current_step >= self.num_seconds
    
    def close(self):
        """Close the environment and SUMO connection."""
        if traci.isConnected():
            traci.close()
    
    def render(self):
        """Render the environment (not implemented for SUMO)."""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current episode metrics."""
        return self.episode_metrics.copy()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information."""
        return {
            'current_step': self.current_step,
            'current_phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'total_vehicles': len(traci.vehicle.getIDList()),
            'queue_lengths': {
                d: get_queue_length(d, self.detection_ranges[d])
                for d in ['north', 'south', 'east', 'west']
            },
            'wait_times': {
                d: get_waiting_time(d)
                for d in ['north', 'south', 'east', 'west']
            }
        }
