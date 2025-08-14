"""
Configuration parameters for DQN-Based Traffic Signal Control
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                 VISUALIZATIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# SUMO Configuration
SUMO_CONFIG = {
    "gui": False,  # Set to True for visualization
    "num_seconds": 3600,  # Simulation duration in seconds
    "yellow_time": 3,  # Yellow light duration
    "min_green": 10,  # Minimum green light duration
    "max_green": 60,  # Maximum green light duration
    "phase_duration": 30,  # Default phase duration
    "detection_range": 50,  # Detection zone range in meters
    "intersection_id": "intersection_0",
    "traffic_light_id": "traffic_light_0"
}

# Traffic Environment Configuration
ENV_CONFIG = {
    "state_size": 20,  # Size of state representation
    "action_size": 4,  # Number of possible actions
    "reward_weights": {
        "throughput": 0.4,
        "wait_time": 0.3,
        "fuel_consumption": 0.2,
        "phase_changes": 0.1
    },
    "normalization": {
        "queue_length": 50,  # Max queue length for normalization
        "wait_time": 300,    # Max wait time for normalization
        "vehicle_count": 100 # Max vehicle count for normalization
    }
}

# DQN Agent Configuration
DQN_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.95,  # Discount factor
    "epsilon": 1.0,  # Initial exploration rate
    "epsilon_min": 0.01,  # Minimum exploration rate
    "epsilon_decay": 0.995,  # Exploration rate decay
    "memory_size": 2000,  # Experience replay buffer size
    "batch_size": 32,  # Training batch size
    "target_update_freq": 100,  # Target network update frequency
    "network_architecture": [128, 64, 32],  # Hidden layer sizes
    "activation": "relu",
    "optimizer": "adam"
}

# Training Configuration
TRAINING_CONFIG = {
    "episodes": 1000,
    "max_steps_per_episode": 1000,
    "save_frequency": 100,  # Save model every N episodes
    "evaluation_frequency": 50,  # Evaluate model every N episodes
    "early_stopping_patience": 50,  # Early stopping patience
    "validation_split": 0.2,  # Validation data split
    "random_seed": 42
}

# Model Configuration
MODEL_CONFIG = {
    "model_name": "dqn_traffic_model",
    "save_format": "pkl",  # or "h5" for TensorFlow models
    "load_best_model": True,
    "model_checkpoint_dir": str(MODELS_DIR / "checkpoints")
}

# Visualization Configuration
VIS_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "viridis",
    "save_format": "png",
    "animation_fps": 10
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "traffic_rl.log"),
    "console": True
}

# Traffic Generation Configuration
TRAFFIC_CONFIG = {
    "vehicle_types": {
        "passenger": {"probability": 0.7, "speed": 13.89},  # 50 km/h
        "truck": {"probability": 0.2, "speed": 11.11},      # 40 km/h
        "bus": {"probability": 0.1, "speed": 8.33}          # 30 km/h
    },
    "traffic_patterns": {
        "peak_morning": {"start": 7, "end": 9, "volume": 2000},
        "off_peak": {"start": 10, "end": 16, "volume": 800},
        "peak_evening": {"start": 17, "end": 19, "volume": 1800},
        "night": {"start": 20, "end": 6, "volume": 300}
    },
    "routes": {
        "north_south": {"probability": 0.4},
        "east_west": {"probability": 0.4},
        "left_turn": {"probability": 0.2}
    }
}

# Evaluation Metrics
METRICS_CONFIG = {
    "primary_metrics": [
        "average_wait_time",
        "throughput",
        "fuel_consumption",
        "co2_emissions",
        "average_queue_length"
    ],
    "secondary_metrics": [
        "total_travel_time",
        "number_of_stops",
        "average_speed",
        "signal_efficiency"
    ]
}

# SHAP Configuration for Explainability
SHAP_CONFIG = {
    "background_samples": 100,
    "explanation_samples": 50,
    "feature_names": [
        "queue_north", "queue_south", "queue_east", "queue_west",
        "wait_time_north", "wait_time_south", "wait_time_east", "wait_time_west",
        "vehicle_count_north", "vehicle_count_south", "vehicle_count_east", "vehicle_count_west",
        "current_phase", "phase_duration", "time_since_phase_change",
        "traffic_density", "emergency_vehicle_present", "pedestrian_waiting",
        "weather_condition", "time_of_day"
    ]
}

# Environment Variables
ENV_VARS = {
    "SUMO_HOME": os.getenv("SUMO_HOME", "/usr/share/sumo"),
    "SUMO_BINARY": os.getenv("SUMO_BINARY", "sumo"),
    "SUMO_GUI_BINARY": os.getenv("SUMO_GUI_BINARY", "sumo-gui"),
    "PYTHONPATH": os.getenv("PYTHONPATH", "")
}

# Web Application Configuration
APP_CONFIG = {
    "title": "DQN Traffic Signal Control",
    "description": "Interactive visualization of DQN-based traffic signal control",
    "port": 8501,
    "host": "localhost",
    "debug": False
}

# Docker Configuration
DOCKER_CONFIG = {
    "base_image": "python:3.9-slim",
    "sumo_version": "1.18.0",
    "expose_port": 8501
}
