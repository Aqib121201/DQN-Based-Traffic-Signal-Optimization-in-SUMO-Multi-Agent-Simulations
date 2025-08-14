"""
DQN-Based Traffic Signal Control in Multi-Agent SUMO Simulations

A reinforcement learning approach to adaptive traffic signal control
using Deep Q-Networks and SUMO traffic simulation.
"""

__version__ = "1.0.0"
__author__ = "Traffic RL Team"
__email__ = "traffic-rl@example.com"

from .config import *
from .environment import TrafficEnvironment
from .dqn_agent import DQNAgent
from .traffic_utils import *
from .visualization import *

__all__ = [
    "TrafficEnvironment",
    "DQNAgent",
    "config",
    "traffic_utils",
    "visualization"
]
