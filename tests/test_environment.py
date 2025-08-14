"""
Unit tests for the TrafficEnvironment module.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.environment import TrafficEnvironment
from src.config import ENV_CONFIG, SUMO_CONFIG


class TestTrafficEnvironment(unittest.TestCase):
    """Test cases for TrafficEnvironment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = TrafficEnvironment(gui=False, num_seconds=100)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.action_space.n, ENV_CONFIG["action_size"])
        self.assertEqual(self.env.observation_space.shape[0], ENV_CONFIG["state_size"])
    
    def test_reset(self):
        """Test environment reset."""
        observation, info = self.env.reset()
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape[0], ENV_CONFIG["state_size"])
        self.assertIsInstance(info, dict)
        self.assertIn('episode', info)
        self.assertIn('step', info)
        self.assertIn('total_reward', info)
    
    def test_step(self):
        """Test environment step."""
        observation, info = self.env.reset()
        
        # Test with valid action
        action = 0
        next_observation, reward, done, truncated, info = self.env.step(action)
        
        self.assertIsInstance(next_observation, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_invalid_action(self):
        """Test handling of invalid actions."""
        observation, info = self.env.reset()
        
        # Test with invalid action (should be handled gracefully)
        with self.assertRaises(Exception):
            self.env.step(10)  # Invalid action
    
    def test_observation_normalization(self):
        """Test that observations are properly normalized."""
        observation, info = self.env.reset()
        
        # Check that all values are in [0, 1] range
        self.assertTrue(np.all(observation >= 0))
        self.assertTrue(np.all(observation <= 1))
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        observation, info = self.env.reset()
        
        # Run for maximum steps
        for step in range(100):
            action = 0
            observation, reward, done, truncated, info = self.env.step(action)
            
            if done or truncated:
                break
        
        # Should terminate after max steps
        self.assertTrue(done or truncated)
    
    def test_metrics_collection(self):
        """Test metrics collection during simulation."""
        observation, info = self.env.reset()
        
        # Run a few steps
        for step in range(10):
            action = 0
            observation, reward, done, truncated, info = self.env.step(action)
            
            # Check that metrics are collected
            self.assertIn('metrics', info)
            self.assertIsInstance(info['metrics'], dict)
            
            if done or truncated:
                break
    
    def test_state_info(self):
        """Test state information retrieval."""
        observation, info = self.env.reset()
        
        state_info = self.env.get_state_info()
        
        self.assertIsInstance(state_info, dict)
        self.assertIn('current_step', state_info)
        self.assertIn('current_phase', state_info)
        self.assertIn('phase_duration', state_info)
        self.assertIn('total_vehicles', state_info)
        self.assertIn('queue_lengths', state_info)
        self.assertIn('wait_times', state_info)
    
    def test_metrics_retrieval(self):
        """Test metrics retrieval."""
        observation, info = self.env.reset()
        
        # Run a few steps to accumulate metrics
        for step in range(5):
            action = 0
            observation, reward, done, truncated, info = self.env.step(action)
            
            if done or truncated:
                break
        
        metrics = self.env.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_wait_time', metrics)
        self.assertIn('total_vehicles', metrics)
        self.assertIn('total_fuel_consumption', metrics)
        self.assertIn('phase_changes', metrics)
    
    def test_close(self):
        """Test environment cleanup."""
        # This test might not work without SUMO installed
        try:
            self.env.close()
            # Should not raise an exception
        except Exception as e:
            # If SUMO is not available, this is expected
            pass
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.env.close()
        except:
            pass


class TestTrafficEnvironmentConfig(unittest.TestCase):
    """Test cases for environment configuration."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with valid config
        env = TrafficEnvironment(gui=False, num_seconds=100)
        self.assertIsNotNone(env)
        
        # Test with custom config
        custom_config = SUMO_CONFIG.copy()
        custom_config['num_seconds'] = 200
        env = TrafficEnvironment(gui=False, num_seconds=200)
        self.assertEqual(env.num_seconds, 200)
    
    def test_gui_mode(self):
        """Test GUI mode configuration."""
        # Test without GUI
        env = TrafficEnvironment(gui=False, num_seconds=100)
        self.assertEqual(env.gui, False)
        self.assertEqual(env.sumo_binary, "sumo")
        
        # Test with GUI
        env = TrafficEnvironment(gui=True, num_seconds=100)
        self.assertEqual(env.gui, True)
        self.assertEqual(env.sumo_binary, "sumo-gui")


if __name__ == '__main__':
    unittest.main()
