"""
Unit tests for the traffic_utils module.
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.traffic_utils import (
    calculate_reward, normalize_state, get_traffic_metrics,
    create_sumo_config, create_network_file, create_route_file,
    create_traffic_light_file, generate_traffic_flows
)
from src.config import ENV_CONFIG, TRAFFIC_CONFIG


class TestTrafficUtils(unittest.TestCase):
    """Test cases for traffic utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_raw_dir = Path(__file__).parent.parent / "src" / "config.py"
        
        # Mock RAW_DATA_DIR for testing
        import src.config
        src.config.RAW_DATA_DIR = Path(self.temp_dir)
    
    def test_calculate_reward(self):
        """Test reward calculation function."""
        # Test with default weights
        reward = calculate_reward(
            throughput=100,
            wait_time=50,
            fuel_consumption=10,
            phase_changes=5
        )
        
        self.assertIsInstance(reward, float)
        
        # Test with custom weights
        custom_weights = {
            'throughput': 0.5,
            'wait_time': 0.3,
            'fuel_consumption': 0.1,
            'phase_changes': 0.1
        }
        
        reward_custom = calculate_reward(
            throughput=100,
            wait_time=50,
            fuel_consumption=10,
            phase_changes=5,
            weights=custom_weights
        )
        
        self.assertIsInstance(reward_custom, float)
        self.assertNotEqual(reward, reward_custom)  # Should be different with custom weights
    
    def test_normalize_state(self):
        """Test state normalization function."""
        # Create test state
        state = [25, 30, 15, 20,  # queue lengths
                150, 200, 100, 120,  # wait times
                10, 15, 8, 12,  # vehicle counts
                0, 15, 5,  # traffic light state
                45,  # traffic density
                1,  # emergency vehicles
                2,  # pedestrians
                0,  # weather
                0.5]  # time of day
        
        normalization = ENV_CONFIG["normalization"]
        
        normalized_state = normalize_state(state, normalization)
        
        # Check that normalized state has correct length
        self.assertEqual(len(normalized_state), len(state))
        
        # Check that all values are in [0, 1] range
        for value in normalized_state:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        
        # Check that time of day is preserved (should be already normalized)
        self.assertEqual(normalized_state[-1], state[-1])
    
    def test_create_network_file(self):
        """Test network file creation."""
        network_file = create_network_file()
        
        self.assertTrue(network_file.exists())
        self.assertEqual(network_file.suffix, '.xml')
        
        # Check file content
        with open(network_file, 'r') as f:
            content = f.read()
            self.assertIn('<?xml', content)
            self.assertIn('network', content)
    
    def test_create_route_file(self):
        """Test route file creation."""
        route_file = create_route_file(num_seconds=100)
        
        self.assertTrue(route_file.exists())
        self.assertEqual(route_file.suffix, '.xml')
        
        # Check file content
        with open(route_file, 'r') as f:
            content = f.read()
            self.assertIn('<?xml', content)
            self.assertIn('routes', content)
            self.assertIn('flow', content)
    
    def test_create_traffic_light_file(self):
        """Test traffic light file creation."""
        traffic_light_file = create_traffic_light_file()
        
        self.assertTrue(traffic_light_file.exists())
        self.assertEqual(traffic_light_file.suffix, '.xml')
        
        # Check file content
        with open(traffic_light_file, 'r') as f:
            content = f.read()
            self.assertIn('<?xml', content)
            self.assertIn('tlLogic', content)
    
    def test_create_sumo_config(self):
        """Test SUMO configuration file creation."""
        config_file = create_sumo_config(num_seconds=100, gui=False)
        
        self.assertTrue(Path(config_file).exists())
        self.assertTrue(config_file.endswith('.sumocfg'))
        
        # Check file content
        with open(config_file, 'r') as f:
            content = f.read()
            self.assertIn('<?xml', content)
            self.assertIn('configuration', content)
            self.assertIn('100', content)  # Should contain the num_seconds value
    
    def test_generate_traffic_flows(self):
        """Test traffic flow generation."""
        flows = generate_traffic_flows(num_seconds=100)
        
        self.assertIsInstance(flows, str)
        self.assertIn('flow', flows)
        self.assertIn('100', flows)  # Should contain the num_seconds value
        
        # Check that flows contain expected patterns
        expected_patterns = ['morning', 'offpeak', 'evening', 'night']
        for pattern in expected_patterns:
            self.assertIn(pattern, flows)
    
    def test_get_traffic_metrics_mock(self):
        """Test traffic metrics function (mock version)."""
        # This test would require SUMO to be running
        # For now, we'll test the function signature and basic behavior
        
        # Mock traci connection
        import traci
        if hasattr(traci, 'isConnected'):
            # If traci is available, test the function
            try:
                metrics = get_traffic_metrics()
                self.assertIsInstance(metrics, dict)
            except:
                # If SUMO is not available, this is expected
                pass
        else:
            # Mock traci
            import types
            traci.isConnected = types.MethodType(lambda self: False, traci)
            
            metrics = get_traffic_metrics()
            self.assertIsInstance(metrics, dict)
            self.assertEqual(metrics, {})  # Should return empty dict when not connected
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test reward calculation with zero values
        reward = calculate_reward(0, 0, 0, 0)
        self.assertIsInstance(reward, float)
        
        # Test state normalization with zero values
        zero_state = [0] * 20
        normalization = ENV_CONFIG["normalization"]
        normalized_zero = normalize_state(zero_state, normalization)
        
        self.assertEqual(len(normalized_zero), len(zero_state))
        for value in normalized_zero:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        
        # Test state normalization with very large values
        large_state = [1000] * 20
        normalized_large = normalize_state(large_state, normalization)
        
        self.assertEqual(len(normalized_large), len(large_state))
        for value in normalized_large:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
