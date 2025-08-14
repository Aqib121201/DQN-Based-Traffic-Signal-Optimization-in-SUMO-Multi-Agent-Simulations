"""
Unit tests for the DQNAgent module.
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

from src.dqn_agent import DQNAgent, DQNTrainer
from src.config import DQN_CONFIG, ENV_CONFIG


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = ENV_CONFIG["state_size"]
        self.action_size = ENV_CONFIG["action_size"]
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=DQN_CONFIG
        )
        
        # Create temporary directory for model files
        self.temp_dir = tempfile.mkdtemp()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.epsilon, DQN_CONFIG["epsilon"])
        self.assertEqual(self.agent.gamma, DQN_CONFIG["gamma"])
    
    def test_model_architecture(self):
        """Test neural network model architecture."""
        model = self.agent.model
        
        # Check input layer
        self.assertEqual(model.layers[0].input_shape[1], self.state_size)
        
        # Check output layer
        self.assertEqual(model.layers[-1].output_shape[1], self.action_size)
        
        # Check number of layers
        expected_layers = 1 + len(DQN_CONFIG["network_architecture"]) + 1  # input + hidden + output
        self.assertEqual(len(model.layers), expected_layers)
    
    def test_target_model(self):
        """Test target network functionality."""
        # Check that target model has same architecture
        self.assertEqual(
            self.agent.model.count_params(),
            self.agent.target_model.count_params()
        )
        
        # Test target model update
        original_weights = self.agent.target_model.get_weights()
        self.agent.update_target_model()
        updated_weights = self.agent.target_model.get_weights()
        
        # Weights should be the same after update
        for orig, updated in zip(original_weights, updated_weights):
            np.testing.assert_array_equal(orig, updated)
    
    def test_remember(self):
        """Test experience replay memory."""
        state = np.random.random(self.state_size)
        action = 0
        reward = 1.0
        next_state = np.random.random(self.state_size)
        done = False
        
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
        # Check that experience was stored correctly
        stored_experience = self.agent.memory[-1]
        self.assertEqual(stored_experience[0].shape, state.shape)
        self.assertEqual(stored_experience[1], action)
        self.assertEqual(stored_experience[2], reward)
        self.assertEqual(stored_experience[3].shape, next_state.shape)
        self.assertEqual(stored_experience[4], done)
    
    def test_act_training_mode(self):
        """Test action selection in training mode."""
        state = np.random.random(self.state_size)
        
        # Test with high epsilon (should explore)
        self.agent.epsilon = 1.0
        actions = []
        for _ in range(100):
            action = self.agent.act(state, training=True)
            actions.append(action)
        
        # Should have some variety in actions due to exploration
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)
        
        # All actions should be valid
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_size)
    
    def test_act_evaluation_mode(self):
        """Test action selection in evaluation mode."""
        state = np.random.random(self.state_size)
        
        # Test without exploration
        actions = []
        for _ in range(10):
            action = self.agent.act(state, training=False)
            actions.append(action)
        
        # Should always return the same action (no exploration)
        self.assertEqual(len(set(actions)), 1)
        
        # Action should be valid
        action = actions[0]
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)
    
    def test_replay(self):
        """Test experience replay training."""
        # Add some experiences to memory
        for _ in range(self.agent.batch_size + 10):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        # Test replay
        initial_loss = self.agent.loss_history[-1] if self.agent.loss_history else 0
        loss = self.agent.replay()
        
        # Should return a loss value
        self.assertIsInstance(loss, (int, float))
        self.assertGreaterEqual(loss, 0)
        
        # Loss history should be updated
        self.assertGreater(len(self.agent.loss_history), 0)
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        initial_epsilon = self.agent.epsilon
        
        # Simulate multiple training steps
        for _ in range(10):
            # Add experiences and replay
            for _ in range(self.agent.batch_size):
                state = np.random.random(self.state_size)
                action = np.random.randint(0, self.action_size)
                reward = np.random.random()
                next_state = np.random.random(self.state_size)
                done = np.random.choice([True, False])
                self.agent.remember(state, action, reward, next_state, done)
            
            self.agent.replay()
        
        # Epsilon should have decreased
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train the model a bit first
        for _ in range(self.agent.batch_size + 10):
            state = np.random.random(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        self.agent.replay()
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.agent.save(model_path)
        
        # Create new agent and load model
        new_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=DQN_CONFIG
        )
        new_agent.load(model_path)
        
        # Test that models are the same
        original_weights = self.agent.model.get_weights()
        loaded_weights = new_agent.model.get_weights()
        
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_equal(orig, loaded)
    
    def test_get_q_values(self):
        """Test Q-value retrieval."""
        state = np.random.random(self.state_size)
        q_values = self.agent.get_q_values(state)
        
        self.assertIsInstance(q_values, np.ndarray)
        self.assertEqual(q_values.shape[0], self.action_size)
    
    def test_get_action_values(self):
        """Test action value retrieval with names."""
        state = np.random.random(self.state_size)
        action_values = self.agent.get_action_values(state)
        
        self.assertIsInstance(action_values, dict)
        self.assertEqual(len(action_values), self.action_size)
        
        expected_actions = ["Extend Green", "Next Phase", "Skip Phase", "Emergency Priority"]
        for action in expected_actions:
            self.assertIn(action, action_values)
            self.assertIsInstance(action_values[action], float)
    
    def test_get_training_stats(self):
        """Test training statistics retrieval."""
        stats = self.agent.get_training_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('loss_history', stats)
        self.assertIn('epsilon_history', stats)
        self.assertIn('training_step', stats)
        self.assertIn('memory_size', stats)
    
    def test_reset_training_stats(self):
        """Test training statistics reset."""
        # Add some training history
        self.agent.loss_history = [1.0, 2.0, 3.0]
        self.agent.epsilon_history = [1.0, 0.9, 0.8]
        self.agent.training_step = 100
        
        self.agent.reset_training_stats()
        
        self.assertEqual(len(self.agent.loss_history), 0)
        self.assertEqual(len(self.agent.epsilon_history), 0)
        self.assertEqual(self.agent.training_step, 0)
    
    def test_set_epsilon(self):
        """Test epsilon setting."""
        # Test setting valid epsilon
        self.agent.set_epsilon(0.5)
        self.assertEqual(self.agent.epsilon, 0.5)
        
        # Test setting epsilon below minimum
        self.agent.set_epsilon(0.0)
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_min)
        
        # Test setting epsilon above maximum
        self.agent.set_epsilon(2.0)
        self.assertEqual(self.agent.epsilon, 1.0)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestDQNTrainer(unittest.TestCase):
    """Test cases for DQNTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = ENV_CONFIG["state_size"]
        self.action_size = ENV_CONFIG["action_size"]
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=DQN_CONFIG
        )
        
        # Create a mock environment (simplified)
        class MockEnvironment:
            def reset(self):
                return np.random.random(self.state_size), {}
            
            def step(self, action):
                return np.random.random(self.state_size), 1.0, False, False, {'metrics': {}}
        
        self.env = MockEnvironment()
        self.trainer = DQNTrainer(self.agent, self.env)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.agent, self.agent)
        self.assertEqual(self.trainer.env, self.env)
    
    def test_train_episode(self):
        """Test single episode training."""
        stats = self.trainer.train_episode(max_steps=10)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('episode_reward', stats)
        self.assertIn('episode_length', stats)
        self.assertIn('epsilon', stats)
        self.assertIn('memory_size', stats)
        
        # Check that episode was recorded
        self.assertEqual(len(self.trainer.episode_rewards), 1)
        self.assertEqual(len(self.trainer.episode_lengths), 1)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train a bit first
        self.trainer.train_episode(max_steps=5)
        
        # Evaluate
        eval_stats = self.trainer.evaluate(episodes=2, max_steps=5)
        
        self.assertIsInstance(eval_stats, dict)
        self.assertIn('mean_reward', eval_stats)
        self.assertIn('std_reward', eval_stats)
        self.assertIn('min_reward', eval_stats)
        self.assertIn('max_reward', eval_stats)
        self.assertIn('mean_metrics', eval_stats)


if __name__ == '__main__':
    unittest.main()
