"""
Deep Q-Network (DQN) Agent for Traffic Signal Control

This module implements a DQN agent with experience replay, target network,
and epsilon-greedy exploration for traffic signal control optimization.
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
from typing import List, Tuple, Dict, Optional
import pickle
import os

from .config import DQN_CONFIG, MODEL_CONFIG


class DQNAgent:
    """
    Deep Q-Network agent for traffic signal control.
    
    This agent learns optimal traffic signal control policies using
    deep reinforcement learning with experience replay and target networks.
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 config: Dict = None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            config: Agent configuration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or DQN_CONFIG.copy()
        
        # Experience replay memory
        self.memory = deque(maxlen=self.config["memory_size"])
        
        # Hyperparameters
        self.gamma = self.config["gamma"]  # Discount factor
        self.epsilon = self.config["epsilon"]  # Exploration rate
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.learning_rate = self.config["learning_rate"]
        self.batch_size = self.config["batch_size"]
        self.target_update_freq = self.config["target_update_freq"]
        
        # Neural networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training statistics
        self.training_step = 0
        self.loss_history = []
        self.epsilon_history = []
        
    def _build_model(self) -> keras.Model:
        """
        Build the neural network model.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.config["network_architecture"][0],
            input_shape=(self.state_size,),
            activation=self.config["activation"],
            name="dense_1"
        ))
        
        # Hidden layers
        for i, units in enumerate(self.config["network_architecture"][1:], 2):
            model.add(layers.Dense(
                units,
                activation=self.config["activation"],
                name=f"dense_{i}"
            ))
        
        # Output layer
        model.add(layers.Dense(
            self.action_size,
            activation='linear',
            name="output"
        ))
        
        # Compile model
        if self.config["optimizer"] == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def update_target_model(self):
        """Update the target network with weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action to take
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int = None) -> float:
        """
        Train the model on a batch of experiences.
        
        Args:
            batch_size: Size of training batch
            
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch_size = batch_size or self.batch_size
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Current Q-values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Target Q-values
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(
            states, target_q_values,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.epsilon_history.append(self.epsilon)
        
        return loss
    
    def load(self, filepath: str):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to model file
        """
        if os.path.exists(filepath):
            if filepath.endswith('.h5'):
                self.model.load_weights(filepath)
                self.target_model.load_weights(filepath)
            elif filepath.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.model.set_weights(data['model_weights'])
                    self.target_model.set_weights(data['model_weights'])
                    self.epsilon = data.get('epsilon', self.epsilon)
                    self.memory = data.get('memory', self.memory)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")
    
    def save(self, filepath: str):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filepath.endswith('.h5'):
            self.model.save_weights(filepath)
        elif filepath.endswith('.pkl'):
            data = {
                'model_weights': self.model.get_weights(),
                'epsilon': self.epsilon,
                'memory': self.memory,
                'config': self.config,
                'training_step': self.training_step,
                'loss_history': self.loss_history,
                'epsilon_history': self.epsilon_history
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a given state.
        
        Args:
            state: Input state
            
        Returns:
            Q-values for all actions
        """
        return self.model.predict(state.reshape(1, -1), verbose=0)[0]
    
    def get_action_values(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get Q-values for all actions in a readable format.
        
        Args:
            state: Input state
            
        Returns:
            Dictionary mapping action names to Q-values
        """
        q_values = self.get_q_values(state)
        action_names = {
            0: "Extend Green",
            1: "Next Phase", 
            2: "Skip Phase",
            3: "Emergency Priority"
        }
        
        return {action_names[i]: float(q_values[i]) for i in range(self.action_size)}
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }
    
    def reset_training_stats(self):
        """Reset training statistics."""
        self.loss_history = []
        self.epsilon_history = []
        self.training_step = 0
    
    def set_epsilon(self, epsilon: float):
        """
        Set exploration rate.
        
        Args:
            epsilon: New exploration rate
        """
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            Model summary as string
        """
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)


class DQNTrainer:
    """
    Trainer class for DQN agent with additional utilities.
    """
    
    def __init__(self, agent: DQNAgent, env, config: Dict = None):
        """
        Initialize the trainer.
        
        Args:
            agent: DQN agent to train
            env: Training environment
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config or {}
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        
    def train_episode(self, max_steps: int = 1000) -> Dict[str, float]:
        """
        Train for one episode.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        state, info = self.env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # Choose action
            action = self.agent.act(state, training=True)
            
            # Take action
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_metrics.append(info.get('metrics', {}))
        
        return {
            'episode_reward': total_reward,
            'episode_length': steps,
            'epsilon': self.agent.epsilon,
            'memory_size': len(self.agent.memory)
        }
    
    def train(self, episodes: int, max_steps: int = 1000, 
              save_frequency: int = 100, model_path: str = None) -> Dict[str, List]:
        """
        Train the agent for multiple episodes.
        
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            save_frequency: How often to save the model
            model_path: Path to save model
            
        Returns:
            Training statistics
        """
        model_path = model_path or f"{MODEL_CONFIG['model_name']}.pkl"
        
        for episode in range(episodes):
            stats = self.train_episode(max_steps)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
            
            if (episode + 1) % save_frequency == 0:
                self.agent.save(model_path)
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_metrics': self.episode_metrics,
            'loss_history': self.agent.loss_history,
            'epsilon_history': self.agent.epsilon_history
        }
    
    def evaluate(self, episodes: int = 10, max_steps: int = 1000) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation statistics
        """
        evaluation_rewards = []
        evaluation_metrics = []
        
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                action = self.agent.act(state, training=False)
                state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            evaluation_rewards.append(total_reward)
            evaluation_metrics.append(info.get('metrics', {}))
        
        return {
            'mean_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards),
            'min_reward': np.min(evaluation_rewards),
            'max_reward': np.max(evaluation_rewards),
            'mean_metrics': self._aggregate_metrics(evaluation_metrics)
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across episodes."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0) for m in metrics_list]
            aggregated[f"mean_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
        
        return aggregated
