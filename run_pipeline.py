#!/usr/bin/env python3
"""
DQN-Based Traffic Signal Control Pipeline

This script orchestrates the complete pipeline for training and evaluating
a DQN agent for traffic signal control in SUMO simulations.
"""

import argparse
import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.environment import TrafficEnvironment
from src.dqn_agent import DQNAgent, DQNTrainer
from src.traffic_utils import get_traffic_metrics, cleanup_sumo_files
from src.visualization import save_all_visualizations, plot_training_progress
from src.config import (
    ENV_CONFIG, DQN_CONFIG, TRAINING_CONFIG, MODEL_CONFIG,
    VISUALIZATIONS_DIR, MODELS_DIR, LOGS_DIR
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "pipeline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_dqn(episodes: int = 1000, 
              max_steps: int = 1000,
              gui: bool = False,
              save_frequency: int = 100,
              model_path: str = None) -> Dict:
    """
    Train the DQN agent for traffic signal control.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        gui: Whether to show SUMO GUI
        save_frequency: How often to save the model
        model_path: Path to save the model
        
    Returns:
        Training statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting DQN training...")
    
    # Create environment
    env = TrafficEnvironment(gui=gui, num_seconds=3600)
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=ENV_CONFIG["state_size"],
        action_size=ENV_CONFIG["action_size"],
        config=DQN_CONFIG
    )
    
    # Create trainer
    trainer = DQNTrainer(agent, env, config=TRAINING_CONFIG)
    
    # Set model path
    if model_path is None:
        model_path = str(MODELS_DIR / f"{MODEL_CONFIG['model_name']}.pkl")
    
    logger.info(f"Training for {episodes} episodes...")
    logger.info(f"Model will be saved to: {model_path}")
    
    # Train the agent
    training_stats = trainer.train(
        episodes=episodes,
        max_steps=max_steps,
        save_frequency=save_frequency,
        model_path=model_path
    )
    
    logger.info("Training completed!")
    logger.info(f"Final average reward: {np.mean(training_stats['episode_rewards'][-10:]):.2f}")
    
    return training_stats


def evaluate_model(model_path: str, 
                  episodes: int = 10,
                  max_steps: int = 1000,
                  gui: bool = False) -> Dict:
    """
    Evaluate a trained DQN model.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        gui: Whether to show SUMO GUI
        
    Returns:
        Evaluation statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    # Create environment
    env = TrafficEnvironment(gui=gui, num_seconds=3600)
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=ENV_CONFIG["state_size"],
        action_size=ENV_CONFIG["action_size"],
        config=DQN_CONFIG
    )
    
    # Load trained model
    agent.load(model_path)
    logger.info(f"Model loaded from: {model_path}")
    
    # Create trainer for evaluation
    trainer = DQNTrainer(agent, env)
    
    # Evaluate the model
    evaluation_stats = trainer.evaluate(episodes=episodes, max_steps=max_steps)
    
    logger.info("Evaluation completed!")
    logger.info(f"Mean reward: {evaluation_stats['mean_reward']:.2f}")
    logger.info(f"Std reward: {evaluation_stats['std_reward']:.2f}")
    
    return evaluation_stats


def run_baseline_comparison(episodes: int = 10,
                           max_steps: int = 1000,
                           gui: bool = False) -> Dict:
    """
    Run baseline comparison with fixed-time traffic control.
    
    Args:
        episodes: Number of episodes for baseline
        max_steps: Maximum steps per episode
        gui: Whether to show SUMO GUI
        
    Returns:
        Baseline statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Running baseline comparison...")
    
    # Create environment
    env = TrafficEnvironment(gui=gui, num_seconds=3600)
    
    baseline_metrics = []
    
    for episode in range(episodes):
        logger.info(f"Running baseline episode {episode + 1}/{episodes}")
        
        state, info = env.reset()
        episode_metrics = []
        
        for step in range(max_steps):
            # Use fixed-time control (action 1: switch to next phase)
            action = 1 if step % 30 == 0 else 0  # Switch every 30 steps
            
            state, reward, done, truncated, info = env.step(action)
            episode_metrics.append(info.get('metrics', {}))
            
            if done or truncated:
                break
        
        # Aggregate episode metrics
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics]
                avg_metrics[f"mean_{key}"] = np.mean(values)
            baseline_metrics.append(avg_metrics)
    
    # Aggregate across episodes
    if baseline_metrics:
        final_metrics = {}
        for key in baseline_metrics[0].keys():
            values = [m.get(key, 0) for m in baseline_metrics]
            final_metrics[key] = np.mean(values)
    else:
        final_metrics = {}
    
    logger.info("Baseline comparison completed!")
    
    return final_metrics


def create_visualizations(training_stats: Dict,
                         evaluation_stats: Dict,
                         baseline_metrics: Dict,
                         model_path: str = None) -> None:
    """
    Create and save all visualizations.
    
    Args:
        training_stats: Training statistics
        evaluation_stats: Evaluation statistics
        baseline_metrics: Baseline comparison metrics
        model_path: Path to trained model for SHAP analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations...")
    
    # Ensure visualizations directory exists
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training progress plot
    if training_stats:
        plot_training_progress(
            training_stats.get('episode_rewards', []),
            training_stats.get('episode_lengths', []),
            training_stats.get('loss_history', []),
            training_stats.get('epsilon_history', []),
            save_path=str(VISUALIZATIONS_DIR / "training_progress.png")
        )
    
    # Save all visualizations
    save_all_visualizations(
        training_stats=training_stats,
        traffic_metrics=evaluation_stats.get('mean_metrics', {}),
        model=None,  # SHAP analysis can be added later
        background_data=None
    )
    
    logger.info("Visualizations created and saved!")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="DQN Traffic Signal Control Pipeline")
    parser.add_argument("--mode", choices=["train", "evaluate", "baseline", "full"], 
                       default="full", help="Pipeline mode")
    parser.add_argument("--episodes", type=int, default=1000, 
                       help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=1000, 
                       help="Maximum steps per episode")
    parser.add_argument("--gui", action="store_true", 
                       help="Show SUMO GUI")
    parser.add_argument("--model_path", type=str, 
                       help="Path to model file")
    parser.add_argument("--save_frequency", type=int, default=100, 
                       help="Model save frequency")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting DQN Traffic Signal Control Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"GUI: {args.gui}")
    
    try:
        if args.mode == "train":
            # Training only
            training_stats = train_dqn(
                episodes=args.episodes,
                max_steps=args.max_steps,
                gui=args.gui,
                save_frequency=args.save_frequency,
                model_path=args.model_path
            )
            
            # Create visualizations
            create_visualizations(training_stats, {}, {})
            
        elif args.mode == "evaluate":
            # Evaluation only
            if not args.model_path:
                logger.error("Model path required for evaluation mode")
                return
            
            evaluation_stats = evaluate_model(
                model_path=args.model_path,
                episodes=args.episodes,
                max_steps=args.max_steps,
                gui=args.gui
            )
            
            logger.info("Evaluation results:")
            for key, value in evaluation_stats.items():
                logger.info(f"  {key}: {value}")
                
        elif args.mode == "baseline":
            # Baseline comparison only
            baseline_metrics = run_baseline_comparison(
                episodes=args.episodes,
                max_steps=args.max_steps,
                gui=args.gui
            )
            
            logger.info("Baseline results:")
            for key, value in baseline_metrics.items():
                logger.info(f"  {key}: {value}")
                
        elif args.mode == "full":
            # Full pipeline: train, evaluate, and compare
            logger.info("Running full pipeline...")
            
            # 1. Train the model
            training_stats = train_dqn(
                episodes=args.episodes,
                max_steps=args.max_steps,
                gui=args.gui,
                save_frequency=args.save_frequency,
                model_path=args.model_path
            )
            
            # 2. Evaluate the trained model
            model_path = args.model_path or str(MODELS_DIR / f"{MODEL_CONFIG['model_name']}.pkl")
            evaluation_stats = evaluate_model(
                model_path=model_path,
                episodes=10,  # Fewer episodes for evaluation
                max_steps=args.max_steps,
                gui=args.gui
            )
            
            # 3. Run baseline comparison
            baseline_metrics = run_baseline_comparison(
                episodes=10,  # Fewer episodes for baseline
                max_steps=args.max_steps,
                gui=args.gui
            )
            
            # 4. Create visualizations
            create_visualizations(training_stats, evaluation_stats, baseline_metrics, model_path)
            
            # 5. Print summary
            logger.info("\n" + "="*50)
            logger.info("PIPELINE SUMMARY")
            logger.info("="*50)
            logger.info(f"Training episodes: {args.episodes}")
            logger.info(f"Final average reward: {np.mean(training_stats['episode_rewards'][-10:]):.2f}")
            logger.info(f"Evaluation mean reward: {evaluation_stats['mean_reward']:.2f}")
            logger.info(f"Baseline metrics: {baseline_metrics}")
            logger.info("="*50)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_sumo_files()


if __name__ == "__main__":
    main()
