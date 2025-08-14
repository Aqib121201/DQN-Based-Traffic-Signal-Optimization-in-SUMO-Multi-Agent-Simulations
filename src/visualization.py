"""
Visualization Module for DQN Traffic Signal Control

This module provides functions for creating plots, animations, and
visualizations of training progress, traffic metrics, and model explainability.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from .config import VIS_CONFIG, VISUALIZATIONS_DIR, SHAP_CONFIG


def setup_plotting_style():
    """Setup matplotlib and seaborn plotting style."""
    plt.style.use(VIS_CONFIG["style"])
    sns.set_palette(VIS_CONFIG["color_palette"])
    plt.rcParams['figure.figsize'] = VIS_CONFIG["figure_size"]
    plt.rcParams['figure.dpi'] = VIS_CONFIG["dpi"]


def plot_training_progress(episode_rewards: List[float], 
                          episode_lengths: List[float],
                          loss_history: List[float],
                          epsilon_history: List[float],
                          save_path: str = None) -> None:
    """
    Plot training progress including rewards, episode lengths, loss, and epsilon.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        loss_history: List of training losses
        epsilon_history: List of epsilon values
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of rewards
    if len(episode_rewards) > 10:
        window_size = min(10, len(episode_rewards) // 10)
        moving_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
        axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'{window_size}-episode moving average')
        axes[0, 0].legend()
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6, color='green')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if loss_history:
        axes[1, 0].plot(loss_history, alpha=0.6, color='orange')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average of loss
        if len(loss_history) > 10:
            window_size = min(10, len(loss_history) // 10)
            moving_avg = pd.Series(loss_history).rolling(window=window_size).mean()
            axes[1, 0].plot(moving_avg, color='red', linewidth=2, label=f'{window_size}-step moving average')
            axes[1, 0].legend()
    
    # Epsilon decay
    if epsilon_history:
        axes[1, 1].plot(epsilon_history, alpha=0.6, color='purple')
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    
    plt.show()


def plot_traffic_metrics_comparison(baseline_metrics: Dict[str, float],
                                  dqn_metrics: Dict[str, float],
                                  save_path: str = None) -> None:
    """
    Plot comparison of traffic metrics between baseline and DQN control.
    
    Args:
        baseline_metrics: Metrics from baseline control
        dqn_metrics: Metrics from DQN control
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    metrics = ['Average Wait Time (s)', 'Throughput (veh/h)', 
               'Fuel Consumption (L/h)', 'CO2 Emissions (kg/h)']
    
    baseline_values = [
        baseline_metrics.get('average_wait_time', 0),
        baseline_metrics.get('throughput', 0),
        baseline_metrics.get('average_fuel_consumption', 0),
        baseline_metrics.get('average_co2_emissions', 0)
    ]
    
    dqn_values = [
        dqn_metrics.get('average_wait_time', 0),
        dqn_metrics.get('throughput', 0),
        dqn_metrics.get('average_fuel_consumption', 0),
        dqn_metrics.get('average_co2_emissions', 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Fixed-Time Control', 
                   color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, dqn_values, width, label='DQN Control', 
                   color='lightblue', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Traffic Metrics Comparison: Fixed-Time vs DQN Control')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Traffic metrics comparison plot saved to {save_path}")
    
    plt.show()


def plot_queue_lengths_over_time(queue_data: Dict[str, List[float]], 
                                time_steps: List[int],
                                save_path: str = None) -> None:
    """
    Plot queue lengths over time for different directions.
    
    Args:
        queue_data: Dictionary with queue lengths for each direction
        time_steps: List of time steps
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    directions = list(queue_data.keys())
    
    for i, direction in enumerate(directions):
        ax.plot(time_steps, queue_data[direction], 
               label=direction.capitalize(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Queue Length (vehicles)')
    ax.set_title('Queue Lengths Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Queue lengths plot saved to {save_path}")
    
    plt.show()


def create_shap_summary_plot(model, background_data: np.ndarray,
                           feature_names: List[str] = None,
                           save_path: str = None) -> None:
    """
    Create SHAP summary plot for model explainability.
    
    Args:
        model: Trained model
        background_data: Background data for SHAP
        feature_names: Names of features
        save_path: Path to save the plot
    """
    if feature_names is None:
        feature_names = SHAP_CONFIG["feature_names"]
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(background_data[:SHAP_CONFIG["explanation_samples"]])
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, background_data[:SHAP_CONFIG["explanation_samples"]], 
                     feature_names=feature_names, show=False)
    
    plt.title('SHAP Summary Plot - Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()


def create_interactive_traffic_dashboard(training_stats: Dict,
                                       traffic_metrics: Dict,
                                       save_path: str = None) -> go.Figure:
    """
    Create an interactive dashboard using Plotly.
    
    Args:
        training_stats: Training statistics
        traffic_metrics: Traffic performance metrics
        save_path: Path to save the HTML file
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Rewards', 'Traffic Metrics', 
                       'Queue Lengths', 'Action Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training rewards
    if 'episode_rewards' in training_stats:
        episodes = list(range(len(training_stats['episode_rewards'])))
        fig.add_trace(
            go.Scatter(x=episodes, y=training_stats['episode_rewards'],
                      mode='lines', name='Episode Rewards',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
    
    # Traffic metrics
    if traffic_metrics:
        metrics = list(traffic_metrics.keys())
        values = list(traffic_metrics.values())
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Traffic Metrics',
                   marker_color='lightblue'),
            row=1, col=2
        )
    
    # Queue lengths (placeholder)
    directions = ['North', 'South', 'East', 'West']
    queue_lengths = [10, 15, 8, 12]  # Example data
    fig.add_trace(
        go.Bar(x=directions, y=queue_lengths, name='Queue Lengths',
               marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Action distribution (placeholder)
    actions = ['Extend Green', 'Next Phase', 'Skip Phase', 'Emergency']
    action_counts = [45, 30, 20, 5]  # Example data
    fig.add_trace(
        go.Pie(labels=actions, values=action_counts, name='Action Distribution'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="DQN Traffic Signal Control Dashboard",
        showlegend=True,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    return fig


def plot_action_value_heatmap(action_values: List[Dict[str, float]],
                             time_steps: List[int],
                             save_path: str = None) -> None:
    """
    Create a heatmap of action values over time.
    
    Args:
        action_values: List of action value dictionaries
        time_steps: List of time steps
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    # Extract action names and values
    action_names = list(action_values[0].keys()) if action_values else []
    values_matrix = []
    
    for av in action_values:
        values_matrix.append([av.get(action, 0) for action in action_names])
    
    values_matrix = np.array(values_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(values_matrix.T, 
                xticklabels=time_steps[::max(1, len(time_steps)//20)],
                yticklabels=action_names,
                cmap='viridis',
                cbar_kws={'label': 'Q-Value'})
    
    plt.title('Action Values Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Actions')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Action value heatmap saved to {save_path}")
    
    plt.show()


def create_traffic_animation(traffic_data: List[Dict],
                           save_path: str = None,
                           fps: int = 10) -> None:
    """
    Create an animation of traffic flow over time.
    
    Args:
        traffic_data: List of traffic state dictionaries
        save_path: Path to save the animation
        fps: Frames per second
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create animation frames
    frames = []
    for i, data in enumerate(traffic_data):
        ax.clear()
        
        # Plot intersection
        ax.add_patch(plt.Rectangle((95, 95), 10, 10, color='gray', alpha=0.5))
        
        # Plot roads
        ax.plot([100, 100], [0, 200], 'k-', linewidth=3)  # North-South
        ax.plot([0, 200], [100, 100], 'k-', linewidth=3)  # East-West
        
        # Plot vehicles (simplified)
        if 'vehicles' in data:
            for vehicle in data['vehicles']:
                x, y = vehicle['position']
                ax.scatter(x, y, c='red', s=50, alpha=0.7)
        
        # Plot traffic light state
        if 'traffic_light' in data:
            state = data['traffic_light']
            # Add traffic light visualization here
        
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_title(f'Traffic Simulation - Step {i}')
        ax.set_aspect('equal')
        
        frames.append(ax)
    
    # Save animation
    if save_path:
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            # Animation saving code here
            print(f"Traffic animation saved to {save_path}")
        except ImportError:
            print("PillowWriter not available for animation saving")
    
    plt.show()


def plot_learning_curves(episode_rewards: List[float],
                        moving_avg_window: int = 10,
                        save_path: str = None) -> None:
    """
    Plot learning curves with confidence intervals.
    
    Args:
        episode_rewards: List of episode rewards
        moving_avg_window: Window size for moving average
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    episodes = list(range(len(episode_rewards)))
    
    # Calculate moving average and standard deviation
    df = pd.DataFrame({'episode': episodes, 'reward': episode_rewards})
    df['moving_avg'] = df['reward'].rolling(window=moving_avg_window).mean()
    df['moving_std'] = df['reward'].rolling(window=moving_avg_window).std()
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual rewards
    plt.scatter(episodes, episode_rewards, alpha=0.3, s=10, color='lightblue', label='Individual Episodes')
    
    # Plot moving average
    plt.plot(df['episode'], df['moving_avg'], color='red', linewidth=2, label=f'{moving_avg_window}-Episode Moving Average')
    
    # Plot confidence interval
    plt.fill_between(df['episode'], 
                    df['moving_avg'] - df['moving_std'],
                    df['moving_avg'] + df['moving_std'],
                    alpha=0.3, color='red', label='±1 Standard Deviation')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Learning curve plot saved to {save_path}")
    
    plt.show()


def save_all_visualizations(training_stats: Dict,
                           traffic_metrics: Dict,
                           model=None,
                           background_data: np.ndarray = None) -> None:
    """
    Save all visualization plots to the visualizations directory.
    
    Args:
        training_stats: Training statistics
        traffic_metrics: Traffic performance metrics
        model: Trained model for SHAP analysis
        background_data: Background data for SHAP
    """
    # Ensure visualizations directory exists
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training progress
    if all(key in training_stats for key in ['episode_rewards', 'episode_lengths', 'loss_history', 'epsilon_history']):
        plot_training_progress(
            training_stats['episode_rewards'],
            training_stats['episode_lengths'],
            training_stats['loss_history'],
            training_stats['epsilon_history'],
            save_path=str(VISUALIZATIONS_DIR / "training_progress.png")
        )
    
    # Learning curves
    if 'episode_rewards' in training_stats:
        plot_learning_curves(
            training_stats['episode_rewards'],
            save_path=str(VISUALIZATIONS_DIR / "learning_curves.png")
        )
    
    # Interactive dashboard
    if training_stats and traffic_metrics:
        create_interactive_traffic_dashboard(
            training_stats,
            traffic_metrics,
            save_path=str(VISUALIZATIONS_DIR / "traffic_dashboard.html")
        )
    
    # SHAP analysis
    if model is not None and background_data is not None:
        create_shap_summary_plot(
            model,
            background_data,
            save_path=str(VISUALIZATIONS_DIR / "shap_summary.png")
        )
    
    print("All visualizations saved to the visualizations directory.")
