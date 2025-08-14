"""
Streamlit Web Application for DQN Traffic Signal Control

This application provides an interactive interface for visualizing and
controlling the DQN-based traffic signal control system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import pickle
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.environment import TrafficEnvironment
from src.dqn_agent import DQNAgent
from src.traffic_utils import get_traffic_metrics
from src.visualization import create_interactive_traffic_dashboard
from src.config import ENV_CONFIG, DQN_CONFIG, MODEL_CONFIG, VISUALIZATIONS_DIR, MODELS_DIR


def load_trained_model(model_path: str) -> DQNAgent:
    """Load a trained DQN model."""
    try:
        agent = DQNAgent(
            state_size=ENV_CONFIG["state_size"],
            action_size=ENV_CONFIG["action_size"],
            config=DQN_CONFIG
        )
        agent.load(model_path)
        return agent
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def run_simulation(agent: DQNAgent, 
                  env: TrafficEnvironment,
                  steps: int = 100) -> Dict:
    """Run a simulation and collect metrics."""
    state, info = env.reset()
    metrics_history = []
    action_history = []
    state_history = []
    
    for step in range(steps):
        # Get action from agent
        action = agent.act(state, training=False)
        
        # Take action
        state, reward, done, truncated, info = env.step(action)
        
        # Collect data
        metrics_history.append(info.get('metrics', {}))
        action_history.append(action)
        state_history.append(state)
        
        if done or truncated:
            break
    
    return {
        'metrics_history': metrics_history,
        'action_history': action_history,
        'state_history': state_history,
        'total_steps': len(metrics_history)
    }


def create_metrics_chart(metrics_history: List[Dict]) -> go.Figure:
    """Create a chart showing metrics over time."""
    if not metrics_history:
        return go.Figure()
    
    # Extract metrics
    df = pd.DataFrame(metrics_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Wait Time', 'Total Vehicles', 
                       'Fuel Consumption', 'Phase Changes'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Total wait time
    if 'total_wait_time' in df.columns:
        fig.add_trace(
            go.Scatter(y=df['total_wait_time'], name='Wait Time',
                      line=dict(color='red')),
            row=1, col=1
        )
    
    # Total vehicles
    if 'total_vehicles' in df.columns:
        fig.add_trace(
            go.Scatter(y=df['total_vehicles'], name='Vehicles',
                      line=dict(color='blue')),
            row=1, col=2
        )
    
    # Fuel consumption
    if 'total_fuel_consumption' in df.columns:
        fig.add_trace(
            go.Scatter(y=df['total_fuel_consumption'], name='Fuel',
                      line=dict(color='green')),
            row=2, col=1
        )
    
    # Phase changes
    if 'phase_changes' in df.columns:
        fig.add_trace(
            go.Scatter(y=df['phase_changes'], name='Phase Changes',
                      line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True)
    return fig


def create_action_distribution_chart(action_history: List[int]) -> go.Figure:
    """Create a chart showing action distribution."""
    if not action_history:
        return go.Figure()
    
    action_names = ['Extend Green', 'Next Phase', 'Skip Phase', 'Emergency']
    action_counts = [action_history.count(i) for i in range(4)]
    
    fig = go.Figure(data=[go.Pie(labels=action_names, values=action_counts)])
    fig.update_layout(title="Action Distribution")
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DQN Traffic Signal Control",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚦 DQN-Based Traffic Signal Control")
    st.markdown("Interactive visualization and control of reinforcement learning-based traffic signal optimization")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_files = list(MODELS_DIR.glob("*.pkl")) if MODELS_DIR.exists() else []
    model_path = st.sidebar.selectbox(
        "Select Trained Model",
        options=[str(f) for f in model_files],
        index=0 if model_files else None,
        help="Choose a trained DQN model to load"
    )
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Parameters")
    simulation_steps = st.sidebar.slider("Simulation Steps", 50, 500, 100)
    show_gui = st.sidebar.checkbox("Show SUMO GUI", value=False)
    
    # Load model
    agent = None
    if model_path and st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            agent = load_trained_model(model_path)
            if agent:
                st.sidebar.success("Model loaded successfully!")
    
    # Main content
    if agent is None:
        st.warning("Please load a trained model to start the simulation.")
        st.info("""
        ### How to use this application:
        1. **Load a Model**: Select a trained DQN model from the sidebar
        2. **Configure Simulation**: Set simulation parameters
        3. **Run Simulation**: Use the controls below to run traffic simulations
        4. **Analyze Results**: View metrics and visualizations
        """)
        return
    
    # Simulation controls
    st.header("Simulation Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚗 Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                # Create environment
                env = TrafficEnvironment(gui=show_gui, num_seconds=3600)
                
                # Run simulation
                results = run_simulation(agent, env, simulation_steps)
                
                # Store results in session state
                st.session_state.simulation_results = results
                st.success("Simulation completed!")
    
    with col2:
        if st.button("📊 Show Model Info"):
            st.session_state.show_model_info = True
    
    with col3:
        if st.button("🧹 Clear Results"):
            if 'simulation_results' in st.session_state:
                del st.session_state.simulation_results
            if 'show_model_info' in st.session_state:
                del st.session_state.show_model_info
            st.success("Results cleared!")
    
    # Model information
    if st.session_state.get('show_model_info', False):
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.code(agent.get_model_summary())
        
        with col2:
            st.subheader("Training Statistics")
            stats = agent.get_training_stats()
            
            stats_df = pd.DataFrame({
                'Metric': ['Training Steps', 'Memory Size', 'Current Epsilon'],
                'Value': [
                    stats.get('training_step', 0),
                    stats.get('memory_size', 0),
                    f"{stats.get('epsilon_history', [1.0])[-1]:.3f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    # Simulation results
    if 'simulation_results' in st.session_state:
        results = st.session_state.simulation_results
        
        st.header("Simulation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_wait_time = sum(m.get('total_wait_time', 0) for m in results['metrics_history'])
            st.metric("Total Wait Time", f"{total_wait_time:.1f} s")
        
        with col2:
            avg_vehicles = np.mean([m.get('total_vehicles', 0) for m in results['metrics_history']])
            st.metric("Avg Vehicles", f"{avg_vehicles:.1f}")
        
        with col3:
            total_fuel = sum(m.get('total_fuel_consumption', 0) for m in results['metrics_history'])
            st.metric("Total Fuel", f"{total_fuel:.2f} L")
        
        with col4:
            phase_changes = sum(m.get('phase_changes', 0) for m in results['metrics_history'])
            st.metric("Phase Changes", phase_changes)
        
        # Charts
        st.subheader("Metrics Over Time")
        metrics_fig = create_metrics_chart(results['metrics_history'])
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Action distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Action Distribution")
            action_fig = create_action_distribution_chart(results['action_history'])
            st.plotly_chart(action_fig, use_container_width=True)
        
        with col2:
            st.subheader("State Analysis")
            if results['state_history']:
                # Create state analysis chart
                states = np.array(results['state_history'])
                
                # Plot queue lengths (first 4 state values)
                queue_fig = go.Figure()
                directions = ['North', 'South', 'East', 'West']
                for i, direction in enumerate(directions):
                    queue_fig.add_trace(
                        go.Scatter(y=states[:, i], name=f'{direction} Queue',
                                 mode='lines')
                    )
                
                queue_fig.update_layout(
                    title="Queue Lengths Over Time",
                    xaxis_title="Time Step",
                    yaxis_title="Queue Length (normalized)"
                )
                st.plotly_chart(queue_fig, use_container_width=True)
        
        # Raw data
        with st.expander("View Raw Data"):
            st.subheader("Metrics History")
            metrics_df = pd.DataFrame(results['metrics_history'])
            st.dataframe(metrics_df)
            
            st.subheader("Action History")
            action_df = pd.DataFrame({
                'Step': range(len(results['action_history'])),
                'Action': results['action_history']
            })
            st.dataframe(action_df)
    
    # Model comparison
    st.header("Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        
        # Example metrics (replace with actual data)
        metrics_data = {
            'Metric': ['Average Wait Time', 'Throughput', 'Fuel Consumption', 'CO2 Emissions'],
            'Fixed-Time': [45.2, 1250, 85.3, 198.7],
            'DQN Control': [28.7, 1680, 72.1, 167.9],
            'Improvement %': [36.5, 34.4, 15.5, 15.5]
        }
        
        comparison_df = pd.DataFrame(metrics_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.subheader("Improvement Visualization")
        
        # Create improvement chart
        improvement_fig = go.Figure()
        
        improvement_fig.add_trace(go.Bar(
            x=metrics_data['Metric'],
            y=metrics_data['Improvement %'],
            name='Improvement %',
            marker_color='lightgreen'
        ))
        
        improvement_fig.update_layout(
            title="Performance Improvement with DQN",
            yaxis_title="Improvement (%)",
            showlegend=False
        )
        
        st.plotly_chart(improvement_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About this Application
    
    This application demonstrates the effectiveness of Deep Q-Network (DQN) 
    reinforcement learning for adaptive traffic signal control. The system 
    optimizes traffic flow by dynamically adjusting signal timings based on 
    real-time traffic conditions.
    
    **Key Features:**
    - 🧠 Deep Q-Network with experience replay
    - 🚦 Multi-phase traffic signal control
    - 📊 Real-time traffic metrics monitoring
    - 📈 Performance comparison with baseline methods
    - 🎯 Multi-objective optimization (throughput, wait time, fuel consumption)
    """)
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Quick Actions
    
    - **Load Model**: Select and load a trained DQN model
    - **Run Simulation**: Execute traffic simulation with current model
    - **View Results**: Analyze simulation metrics and performance
    - **Compare Performance**: See improvements over baseline methods
    """)
    
    st.sidebar.markdown("**Version:** 1.0.0")
    st.sidebar.markdown("**Last Updated:** December 2024")


if __name__ == "__main__":
    main()
