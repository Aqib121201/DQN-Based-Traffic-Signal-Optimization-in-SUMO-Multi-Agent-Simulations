"""
Utility functions for the Streamlit web application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import json
import os
from pathlib import Path


def load_model_info(model_path: str) -> Dict:
    """Load model information from file."""
    try:
        with open(model_path, 'rb') as f:
            import pickle
            data = pickle.load(f)
            return {
                'config': data.get('config', {}),
                'training_step': data.get('training_step', 0),
                'epsilon': data.get('epsilon', 0),
                'memory_size': len(data.get('memory', [])),
                'loss_history': data.get('loss_history', []),
                'epsilon_history': data.get('epsilon_history', [])
            }
    except Exception as e:
        st.error(f"Failed to load model info: {e}")
        return {}


def create_performance_comparison_chart(baseline_metrics: Dict, dqn_metrics: Dict) -> go.Figure:
    """Create a performance comparison chart."""
    metrics = ['Average Wait Time', 'Throughput', 'Fuel Consumption', 'CO2 Emissions']
    
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fixed-Time Control',
        x=metrics,
        y=baseline_values,
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        name='DQN Control',
        x=metrics,
        y=dqn_values,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Performance Comparison: Fixed-Time vs DQN Control',
        barmode='group',
        xaxis_title='Metrics',
        yaxis_title='Values'
    )
    
    return fig


def create_training_progress_chart(training_stats: Dict) -> go.Figure:
    """Create training progress visualization."""
    if not training_stats.get('episode_rewards'):
        return go.Figure()
    
    episodes = list(range(len(training_stats['episode_rewards'])))
    
    fig = go.Figure()
    
    # Episode rewards
    fig.add_trace(go.Scatter(
        x=episodes,
        y=training_stats['episode_rewards'],
        mode='lines',
        name='Episode Rewards',
        line=dict(color='blue', width=1)
    ))
    
    # Moving average
    if len(episodes) > 10:
        window_size = min(10, len(episodes) // 10)
        moving_avg = pd.Series(training_stats['episode_rewards']).rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=episodes,
            y=moving_avg,
            mode='lines',
            name=f'{window_size}-Episode Moving Average',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title='Training Progress - Episode Rewards',
        xaxis_title='Episode',
        yaxis_title='Reward',
        showlegend=True
    )
    
    return fig


def create_metrics_dashboard(metrics_history: List[Dict]) -> go.Figure:
    """Create a comprehensive metrics dashboard."""
    if not metrics_history:
        return go.Figure()
    
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


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric values for display."""
    if 'time' in metric_name.lower():
        return f"{value:.1f} s"
    elif 'consumption' in metric_name.lower():
        return f"{value:.2f} L"
    elif 'emissions' in metric_name.lower():
        return f"{value:.1f} kg"
    elif 'throughput' in metric_name.lower():
        return f"{value:.0f} veh/h"
    else:
        return f"{value:.1f}"


def calculate_improvement(baseline: float, dqn: float) -> float:
    """Calculate improvement percentage."""
    if baseline == 0:
        return 0
    return ((baseline - dqn) / baseline) * 100


def create_improvement_summary(baseline_metrics: Dict, dqn_metrics: Dict) -> pd.DataFrame:
    """Create improvement summary table."""
    metrics = {
        'Average Wait Time': ('average_wait_time', 's'),
        'Throughput': ('throughput', 'veh/h'),
        'Fuel Consumption': ('average_fuel_consumption', 'L'),
        'CO2 Emissions': ('average_co2_emissions', 'kg')
    }
    
    data = []
    for metric_name, (key, unit) in metrics.items():
        baseline_val = baseline_metrics.get(key, 0)
        dqn_val = dqn_metrics.get(key, 0)
        improvement = calculate_improvement(baseline_val, dqn_val)
        
        data.append({
            'Metric': metric_name,
            'Baseline': f"{baseline_val:.1f} {unit}",
            'DQN': f"{dqn_val:.1f} {unit}",
            'Improvement': f"{improvement:.1f}%"
        })
    
    return pd.DataFrame(data)


def save_simulation_results(results: Dict, filename: str = None) -> str:
    """Save simulation results to file."""
    if filename is None:
        filename = f"simulation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            serializable_results[key] = [v.tolist() for v in value]
        else:
            serializable_results[key] = value
    
    filepath = Path("data/processed") / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return str(filepath)


def load_simulation_results(filepath: str) -> Dict:
    """Load simulation results from file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        if 'state_history' in data:
            data['state_history'] = [np.array(state) for state in data['state_history']]
        
        return data
    except Exception as e:
        st.error(f"Failed to load simulation results: {e}")
        return {}


def get_available_models() -> List[str]:
    """Get list of available trained models."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5"))
    return [str(f) for f in model_files]


def get_available_results() -> List[str]:
    """Get list of available simulation results."""
    results_dir = Path("data/processed")
    if not results_dir.exists():
        return []
    
    result_files = list(results_dir.glob("simulation_results_*.json"))
    return [str(f) for f in result_files]


def create_state_analysis_chart(state_history: List[np.ndarray]) -> go.Figure:
    """Create state analysis visualization."""
    if not state_history:
        return go.Figure()
    
    states = np.array(state_history)
    
    # Feature names for the first 8 state components (queue lengths and wait times)
    feature_names = [
        'Queue North', 'Queue South', 'Queue East', 'Queue West',
        'Wait North', 'Wait South', 'Wait East', 'Wait West'
    ]
    
    fig = go.Figure()
    
    for i, name in enumerate(feature_names[:8]):
        fig.add_trace(go.Scatter(
            y=states[:, i],
            name=name,
            mode='lines'
        ))
    
    fig.update_layout(
        title='State Components Over Time',
        xaxis_title='Time Step',
        yaxis_title='Normalized Value',
        showlegend=True
    )
    
    return fig
