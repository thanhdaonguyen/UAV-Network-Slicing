import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import matplotlib.pyplot as plt

def load_metrics_from_csv(metrics_file_path: str) -> Dict[str, List[float]]:
    """Load training metrics from CSV file"""
    import csv
    metrics = {}
    
    try:
        with open(metrics_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    if key not in metrics:
                        metrics[key] = []
                    try:
                        metrics[key].append(float(value))
                    except ValueError:
                        metrics[key].append(value)
        print(f"✓ Loaded metrics from {metrics_file_path}")
    except FileNotFoundError:
        print(f"❌ Metrics file not found: {metrics_file_path}")
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
    
    return metrics

def load_metrics_from_csv_list(metrics_file_paths: List[str], 
                              continuous_steps: bool = True) -> Dict[str, List[float]]:
    """Load and concatenate metrics from multiple CSV files"""
    merged_metrics = {}
    step_offset = 0
    
    for file_path in metrics_file_paths:
        file_metrics = load_metrics_from_csv(file_path)
        
        if not file_metrics:
            continue
            
        for key, values in file_metrics.items():
            if key not in merged_metrics:
                merged_metrics[key] = []
            
            # Adjust step count if needed
            if key == 'steps' and continuous_steps and merged_metrics[key]:
                step_offset = merged_metrics['steps'][-1]
                adjusted_values = [v + step_offset for v in values]
                merged_metrics[key].extend(adjusted_values)
            else:
                merged_metrics[key].extend(values)
    
    return merged_metrics

def moving_average(data: List[float], window_size: int) -> List[float]:
    """Compute moving average for smoothing"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid').tolist()

def plot_training_progress(step_metrics_paths: Union[str, List[str]], 
                          training_metrics_paths: Union[str, List[str], None] = None,
                          save_path: Optional[str] = None):
    """
    Plot comprehensive training progress from step and training metrics files
    
    Args:
        step_metrics_paths: Path(s) to step_metrics.csv file(s)
        training_metrics_paths: Path(s) to training_metrics.csv file(s) (optional)
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    # Handle single file or list of files
    if isinstance(step_metrics_paths, str):
        step_metrics_paths = [step_metrics_paths]
    
    if isinstance(training_metrics_paths, str):
        training_metrics_paths = [training_metrics_paths]
    
    # Load step metrics
    step_metrics = load_metrics_from_csv_list(step_metrics_paths)
    
    # Load training metrics if provided
    training_metrics = {}
    if training_metrics_paths:
        training_metrics = load_metrics_from_csv_list(training_metrics_paths)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle(f'Training Progress', fontsize=16)
    
    steps = step_metrics.get('steps', [])
    
    if not steps:
        print("No step data found!")
        return
    
    # Plot 1: Reward progression
    if 'rewards' in step_metrics:
        axes[0, 0].plot(steps, step_metrics['rewards'], alpha=0.3, color='blue')
        if len(steps) > 100:
            smoothed = moving_average(step_metrics['rewards'], 1000)
            axes[0, 0].plot(steps[-len(smoothed):], smoothed, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: QoS satisfaction
    if 'qos_satisfaction' in step_metrics:
        axes[0, 1].plot(steps, step_metrics['qos_satisfaction'], 'g-', alpha=0.6)
        if len(steps) > 100:
            smoothed = moving_average(step_metrics['qos_satisfaction'], 1000)
            axes[0, 1].plot(steps[-len(smoothed):], smoothed, 'darkgreen', linewidth=2)
        axes[0, 1].set_title('QoS Satisfaction')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('QoS Satisfaction')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Energy efficiency
    if 'energy_efficiency' in step_metrics:
        axes[0, 2].plot(steps, step_metrics['energy_efficiency'], 'orange', alpha=0.6)
        if len(steps) > 100:
            smoothed = moving_average(step_metrics['energy_efficiency'], 1000)
            axes[0, 2].plot(steps[-len(smoothed):], smoothed, 'darkorange', linewidth=2)
        axes[0, 2].set_title('Energy Consumption Rate')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Energy Consumption Rate')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Active UEs
    if 'active_ues' in step_metrics:
        axes[1, 0].plot(steps, step_metrics['active_ues'], 'purple', alpha=0.6)
        if len(steps) > 100:
            smoothed = moving_average(step_metrics['active_ues'], 1000)
            axes[1, 0].plot(steps[-len(smoothed):], smoothed, 'darkviolet', linewidth=2)
        axes[1, 0].set_title('Active UEs')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Number of UEs')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Actor losses (from training_metrics if available)
    if training_metrics and 'actor_losses' in training_metrics:
        training_steps = training_metrics.get('steps', [])
        axes[1, 1].plot(training_steps, training_metrics['actor_losses'], 
                       'red', alpha=0.6, marker='o', markersize=2)
        axes[1, 1].set_title('Actor Losses')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No training metrics available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Actor Losses')
    
    # Plot 6: Critic losses (from training_metrics if available)
    if training_metrics and 'critic_losses' in training_metrics:
        training_steps = training_metrics.get('steps', [])
        axes[1, 2].plot(training_steps, training_metrics['critic_losses'], 
                       'blue', alpha=0.6, marker='o', markersize=2)
        axes[1, 2].set_title('Critic Losses')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No training metrics available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Critic Losses')
    
    # Plot 7: Fairness Level
    if 'fairness_level' in step_metrics:
        axes[2, 0].plot(steps, step_metrics['fairness_level'], 'brown', alpha=0.6)
        if len(steps) > 100:
            smoothed = moving_average(step_metrics['fairness_level'], 1000)
            axes[2, 0].plot(steps[-len(smoothed):], smoothed, 'saddlebrown', linewidth=2)
        axes[2, 0].set_title('Average Fairness Level')
        axes[2, 0].set_xlabel('Steps')
        axes[2, 0].set_ylabel('Fairness Level')
        axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Exploration Noise
    if 'noise' in step_metrics:
        axes[2, 1].plot(steps, step_metrics['noise'], 'cyan', alpha=0.6)
        axes[2, 1].set_title('Exploration Noise')
        axes[2, 1].set_xlabel('Steps')
        axes[2, 1].set_ylabel('Noise Level')
        axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Power usage of agents
    agent_power_found = False
    for i in range(10):  # Check up to 10 agents
        key = f'agent_{i}_power'
        if key in step_metrics:
            axes[2, 2].plot(steps, step_metrics[key], alpha=0.6, label=f'Agent {i}')
            agent_power_found = True
    
    if agent_power_found:
        axes[2, 2].set_title('Power Usage of Agents')
        axes[2, 2].set_xlabel('Steps')
        axes[2, 2].set_ylabel('Power (normalized)')
        axes[2, 2].legend(loc='best')
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'No agent power data available', 
                       ha='center', va='center', transform=axes[2, 2].transAxes)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Example usage of the plotting functions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training progress from CSV metrics')
    parser.add_argument('--step-metrics', nargs='+', required=True,
                       help='Path(s) to step_metrics.csv file(s)')
    parser.add_argument('--training-metrics', nargs='+', default=None,
                       help='Path(s) to training_metrics.csv file(s)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the plot')
    parser.add_argument('--continuous', action='store_true',
                       help='Make step counts continuous across multiple files')
    
    args = parser.parse_args()
    
    # Plot the metrics
    plot_training_progress(
        step_metrics_paths=args.step_metrics,
        training_metrics_paths=args.training_metrics,
        save_path=args.save
    )

if __name__ == "__main__":
    # Example usage - modify these paths as needed
    
    # Single model example
    # plot_training_progress(
    #     step_metrics_paths="saved_models/model47/step_metrics.csv",
    #     training_metrics_paths="saved_models/model47/training_metrics.csv"
    # )
    
    # Multiple models example
    plot_training_progress(
        step_metrics_paths=[
            "saved_models/model86/step_metrics.csv"
        ],
        training_metrics_paths=[
            "saved_models/model86/training_metrics.csv"
        ]
    )
    
    # Or use command line:
    # main()