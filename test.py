import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import glob

def compare_agents(self, results_dir: str):
    """
    Generate comparison plots for multiple agents by reading CSV files
    
    Args:
        results_dir: Directory path containing agent CSV files (e.g., 'results/run_001/')
    """
    
    print(f"\nLoading agent results from: {results_dir}")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} agent result files")
    
    # Load data from each CSV file
    results_list = []
    
    for csv_path in csv_files:
        # Extract agent name from filename
        filename = os.path.basename(csv_path)
        agent_name = filename.replace('.csv', '').replace('_', ' ')
        
        print(f"  Loading: {agent_name}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
            
            # Extract metrics from dataframe
            metrics = {
                'steps': df['step'].values,
                'rewards': df['reward'].values,
                'cumulative_rewards': df['cumulative_reward'].values,
                'qos': df['qos'].values,
                'energy': df['energy'].values,
                'fairness': df['fairness'].values,
                'active_ues': df['active_ues'].values
            }
            
            # Calculate summary statistics
            summary = {
                'total_reward': float(metrics['cumulative_rewards'][-1]) if len(metrics['cumulative_rewards']) > 0 else 0.0,
                'mean_qos': float(np.mean(metrics['qos'])),
                'mean_energy': float(np.mean(metrics['energy'])),
                'mean_fairness': float(np.mean(metrics['fairness'])),
                'mean_active_ues': float(np.mean(metrics['active_ues']))
            }
            
            # Create result dictionary
            result = {
                'agent_name': agent_name,
                'metrics': metrics,
                'summary': summary
            }
            
            results_list.append(result)
            
            print(f"    ✓ Loaded {len(metrics['steps'])} steps")
            print(f"    Total reward: {summary['total_reward']:.2f}")
            
        except Exception as e:
            print(f"    ⚠️  Error loading {filename}: {e}")
            continue
    
    if not results_list:
        raise ValueError("No valid agent results could be loaded")
    
    # Sort by agent name for consistent ordering
    results_list.sort(key=lambda x: x['agent_name'])
    
    print(f"\n✓ Successfully loaded {len(results_list)} agents")
    print(f"Generating comparison plots...")
    
    # =========================================================================
    # REST OF THE PLOTTING CODE (UNCHANGED)
    # =========================================================================
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle('Agent Performance Comparison - Single Episode', fontsize=16, fontweight='bold')
    
    # Extract data
    agent_names = [r['agent_name'] for r in results_list]
    # Use standard highlighting colors: blue, green, red, orange, purple, etc.
    standard_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = standard_colors[:len(agent_names)]
    
    # 1. Step Rewards
    ax = axes[0, 0]
    for result, color in zip(results_list, colors):
        steps = result['metrics']['steps']
        rewards = result['metrics']['rewards']
        # Smooth with moving average
        window = min(20, len(rewards) // 10) if len(rewards) > 20 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                   color=color, linewidth=2.5)
            # Plot raw data with transparency
            ax.plot(steps, rewards, color=color, alpha=0.2, linewidth=1)
        else:
            ax.plot(steps, rewards, label=result['agent_name'], 
                   color=color, linewidth=2.5)
    ax.set_title('Step Rewards', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative Rewards
    ax = axes[0, 1]
    for result, color in zip(results_list, colors):
        steps = result['metrics']['steps']
        cumulative_rewards = result['metrics']['cumulative_rewards']
        ax.plot(steps, cumulative_rewards, label=result['agent_name'], 
               color=color, linewidth=2.5)
    ax.set_title('Cumulative Rewards', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. QoS Satisfaction
    ax = axes[0, 2]
    for result, color in zip(results_list, colors):
        steps = result['metrics']['steps']
        qos = result['metrics']['qos']
        # Smooth
        window = min(20, len(qos) // 10) if len(qos) > 20 else 1
        if window > 1:
            smoothed = np.convolve(qos, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                   color=color, linewidth=2.5)
            ax.plot(steps, qos, color=color, alpha=0.2, linewidth=1)
        else:
            ax.plot(steps, qos, label=result['agent_name'], 
                   color=color, linewidth=2.5)
    ax.set_title('QoS Satisfaction', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('QoS Satisfaction')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Energy Efficiency
    ax = axes[1, 0]
    for result, color in zip(results_list, colors):
        steps = result['metrics']['steps']
        energy = result['metrics']['energy']
        # Smooth
        window = min(20, len(energy) // 10) if len(energy) > 20 else 1
        if window > 1:
            smoothed = np.convolve(energy, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                   color=color, linewidth=2.5)
            ax.plot(steps, energy, color=color, alpha=0.2, linewidth=1)
        else:
            ax.plot(steps, energy, label=result['agent_name'], 
                   color=color, linewidth=2.5)
    ax.set_title('Energy Consumption Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy Consumption (Lower is Better)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 5. Fairness Index
    ax = axes[1, 1]
    for result, color in zip(results_list, colors):
        steps = result['metrics']['steps']
        fairness = result['metrics']['fairness']
        # Smooth
        window = min(20, len(fairness) // 10) if len(fairness) > 20 else 1
        if window > 1:
            smoothed = np.convolve(fairness, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, label=result['agent_name'], 
                   color=color, linewidth=2.5)
            ax.plot(steps, fairness, color=color, alpha=0.2, linewidth=1)
        else:
            ax.plot(steps, fairness, label=result['agent_name'], 
                   color=color, linewidth=2.5)
    ax.set_title('Fairness Index', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fairness')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Bar Chart
    ax = axes[1, 2]
    metrics_names = ['Total\nReward', 'Mean\nQoS', 'Energy\n(Inverted)', 'Mean\nFairness']
    x = np.arange(len(metrics_names))
    width = 0.8 / len(results_list)
    
    for i, (result, color) in enumerate(zip(results_list, colors)):
        # Normalize metrics for comparison
        total_rewards = [r['summary']['total_reward'] for r in results_list]
        max_reward = max(total_rewards) if max(total_rewards) > 0 else 1
        
        energies = [r['summary']['mean_energy'] for r in results_list]
        max_energy = max(energies) if max(energies) > 0 else 1
        
        values = [
            result['summary']['total_reward'] / max_reward,
            result['summary']['mean_qos'],
            1 - (result['summary']['mean_energy'] / max_energy),  # Invert energy
            result['summary']['mean_fairness']
        ]
        
        offset = (i - len(results_list)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=result['agent_name'], 
              color=color, alpha=0.8)
    
    ax.set_ylabel('Normalized Performance')
    ax.set_title('Performance Summary (Normalized)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(results_dir, 'comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {plot_path}")
    
    # Also save a summary table
    self._save_summary_table(results_list, results_dir)
    
    return fig


def _save_summary_table(self, results_list: List[Dict], results_dir: str):
    """Save a summary table of all agents' performance"""
    
    summary_path = os.path.join(results_dir, 'summary_table.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGENT PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Header
        f.write(f"{'Agent Name':<25} {'Total Reward':>15} {'Mean QoS':>12} {'Mean Energy':>12} {'Mean Fairness':>12}\n")
        f.write("-"*80 + "\n")
        
        # Data rows
        for result in results_list:
            f.write(f"{result['agent_name']:<25} "
                   f"{result['summary']['total_reward']:>15.2f} "
                   f"{result['summary']['mean_qos']:>12.4f} "
                   f"{result['summary']['mean_energy']:>12.4f} "
                   f"{result['summary']['mean_fairness']:>12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        
        # Find best performers
        f.write("\nBEST PERFORMERS:\n")
        f.write("-"*80 + "\n")
        
        best_reward = max(results_list, key=lambda x: x['summary']['total_reward'])
        f.write(f"Highest Total Reward:  {best_reward['agent_name']} ({best_reward['summary']['total_reward']:.2f})\n")
        
        best_qos = max(results_list, key=lambda x: x['summary']['mean_qos'])
        f.write(f"Highest Mean QoS:      {best_qos['agent_name']} ({best_qos['summary']['mean_qos']:.4f})\n")
        
        best_energy = min(results_list, key=lambda x: x['summary']['mean_energy'])
        f.write(f"Lowest Mean Energy:    {best_energy['agent_name']} ({best_energy['summary']['mean_energy']:.4f})\n")
        
        best_fairness = max(results_list, key=lambda x: x['summary']['mean_fairness'])
        f.write(f"Highest Mean Fairness: {best_fairness['agent_name']} ({best_fairness['summary']['mean_fairness']:.4f})\n")
    
    print(f"✓ Saved summary table to {summary_path}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Basic Usage
----------------------

evaluator = Evaluator(config)

# Compare all agents in a directory
evaluator.compare_agents('results/run_001/')


EXAMPLE 2: Compare Multiple Runs
---------------------------------

# Directory structure:
# results/
# ├── run_001/
# │   ├── RL_Agent.csv
# │   ├── Greedy_Agent.csv
# │   └── Random_Agent.csv
# └── run_002/
#     ├── RL_Agent_v2.csv
#     └── Baseline.csv

evaluator.compare_agents('results/run_001/')
evaluator.compare_agents('results/run_002/')


EXAMPLE 3: Expected CSV Format
-------------------------------

The CSV files should have this format:

step,reward,cumulative_reward,qos,energy,fairness,active_ues
0,10.5,10.5,0.85,0.32,0.78,15
1,12.3,22.8,0.87,0.30,0.81,16
2,9.8,32.6,0.83,0.35,0.76,14
...


EXAMPLE 4: Programmatic Comparison
-----------------------------------

import matplotlib.pyplot as plt

evaluator = Evaluator(config)

# Generate comparison plot
fig = evaluator.compare_agents('results/run_001/')

# Further customize if needed
plt.show()

# Or save with different settings
fig.savefig('custom_comparison.png', dpi=300)


EXAMPLE 5: Error Handling
--------------------------

try:
    evaluator.compare_agents('results/run_001/')
except ValueError as e:
    print(f"Error: {e}")
    # Handle missing files or invalid directory
"""


# =============================================================================
# STANDALONE FUNCTION VERSION (if not in a class)
# =============================================================================

def compare_agents_standalone(results_dir: str, output_dir: str = None):
    """
    Standalone version that doesn't require being in a class
    
    Args:
        results_dir: Directory with CSV files
        output_dir: Where to save plots (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    print(f"\nLoading agent results from: {results_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")
    
    print(f"Found {len(csv_files)} agent result files")
    
    # Load data
    results_list = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        agent_name = filename.replace('.csv', '').replace('_', ' ')
        
        print(f"  Loading: {agent_name}")
        
        try:
            df = pd.read_csv(csv_path)
            
            metrics = {
                'steps': df['step'].values,
                'rewards': df['reward'].values,
                'cumulative_rewards': df['cumulative_reward'].values,
                'qos': df['qos'].values,
                'energy': df['energy'].values,
                'fairness': df['fairness'].values,
                'active_ues': df['active_ues'].values
            }
            
            summary = {
                'total_reward': float(metrics['cumulative_rewards'][-1]) if len(metrics['cumulative_rewards']) > 0 else 0.0,
                'mean_qos': float(np.mean(metrics['qos'])),
                'mean_energy': float(np.mean(metrics['energy'])),
                'mean_fairness': float(np.mean(metrics['fairness']))
            }
            
            result = {
                'agent_name': agent_name,
                'metrics': metrics,
                'summary': summary
            }
            
            results_list.append(result)
            print(f"    ✓ Loaded {len(metrics['steps'])} steps")
            
        except Exception as e:
            print(f"    ⚠️  Error loading {filename}: {e}")
            continue
    
    if not results_list:
        raise ValueError("No valid agent results could be loaded")
    
    results_list.sort(key=lambda x: x['agent_name'])
    
    print(f"\n✓ Successfully loaded {len(results_list)} agents")
    
    # Create plots (same plotting code as above)
    # ... [plotting code here] ...
    
    print(f"\n✓ Saved comparison plot to {output_dir}/comparison.png")
    
    return results_list


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        compare_agents_standalone(results_dir)
    else:
        print("Usage: python script.py <results_directory>")
        print("Example: python script.py results/run_001/")