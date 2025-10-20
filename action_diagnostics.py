"""
Diagnostic script to analyze why MADRL performs worse than Greedy
"""

import numpy as np
import matplotlib.pyplot as plt
from baseline import RandomAgent, GreedyAgent
from environment import NetworkSlicingEnv
from agents import MADRLAgent
from utils import Configuration
import os

def collect_action_statistics(agent, agent_name, env, num_steps=500):
    """Collect detailed statistics about agent actions"""
    
    np.random.seed(42)
    obs = env.reset()
    
    stats = {
        'power_levels': [],
        'power_per_uav': {i: [] for i in range(env.num_uavs)},
        'bandwidth_allocations': [],
        'position_changes': [],
        'num_ues_per_uav': {i: [] for i in range(env.num_uavs)},
        'qos_per_step': [],
        'fairness_per_step': [],
        'energy_per_step': []
    }
    
    for step in range(num_steps):
        actions = agent.select_actions(obs, explore=False)
        
        # Collect action statistics
        for uav_id, action in actions.items():
            stats['power_levels'].append(action[3])
            stats['power_per_uav'][uav_id].append(action[3])
            stats['position_changes'].append(np.linalg.norm(action[0:3]))
            if len(action) > 4:
                stats['bandwidth_allocations'].extend(action[4:])
        
        # Step environment
        next_obs, reward, done, info = env.step(actions)
        
        # Collect environment statistics
        for uav_id in range(env.num_uavs):
            assigned_ues = len([ue for ue in env.ues.values() 
                              if ue.assigned_uav == uav_id and ue.is_active])
            stats['num_ues_per_uav'][uav_id].append(assigned_ues)
        
        stats['qos_per_step'].append(info['qos_satisfaction'])
        stats['fairness_per_step'].append(info['fairness_level'])
        stats['energy_per_step'].append(info['energy_efficiency'])
        
        obs = next_obs
        if done:
            break
    
    # Calculate summary statistics
    summary = {
        'agent_name': agent_name,
        'avg_power': np.mean(stats['power_levels']),
        'std_power': np.std(stats['power_levels']),
        'min_power': np.min(stats['power_levels']),
        'max_power': np.max(stats['power_levels']),
        'avg_position_change': np.mean(stats['position_changes']),
        'avg_bandwidth_alloc': np.mean(stats['bandwidth_allocations']) if stats['bandwidth_allocations'] else 0,
        'std_bandwidth_alloc': np.std(stats['bandwidth_allocations']) if stats['bandwidth_allocations'] else 0,
        'avg_qos': np.mean(stats['qos_per_step']),
        'avg_fairness': np.mean(stats['fairness_per_step']),
        'avg_energy': np.mean(stats['energy_per_step']),
        'power_per_uav_avg': {k: np.mean(v) for k, v in stats['power_per_uav'].items()},
        'ue_distribution_std': np.std([np.mean(v) for v in stats['num_ues_per_uav'].values()])
    }
    
    return stats, summary

def compare_agent_behaviors(summaries):
    """Print comparison table"""
    print("\n" + "="*100)
    print("AGENT BEHAVIOR COMPARISON")
    print("="*100)
    
    headers = ["Metric", "Random", "Greedy", "MADRL (Trained)"]
    print(f"{headers[0]:<30} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20}")
    print("-"*100)
    
    metrics = [
        ('avg_power', 'Avg Power', '.3f'),
        ('std_power', 'Power Std Dev', '.3f'),
        ('min_power', 'Min Power', '.3f'),
        ('max_power', 'Max Power', '.3f'),
        ('avg_position_change', 'Avg Movement', '.3f'),
        ('avg_bandwidth_alloc', 'Avg Bandwidth Alloc', '.3f'),
        ('std_bandwidth_alloc', 'Bandwidth Std Dev', '.3f'),
        ('avg_qos', 'Mean QoS', '.3f'),
        ('avg_fairness', 'Mean Fairness', '.3f'),
        ('avg_energy', 'Mean Energy', '.3f'),
        ('ue_distribution_std', 'UE Load Balance', '.3f')
    ]
    
    for key, label, fmt in metrics:
        values = [f"{s[key]:{fmt}}" for s in summaries]
        print(f"{label:<30} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print("="*100)
    
    # Highlight key differences
    print("\n🔍 KEY INSIGHTS:")
    
    madrl_power = summaries[2]['avg_power']
    greedy_power = summaries[1]['avg_power']
    
    if madrl_power < greedy_power * 0.7:
        print(f"⚠️  MADRL uses {(1 - madrl_power/greedy_power)*100:.1f}% LESS power than Greedy")
        print(f"   → This explains better energy efficiency but worse QoS!")
        print(f"   → Solution: Increase QoS reward weight or add minimum power constraint")
    
    madrl_power_std = summaries[2]['std_power']
    greedy_power_std = summaries[1]['std_power']
    
    if madrl_power_std < greedy_power_std * 0.5:
        print(f"\n⚠️  MADRL power variance is very low ({madrl_power_std:.3f} vs {greedy_power_std:.3f})")
        print(f"   → Agent is being too conservative/uniform")
        print(f"   → Solution: Encourage exploration or reduce power action clipping")
    
    madrl_bw_std = summaries[2]['std_bandwidth_alloc']
    greedy_bw_std = summaries[1]['std_bandwidth_alloc']
    
    if madrl_bw_std < greedy_bw_std * 0.5:
        print(f"\n⚠️  MADRL bandwidth allocation is too uniform ({madrl_bw_std:.3f} vs {greedy_bw_std:.3f})")
        print(f"   → Not adapting allocations to demand")
        print(f"   → Solution: Add per-DA QoS tracking to reward")

def plot_power_usage_comparison(all_stats, summaries):
    """Plot power usage patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Power Usage Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # 1. Power level distribution
    ax = axes[0, 0]
    for stats, summary, color in zip(all_stats, summaries, colors):
        ax.hist(stats['power_levels'], bins=30, alpha=0.5, 
               label=summary['agent_name'], color=color, density=True)
    ax.set_xlabel('Power Level')
    ax.set_ylabel('Density')
    ax.set_title('Power Level Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Power over time
    ax = axes[0, 1]
    for stats, summary, color in zip(all_stats, summaries, colors):
        steps = range(len(stats['power_levels'][:500]))
        powers = stats['power_levels'][:500]
        # Smooth
        window = 20
        if len(powers) > window:
            smoothed = np.convolve(powers, np.ones(window)/window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, label=summary['agent_name'], 
                   color=color, linewidth=2)
        else:
            ax.plot(steps, powers, label=summary['agent_name'], 
                   color=color, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Power')
    ax.set_title('Power Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Power per UAV
    ax = axes[1, 0]
    num_uavs = len(all_stats[0]['power_per_uav'])
    x = np.arange(num_uavs)
    width = 0.25
    
    for i, (stats, summary, color) in enumerate(zip(all_stats, summaries, colors)):
        powers = [np.mean(stats['power_per_uav'][uav_id]) for uav_id in range(num_uavs)]
        ax.bar(x + i*width, powers, width, label=summary['agent_name'], 
              color=color, alpha=0.8)
    
    ax.set_xlabel('UAV ID')
    ax.set_ylabel('Average Power')
    ax.set_title('Average Power per UAV')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'UAV {i}' for i in range(num_uavs)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. QoS vs Power scatter
    ax = axes[1, 1]
    for stats, summary, color in zip(all_stats, summaries, colors):
        # Sample every 10 steps to avoid overcrowding
        powers = stats['power_levels'][::10]
        qos = stats['qos_per_step'][::10]
        min_len = min(len(powers), len(qos))
        ax.scatter(powers[:min_len], qos[:min_len], alpha=0.3, 
                  label=summary['agent_name'], color=color, s=20)
    
    ax.set_xlabel('Power Level')
    ax.set_ylabel('QoS Satisfaction')
    ax.set_title('QoS vs Power Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved power analysis to power_analysis.png")
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_config', type=str, default='config/environment/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained MADRL model checkpoint')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    
    # Load config and create environment
    env_config = Configuration(args.env_config)
    num_agents = env_config.system.num_uavs
    env = NetworkSlicingEnv(config_path=args.env_config)
    
    obs_sample = env.reset()
    obs_dim = len(list(obs_sample.values())[0])
    action_dim = 4 + env_config.system.num_das_per_slice * 3
    
    print(f"\n{'='*60}")
    print(f"ACTION DIAGNOSTICS")
    print(f"{'='*60}\n")
    
    # Initialize agents
    random_agent = RandomAgent(num_agents, obs_dim, action_dim)
    greedy_agent = GreedyAgent(num_agents, obs_dim, action_dim, env)
    
    madrl_agent = MADRLAgent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        training=False
    )
    madrl_agent.load_models(args.checkpoint)
    
    agents = [
        (random_agent, "Random"),
        (greedy_agent, "Greedy"),
        (madrl_agent, "MADRL (Trained)")
    ]
    
    # Collect statistics
    all_stats = []
    all_summaries = []
    
    for agent, name in agents:
        print(f"Analyzing {name}...")
        np.random.seed(42)  # Same environment for all
        stats, summary = collect_action_statistics(agent, name, env, args.steps)
        all_stats.append(stats)
        all_summaries.append(summary)
    
    # Compare behaviors
    compare_agent_behaviors(all_summaries)
    
    # Plot analysis
    plot_power_usage_comparison(all_stats, all_summaries)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()