"""
Diagnostic script to understand reward component magnitudes and suggest optimal weights
"""

import numpy as np
import matplotlib.pyplot as plt
from baseline import RandomAgent, GreedyAgent
from environment import NetworkSlicingEnv
from agents import MADRLAgent
from utils import Configuration
import os

def collect_reward_components(agent, agent_name, env, num_steps=500):
    """Collect raw reward components before weighting"""
    
    np.random.seed(42)
    obs = env.reset()
    
    components = {
        'qos': [],
        'energy': [],
        'fairness': [],
        'total_reward': []
    }
    
    # Temporarily store original weights
    original_weights = {
        'qos': env.reward_weights.qos,
        'energy': env.reward_weights.energy,
        'fairness': env.reward_weights.fairness
    }
    
    for step in range(num_steps):
        actions = agent.select_actions(obs, explore=False)
        
        # Step environment
        next_obs, reward, done, info = env.step(actions)
        
        # Store components (from info)
        components['qos'].append(info['qos_satisfaction'])
        components['energy'].append(info['energy_efficiency'])
        components['fairness'].append(info['fairness_level'])
        components['total_reward'].append(reward)
        
        obs = next_obs
        if done:
            break
    
    # Calculate statistics
    stats = {
        'agent_name': agent_name,
        'qos_mean': np.mean(components['qos']),
        'qos_std': np.std(components['qos']),
        'qos_range': (np.min(components['qos']), np.max(components['qos'])),
        'energy_mean': np.mean(components['energy']),
        'energy_std': np.std(components['energy']),
        'energy_range': (np.min(components['energy']), np.max(components['energy'])),
        'fairness_mean': np.mean(components['fairness']),
        'fairness_std': np.std(components['fairness']),
        'fairness_range': (np.min(components['fairness']), np.max(components['fairness'])),
        'total_reward_mean': np.mean(components['total_reward']),
        'components': components
    }
    
    return stats, original_weights

def analyze_reward_contributions(stats_list, weights):
    """Analyze how much each component contributes to total reward"""
    
    print("\n" + "="*100)
    print("REWARD COMPONENT ANALYSIS")
    print("="*100)
    print(f"\nCurrent Weights: QoS={weights['qos']}, Energy={weights['energy']}, Fairness={weights['fairness']}")
    print("\n" + "-"*100)
    
    # Header
    print(f"{'Agent':<20} {'Component':<15} {'Mean':<12} {'Std':<12} {'Range':<20} {'Weighted':<15}")
    print("-"*100)
    
    for stats in stats_list:
        name = stats['agent_name']
        
        # QoS
        qos_weighted = stats['qos_mean'] * weights['qos']
        print(f"{name:<20} {'QoS':<15} {stats['qos_mean']:<12.4f} {stats['qos_std']:<12.4f} "
              f"[{stats['qos_range'][0]:.3f}, {stats['qos_range'][1]:.3f}]  {qos_weighted:<15.4f}")
        
        # Energy (negative contribution)
        energy_weighted = -stats['energy_mean'] * weights['energy']
        print(f"{'':<20} {'Energy':<15} {stats['energy_mean']:<12.4f} {stats['energy_std']:<12.4f} "
              f"[{stats['energy_range'][0]:.3f}, {stats['energy_range'][1]:.3f}]  {energy_weighted:<15.4f}")
        
        # Fairness
        fairness_weighted = stats['fairness_mean'] * weights['fairness']
        print(f"{'':<20} {'Fairness':<15} {stats['fairness_mean']:<12.4f} {stats['fairness_std']:<12.4f} "
              f"[{stats['fairness_range'][0]:.3f}, {stats['fairness_range'][1]:.3f}]  {fairness_weighted:<15.4f}")
        
        # Total
        total_weighted = qos_weighted + energy_weighted + fairness_weighted
        print(f"{'':<20} {'TOTAL':<15} {stats['total_reward_mean']:<12.4f} {'':<12} {'':<20} {total_weighted:<15.4f}")
        print("-"*100)
    
    print("\n🔍 KEY INSIGHTS:\n")
    
    # Compare component contributions
    madrl_stats = next(s for s in stats_list if "MADRL" in s['agent_name'])
    greedy_stats = next(s for s in stats_list if "Greedy" in s['agent_name'])
    
    madrl_qos_contrib = madrl_stats['qos_mean'] * weights['qos']
    madrl_energy_contrib = -madrl_stats['energy_mean'] * weights['energy']
    madrl_fairness_contrib = madrl_stats['fairness_mean'] * weights['fairness']
    
    greedy_qos_contrib = greedy_stats['qos_mean'] * weights['qos']
    greedy_energy_contrib = -greedy_stats['energy_mean'] * weights['energy']
    greedy_fairness_contrib = greedy_stats['fairness_mean'] * weights['fairness']
    
    print(f"MADRL Weighted Contributions:")
    print(f"  QoS:      {madrl_qos_contrib:+.4f}")
    print(f"  Energy:   {madrl_energy_contrib:+.4f}")
    print(f"  Fairness: {madrl_fairness_contrib:+.4f}")
    print(f"  TOTAL:    {madrl_qos_contrib + madrl_energy_contrib + madrl_fairness_contrib:+.4f}")
    
    print(f"\nGreedy Weighted Contributions:")
    print(f"  QoS:      {greedy_qos_contrib:+.4f}")
    print(f"  Energy:   {greedy_energy_contrib:+.4f}")
    print(f"  Fairness: {greedy_fairness_contrib:+.4f}")
    print(f"  TOTAL:    {greedy_qos_contrib + greedy_energy_contrib + greedy_fairness_contrib:+.4f}")
    
    print("\n📊 Component Comparison (MADRL vs Greedy):")
    qos_diff = madrl_qos_contrib - greedy_qos_contrib
    energy_diff = madrl_energy_contrib - greedy_energy_contrib
    fairness_diff = madrl_fairness_contrib - greedy_fairness_contrib
    
    print(f"  QoS difference:      {qos_diff:+.4f} {'✅ Better' if qos_diff > 0 else '❌ Worse'}")
    print(f"  Energy difference:   {energy_diff:+.4f} {'✅ Better' if energy_diff > 0 else '❌ Worse'}")
    print(f"  Fairness difference: {fairness_diff:+.4f} {'✅ Better' if fairness_diff > 0 else '❌ Worse'}")
    
    # Diagnose the problem
    print("\n💡 DIAGNOSIS:\n")
    
    if abs(qos_diff) > abs(energy_diff):
        print(f"⚠️  QoS loss ({abs(qos_diff):.4f}) is LARGER than energy gain ({abs(energy_diff):.4f})")
        print(f"    → Agent is making a BAD trade-off!")
        print(f"    → The QoS weight is too LOW relative to its importance")
    else:
        print(f"✓  Energy gain ({abs(energy_diff):.4f}) is larger than QoS loss ({abs(qos_diff):.4f})")
        print(f"    → Agent is making a reasonable trade-off given current weights")
        print(f"    → But if you want better QoS, increase its weight")
    
    # Suggest new weights
    print("\n🎯 SUGGESTED WEIGHT ADJUSTMENTS:\n")
    
    # Calculate ideal weights to match greedy's component balance
    greedy_total = greedy_qos_contrib + greedy_energy_contrib + greedy_fairness_contrib
    
    # What weights would make MADRL's components balanced like Greedy?
    target_qos_contrib = greedy_qos_contrib
    target_fairness_contrib = greedy_fairness_contrib
    
    new_qos_weight = target_qos_contrib / madrl_stats['qos_mean'] if madrl_stats['qos_mean'] > 0 else weights['qos']
    new_fairness_weight = target_fairness_contrib / madrl_stats['fairness_mean'] if madrl_stats['fairness_mean'] > 0 else weights['fairness']
    
    print("Option 1: Match Greedy's component balance")
    print(f"  qos: {new_qos_weight:.2f} (currently {weights['qos']})")
    print(f"  energy: {weights['energy']:.2f} (keep same)")
    print(f"  fairness: {new_fairness_weight:.2f} (currently {weights['fairness']})")
    
    print("\nOption 2: Prioritize QoS (Conservative)")
    print(f"  qos: {weights['qos'] * 3:.2f} (3x increase)")
    print(f"  energy: {weights['energy']:.2f} (keep same)")
    print(f"  fairness: {weights['fairness'] * 2:.2f} (2x increase)")
    
    print("\nOption 3: Aggressive QoS focus")
    print(f"  qos: {weights['qos'] * 10:.2f} (10x increase)")
    print(f"  energy: {weights['energy'] * 0.5:.2f} (halve)")
    print(f"  fairness: {weights['fairness'] * 5:.2f} (5x increase)")

def plot_reward_components(stats_list, weights):
    """Visualize reward components"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Reward Component Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    component_names = ['QoS', 'Energy', 'Fairness']
    
    # Plot 1-3: Time series of each component
    for idx, comp_key in enumerate(['qos', 'energy', 'fairness']):
        ax = axes[0, idx]
        for stats, color in zip(stats_list, colors):
            data = stats['components'][comp_key]
            steps = range(len(data))
            
            # Smooth
            window = 20
            if len(data) > window:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(steps[:len(smoothed)], smoothed, label=stats['agent_name'], 
                       color=color, linewidth=2.5)
                ax.plot(steps, data, color=color, alpha=0.2, linewidth=1)
            else:
                ax.plot(steps, data, label=stats['agent_name'], 
                       color=color, linewidth=2.5)
        
        ax.set_title(f'{component_names[idx]} Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel(component_names[idx])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Component distribution (box plot style)
    ax = axes[1, 0]
    x_pos = np.arange(len(stats_list))
    width = 0.25
    
    for i, comp_key in enumerate(['qos', 'energy', 'fairness']):
        values = [s[f'{comp_key}_mean'] for s in stats_list]
        offset = (i - 1) * width
        ax.bar(x_pos + offset, values, width, label=component_names[i], alpha=0.8)
    
    ax.set_ylabel('Mean Value')
    ax.set_title('Component Means Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s['agent_name'] for s in stats_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Weighted contributions
    ax = axes[1, 1]
    x_pos = np.arange(len(stats_list))
    width = 0.25
    
    qos_weighted = [s['qos_mean'] * weights['qos'] for s in stats_list]
    energy_weighted = [-s['energy_mean'] * weights['energy'] for s in stats_list]
    fairness_weighted = [s['fairness_mean'] * weights['fairness'] for s in stats_list]
    
    ax.bar(x_pos - width, qos_weighted, width, label='QoS (weighted)', alpha=0.8, color='green')
    ax.bar(x_pos, energy_weighted, width, label='Energy (weighted)', alpha=0.8, color='red')
    ax.bar(x_pos + width, fairness_weighted, width, label='Fairness (weighted)', alpha=0.8, color='blue')
    
    ax.set_ylabel('Weighted Contribution')
    ax.set_title('Weighted Component Contributions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s['agent_name'] for s in stats_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 6: Total reward comparison
    ax = axes[1, 2]
    total_rewards = [s['total_reward_mean'] for s in stats_list]
    bars = ax.bar(range(len(stats_list)), total_rewards, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Total Reward')
    ax.set_title('Total Reward Comparison')
    ax.set_xticks(range(len(stats_list)))
    ax.set_xticklabels([s['agent_name'] for s in stats_list])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, total_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reward_component_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved analysis to reward_component_analysis.png")
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze reward components and suggest weights')
    parser.add_argument('--env_config', type=str, default='config/environment/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    
    # Load config and environment
    env_config = Configuration(args.env_config)
    num_agents = env_config.system.num_uavs
    env = NetworkSlicingEnv(config_path=args.env_config)
    
    obs_sample = env.reset()
    obs_dim = len(list(obs_sample.values())[0])
    action_dim = 4 + env_config.system.num_das_per_slice * 3
    
    print(f"\n{'='*60}")
    print(f"REWARD COMPONENT DIAGNOSTICS")
    print(f"{'='*60}\n")
    
    # Initialize agents
    random_agent = RandomAgent(num_agents, obs_dim, action_dim)
    greedy_agent = GreedyAgent(num_agents, obs_dim, action_dim, env)
    madrl_agent = MADRLAgent(num_agents, obs_dim, action_dim, training=False)
    madrl_agent.load_models(args.checkpoint)
    
    agents = [
        (random_agent, "Random"),
        (greedy_agent, "Greedy"),
        (madrl_agent, "MADRL (Trained)")
    ]
    
    # Collect statistics
    stats_list = []
    weights = None
    
    for agent, name in agents:
        print(f"Analyzing {name}...")
        np.random.seed(42)
        stats, weights = collect_reward_components(agent, name, env, args.steps)
        stats_list.append(stats)
    
    # Analyze and suggest
    analyze_reward_contributions(stats_list, weights)
    
    # Plot
    plot_reward_components(stats_list, weights)
    
    print(f"\n{'='*60}")
    print("Diagnostics complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()