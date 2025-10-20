# train.py
import numpy as np
from environment import NetworkSlicingEnv
from agents import BCWeightScheduler, MADRLAgent
from visualizer import Network3DVisualizer
import os
from tqdm import tqdm
import re
from utils import Configuration
import cProfile
import pstats
import io
from pstats import SortKey
from baseline import RandomAgent, GreedyAgent

class TrainingManager:
    """Manages the training process for UAV network slicing MADRL"""

    def __init__(self, env_config_path: str = None, train_config_path: str = None, 
             checkpoint_path: str = None, enable_visualization: bool = False):
        # Load configurations using your Configuration class
        env_config = Configuration(env_config_path)
        train_config = Configuration(train_config_path)

        self.config = {
            # Environment parameters
            'num_uavs': env_config.system.num_uavs,
            'num_ues': env_config.system.num_ues,
            'service_area': tuple(env_config.system.service_area),
            'uav_fly_range_h': tuple(env_config.system.uav_fly_range_h),
            'num_das_per_slice': env_config.system.num_das_per_slice,

            # Step-based training parameters
            'total_training_steps': train_config.total_training_steps,

            # Step-based intervals
            'log_interval': train_config.log_interval,
            'save_interval': train_config.save_interval,
            'evaluation_interval': train_config.evaluation_interval,
            'plot_interval': train_config.plot_interval,
            'exploration_noise_update_interval': train_config.exploration_noise_update_interval,
            'greedy_experience_gathering_steps': train_config.greedy_experience_gathering_steps,

            # Paths
            'save_dir': train_config.save_dir,
            'log_dir': train_config.log_dir,
            'tensorboard_dir': train_config.tensorboard_dir,
            
            # Config paths
            'env_config_path': env_config_path,
            'train_config_path': train_config_path,

            # Training frequency parameters
            'train_frequency': train_config.train_frequency,
            'train_iterations': train_config.train_iterations,
            'min_buffer_size': train_config.min_buffer_size,

            # BC Regularization
            'bc_regularization': {
                'initial_weight': train_config.bc_regularization.initial_weight,
                'final_weight': train_config.bc_regularization.final_weight,
                'transition_steps': train_config.bc_regularization.transition_steps,
                'schedule_type': train_config.bc_regularization.schedule_type
            }
        }

        # Initialize environment and agent
        self.env = NetworkSlicingEnv(config_path=self.config['env_config_path'])


        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)

        # Create unique model directory
        self.model_dir = self._get_next_model_dir(self.config['save_dir'])
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'training_progress'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'uav_throughput'), exist_ok=True)

        # Initialize metrics storage
        self._initialize_metrics()
        
        # Get observation and action dimensions
        obs_sample = self.env.reset()
        obs_dim = len(list(obs_sample.values())[0])
        print(f"Observation dimension: {obs_dim}")

        action_dim = 4 + self.config['num_das_per_slice'] * 3  # pos + power + bandwidth per DA
        
        self.MADRLagent = MADRLAgent(
            num_agents=self.config['num_uavs'],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # ADD THIS RIGHT AFTER:
        print(f"✅ Agent initialized on: {self.MADRLagent.device}")
        if self.MADRLagent.device.type == 'cpu':
            print("   ⚠️  WARNING: Training on CPU will be VERY SLOW!")
            print("   Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x2")

        # BC weight scheduler
        self.bc_scheduler = BCWeightScheduler(
            initial_weight=self.config['bc_regularization']['initial_weight'],      # 100% BC initially
            final_weight=self.config['bc_regularization']['final_weight'],        # 10% BC finally
            transition_steps=self.config['bc_regularization']['transition_steps'],
            schedule_type=self.config['bc_regularization']['schedule_type']
        )


        # Load checkpoint if provided
        if checkpoint_path:
            self.MADRLagent.load_models(checkpoint_path)
            self.MADRLagent.reset_optimizer_state()  # Reset optimizers to avoid issues
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.GreedyAgent = GreedyAgent(
            num_agents=self.config['num_uavs'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            env=self.env
        )

        # Save config files
        self._save_config_files()


        # Visualization
        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            self.visualizer = Network3DVisualizer(self.env, self.MADRLagent)
        else:
            self.visualizer = None

    def _initialize_metrics(self):  
        # Initialize tracking variables for step-based training
        self.step_metrics = {
            'steps': [],
            'rewards': [],
            'qos_satisfaction': [],
            'energy_efficiency': [],
            'fairness_level': [],
            'active_ues': [],
            'noise': [],
        }

        # Add separate training metrics (NEW)
        self.training_metrics = {
            'steps': [],           # Steps when training occurred
            'actor_losses': [],    # Average actor loss per training session
            'critic_losses': [],   # Average critic loss per training session
            'iterations': [],      # Number of iterations performed
            'buffer_size': []      # Buffer size at training time
        }

        # Initialize per-agent action metrics
        for i in range(self.config['num_uavs']):
            self.step_metrics[f'agent_{i}_pos_x'] = []
            self.step_metrics[f'agent_{i}_pos_y'] = []
            self.step_metrics[f'agent_{i}_pos_z'] = []
            self.step_metrics[f'agent_{i}_power'] = []
            for j in range(int(len(self.env.demand_areas) / self.config['num_uavs'])):
                self.step_metrics[f'agent_{i}_da_{j}'] = []

    def _get_next_model_dir(self, save_dir):
        """Find the next available model directory name (model1, model2, ...)"""
        existing = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
        numbers = []
        for name in existing:
            m = re.match(r'model(\d+)', name)
            if m:
                numbers.append(int(m.group(1)))
        next_num = 1
        while f"model{next_num}" in existing:
            next_num += 1
        return os.path.join(save_dir, f"model{next_num}")

    def _save_config_files(self):
        """Save copies of the config files used for this training run"""
        import shutil
        shutil.copy(self.config['env_config_path'], os.path.join(self.model_dir, 'env_config.yaml'))
        shutil.copy(self.config['train_config_path'], os.path.join(self.model_dir, 'train_config.yaml'))
        shutil.copy('./config/agents/default.yaml', os.path.join(self.model_dir, 'agents_config.yaml'))
        print(f"✓ Saved config files to {self.model_dir}")

    def train(self):

        """Step-based training loop with comprehensive logging"""
        print(f"Starting step-based training for {self.config['total_training_steps']} steps")
        
        # Initialize environment
        observations = self.env.reset()
        
        # Main training loop
        # In this simulation, each training step corresponds to one large time step
        progress_bar = tqdm(range(self.config['total_training_steps']), desc="Training Steps")
        
        for global_step in progress_bar:
            # Collect experiences
            # greedy_prob = max(0.0, 1.0 - global_step / 100000)
            
            # if global_step < self.config['greedy_experience_gathering_steps']:
            greedy_actions = self.GreedyAgent.select_actions(observations)
            actions = self.MADRLagent.select_actions(observations, explore=True)
            
            next_observations, reward, done, info = self.env.step(actions)
            self.MADRLagent.store_transition(observations, actions, greedy_actions, reward, next_observations, done)

            # Check if we should train
            should_train = (
                (global_step + 1) % self.config['train_frequency'] == 0 and
                len(self.MADRLagent.buffer) >= self.config['min_buffer_size']
            )
            
            if should_train:
                # Perform multiple training iterations
                actor_losses = []
                critic_losses = []
                rl_losses_list = []
                bc_losses_list = []

                bc_weight = self.bc_scheduler.get_weight()
                self.bc_scheduler.step()
                
                for _ in range(self.config['train_iterations']):
                    train_info = self.MADRLagent.train_rl_with_bc_regularization(
                        bc_weight=bc_weight
                    )
                    if train_info:
                        # Collect losses from all agents
                        actor_losses.append(np.mean(train_info['actor_losses']))
                        critic_losses.append(train_info['critic_loss'])
                        rl_losses_list.append(np.mean(train_info['rl_losses']))
                        bc_losses_list.append(np.mean(train_info['bc_losses']))
                
                

                # Store training metrics
                if actor_losses:  # Only store if training actually happened
                    self.training_metrics['steps'].append(global_step)
                    self.training_metrics['actor_losses'].append(np.mean(actor_losses))
                    self.training_metrics['critic_losses'].append(np.mean(critic_losses))
                    self.training_metrics['iterations'].append(self.config['train_iterations'])
                    self.training_metrics['buffer_size'].append(len(self.MADRLagent.buffer))

                    # Add metrics for BC regularization
                    if 'rl_losses' not in self.training_metrics:  
                        self.training_metrics['rl_losses'] = []
                        self.training_metrics['bc_losses'] = []
                        self.training_metrics['bc_weights'] = []

                    self.training_metrics['rl_losses'].append(np.mean(rl_losses_list))
                    self.training_metrics['bc_losses'].append(np.mean(bc_losses_list))
                    self.training_metrics['bc_weights'].append(bc_weight)
                    
                    # Update progress bar with training info
                    progress_bar.set_postfix({
                        'Reward': f'{reward:.3f}',
                        'BC Weight': f'{bc_weight:.3f}',
                        'Actor Loss': f'{np.mean(actor_losses):.4f}',
                        'RL Loss': f'{np.mean(rl_losses_list):.4f}',
                        'BC Loss': f'{np.mean(bc_losses_list):.4f}',
                        'Buffer': len(self.MADRLagent.buffer)
                    })
            
            # Store environment metrics (every step)
            self._store_step_metrics(global_step, actions, reward, info)
            
            # Update visualization if enabled
            if self.enable_visualization:
                # if global_step % 10 == 0:  # Update visualization every 5 steps to reduce overhead
                    self._update_visualization()
            
            observations = next_observations
            
            # Periodic operations
            if (global_step + 1) % self.config['save_interval'] == 0 and global_step > 0:
                self._save_checkpoint(global_step + 1)
                self._save_metrics_to_csv()
            
            if (global_step + 1) % self.config['plot_interval'] == 0 and global_step > 0:
                
                self._plot_training_progress(global_step + 1)
            
            # Update exploration noise less frequently
            if  (global_step + 1) % self.config['exploration_noise_update_interval'] == 0:
                self.MADRLagent.exploration_noise = max(
                    self.MADRLagent.min_noise,
                    self.MADRLagent.exploration_noise * self.MADRLagent.noise_decay
                )
        
        self.env.print_cache_stats()
        print("Training completed!")

    def _store_step_metrics(self, step, actions, reward, info):
        """Store metrics for this step"""
        self.step_metrics['steps'].append(step)
        self.step_metrics['rewards'].append(reward)
        self.step_metrics['qos_satisfaction'].append(info['qos_satisfaction'])
        self.step_metrics['energy_efficiency'].append(info['energy_efficiency'])
        self.step_metrics['fairness_level'].append(info['fairness_level'])
        self.step_metrics['active_ues'].append(info['active_ues'])
        self.step_metrics['noise'].append(getattr(self.MADRLagent, 'exploration_noise', 0))

        # Store actions of each agent for analysis if needed
        for i, action in actions.items():
            self.step_metrics[f'agent_{i}_pos_x'].append(action[0])
            self.step_metrics[f'agent_{i}_pos_y'].append(action[1])
            self.step_metrics[f'agent_{i}_pos_z'].append(action[2])
            self.step_metrics[f'agent_{i}_power'].append(action[3])
            for j in range(int(len(self.env.demand_areas) / self.config['num_uavs'])):
                self.step_metrics[f'agent_{i}_da_{j}'].append(action[4 + j])

    def _plot_training_progress(self, current_step):
        """Plot comprehensive training progress"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Step {current_step}', fontsize=16)
        
        # Read data from CSV files
        step_metrics_path = f'{self.model_dir}/step_metrics.csv'
        training_metrics_path = f'{self.model_dir}/training_metrics.csv'
        
        if not os.path.exists(step_metrics_path):
            print(f"Warning: {step_metrics_path} not found")
            plt.close()
            return
        
        # Load step metrics
        step_df = pd.read_csv(step_metrics_path)
        steps = step_df['steps'].values
        
        # Load training metrics if exists
        training_df = None
        if os.path.exists(training_metrics_path):
            training_df = pd.read_csv(training_metrics_path)
        
        # Plot 1: Reward progression
        rewards = step_df['rewards'].values
        axes[0, 0].plot(steps, rewards, alpha=0.3, color='blue')
        if len(steps) > 100:
            smoothed_rewards = self._moving_average(rewards, 1000)
            axes[0, 0].plot(steps[-len(smoothed_rewards):], smoothed_rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: QoS satisfaction
        qos_satisfaction = step_df['qos_satisfaction'].values
        axes[0, 1].plot(steps, qos_satisfaction, 'g-', alpha=0.6)
        if len(steps) > 100:
            smoothed_qos = self._moving_average(qos_satisfaction, 1000)
            axes[0, 1].plot(steps[-len(smoothed_qos):], smoothed_qos, 'darkgreen', linewidth=2)
        axes[0, 1].set_title('QoS Satisfaction')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('QoS Satisfaction')
        axes[0, 1].grid(True)
        
        # Plot 3: Energy efficiency
        energy_efficiency = step_df['energy_efficiency'].values
        axes[0, 2].plot(steps, energy_efficiency, 'orange', alpha=0.6)
        if len(steps) > 100:
            smoothed_energy = self._moving_average(energy_efficiency, 1000)
            axes[0, 2].plot(steps[-len(smoothed_energy):], smoothed_energy, 'darkorange', linewidth=2)
        axes[0, 2].set_title('Energy Consumption Rate')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Energy Consumption Rate')
        axes[0, 2].grid(True)
        
        # Plot 4: Active UEs
        active_ues = step_df['active_ues'].values
        axes[1, 0].plot(steps, active_ues, 'purple', alpha=0.6)
        if len(steps) > 100:
            smoothed_ues = self._moving_average(active_ues, 1000)
            axes[1, 0].plot(steps[-len(smoothed_ues):], smoothed_ues, 'darkviolet', linewidth=2)
        axes[1, 0].set_title('Active UEs')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Number of UEs')
        axes[1, 0].grid(True)
        
        # Plot 5: Actor losses (only where training occurred)
        if training_df is not None and 'actor_losses' in training_df.columns:
            training_steps = training_df['steps'].values
            actor_losses = training_df['actor_losses'].values
            axes[1, 1].plot(training_steps, actor_losses, 
                        'red', alpha=0.6, marker='o', markersize=3)
            axes[1, 1].set_title('Actor Losses')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        # Plot 6: Critic losses (only where training occurred)
        if training_df is not None and 'critic_losses' in training_df.columns:
            training_steps = training_df['steps'].values
            critic_losses = training_df['critic_losses'].values
            axes[1, 2].plot(training_steps, critic_losses, 
                        'blue', alpha=0.6, marker='o', markersize=3)
            axes[1, 2].set_title('Critic Losses')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True)

        # Plot 7: Fairness Level
        fairness_level = step_df['fairness_level'].values
        axes[2, 0].plot(steps, fairness_level, 'brown', alpha=0.6)
        if len(steps) > 100:
            smoothed_fairness = self._moving_average(fairness_level, 1000)
            axes[2, 0].plot(steps[-len(smoothed_fairness):], smoothed_fairness, 'saddlebrown', linewidth=2)
        axes[2, 0].set_title('Average Fairness Level')
        axes[2, 0].set_xlabel('Steps')
        axes[2, 0].set_ylabel('Average Fairness Level')
        axes[2, 0].grid(True)

        # Plot 8: Exploration Noise
        noise = step_df['noise'].values
        axes[2, 1].plot(steps, noise, 'cyan', alpha=0.6)
        axes[2, 1].set_title('Exploration Noise')
        axes[2, 1].set_xlabel('Steps')
        axes[2, 1].set_ylabel('Noise Level')
        axes[2, 1].grid(True)

        # Plot 9: Power usage of agents
        for i in range(self.config['num_uavs']):
            col_name = f'agent_{i}_power'
            if col_name in step_df.columns:
                agent_power = step_df[col_name].values
                axes[2, 2].plot(steps, agent_power, alpha=0.6, label=f'Agent {i}')
        axes[2, 2].set_title('Power Usage of Agents')
        axes[2, 2].set_xlabel('Steps')
        axes[2, 2].set_ylabel('Power Usage (Watts)')
        axes[2, 2].legend(loc='lower left')
        axes[2, 2].grid(True)

        if hasattr(self, 'training_metrics') and 'bc_weights' in self.training_metrics:
            training_df = pd.read_csv(f'{self.model_dir}/training_metrics.csv')
        
            if 'bc_weights' in training_df.columns:
                axes[0, 3].plot(
                    training_df['steps'], 
                    training_df['bc_weights'], 
                    'purple', 
                    alpha=0.8,
                    linewidth=3
                )
                axes[0, 3].set_title('BC Weight Schedule')
                axes[0, 3].set_xlabel('Steps')
                axes[0, 3].set_ylabel('BC Weight')
                axes[0, 3].set_ylim([0, 1])
                axes[0, 3].grid(True)

            # NEW: Plot RL Loss vs BC Loss
            if 'rl_losses' in training_df.columns and 'bc_losses' in training_df.columns:
                ax_twin = axes[1, 3].twinx()
                
                line1 = axes[1, 3].plot(
                    training_df['steps'], 
                    training_df['rl_losses'],
                    'red', 
                    alpha=0.6, 
                    label='RL Loss'
                )
                line2 = ax_twin.plot(
                    training_df['steps'], 
                    training_df['bc_losses'],
                    'blue', 
                    alpha=0.6, 
                    label='BC Loss'
                )
                
                axes[1, 3].set_title('RL Loss vs BC Loss')
                axes[1, 3].set_xlabel('Steps')
                axes[1, 3].set_ylabel('RL Loss', color='red')
                ax_twin.set_ylabel('BC Loss', color='blue')
                axes[1, 3].tick_params(axis='y', labelcolor='red')
                ax_twin.tick_params(axis='y', labelcolor='blue')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[1, 3].legend(lines, labels, loc='upper right')
                axes[1, 3].grid(True)

        plt.tight_layout()
        
        # Ensure directory exists
        progress_dir = f'{self.model_dir}/training_progress'
        os.makedirs(progress_dir, exist_ok=True)
        
        plt.savefig(f'{progress_dir}/training_progress_step_{current_step}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory

    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', f'checkpoint_step_{step}.pth')
        # Assuming your agent has a save method
        if hasattr(self.MADRLagent, 'save_models'):
            self.MADRLagent.save_models(checkpoint_path)
        print(f"Checkpoint saved at step {step}")

    def _plot_the_throughput_state_of_uavs(self, step):
        import matplotlib.pyplot as plt
        """Plot throughput state of UAVs for debugging"""
        fig, axes = plt.subplots(self.env.num_uavs, 1, figsize=(10, 5 * self.env.num_uavs))
        fig.suptitle('UAV Throughput per Demand Area', fontsize=16)

        for uav_id, ax in zip(self.env.uavs.keys(), axes):
            uav = self.env.uavs[uav_id]
            da_throughputs = []
            da_labels = []

            for da in self.env.demand_areas.values():
                if da.uav_id == uav_id and len(da.user_ids) > 0:
                    total_throughput = 0.0
                    for ue_id in da.user_ids:
                        ue = self.env.ues[ue_id]
                        if len(ue.assigned_rb) == 0:
                            continue
                        for rb in ue.assigned_rb:
                            sinr_db = self.env._calculate_sinr(ue, uav, rb)
                            sinr_linear = 10 ** (sinr_db / 10)
                            throughput = rb.bandwidth * np.log2(1 + sinr_linear)
                            total_throughput += throughput
                    da_throughputs.append(total_throughput / len(da.user_ids))
                    da_labels.append(f"DA {da.id} ({da.slice_type}, {da.distance_level})")

            ax.bar(da_labels, da_throughputs, color='skyblue')
            ax.set_title(f'UAV {uav_id} Throughput per Demand Area')
            ax.set_ylabel('Average Throughput (bps)')
            ax.set_xticklabels(da_labels, rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{self.model_dir}/uav_throughput/uav_throughput_step_{step}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory

    def _save_metrics_to_csv(self):
        """Robust version with validation and error handling"""
        import csv
        import os
        
        def save_dict_of_lists_to_csv(metrics_dict, csv_path):
            """Helper function to save dict of lists structure to CSV"""
            if not metrics_dict or not metrics_dict.get('steps'):
                return False
            
            try:
                # Validate all lists have same length
                first_key = next(iter(metrics_dict))
                expected_length = len(metrics_dict[first_key])
                
                for key, values in metrics_dict.items():
                    if len(values) != expected_length:
                        print(f"Warning: Inconsistent length for {key}: {len(values)} vs {expected_length}")
                        # Pad or truncate to match
                        if len(values) < expected_length:
                            values.extend([None] * (expected_length - len(values)))
                        else:
                            metrics_dict[key] = values[:expected_length]
                
                # Convert to list of dicts
                records = []
                for i in range(expected_length):
                    record = {key: values[i] for key, values in metrics_dict.items()}
                    records.append(record)
                
                # Write to CSV
                mode = 'a' if os.path.exists(csv_path) else 'w'
                with open(csv_path, mode, newline='') as csvfile:
                    fieldnames = list(metrics_dict.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if mode == 'w':
                        writer.writeheader()
                    
                    writer.writerows(records)
                
                return True
                
            except Exception as e:
                print(f"Error saving metrics to {csv_path}: {e}")
                return False
        
        # Save step metrics
        if save_dict_of_lists_to_csv(self.step_metrics, f'{self.model_dir}/step_metrics.csv'):
            # Clear on successful save
            for key in self.step_metrics:
                self.step_metrics[key].clear()
        
        # Save training metrics
        if save_dict_of_lists_to_csv(self.training_metrics, f'{self.model_dir}/training_metrics.csv'):
            # Clear on successful save
            for key in self.training_metrics:
                self.training_metrics[key].clear()

    def _update_visualization(self):
        """Update the visualization window"""
        import pygame
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.enable_visualization = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    self.enable_visualization = False
                    return
                elif event.key == pygame.K_c:
                    self.visualizer.show_connections = not self.visualizer.show_connections
                elif event.key == pygame.K_p:
                    self.visualizer.show_paths = not self.visualizer.show_paths
                elif event.key == pygame.K_d:
                    self.visualizer.show_das = not self.visualizer.show_das
                elif event.key == pygame.K_g:
                    self.visualizer.show_grid = not self.visualizer.show_grid
                elif event.key == pygame.K_TAB:
                    if self.visualizer.selected_uav is None:
                        self.visualizer.selected_uav = 0
                    else:
                        self.visualizer.selected_uav = (self.visualizer.selected_uav + 1) % len(self.visualizer.env.uavs)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.visualizer.camera.zoom(-50)
                elif event.button == 5:  # Scroll down
                    self.visualizer.camera.zoom(50)
        
        # Handle mouse dragging
        self.visualizer.handle_mouse()
        
        # Redraw the scene
        self.visualizer.draw_3d_scene()
        self.visualizer.draw_2d_overlay()
        self.visualizer.animation_time += 0.016
        pygame.display.flip()
        self.visualizer.clock.tick(60)

def profile_training_step():
    """Profile a single training step to identify bottlenecks"""
    
    # Initialize environment and agent
    env = NetworkSlicingEnv(config_path="config/environment/default.yaml")
    obs = env.reset()
    
    agent = MADRLAgent(
        num_agents=env.num_uavs,
        obs_dim=len(list(obs.values())[0]),
        action_dim=4 + env.num_das_per_slice * 3
    )
    
    # Warm up (fill buffer)
    print("Warming up buffer...")
    for _ in range(1000):
        actions = agent.select_actions(obs, explore=True)
        next_obs, reward, done, info = env.step(actions)
        agent.store_transition(obs, actions, actions, reward, next_obs, done)
        obs = next_obs
    
    # Profile the actual training loop
    print("Starting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    

    progress_bar = tqdm(range(10000), desc="Training Steps")
    # Profile 100 steps
    for i in progress_bar:
        # Environment step
        actions = agent.select_actions(obs, explore=True)
        next_obs, reward, done, info = env.step(actions)
        agent.store_transition(obs, actions, actions, reward, next_obs, done)
        
        # Training (every 10 steps)
        if (i + 1) % 10 == 0:
            for _ in range(10):
                agent.train()
        
        obs = next_obs
    
    profiler.disable()
    
    # Print results
    print("\n" + "="*80)
    print("PROFILING RESULTS - TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)
    
    print(s.getvalue())
    
    return stats

def main():
    """Main function to run training"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Profile the code')
    parser.add_argument('--visualize', action='store_true', help='Show visualization during training')
    args = parser.parse_args()



    # You can specify a custom config file
    env_config_file = "config/environment/default.yaml"  # or 'config.json' if you have one
    train_config_file = 'config/train/default.yaml'
    checkpoint_file = None
    # checkpoint_file = "saved_models/model130/checkpoints/checkpoint_step_190000.pth"  # or specify a checkpoint path if resuming training
    # Run profiling if specified
    if args.profile:
        # Run profiling instead of training
        profile_training_step()
        return
    
    # Create training manager
    trainer = TrainingManager(env_config_path=env_config_file, 
                              train_config_path=train_config_file,
                              checkpoint_path=checkpoint_file,
                              enable_visualization=args.visualize)


    # Run training
    trainer.train()

if __name__ == "__main__":
    main()
