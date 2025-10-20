# agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from utils import Configuration

class TemperatureSoftmax(nn.Module):
    def __init__(self, dim=-1, init_temp=5.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        self.dim = dim
    
    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=self.dim)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent communication"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_linear(context)
        return output, attention_weights

class ActorNetwork(nn.Module):
    """
    Enhanced Actor with modular encoders for different observation components
    Total obs_dim = 107:
      - UAV state: 5 dims
      - DA info: 90 dims (9 DAs x 10 features)
      - Handover state: 4 dims
      - Surrounding: 8 dims
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Observation component dimensions
        self.uav_state_dim = 5
        self.da_info_dim = 63      # 9 DAs x 7 features
        self.handover_dim = 4
        self.load_pred_dim = 0
        self.surrounding_dim = 8
        
        # Verify total
        total_expected = (self.uav_state_dim + self.da_info_dim + 
                         self.handover_dim + self.load_pred_dim + 
                         self.surrounding_dim)
        assert obs_dim == total_expected, \
            f"obs_dim mismatch: expected {total_expected}, got {obs_dim}"
        
        # ============================================
        # Separate Encoders for Each Component
        # ============================================
        
        # 1. UAV State Encoder (position, power, battery)
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_state_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # 2. DA Info Encoder (CRITICAL for bandwidth allocation!)
        # This is the most important encoder - processes queue, delay, utilization
        self.da_encoder = nn.Sequential(
            nn.Linear(self.da_info_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # 3. Handover State Encoder (CRITICAL for movement decisions!)
        self.handover_encoder = nn.Sequential(
            nn.Linear(self.handover_dim, 16),
            nn.ReLU()
        )
        
        # 4. Load Prediction Encoder (for proactive decisions!)
        # self.load_encoder = nn.Sequential(
        #     nn.Linear(self.load_pred_dim, 16),
        #     nn.ReLU()
        # )
        
        # 5. Surrounding Encoder (for coordination)
        self.surrounding_encoder = nn.Sequential(
            nn.Linear(self.surrounding_dim, 16),
            nn.ReLU()
        )
        
        # ============================================
        # Fusion and Policy Network
        # ============================================
        
        # Combined feature dimension
        combined_dim = 32 + 64 + 16 + 16  # = 128
        
        # Attention mechanism for DA features (helps bandwidth allocation)
        self.da_attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4,
            batch_first=True
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim)
        )
        
        # Learnable temperature for bandwidth softmax
        self.log_bandwidth_temperature = nn.Parameter(
            torch.log(torch.tensor(10.0))
        )
    
    def forward(self, obs, return_entropy=False):
        batch_size = obs.size(0)
        
        # ============================================
        # Split observation into components
        # ============================================
        idx = 0
        
        # UAV state (5)
        uav_state = obs[:, idx:idx+self.uav_state_dim]
        idx += self.uav_state_dim
        
        # DA info (90)
        da_info = obs[:, idx:idx+self.da_info_dim]
        idx += self.da_info_dim
        
        # Handover state (4)
        handover_state = obs[:, idx:idx+self.handover_dim]
        idx += self.handover_dim
        
        # # Load prediction (5)
        # load_state = obs[:, idx:idx+self.load_pred_dim]
        # idx += self.load_pred_dim
        
        # Surrounding (8)
        surrounding_state = obs[:, idx:idx+self.surrounding_dim]
        
        # ============================================
        # Encode each component
        # ============================================
        
        uav_encoded = self.uav_encoder(uav_state)
        handover_encoded = self.handover_encoder(handover_state)
        # load_encoded = self.load_encoder(load_state)
        surrounding_encoded = self.surrounding_encoder(surrounding_state)
        
        # DA encoding with self-attention (helps learn DA relationships)
        da_encoded = self.da_encoder(da_info)
        
        # Apply attention to DA features
        # Reshape for attention: (batch, 9 DAs, 64 features)
        # This helps the network focus on critical DAs (e.g., URLLC with high queue)
        da_attended, _ = self.da_attention(
            da_encoded.unsqueeze(1),
            da_encoded.unsqueeze(1),
            da_encoded.unsqueeze(1)
        )
        da_attended = da_attended.squeeze(1)
        
        # ============================================
        # Combine all features
        # ============================================
        
        combined = torch.cat([
            uav_encoded,         # 32
            da_attended,         # 64
            handover_encoded,    # 16
            # load_encoded,      # 16
            surrounding_encoded  # 16
        ], dim=-1)  # Total: 128

        # ============================================
        # Generate actions
        # ============================================
        
        out = self.policy_net(combined)
        
        # Split into action components
        position = torch.tanh(out[:, :3])  # [-1, 1] for movement
        power = torch.sigmoid(out[:, 3:4])  # [0, 1] for power
        
        # Bandwidth allocation (9 DAs)
        bandwidth_logits = out[:, 4:]
        
        # Temperature-scaled softmax
        temperature = torch.exp(self.log_bandwidth_temperature).clamp(2.0, 30.0)
        bandwidth = F.softmax(bandwidth_logits / temperature, dim=-1)
        
        # Combine all actions
        actions = torch.cat([position, power, bandwidth], dim=-1)
        
        if return_entropy:
            # Calculate entropy for exploration bonus
            bandwidth_dist = torch.distributions.Categorical(probs=bandwidth)
            entropy = bandwidth_dist.entropy()
            return actions, entropy
        
        return actions
    
class CriticNetwork(nn.Module):
    """
    Centralized Critic with proper dimensions for 112-dim observations
    Input: all states (112 × num_agents) + all actions (13 × num_agents)
    """
    def __init__(self, state_dim, action_dim, num_agents=5):  # Changed to 5 UAVs
        super().__init__()
        
        # Input: concatenated states and actions from all agents
        input_dim = (state_dim + action_dim) * num_agents
        # For 5 agents: (112 + 13) × 5 = 625
        
        # Deeper network for better value estimation
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
        # Initialize last layer with small weights
        self.net[-1].weight.data.mul_(0.1)
        self.net[-1].bias.data.mul_(0.1)
    
    def forward(self, states, actions):
        """
        Args:
            states: (batch, num_agents, state_dim)
            actions: (batch, num_agents, action_dim)
        Returns:
            Q-values: (batch, 1)
        """
        # Flatten agents dimension
        x = torch.cat([states.flatten(1), actions.flatten(1)], dim=1)
        return self.net(x)

class ExperienceBuffer:
    """Experience replay buffer for MADRL"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        """Add experience tuple to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.FloatTensor(np.array([e[1] for e in experiences]))
        greedy_actions = torch.FloatTensor(np.array([e[2] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[3] for e in experiences]))
        next_states = torch.FloatTensor(np.array([e[4] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[5] for e in experiences]))

        return states, actions, greedy_actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class BCWeightScheduler:
    """
    Schedule BC weight over training to gradually shift from BC to RL
    """
    def __init__(self, 
                 initial_weight,
                 final_weight,
                 transition_steps,
                 schedule_type):
        """
        Args:
            initial_weight: Starting BC weight (e.g., 0.8 = 80% BC, 20% RL)
            final_weight: Ending BC weight (e.g., 0.2 = 20% BC, 80% RL)
            transition_steps: Steps to transition from initial to final
            schedule_type: 'linear', 'cosine', or 'exponential'
        """
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.transition_steps = transition_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def step(self):
        """Advance scheduler by one step"""
        self.current_step += 1
    
    def get_weight(self):
        """Get current BC weight"""
        if self.current_step >= self.transition_steps:
            return self.final_weight
        
        progress = self.current_step / self.transition_steps
        
        if self.schedule_type == 'linear':
            # Linear decay
            weight = self.initial_weight - progress * (self.initial_weight - self.final_weight)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing (smooth decay)
            import math
            weight = self.final_weight + 0.5 * (self.initial_weight - self.final_weight) * \
                     (1 + math.cos(math.pi * progress))
        
        elif self.schedule_type == 'exponential':
            # Exponential decay
            import math
            decay_rate = math.log(self.final_weight / self.initial_weight) / self.transition_steps
            weight = self.initial_weight * math.exp(decay_rate * self.current_step)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return weight
    
    def __repr__(self):
        return (f"BCWeightScheduler(current={self.get_weight():.3f}, "
                f"step={self.current_step}/{self.transition_steps})")

class MADRLAgent:
    """Multi-Agent Deep Reinforcement Learning Agent for UAV Network Slicing"""
    def __init__(self, 
                 num_agents: int,
                 obs_dim: int,
                 action_dim: int,
                 training: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config_path = "./config/agents/default.yaml"):
        
        
        self.config = Configuration(config_path)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = self.config.maddpg.gamma
        self.tau = self.config.maddpg.tau
        self.buffer_size = self.config.maddpg.buffer_size
        self.batch_size = self.config.maddpg.batch_size
        self.lr_actor = self.config.maddpg.actor_lr
        self.lr_critic = self.config.maddpg.critic_lr
        self.device = torch.device(device)
    

        # Initialize actor networks for each agent
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []
        

        for i in range(num_agents):
            actor = ActorNetwork(obs_dim, action_dim).to(self.device)
            actor_target = ActorNetwork(obs_dim, action_dim).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.lr_actor))

        # Initialize centralized critic
        self.critic = CriticNetwork(obs_dim, action_dim, num_agents).to(self.device)
        self.critic_target = CriticNetwork(obs_dim, action_dim, num_agents).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Experience buffer
        self.buffer = ExperienceBuffer(self.buffer_size)
        
        # Exploration parameters
        
        self.exploration_noise = self.config.exploration.gaussian_init
        self.noise_decay = self.config.exploration.gaussian_decay
        self.min_noise = self.config.exploration.gaussian_min
        if not training:
            self.exploration_noise = 0.0  # No exploration during evaluation

    def select_actions(self, observations: Dict[int, np.ndarray], 
                    explore: bool = True) -> Dict[int, np.ndarray]:
        actions = {}
        
        for agent_id, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            self.actors[agent_id].eval()
            
            with torch.no_grad():
                action = self.actors[agent_id](obs_tensor).cpu().numpy()[0]
            
            # Add SMALL additive noise (not replacement)
            if explore and self.exploration_noise > 0:
                # Position: small Gaussian noise
                pos_noise = np.random.normal(0, self.exploration_noise, size=3)
                action[:3] = np.clip(action[:3] + pos_noise, -1, 1)
                
                # Power: small Gaussian noise
                power_noise = np.random.normal(0, self.exploration_noise, size=1)
                action[3:4] = np.clip(action[3:4] + power_noise, 0, 1)
                
                # Bandwidth: Dirichlet noise for valid distribution
                if len(action) > 4:
                    alpha = 20.0 / (self.exploration_noise + 0.1)  # Higher alpha = less noise
                    noise = np.random.dirichlet(alpha * action[4:] + 0.1)
                    action[4:] = 0.6 * action[4:] + 0.4 * noise
                    action[4:] = action[4:] / action[4:].sum()
                
            actions[agent_id] = action
        
        return actions
    
    def store_transition(self, observations: Dict[int, np.ndarray],
                        actions: Dict[int, np.ndarray],
                        greedy_actions: Dict[int, np.ndarray],
                        reward: float,
                        next_observations: Dict[int, np.ndarray],
                        done: bool):
        """Store transition in replay buffer"""
        # Convert dictionaries to arrays
        obs_array = np.concatenate([observations[i] for i in range(self.num_agents)])
        action_array = np.concatenate([actions[i] for i in range(self.num_agents)])
        greedy_action_array = np.concatenate([greedy_actions[i] for i in range(self.num_agents)])
        next_obs_array = np.concatenate([next_observations[i] for i in range(self.num_agents)])

        self.buffer.push((obs_array, action_array, greedy_action_array, reward, next_obs_array, done))

    def train(self):
        """Train the MADRL agent"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Set all models to training mode
        for actor in self.actors:
            actor.train()
        for actor_target in self.actor_targets:
            actor_target.train()  # Targets should also be in train mode
        self.critic.train()
        self.critic_target.train()
        
        # Sample batch
        states, actions, greedy_actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(-1)
        
        # Reshape for individual agents
        states_per_agent = states.view(self.batch_size, self.num_agents, self.obs_dim)
        actions_per_agent = actions.view(self.batch_size, self.num_agents, self.action_dim)
        greedy_actions_per_agent = greedy_actions.view(self.batch_size, self.num_agents, self.action_dim)
        next_states_per_agent = next_states.view(self.batch_size, self.num_agents, self.obs_dim)
        
        # Update critic
        with torch.no_grad():
            # Get target actions from target actors
            current_actionss = []
            for i in range(self.num_agents):
                current_actions = self.actor_targets[i](next_states_per_agent[:, i, :])
                current_actionss.append(current_actions)
            
            current_actionss = torch.stack(current_actionss, dim=1)
            current_actionss = current_actionss.view(self.batch_size, -1)
            
            # Compute target Q-value
            target_q = self.critic_target(next_states_per_agent.view(self.batch_size, -1), current_actionss)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actors
        actor_losses = []

        for i in range(self.num_agents):
            # Get current actions from all agents
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # For agent i, use its actor network
                    action = self.actors[j](states_per_agent[:, j, :])
                else:
                    # For other agents, use their current actions (detached)
                    action = actions_per_agent[:, j, :].detach()
                current_actions.append(action)
            
            current_actions = torch.stack(current_actions, dim=1)
            current_actions = current_actions.view(self.batch_size, -1)
            
            # Actor loss (negative Q-value for gradient descent)
            actor_loss = -self.critic(states, current_actions).mean() / 200
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            actor_losses.append(actor_loss.item())
        
        # Soft update target networks
        self._soft_update()
        
        return {
            'actor_losses': actor_losses,
            'critic_loss': critic_loss.item()
        }
    
    def train_BC(self):
        """Train the MADRL agent"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Set all models to training mode
        for actor in self.actors:
            actor.train()
        for actor_target in self.actor_targets:
            actor_target.train()  # Targets should also be in train mode
        self.critic.train()
        self.critic_target.train()
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(-1)
        
        # Reshape for individual agents
        states_per_agent = states.view(self.batch_size, self.num_agents, self.obs_dim)
        actions_per_agent = actions.view(self.batch_size, self.num_agents, self.action_dim)
        next_states_per_agent = next_states.view(self.batch_size, self.num_agents, self.obs_dim)
        
        # Update critic
        with torch.no_grad():
            # Get target actions from target actors
            current_actionss = []
            for i in range(self.num_agents):
                current_actions = self.actor_targets[i](next_states_per_agent[:, i, :])
                current_actionss.append(current_actions)
            
            current_actionss = torch.stack(current_actionss, dim=1)
            current_actionss = current_actionss.view(self.batch_size, -1)
            
            # Compute target Q-value
            target_q = self.critic_target(next_states_per_agent.view(self.batch_size, -1), current_actionss)
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actors
        actor_losses = []

        for i in range(self.num_agents):
            # Get current actions from all agents
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # For agent i, use its actor network
                    action = self.actors[j](states_per_agent[:, j, :])
                else:
                    # For other agents, use their current actions (detached)
                    action = actions_per_agent[:, j, :].detach()
                current_actions.append(action)
            
            current_actions = torch.stack(current_actions, dim=1)
            current_actions = current_actions.view(self.batch_size, -1)
        
            # RL loss
            rl_loss = -self.critic(states, current_actions).mean()
            
            # ADD THIS: Behavioral Cloning loss
            # actions_per_agent[:, i, :] are greedy's actions
            greedy_action = self.actors[i](states_per_agent[:, i, :])
            current_actions = actions_per_agent[:, i, :].detach()
            
            # MSE loss for continuous actions
            bc_loss = F.mse_loss(greedy_action, current_actions)
            
            # Or for bandwidth specifically (since it's critical):
            bandwidth_pred = greedy_action[:, 4:]
            bandwidth_target = current_actions[:, 4:]
            bandwidth_bc_loss = F.mse_loss(bandwidth_pred, bandwidth_target)
            
            # Combined loss
            # total_loss = rl_loss + 0.5 * bc_loss  # Adjust weight
            # total_loss = rl_loss + 0.3 * bc_loss + 0.2 * bandwidth_bc_loss
            total_loss = rl_loss * (1 - self.bc_weight) + (0.6 * bc_loss + 0.4 * bandwidth_bc_loss) * self.bc_weight
            
            self.actor_optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            actor_losses.append(total_loss.item())
        
        # Soft update target networks
        self._soft_update()
        
        return {
            'actor_losses': actor_losses,
            'critic_loss': critic_loss.item()
        }
    
    # agents.py - Add to MADRLAgent class

    def train_rl_with_bc_regularization(self, bc_weight):
        """
        RL training with BC regularization to prevent catastrophic forgetting
        
        Args:
            bc_weight: Weight for BC loss (0.0-1.0)
                    - 0.0 = pure RL (dangerous!)
                    - 0.5 = balanced BC + RL (recommended)
                    - 1.0 = pure BC (no improvement)
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Set to training mode
        for actor in self.actors:
            actor.train()
        for actor_target in self.actor_targets:
            actor_target.eval()
        self.critic.train()
        self.critic_target.eval()
        
        # Sample batch
        states, actions, greedy_actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        greedy_actions = greedy_actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(-1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(-1)
        
        # Reshape for individual agents
        states_per_agent = states.view(self.batch_size, self.num_agents, self.obs_dim)
        actions_per_agent = actions.view(self.batch_size, self.num_agents, self.action_dim)
        greedy_actions_per_agent = greedy_actions.view(self.batch_size, self.num_agents, self.action_dim)
        next_states_per_agent = next_states.view(self.batch_size, self.num_agents, self.obs_dim)
        
        # ============================================
        # Update Critic (Standard MADDPG)
        # ============================================
        with torch.no_grad():
            # Get target actions from target actors
            current_actionss = []
            for i in range(self.num_agents):
                current_actions = self.actor_targets[i](next_states_per_agent[:, i, :])
                current_actionss.append(current_actions)
            
            current_actionss = torch.stack(current_actionss, dim=1)
            current_actionss_flat = current_actionss.view(self.batch_size, -1)
            
            # Compute target Q-value
            target_q = self.critic_target(
                next_states_per_agent.view(self.batch_size, -1), 
                current_actionss_flat
            )
            target_value = rewards + self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ============================================
        # Update Actors with BC Regularization
        # ============================================
        actor_losses = []
        rl_losses = []
        bc_losses = []
        
        for i in range(self.num_agents):
            # Get current actions from all agents
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    # For agent i, use its actor network
                    action = self.actors[j](states_per_agent[:, j, :])
                else:
                    # For other agents, use their actions from buffer (detached)
                    action = actions_per_agent[:, j, :].detach()
                current_actions.append(action)
            
            current_actions = torch.stack(current_actions, dim=1)
            current_actions_flat = current_actions.view(self.batch_size, -1)
            
            # Get actor's predicted action
            greedy_action = greedy_actions_per_agent[:, i, :].detach()
            
            
            # ============================================
            # RL Loss: Maximize Q-values
            # ============================================
            q_values = self.critic(states, current_actions_flat)
            rl_loss_raw = -q_values.mean()
            
            # Normalize RL loss to BC scale
            # Option 1: Simple scaling
            rl_loss_normalized = rl_loss_raw / 20000.0  # Adjust based on your Q-value scale
            
            
            # ============================================
            # BC Loss: Stay close to greedy
            # ============================================
            # Overall BC loss
            # bc_loss_overall = F.mse_loss(greedy_action, current_actions)
            
            # Component-specific BC losses (for better control)
            action = self.actors[i](states_per_agent[:, i, :])
            position_bc = F.mse_loss(greedy_action[:, :3], action[:, :3])
            power_bc = F.mse_loss(greedy_action[:, 3:4], action[:, 3:4])
            bandwidth_bc = F.mse_loss(greedy_action[:, 4:], action[:, 4:])
            
            # Weighted BC loss (emphasize bandwidth since it's critical)
            bc_loss = (
                0.3 * position_bc + 
                0.2 * power_bc + 
                0.5 * bandwidth_bc
            )
            
            # ============================================
            # Combined Loss
            # ============================================
            total_loss = (1 - bc_weight) * rl_loss_normalized + bc_weight * bc_loss
            
            # Backward pass
            self.actor_optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            # Record losses
            actor_losses.append(total_loss.item())
            rl_losses.append(rl_loss_normalized.item())
            bc_losses.append(bc_loss.item())
        
        # Soft update target networks
        self._soft_update()
        
        return {
            'actor_losses': actor_losses,
            'critic_loss': critic_loss.item(),
            'rl_losses': rl_losses,
            'bc_losses': bc_losses,
            'bc_weight': bc_weight
        }
    
    def _soft_update(self):
        """Soft update target networks"""
        # Update critic target
        for param, target_param in zip(self.critic.parameters(), 
                                      self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)
        
        # Update actor targets
        for i in range(self.num_agents):
            for param, target_param in zip(self.actors[i].parameters(), 
                                          self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + 
                                      (1 - self.tau) * target_param.data)
    
    def save_models(self, path: str):
        """Save all models"""
        checkpoint = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actors': [actor.state_dict() for actor in self.actors],
            'actor_targets': [actor.state_dict() for actor in self.actor_targets],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'exploration_noise': self.exploration_noise
        }
        torch.save(checkpoint, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.actor_targets[i].load_state_dict(checkpoint['actor_targets'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
        
        self.exploration_noise = checkpoint['exploration_noise']
        print(f"Models loaded from {path}")

    def reset_optimizer_state(self):
        """Reset Adam's momentum when switching from BC to RL"""
        print("Resetting optimizer state for BC → RL transition...")
        
        # Reset actor optimizers
        for i, opt in enumerate(self.actor_optimizers):
            state = opt.state_dict()
            # Clear the state dict (momentum, variance)
            opt.state_dict()['state'] = {}
            opt.load_state_dict(state)
            print(f"  Actor {i} optimizer reset")
        
        # Also reset critic optimizer
        state = self.critic_optimizer.state_dict()
        self.critic_optimizer.state_dict()['state'] = {}
        self.critic_optimizer.load_state_dict(state)
        print("  Critic optimizer reset")

        # Also reset exploration noise to initial value
        self.exploration_noise = self.config.exploration.gaussian_init
        print(f"  Exploration noise reset to {self.exploration_noise}")

    def prepare_for_rl_training(self):
        """CRITICAL: Reset everything for RL"""
        
        print("="*60)
        print("PREPARING FOR RL (Preventing Catastrophic Forgetting)")
        print("="*60)
        
        # 1. CREATE FRESH OPTIMIZERS (no BC momentum!)
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=1e-5)  # Fresh state!
            for actor in self.actors
        ]
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=3e-5
        )
        
        # 2. Initialize Q-value normalization
        self.q_mean_ema = 0.0
        self.q_std_ema = 1.0
        
        print("✓ Created fresh optimizers (no BC momentum)")
        print("✓ Initialized Q-value normalization")
        print("="*60)

    

                                             
