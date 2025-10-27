import numpy as np
import matplotlib.pyplot as plt
from environment import NetworkSlicingEnv
from agents import MADRLAgent
from utils import Configuration
import os
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import json
from datetime import datetime
from tqdm import tqdm
import glob
import pandas as pd


class BaselineAgent(ABC):
    """Abstract base class for baseline agents"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select actions for all agents"""
        pass
    
    def get_name(self) -> str:
        """Return agent name for logging"""
        return self.__class__.__name__


class BaselineAgent(ABC):
    """Abstract base class for baseline agents"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select actions for all agents"""
        pass
    
    def get_name(self) -> str:
        """Return agent name for logging"""
        return self.__class__.__name__

class RandomAgent(BaselineAgent):
    """Random baseline that selects actions uniformly at random with reasonable constraints"""
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select random actions with constraints to avoid invalid states"""
        actions = {}
        for agent_id in observations.keys():
            action = np.zeros(self.action_dim)
            
            # Position changes: random but reasonable [-1, 1]
            action[0:3] = np.random.uniform(-1, 1, 3)
            # Power: random but not too low (minimum 0% power to avoid SINR issues)
            action[3] = np.random.uniform(0, 1.0)
            
            # Bandwidth allocations: random positive values
            if self.action_dim > 4:
                action[4:] = np.random.uniform(0.01, 1.0, self.action_dim - 4)
                # Normalize to sum to reasonable value
                if np.sum(action[4:]) > 0:
                    action[4:] = action[4:] / np.sum(action[4:])
            
            actions[agent_id] = action
        return actions

class GreedyAgent(BaselineAgent):
    """Greedy baseline that makes locally optimal decisions"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env: NetworkSlicingEnv):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        self.power_strategy = "proportional"  # "max", "proportional", or "adaptive"
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        """Select greedy actions based on heuristics"""
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # 1. Position: Move towards center of mass of assigned UEs
            assigned_ues = [ue for ue in self.env.ues.values() 
                          if ue.assigned_uav == uav_id and ue.is_active]
            
            if assigned_ues:
                # Calculate center of mass
                ue_positions = np.array([ue.position for ue in assigned_ues])
                center_of_mass = np.mean(ue_positions, axis=0)
                
                # Direction to center of mass
                direction = center_of_mass - uav.position
                distance = np.linalg.norm(direction)
                
                if distance > 10:  # Only move if significantly far
                    direction = direction / (distance + 1e-6)  # Normalize
                    # Scale to action space [-1, 1]
                    action[0] = np.clip(direction[0] * 0.5, -1, 1)
                    action[1] = np.clip(direction[1] * 0.5, -1, 1)
                    action[2] = np.clip(direction[2] * 0.3, -1, 1)  # Less aggressive on height
            
            # 2. Power: Use proportional to number of UEs or max
            if self.power_strategy == "max":
                action[3] = 1.0  # Maximum power
            elif self.power_strategy == "proportional":
                # Proportional to number of assigned UEs
                ue_ratio = len(assigned_ues) / max(1, len(self.env.ues))
                action[3] = np.clip(ue_ratio * 2, 0.3, 1.0)  # At least 30% power
            else:  # adaptive
                # Based on average distance to UEs
                if assigned_ues:
                    avg_distance = np.mean([np.linalg.norm(ue.position - uav.position) 
                                          for ue in assigned_ues])
                    # More power for farther UEs
                    action[3] = np.clip(avg_distance / 500, 0.3, 1.0)
                else:
                    action[3] = 0.3
            
            # 3. Bandwidth allocation: Proportional to number of UEs in each DA
            uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav_id]
            
            if uav_das:
                total_ues = sum(len(da.user_ids) for da in uav_das)
                
                for i, da in enumerate(uav_das):
                    if i + 4 < self.action_dim:
                        # Allocate bandwidth proportionally to number of users
                        if total_ues > 0:
                            ue_ratio = len(da.user_ids) / total_ues
                            
                            # Boost allocation for high-priority slices
                            if da.slice_type == "urllc":
                                ue_ratio *= 1.5
                            elif da.slice_type == "embb":
                                ue_ratio *= 1.2
                            
                            # Boost allocation for far UEs (high SINR level needs more resources)
                            if da.distance_level == "Far":
                                ue_ratio *= 1.3
                            
                            action[4 + i] = np.clip(ue_ratio, 0, 1)
                        else:
                            action[4 + i] = 0.1  # Small baseline allocation
            
            actions[uav_id] = action
        
        return actions

# ============================================================================
# ENHANCED GREEDY VARIANTS
# ============================================================================

class MaxMinFairnessGreedyAgent(BaselineAgent):
    """
    Max-Min Fairness Greedy: Prioritize worst-served users
    - Position: Target users with lowest resource allocation
    - Power: Moderate usage for stability
    - Bandwidth: Inverse proportion - fewer resources → higher priority
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # 1. Position: Move towards worst-served users
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if assigned_ues:
                # Rank users by resource allocation
                ue_scores = [(ue, len(ue.assigned_rb) if ue.assigned_rb else 0) 
                            for ue in assigned_ues]
                ue_scores.sort(key=lambda x: x[1])
                
                # Target worst 30% of users
                worst_count = max(1, len(ue_scores) // 3)
                worst_ues = [ue for ue, _ in ue_scores[:worst_count]]
                
                target_pos = np.mean([ue.position for ue in worst_ues], axis=0)
                direction = target_pos - uav.position
                distance = np.linalg.norm(direction)
                
                if distance > 5.0:
                    direction = (direction / distance) * 0.35
                    action[0:3] = np.clip(direction, -1.0, 1.0)
            
            # 2. Power: Moderate and stable
            num_users = len(assigned_ues)
            if num_users > 0:
                action[3] = 0.7
            else:
                action[3] = 0.4
            
            # 3. Bandwidth: Max-min fairness allocation
            uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav_id]
            
            if uav_das:
                allocations = []
                for da in sorted(uav_das, key=lambda d: d.id):
                    if len(da.user_ids) == 0:
                        allocations.append(0.02)
                    else:
                        # Inverse proportion: fewer existing RBs → higher priority
                        avg_rbs = len(da.RB_ids_list) / max(1, len(da.user_ids))
                        urgency = 1.0 / (avg_rbs + 0.5)
                        slice_weight = self.env.slice_weights[da.slice_type]
                        allocations.append(urgency * slice_weight)
                
                allocations = np.array(allocations)
                action[4:] = allocations / (allocations.sum() + 1e-6)
            
            actions[uav_id] = action
        
        return actions

class ProportionalFairGreedyAgent(BaselineAgent):
    """
    Proportional Fair Greedy: Balance efficiency and fairness
    - Position: Towards geometric center with efficiency focus
    - Power: High but battery-aware
    - Bandwidth: Proportional fairness metric maximization
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # 1. Position: Geometric center
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if assigned_ues:
                center = np.mean([ue.position for ue in assigned_ues], axis=0)
                direction = center - uav.position
                distance = np.linalg.norm(direction)
                
                if distance > 10.0:
                    direction = (direction / distance) * 0.45
                    action[0:3] = np.clip(direction, -1.0, 1.0)
            
            # 2. Power: High utilization with battery awareness
            battery_ratio = uav.current_battery / uav.battery_capacity
            if len(assigned_ues) > 0:
                base_power = 0.85
                battery_penalty = 1.0 if battery_ratio > 0.5 else 0.7
                action[3] = base_power * battery_penalty
            else:
                action[3] = 0.35
            
            # 3. Bandwidth: Proportional fairness
            uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav_id]
            
            if uav_das:
                allocations = []
                for da in sorted(uav_das, key=lambda d: d.id):
                    if len(da.user_ids) == 0:
                        allocations.append(0.01)
                    else:
                        # Proportional fair: log utility approximation
                        current_rbs = len(da.RB_ids_list)
                        num_users = len(da.user_ids)
                        
                        # Marginal utility (derivative of log)
                        marginal_utility = num_users / (current_rbs + 1.0)
                        
                        # Weight by slice priority
                        slice_weight = self.env.slice_weights[da.slice_type]
                        
                        allocations.append(marginal_utility * slice_weight)
                
                allocations = np.array(allocations)
                action[4:] = allocations / (allocations.sum() + 1e-6)
            
            actions[uav_id] = action
        
        return actions

class LoadBalancingGreedyAgent(BaselineAgent):
    """
    Load Balancing Greedy: Distribute load evenly across UAVs
    - Position: Avoid clustering with other UAVs
    - Power: Balanced based on relative load
    - Bandwidth: Equal per-user allocation within each DA
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        self.repulsion_factor = 0.2
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # 1. Position: Attraction to users + Repulsion from other UAVs
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            attraction = np.zeros(3)
            if assigned_ues:
                center = np.mean([ue.position for ue in assigned_ues], axis=0)
                attraction = center - uav.position
                if np.linalg.norm(attraction) > 1e-6:
                    attraction = attraction / np.linalg.norm(attraction)
            
            # Repulsion from other UAVs
            repulsion = np.zeros(3)
            for other_uav in self.env.uavs.values():
                if other_uav.id != uav_id:
                    direction = uav.position - other_uav.position
                    distance = np.linalg.norm(direction)
                    if distance < 300.0 and distance > 1e-6:
                        repulsion += (direction / distance) * (1.0 - distance / 300.0)
            
            if np.linalg.norm(repulsion) > 1e-6:
                repulsion = (repulsion / np.linalg.norm(repulsion)) * self.repulsion_factor
            
            combined = attraction + repulsion
            if np.linalg.norm(combined) > 1e-6:
                action[0:3] = np.clip(combined / np.linalg.norm(combined) * 0.4, -1.0, 1.0)
            
            # 2. Power: Proportional to relative load
            my_load = len(assigned_ues)
            total_load = sum(len([ue for ue in self.env.ues.values() 
                                 if ue.assigned_uav == other_uav.id and ue.is_active])
                           for other_uav in self.env.uavs.values())
            
            if total_load > 0:
                load_ratio = my_load / total_load * self.num_agents
                action[3] = np.clip(0.5 + load_ratio * 0.3, 0.4, 0.9)
            else:
                action[3] = 0.4
            
            # 3. Bandwidth: Equal per-user allocation
            uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav_id]
            
            if uav_das:
                # Total users across all DAs
                total_users = sum(len(da.user_ids) for da in uav_das)
                
                allocations = []
                for da in sorted(uav_das, key=lambda d: d.id):
                    if total_users > 0:
                        allocations.append(len(da.user_ids) / total_users)
                    else:
                        allocations.append(1.0 / len(uav_das))
                
                action[4:] = np.array(allocations)
            
            actions[uav_id] = action
        
        return actions

class EnergyAwareGreedyAgent(BaselineAgent):
    """
    Energy-Aware Greedy: Optimize for energy efficiency
    - Position: Minimize movement distance
    - Power: Conservative, scaled by necessity
    - Bandwidth: Efficient allocation to reduce interference
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # 1. Position: Minimal movement
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if assigned_ues:
                center = np.mean([ue.position for ue in assigned_ues], axis=0)
                direction = center - uav.position
                distance = np.linalg.norm(direction)
                
                # Only move if really necessary
                if distance > 50.0:
                    direction = (direction / distance) * 0.25
                    action[0:3] = np.clip(direction, -1.0, 1.0)
            
            # 2. Power: Conservative with minimum viable power
            battery_ratio = uav.current_battery / uav.battery_capacity
            num_users = len(assigned_ues)
            
            if num_users > 0:
                # Calculate minimum power needed
                avg_distance = np.mean([np.linalg.norm(ue.position - uav.position) 
                                       for ue in assigned_ues])
                
                # Power scales with distance and battery state
                base_power = np.clip(avg_distance / 500.0, 0.3, 0.7)
                battery_scaling = 0.8 if battery_ratio < 0.4 else 1.0
                
                action[3] = base_power * battery_scaling
            else:
                action[3] = 0.3
            
            # 3. Bandwidth: Focus on nearby users (lower power needs)
            uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav_id]
            
            if uav_das:
                allocations = []
                for da in sorted(uav_das, key=lambda d: d.id):
                    if len(da.user_ids) == 0:
                        allocations.append(0.01)
                    else:
                        # Prioritize nearby users (lower transmission energy)
                        distance_bonus = {'Near': 0.6, 'Medium': 1.0, 'Far': 1.5}[da.distance_level]
                        slice_weight = self.env.slice_weights[da.slice_type]
                        num_users = len(da.user_ids)
                        
                        allocations.append(num_users * slice_weight * distance_bonus)
                
                allocations = np.array(allocations)
                action[4:] = allocations / (allocations.sum() + 1e-6)
            
            actions[uav_id] = action
        
        return actions

class AdaptiveHeightGreedyAgent(BaselineAgent):
    """
    Adaptive Height Strategy:
    - Spread out users → Fly higher (wider coverage, beam angle covers more area)
    - Clustered users → Fly lower (better signal quality, higher SINR)
    - Adjusts height dynamically based on user spatial distribution
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        self.min_height = env.uav_fly_range_h[0]
        self.max_height = env.uav_fly_range_h[1]
        self.position_aggressiveness = 0.4
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            # Get assigned UEs
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4  # Minimal power
                action[4:] = 1.0 / (self.action_dim - 4)  # Equal bandwidth
                actions[uav_id] = action
                continue
            
            # 1. Calculate optimal height based on user spread
            target_height = self._compute_optimal_height(uav, assigned_ues)
            
            # 2. Position: Move towards weighted centroid (XY) + target height (Z)
            position_action = self._compute_smart_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Power: Adjust based on height (higher altitude needs more power)
            power_action = self._compute_height_aware_power(uav, assigned_ues, target_height)
            action[3] = power_action
            
            # 4. Bandwidth: Standard allocation
            bandwidth_actions = self._compute_intelligent_bandwidth(uav)
            action[4:] = bandwidth_actions
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_optimal_height(self, uav, assigned_ues) -> float:
        """
        Calculate optimal height based on user spatial distribution.
        
        Strategy:
        - High user spread → Fly higher (beam covers more area)
        - Low user spread → Fly lower (better SINR for clustered users)
        - Consider beam angle constraint
        """
        if not assigned_ues:
            return (self.min_height + self.max_height) / 2
        
        # Calculate user spread (standard deviation of distances from centroid)
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        centroid_2d = np.mean(ue_positions, axis=0)
        
        distances = [np.linalg.norm(ue.position[:2] - centroid_2d) for ue in assigned_ues]
        user_spread = np.std(distances) if len(distances) > 1 else 0
        
        # Calculate max distance to farthest user
        max_distance = max(distances) if distances else 0
        
        # Calculate required height for beam coverage
        # Given beam angle θ and distance d, minimum height h = d / tan(θ)
        beam_angle_rad = np.radians(uav.beam_angle)
        min_height_for_coverage = max_distance / np.tan(beam_angle_rad) if beam_angle_rad > 0 else self.min_height
        
        # Strategy: Balance between coverage and quality
        if user_spread > 100:  # Highly spread out
            # Prioritize coverage - fly higher
            target_height = max(min_height_for_coverage * 1.2, self.max_height * 0.7)
        elif user_spread > 50:  # Moderately spread
            # Balanced approach
            target_height = max(min_height_for_coverage * 1.1, self.max_height * 0.5)
        else:  # Clustered users
            # Prioritize signal quality - fly lower
            target_height = max(min_height_for_coverage * 0.9, self.min_height * 1.5)
        
        # Clamp to allowed range
        target_height = np.clip(target_height, self.min_height, self.max_height)
        
        return target_height
    
    def _compute_smart_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Position with height awareness"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Weighted centroid in XY plane
        weighted_positions = []
        total_weight = 0.0
        
        for ue in assigned_ues:
            slice_weight = self.env.slice_weights[ue.slice_type]
            distance = np.linalg.norm(ue.position[:2] - uav.position[:2])
            distance_weight = 1.0 / (distance + 20.0)
            
            combined_weight = slice_weight * distance_weight
            weighted_positions.append(ue.position[:2] * combined_weight)
            total_weight += combined_weight
        
        target_xy = sum(weighted_positions) / total_weight if total_weight > 0 else uav.position[:2]
        
        # XY direction
        direction_xy = target_xy - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * self.position_aggressiveness
        else:
            direction_xy = np.zeros(2)
        
        # Z direction (height adjustment)
        current_height = uav.position[2]
        height_diff = target_height - current_height
        
        # Smooth height adjustment (slower than XY movement)
        height_action = np.clip(height_diff / 100.0, -0.3, 0.3)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_height_aware_power(self, uav, assigned_ues, target_height) -> float:
        """Power allocation considering altitude"""
        if not assigned_ues:
            return 0.4
        
        # Base power factors
        num_users = len(assigned_ues)
        demand_factor = min(num_users / 20.0, 1.0)
        
        # Height penalty: Higher altitude needs more power
        # Path loss increases with distance, so compensate
        height_factor = 1.0 + (target_height - self.min_height) / (self.max_height - self.min_height) * 0.3
        
        # Average distance to users
        avg_distance_2d = np.mean([np.linalg.norm(ue.position[:2] - uav.position[:2]) 
                                   for ue in assigned_ues])
        # 3D distance will be sqrt(distance_2d^2 + height^2)
        avg_distance_3d = np.sqrt(avg_distance_2d**2 + target_height**2)
        distance_factor = np.clip(avg_distance_3d / 400.0, 0.5, 1.0)
        
        # Slice priority
        slice_priorities = [self.env.slice_weights[ue.slice_type] for ue in assigned_ues]
        avg_priority = np.mean(slice_priorities)
        
        # Battery constraint
        battery_ratio = uav.current_battery / uav.battery_capacity
        battery_factor = 0.6 if battery_ratio < 0.3 else (0.8 if battery_ratio < 0.5 else 1.0)
        
        # Combine
        target_power = 0.4 + 0.45 * demand_factor * height_factor * distance_factor * avg_priority * battery_factor
        
        return np.clip(target_power, 0.4, 0.95)
    
    def _compute_intelligent_bandwidth(self, uav) -> np.ndarray:
        """Standard bandwidth allocation"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                priority = (len(da.user_ids) * 
                           self.env.slice_weights[da.slice_type] *
                           {'Near': 1.0, 'Medium': 1.4, 'Far': 1.8}[da.distance_level])
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        
        # Minimum allocation
        for i, da in enumerate(sorted(uav_das, key=lambda d: d.id)):
            if len(da.user_ids) > 0:
                allocations[i] = max(allocations[i], 0.03)
        
        return allocations / allocations.sum()

class CoverageMaximizationGreedyAgent(BaselineAgent):
    """
    Coverage Maximization Strategy:
    - Primary goal: Maximize number of UEs within beam coverage
    - Adjusts height to ensure all UEs are covered by beam angle
    - Sacrifices some SINR for broader coverage
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4
                action[4:] = 1.0 / (self.action_dim - 4)
                actions[uav_id] = action
                continue
            
            # 1. Find height that covers ALL users
            target_height = self._compute_coverage_maximizing_height(uav, assigned_ues)
            
            # 2. Position towards coverage center
            position_action = self._compute_coverage_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Power for wide coverage
            action[3] = self._compute_coverage_power(uav, assigned_ues, target_height)
            
            # 4. Bandwidth prioritizing uncovered users
            action[4:] = self._compute_coverage_bandwidth(uav)
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_coverage_maximizing_height(self, uav, assigned_ues) -> float:
        """
        Find minimum height that covers all assigned UEs.
        Strategy: h = max_distance / tan(beam_angle)
        """
        if not assigned_ues:
            return self.env.uav_fly_range_h[0]
        
        # Find center of all users
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        center_2d = np.mean(ue_positions, axis=0)
        
        # Find maximum distance from center
        max_distance = max([np.linalg.norm(ue.position[:2] - center_2d) for ue in assigned_ues])
        
        # Calculate required height for beam coverage
        beam_angle_rad = np.radians(uav.beam_angle)
        
        if beam_angle_rad > 0:
            # h = d / tan(θ), add 10% margin for safety
            required_height = (max_distance / np.tan(beam_angle_rad)) * 1.1
        else:
            required_height = self.env.uav_fly_range_h[1]
        
        # Clamp to allowed range
        target_height = np.clip(required_height, 
                               self.env.uav_fly_range_h[0], 
                               self.env.uav_fly_range_h[1])
        
        return target_height
    
    def _compute_coverage_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Move to position that maximizes coverage"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Target is centroid of all users
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        centroid = np.mean(ue_positions, axis=0)
        
        # XY direction
        direction_xy = centroid - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * 0.45
        else:
            direction_xy = np.zeros(2)
        
        # Z adjustment
        height_diff = target_height - uav.position[2]
        height_action = np.clip(height_diff / 80.0, -0.4, 0.4)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_coverage_power(self, uav, assigned_ues, target_height) -> float:
        """High power for wide coverage"""
        # Higher altitude needs more power
        height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
                      (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])

        base_power = np.sin(height_ratio * np.pi / 2)  # Between 0.4 and 1.0

        
        return np.clip(base_power, 0, 1)
    
    def _compute_coverage_bandwidth(self, uav) -> np.ndarray:
        """Prioritize users that are harder to cover (Far distance)"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                # Extra weight for Far users (harder to cover)
                distance_weight = {'Near': 1.0, 'Medium': 1.3, 'Far': 2.0}[da.distance_level]
                priority = len(da.user_ids) * self.env.slice_weights[da.slice_type] * distance_weight
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        return allocations / allocations.sum()

class QoSAwareHeightGreedyAgent(BaselineAgent):
    """
    QoS-Aware Height Strategy:
    - URLLC users → Fly lower (better SINR, lower latency)
    - eMBB users → Medium height (balance throughput and coverage)
    - mMTC users → Can fly higher (less demanding QoS)
    - Dynamically adjusts height based on slice type mix
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        
        # Preferred heights for each slice type
        self.preferred_heights = {
            'urllc': 0.3,   # 30% of height range (low altitude)
            'embb': 0.5,    # 50% of height range (medium)
            'mmtc': 0.7     # 70% of height range (high)
        }
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4
                action[4:] = 1.0 / (self.action_dim - 4)
                actions[uav_id] = action
                continue
            
            # 1. Calculate QoS-aware optimal height
            target_height = self._compute_qos_aware_height(uav, assigned_ues)
            
            # 2. Position with QoS priority weighting
            position_action = self._compute_qos_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Power based on QoS requirements
            action[3] = self._compute_qos_power(uav, assigned_ues, target_height)
            
            # 4. Bandwidth with QoS priorities
            action[4:] = self._compute_qos_bandwidth(uav)
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_qos_aware_height(self, uav, assigned_ues) -> float:
        """Calculate height based on slice type distribution"""
        if not assigned_ues:
            return (self.env.uav_fly_range_h[0] + self.env.uav_fly_range_h[1]) / 2
        
        # Count users by slice type
        slice_counts = {'urllc': 0, 'embb': 0, 'mmtc': 0}
        for ue in assigned_ues:
            slice_counts[ue.slice_type] += 1
        
        total_users = len(assigned_ues)
        
        # Weighted average of preferred heights
        target_height_ratio = 0.0
        for slice_type, count in slice_counts.items():
            if count > 0:
                weight = count / total_users
                # Extra weight for URLLC (critical)
                if slice_type == 'urllc':
                    weight *= 2.0
                target_height_ratio += weight * self.preferred_heights[slice_type]
        
        # Normalize if URLLC was over-weighted
        target_height_ratio = min(target_height_ratio, 1.0)
        
        # Convert ratio to actual height
        height_range = self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0]
        target_height = self.env.uav_fly_range_h[0] + height_range * target_height_ratio
        
        # Also consider beam coverage constraint
        ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
        centroid = np.mean(ue_positions, axis=0)
        max_distance = max([np.linalg.norm(ue.position[:2] - centroid) for ue in assigned_ues])
        
        beam_angle_rad = np.radians(uav.beam_angle)
        min_height_for_coverage = (max_distance / np.tan(beam_angle_rad)) * 1.05 if beam_angle_rad > 0 else self.env.uav_fly_range_h[0]
        
        # Ensure coverage, but prefer QoS-optimal height
        target_height = max(target_height, min_height_for_coverage)
        
        return np.clip(target_height, self.env.uav_fly_range_h[0], self.env.uav_fly_range_h[1])
    
    def _compute_qos_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Position prioritizing high-QoS users"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Weighted centroid with QoS priority
        weighted_positions = []
        total_weight = 0.0
        
        for ue in assigned_ues:
            # URLLC gets 3x weight, eMBB gets 2x, mMTC gets 1x
            qos_weight = {'urllc': 3.0, 'embb': 2.0, 'mmtc': 1.0}[ue.slice_type]
            slice_weight = self.env.slice_weights[ue.slice_type]
            
            combined_weight = qos_weight * slice_weight
            weighted_positions.append(ue.position[:2] * combined_weight)
            total_weight += combined_weight
        
        target_xy = sum(weighted_positions) / total_weight if total_weight > 0 else uav.position[:2]
        
        # XY movement
        direction_xy = target_xy - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * 0.4
        else:
            direction_xy = np.zeros(2)
        
        # Z movement (prioritize reaching QoS-optimal height)
        height_diff = target_height - uav.position[2]
        height_action = np.clip(height_diff / 70.0, -0.35, 0.35)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_qos_power(self, uav, assigned_ues, target_height) -> float:
        """Power allocation for QoS satisfaction"""
        if not assigned_ues:
            return 0.4
        
        # Check if we have URLLC users (need high power)
        has_urllc = any(ue.slice_type == 'urllc' for ue in assigned_ues)
        
        # Base power higher for URLLC
        base_power = 0.8 if has_urllc else 0.65
        
        # Height adjustment
        height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
                      (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])
        power_boost = 0.15 * height_ratio
        
        # Battery constraint
        battery_ratio = uav.current_battery / uav.battery_capacity
        battery_factor = 0.7 if battery_ratio < 0.25 else 1.0
        
        return np.clip((base_power + power_boost) * battery_factor, 0.5, 0.95)
    
    def _compute_qos_bandwidth(self, uav) -> np.ndarray:
        """Bandwidth with strong QoS priorities"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                # Strong emphasis on slice weights (URLLC >> eMBB > mMTC)
                slice_priority = self.env.slice_weights[da.slice_type] ** 2  # Squared for emphasis
                user_count = len(da.user_ids)
                distance_weight = {'Near': 1.0, 'Medium': 1.2, 'Far': 1.5}[da.distance_level]
                
                priority = user_count * slice_priority * distance_weight
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        
        # Ensure minimum for non-empty DAs
        for i, da in enumerate(sorted(uav_das, key=lambda d: d.id)):
            if len(da.user_ids) > 0:
                allocations[i] = max(allocations[i], 0.02)
        
        return allocations / allocations.sum()

class DynamicHeightGreedyAgent(BaselineAgent):
    """
    Dynamic Height Strategy:
    - Continuously adjusts height based on real-time metrics
    - Monitors uncovered UEs and adjusts height to improve coverage
    - Tracks satisfaction per distance level and adjusts accordingly
    - Most responsive to changing network conditions
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, env):
        super().__init__(num_agents, obs_dim, action_dim)
        self.env = env
        self.height_adjustment_rate = 0.3  # How aggressively to adjust height
    
    def select_actions(self, observations: Dict[int, np.ndarray], explore: bool = False) -> Dict[int, np.ndarray]:
        actions = {}
        
        for uav_id, obs in observations.items():
            action = np.zeros(self.action_dim)
            uav = self.env.uavs[uav_id]
            
            assigned_ues = [ue for ue in self.env.ues.values() 
                           if ue.assigned_uav == uav_id and ue.is_active]
            
            if not assigned_ues:
                action[3] = 0.4
                action[4:] = 1.0 / (self.action_dim - 4)
                actions[uav_id] = action
                continue
            
            # 1. Calculate dynamic height based on current performance
            target_height = self._compute_dynamic_height(uav, assigned_ues)
            
            # 2. Responsive positioning
            position_action = self._compute_dynamic_position(uav, assigned_ues, target_height)
            action[0:3] = position_action
            
            # 3. Adaptive power
            action[3] = self._compute_dynamic_power(uav, assigned_ues, target_height)
            
            # 4. Performance-based bandwidth
            action[4:] = self._compute_dynamic_bandwidth(uav)
            
            actions[uav_id] = action
        
        return actions
    
    def _compute_dynamic_height(self, uav, assigned_ues) -> float:
        """Adjust height based on real-time performance metrics"""
        if not assigned_ues:
            return uav.position[2]  # Stay at current height
        
        # Check coverage issues
        uncovered_ues = []
        for ue in assigned_ues:
            beam_status = self.env.get_ue_beam_status(ue.id)
            if beam_status and not beam_status['covered']:
                uncovered_ues.append(ue)
        
        uncoverage_ratio = len(uncovered_ues) / len(assigned_ues)
        
        # Check satisfaction by distance level
        ue_info = self.env.get_UEs_throughput_demand_and_satisfaction()
        near_satisfaction = []
        far_satisfaction = []
        
        for ue in assigned_ues:
            if ue.id in ue_info:
                _, _, satisfaction = ue_info[ue.id]
                if ue.assigned_da in self.env.demand_areas:
                    da = self.env.demand_areas[ue.assigned_da]
                    if da.distance_level == 'Near':
                        near_satisfaction.append(satisfaction)
                    elif da.distance_level == 'Far':
                        far_satisfaction.append(satisfaction)
        
        avg_near_sat = np.mean(near_satisfaction) if near_satisfaction else 1.0
        avg_far_sat = np.mean(far_satisfaction) if far_satisfaction else 1.0
        
        # Decision logic
        current_height = uav.position[2]
        height_range = self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0]
        
        # Case 1: High uncoverage → Increase height significantly
        if uncoverage_ratio > 0.2:
            target_height = current_height + height_range * 0.3
        
        # Case 2: Far users unsatisfied → Increase height moderately
        elif avg_far_sat < 0.5 and avg_near_sat > 0.7:
            target_height = current_height + height_range * 0.2
        
        # Case 3: Near users unsatisfied but far users ok → Decrease height
        elif avg_near_sat < 0.6 and avg_far_sat > 0.7:
            target_height = current_height - height_range * 0.15
        
        # Case 4: Everything ok → Small adjustment towards optimal
        else:
            # Calculate spatial optimal height
            ue_positions = np.array([ue.position[:2] for ue in assigned_ues])
            centroid = np.mean(ue_positions, axis=0)
            max_distance = max([np.linalg.norm(ue.position[:2] - centroid) for ue in assigned_ues])
            
            beam_angle_rad = np.radians(uav.beam_angle)
            optimal_height = (max_distance / np.tan(beam_angle_rad)) * 1.1 if beam_angle_rad > 0 else current_height
            
            # Gradual adjustment
            target_height = current_height + (optimal_height - current_height) * 0.1
        
        # Clamp
        target_height = np.clip(target_height, 
                               self.env.uav_fly_range_h[0], 
                               self.env.uav_fly_range_h[1])
        
        return target_height
    
    def _compute_dynamic_position(self, uav, assigned_ues, target_height) -> np.ndarray:
        """Responsive position adjustment"""
        if not assigned_ues:
            return np.zeros(3)
        
        # Weight uncovered users more heavily
        weighted_positions = []
        total_weight = 0.0
        
        for ue in assigned_ues:
            beam_status = self.env.get_ue_beam_status(ue.id)
            
            # Uncovered UEs get 3x weight
            coverage_weight = 3.0 if (beam_status and not beam_status['covered']) else 1.0
            slice_weight = self.env.slice_weights[ue.slice_type]
            
            combined_weight = coverage_weight * slice_weight
            weighted_positions.append(ue.position[:2] * combined_weight)
            total_weight += combined_weight
        
        target_xy = sum(weighted_positions) / total_weight if total_weight > 0 else uav.position[:2]
        
        # XY
        direction_xy = target_xy - uav.position[:2]
        distance_xy = np.linalg.norm(direction_xy)
        
        if distance_xy > 5.0:
            direction_xy = (direction_xy / distance_xy) * 0.45
        else:
            direction_xy = np.zeros(2)
        
        # Z - more aggressive adjustment
        height_diff = target_height - uav.position[2]
        height_action = np.clip(height_diff / 60.0 * self.height_adjustment_rate, -0.5, 0.5)
        
        return np.array([
            np.clip(direction_xy[0], -1.0, 1.0),
            np.clip(direction_xy[1], -1.0, 1.0),
            height_action
        ])
    
    def _compute_dynamic_power(self, uav, assigned_ues, target_height) -> float:
        """Adaptive power based on current conditions"""
        if not assigned_ues:
            return 0.4
        
        # Check if we have coverage issues
        beam_statuses = [self.env.get_ue_beam_status(ue.id) for ue in assigned_ues]
        uncovered_count = sum(1 for status in beam_statuses if status and not status['covered'])
        
        # Base power higher if coverage issues
        base_power = 0.85 if uncovered_count > 0 else 0.7
        
        # Height compensation
        height_ratio = (target_height - self.env.uav_fly_range_h[0]) / \
                      (self.env.uav_fly_range_h[1] - self.env.uav_fly_range_h[0])
        power_boost = 0.1 * height_ratio
        
        # Battery
        battery_ratio = uav.current_battery / uav.battery_capacity
        battery_factor = 0.75 if battery_ratio < 0.3 else 1.0
        
        return np.clip((base_power + power_boost) * battery_factor, 0.5, 0.95)
    
    def _compute_dynamic_bandwidth(self, uav) -> np.ndarray:
        """Bandwidth based on real-time satisfaction"""
        uav_das = [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]
        
        if not uav_das:
            return np.ones(self.action_dim - 4) / (self.action_dim - 4)
        
        priorities = []
        ue_info = self.env.get_UEs_throughput_demand_and_satisfaction()
        
        for da in sorted(uav_das, key=lambda d: d.id):
            if len(da.user_ids) == 0:
                priorities.append(0.01)
            else:
                # Calculate average satisfaction for this DA
                satisfactions = []
                for ue_id in da.user_ids:
                    if ue_id in ue_info:
                        _, _, satisfaction = ue_info[ue_id]
                        satisfactions.append(satisfaction)
                
                avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.5
                
                # Urgency: lower satisfaction → higher priority
                urgency = 2.0 - avg_satisfaction  # Range: [1.0, 2.0]
                
                priority = (len(da.user_ids) * 
                           self.env.slice_weights[da.slice_type] *
                           urgency *
                           {'Near': 1.0, 'Medium': 1.3, 'Far': 1.6}[da.distance_level])
                priorities.append(priority)
        
        allocations = np.array(priorities) / (sum(priorities) + 1e-6)
        
        for i, da in enumerate(sorted(uav_das, key=lambda d: d.id)):
            if len(da.user_ids) > 0:
                allocations[i] = max(allocations[i], 0.02)
        
        return allocations / allocations.sum()

class EvaluationFramework:
    """Framework for evaluating and comparing agents"""
    
    def __init__(self, env_config_path: str, results_dir: str = "baseline_results"):
        self.env_config_path = env_config_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
    
    def evaluate_agent(self, agent, agent_name: str, env: NetworkSlicingEnv, 
                      max_steps: int = 500) -> Dict:
        """Evaluate an agent on a single episode with fixed environment"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_name}")
        print(f"{'='*60}")
        
        # Reset environment (will use same seed as set externally)
        obs = env.reset()
        
        # Storage for step-by-step metrics
        metrics = {
            'steps': [],
            'rewards': [],
            'cumulative_rewards': [],
            'qos': [],
            'energy': [],
            'fairness': [],
            'active_ues': []
        }
        
        cumulative_reward = 0
        
        for step in tqdm(range(max_steps), desc=f"{agent_name}", leave=True):
            # Select actions
            actions = agent.select_actions(obs, explore=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(actions)
            
            cumulative_reward += reward
            
            # Store metrics
            metrics['steps'].append(step)
            metrics['rewards'].append(reward)
            metrics['cumulative_rewards'].append(cumulative_reward)
            metrics['qos'].append(info['qos_satisfaction'])
            metrics['energy'].append(info['energy_usage_level'])
            metrics['fairness'].append(info['fairness_level'])
            metrics['active_ues'].append(info['active_ues'])
            
            obs = next_obs
            
            if done:
                break
        
        # Compile results
        results = {
            'agent_name': agent_name,
            'metrics': metrics,
            'summary': {
                'total_reward': cumulative_reward,
                'mean_reward': np.mean(metrics['rewards']),
                'mean_qos': np.mean(metrics['qos']),
                'mean_energy': np.mean(metrics['energy']),
                'mean_fairness': np.mean(metrics['fairness']),
                'mean_active_ues': np.mean(metrics['active_ues']),
                'final_qos': metrics['qos'][-1] if metrics['qos'] else 0,
                'final_energy': metrics['energy'][-1] if metrics['energy'] else 0,
                'final_fairness': metrics['fairness'][-1] if metrics['fairness'] else 0
            }
        }
        
        # Print summary
        print(f"\n{agent_name} Results:")
        print(f"  Total Reward: {results['summary']['total_reward']:.4f}")
        print(f"  Mean Step Reward: {results['summary']['mean_reward']:.4f}")
        print(f"  Mean QoS: {results['summary']['mean_qos']:.4f}")
        print(f"  Mean Energy Efficiency: {results['summary']['mean_energy']:.4f}")
        print(f"  Mean Fairness: {results['summary']['mean_fairness']:.4f}")
        print(f"  Steps Completed: {len(metrics['steps'])}")
        
        return results
    
    def compare_agents(self, results_dir: str):
        """Generate comparison plots for multiple agents"""

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
        
        print(f"\nGenerating comparison plots...")
        
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
        plot_path = os.path.join(self.run_dir, 'comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
        
        return fig
    
    def save_results(self, results_list: List[Dict]):
        """Save results to JSON file"""
        # Convert results for JSON serialization (exclude large arrays)
        serializable_results = []
        for result in results_list:
            serializable = {
                'agent_name': result['agent_name'],
                'summary': result['summary'],
                'num_steps': len(result['metrics']['steps'])
            }
            serializable_results.append(serializable)
        
        results_path = os.path.join(self.run_dir, 'results_summary.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved results summary to {results_path}")
        
        # Save detailed metrics as CSV for each agent
        for result in results_list:
            agent_name = result['agent_name'].replace(' ', '_').replace('(', '').replace(')', '')
            csv_path = os.path.join(self.run_dir, f'{agent_name}.csv')
            
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'reward', 'cumulative_reward', 'qos', 'energy', 'fairness', 'active_ues'])
                
                metrics = result['metrics']
                for i in range(len(metrics['steps'])):
                    writer.writerow([
                        metrics['steps'][i],
                        metrics['rewards'][i],
                        metrics['cumulative_rewards'][i],
                        metrics['qos'][i],
                        metrics['energy'][i],
                        metrics['fairness'][i],
                        metrics['active_ues'][i]
                    ])
            
            print(f"Saved detailed metrics to {csv_path}")

class AdaptiveMultiObjectiveGreedy:
    """
    Ultimate greedy baseline that optimizes for QoS, energy, and fairness
    in a single decision framework with adaptive weighting
    """
    
    def select_actions(self, observation):
        # Parse observation
        uav_state = observation[:5]  # position, power, battery
        da_info = observation[5:68].reshape(9, 7)  # 9 DAs x 7 features
        handover_state = observation[68:72]
        surrounding = observation[72:80]
        
        # Extract critical metrics per DA
        queues = da_info[:, 0]  # Queue lengths
        delays = da_info[:, 1]  # Delays
        throughputs = da_info[:, 2]  # Throughput
        qos_satisfaction = da_info[:, 3]  # Current QoS
        loads = da_info[:, 4]  # Load
        da_positions = da_info[:, 5:7]  # DA positions
        
        # 1. MOVEMENT: Move toward most critical DA
        # Criticality score combines QoS violation + queue + delay
        qos_violations = 1.0 - qos_satisfaction  # Higher = worse
        normalized_queues = queues / (queues.max() + 1e-6)
        normalized_delays = delays / (delays.max() + 1e-6)
        
        criticality = (0.5 * qos_violations + 
                      0.3 * normalized_queues + 
                      0.2 * normalized_delays)
        
        most_critical_da = np.argmax(criticality)
        target_position = da_positions[most_critical_da]
        
        # Calculate direction to most critical DA
        current_pos = uav_state[:2]
        direction = target_position - current_pos
        distance = np.linalg.norm(direction)
        
        # Adaptive movement: faster toward critical, slower when close
        if distance > 0:
            movement = direction / distance  # Normalize
            speed_factor = min(1.0, distance / 50.0)  # Slow down when close
            movement = movement * speed_factor
        else:
            movement = np.array([0.0, 0.0])
        
        # Height adjustment based on coverage needs
        if criticality.max() > 0.5:  # High criticality
            altitude = 0.3  # Lower for better signal
        else:
            altitude = 0.0  # Medium height
        
        position_action = np.array([movement[0], movement[1], altitude])
        
        # 2. POWER: Adaptive based on demand and battery
        battery_level = uav_state[4]
        avg_qos = qos_satisfaction.mean()
        
        if battery_level < 0.3:  # Low battery - conserve
            power = 0.3
        elif avg_qos < 0.6:  # Poor QoS - boost power
            power = 0.9
        elif criticality.max() > 0.7:  # Critical DA exists
            power = 0.8
        else:  # Normal operation
            power = 0.6
        
        power_action = np.array([power])
        
        # 3. BANDWIDTH: Proportional to weighted demand with fairness
        # Demand score combines multiple factors
        demand_scores = (
            0.4 * normalized_queues +  # Queue pressure
            0.3 * qos_violations +      # QoS violations
            0.2 * normalized_delays +   # Delay issues
            0.1 * (loads / (loads.max() + 1e-6))  # Current load
        )
        
        # Apply fairness: ensure minimum allocation
        min_allocation = 0.05  # 5% minimum per DA
        available_bandwidth = 1.0 - (9 * min_allocation)
        
        # Distribute available bandwidth proportionally to demand
        if demand_scores.sum() > 0:
            proportional_allocation = (demand_scores / demand_scores.sum()) * available_bandwidth
            bandwidth = proportional_allocation + min_allocation
        else:
            bandwidth = np.ones(9) / 9  # Equal if no demand info
        
        # Ensure normalization
        bandwidth = bandwidth / bandwidth.sum()
        
        # Combine all actions
        action = np.concatenate([position_action, power_action, bandwidth])
        
        return action

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline agents')
    parser.add_argument('--env_config', type=str, default='config/environment/default.yaml',
                       help='Path to environment config')
    parser.add_argument('--checkpoint', type=str, default='saved_models/model6/checkpoints/checkpoint_step_190000.pth',
                       help='Path to trained MADRL model checkpoint')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Number of steps for evaluation')
    parser.add_argument('--seed', type=int, default=45,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    
    # Load environment config to get dimensions
    env_config = Configuration(args.env_config)
    num_agents = env_config.system.num_uavs
    
    # Create environment (will be reset with same seed for each agent)
    env = NetworkSlicingEnv(config_path=args.env_config)
    
    # Get observation/action dimensions
    obs_sample = env.reset()
    obs_dim = len(list(obs_sample.values())[0])
    action_dim = 4 + env_config.system.num_das_per_slice * 3
    
    print(f"\n{'='*60}")
    print(f"Environment Setup (Seed: {args.seed})")
    print(f"{'='*60}")
    print(f"  Num Agents: {num_agents}")
    print(f"  Observation Dim: {obs_dim}")
    print(f"  Action Dim: {action_dim}")
    print(f"  Evaluation Steps: {args.steps}")
    print(f"{'='*60}\n")
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(env_config_path=args.env_config)
    
    # Initialize agents
    agents = []
    
    # Random Agent
    # random_agent = RandomAgent(num_agents, obs_dim, action_dim)
    # agents.append((random_agent, "Random"))

    # # Additional Baseline Agents
    # smart_greedy_agent = SmartGreedyAgent(num_agents, obs_dim, action_dim, env)
    # agents.append((smart_greedy_agent, "Smart Greedy"))

    # qos_greedy_agent = QoSAwareHeightGreedyAgent(num_agents, obs_dim, action_dim, env)
    # agents.append((qos_greedy_agent, "QoS-Aware Greedy"))

    # energy_aware_agent = EnergyAwareGreedyAgent(num_agents, obs_dim, action_dim, env)
    # agents.append((energy_aware_agent, "Energy-Aware Greedy"))

    # dynamic_height_agent = DynamicHeightGreedyAgent(num_agents, obs_dim, action_dim, env)
    # agents.append((dynamic_height_agent, "Dynamic Height Greedy"))

    coverage_greedy_agent = CoverageMaximizationGreedyAgent(num_agents, obs_dim, action_dim, env)
    agents.append((coverage_greedy_agent, "Coverage Greedy"))

    # adaptive_height_agent = AdaptiveHeightGreedyAgent(num_agents, obs_dim, action_dim, env)
    # agents.append((adaptive_height_agent, "Adaptive Height Greedy"))

    # 3. Trained MADRL Agent (if checkpoint provided)
    if args.checkpoint:
        trained_agent = MADRLAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training=False
        )
        trained_agent.load_models(args.checkpoint)
        agents.append((trained_agent, "MADRL (Trained)"))
        print(f"✓ Loaded trained model from {args.checkpoint}\n")

    # Another trained agent
    agent2 = MADRLAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            training=False
        )
    agent2.load_models('saved_models/model6/checkpoints/checkpoint_step_372000.pth')
    # agents.append((agent2, "MADRL (Model 6)"))

    # Evaluate all agents on the same environment
    results_list = []
    for agent, agent_name in agents:
        # Reset environment with same seed for fair comparison
        np.random.seed(args.seed)
        
        results = evaluator.evaluate_agent(
            agent, 
            agent_name,
            env=env,
            max_steps=args.steps
        )
        results_list.append(results)
    
    # # Save results
    evaluator.save_results(results_list)

    # Generate comparison plots
    evaluator.compare_agents(evaluator.run_dir)
    # evaluator.compare_agents("baseline_results/run_20251025_171841")
    

    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Results saved to: {evaluator.run_dir}")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()