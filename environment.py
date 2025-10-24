
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils import Configuration, QueueingModel, Packet
from dataclasses import dataclass, field
from numba import jit, prange
import heapq
from collections import defaultdict



class HandoverTracker:
    '''Track handovers and calculate penalties'''
    def __init__(self):
        self.handover_delay_ms = 50.0
        self.handover_failure_rate = 0.01
        self.recent_handovers = {}
        self.ping_pong_threshold = 5.0
        
        # Statistics
        self.total_handovers = 0
        self.ping_pong_handovers = 0
        
        # FIX: Cache the handover rate instead of recalculating
        self._cached_handover_rate = 0.0
        self._last_rate_update_time = 0.0
        self._rate_cache_ttl = 1.0  # Update every second
    
    def check_handover(self, ue_id, old_uav, new_uav, current_time):
        if old_uav is None or old_uav == new_uav or not new_uav:
            return {'handover': False, 'delay_penalty_ms': 0.0, 'ping_pong': False}
        
        self.total_handovers += 1
        
        is_ping_pong = False
        if ue_id in self.recent_handovers:
            time_since_last = current_time - self.recent_handovers[ue_id]
            if time_since_last < self.ping_pong_threshold:
                is_ping_pong = True
                self.ping_pong_handovers += 1
        
        self.recent_handovers[ue_id] = current_time
        
        # Update cached rate if needed
        if current_time - self._last_rate_update_time >= self._rate_cache_ttl:
            self._update_handover_rate_cache(current_time)
        
        handover_delay_ms = self.handover_delay_ms
        if is_ping_pong:
            handover_delay_ms *= 1.5
        
        return {
            'handover': True,
            'delay_penalty_ms': handover_delay_ms,
            'packet_loss_prob': self.handover_failure_rate,
            'ping_pong': is_ping_pong
        }
    
    def _update_handover_rate_cache(self, current_time, time_window=60.0):
        """Update cached handover rate"""
        recent = sum(1 for t in self.recent_handovers.values() 
                    if current_time - t <= time_window)
        self._cached_handover_rate = recent / (time_window / 60.0)
        self._last_rate_update_time = current_time
    
    def get_handover_rate(self, time_window=60.0):
        """OPTIMIZED: Return cached rate instead of recalculating"""
        # This is called 30,000 times costing 135 seconds!
        # Original had O(n) iteration over all handovers
        return self._cached_handover_rate  # O(1) lookup!

@dataclass
class QoSProfile:
    """Quality of Service profile for network slices"""
    min_rate: float  # bps
    max_latency: float  # ms
    min_reliability: float  # percentage

@dataclass
class UAV:
    """UAV entity with position and resource constraints"""
    id: int
    position: np.ndarray  # [x, y, h]
    max_bandwidth: float  # MHz
    max_power: float  # Watts
    current_power: float
    battery_capacity: float  # Joules
    current_battery: float
    velocity_max: float  # m/s
    buffer_size: int  # bits
    energy_used: Dict[str, float] = field(default_factory=dict)  # Joules used in last step
    RBs: List['ResourceBlock'] = None  # List of ResourceBlocks
    beam_angle: float = 60.0  # NEW: Maximum beam angle in degrees (half-angle from vertical)
    beam_direction: np.ndarray = None  # NEW: Beam pointing direction (default: straight down)
    avg_queuing_delay_ms: float = 0.0  # NEW: Average queuing delay at the UAV
    avg_drop_rate: float = 0.0  # NEW: Average packet drop rate at the UAV

    def __post_init__(self):
        if self.RBs is None:
            self.RBs = []
        if self.beam_direction is None:
            self.beam_direction = np.array([0.0, 0.0, -1.0])  # Default: pointing straight down

    def update_position(self, delta_pos: np.ndarray):
        """Update UAV position with velocity constraints"""
        self.position += delta_pos

class LEOSat:
    """LEO Satellite entity"""
    def __init__(self, id: int, position: np.ndarray, coverage_radius: float):
        self.id = id
        self.position = position  # [x, y, h]
        self.coverage_radius = coverage_radius  # meters

@dataclass
class ResourceBlock:
    """Resource Block allocation"""
    id: int
    bandwidth: float  # Hz
    frequency: Optional[float] = 3.5e6  # Frequency in MHz
    allocated_ue_id: int = -1  # User Equipment ID
    allocated_da_id: Optional[int] = -1  # DemandArea ID

@dataclass
class UE:
    """User Equipment entity"""
    id: int
    position: np.ndarray  # [x, y, h]
    slice_type: str  # embb, urllc, mmtc
    assigned_uav: Optional[int] = None
    assigned_da: Optional[int] = None
    assigned_rb: List[ResourceBlock] = None
    throughput: float = 0.0  # bps
    latency_ms: float = 0.0  # ms
    reliability: float = 0.0  # percentage
    throughput_satisfaction = 0.0
    delay_satisfaction = 0.0
    reliability_satisfaction = 0.0
    is_active: bool = True  # NEW: whether UE is currently active
    velocity: np.ndarray = None  # NEW: velocity vector for movement
    traffic_pattern: List[float] = None # The traffic coming to this UE in each T_L period
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.assigned_rb is None:
            self.assigned_rb = []

@dataclass
class DemandArea:
    """Demand Area for clustering users"""
    id: int
    uav_id: int
    slice_type: str
    user_ids: List[int]
    allocated_bandwidth: float = 2.0
    raw_allocated_action: float = 0.0  # Store raw action before normalization
    center_position: Optional[np.ndarray] = None
    distance_level: str = "Medium"  # New field: "Low", "Medium", "High"
    RB_ids_list: List[int] = None  # List of Resource Block IDs allocated to this Demand Area
    def __post_init__(self):
        if self.center_position is None:
            self.center_position = np.zeros(3)
        if self.RB_ids_list is None:
            self.RB_ids_list = []

@jit(nopython=True, parallel=True, fastmath=True, cache=True)  # Add cache=True!
def calculate_sinr_batch_numba(ue_positions, rb_frequencies, uav_positions, 
                                uav_powers, serving_indices, noise_power, 
                                path_loss_exponent):
    """Ultra-fast SINR calculation - NOW WITH CACHING"""
    
    n_calculations = ue_positions.shape[0]
    n_uavs = uav_positions.shape[0]
    sinr_db = np.empty(n_calculations, dtype=np.float32)
    
    # Parallel loop
    for i in prange(n_calculations):
        ue_pos = ue_positions[i]
        rb_freq = rb_frequencies[i]
        serving_idx = serving_indices[i]
        
        wavelength = 3e8 / rb_freq
        factor = wavelength / (4.0 * 3.141592653589793)  # Hardcode pi
        
        signal_power = 0.0
        interference_power = 0.0
        
        for j in range(n_uavs):
            dx = ue_pos[0] - uav_positions[j, 0]
            dy = ue_pos[1] - uav_positions[j, 1]
            dz = ue_pos[2] - uav_positions[j, 2]
            distance = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            path_loss = (factor / distance) ** path_loss_exponent
            rx_power = path_loss * uav_powers[j]
            
            if j == serving_idx:
                signal_power = rx_power
            else:
                angle_to_ue = np.arccos(-dz / distance) * (180.0 / 3.141592653589793)  # in degrees
                if angle_to_ue <= 60.0:  # Assuming beam angle of 60 degrees
                    interference_power += rx_power

        sinr = signal_power / (interference_power + noise_power)
        sinr_db[i] = 10.0 * 0.4342944819032518 * np.log(sinr + 1e-10)  # log10(x) = log(x) / log(10)
        
        # Clip
        if sinr_db[i] < -10.0:
            sinr_db[i] = -10.0
        elif sinr_db[i] > 50.0:
            sinr_db[i] = 50.0
    
    return sinr_db


class NetworkSlicingEnv:
    """
    UAV-based network slicing environment with CTDE support
    """
    def __init__(self, config_path):

        env_config = Configuration(config_path)

        # Load configuration parameters
        self.num_uavs = env_config.system.num_uavs
        self.num_ues = env_config.system.num_ues
        self.service_area = tuple(env_config.system.service_area)  # (width, height)
        self.uav_fly_range_x = tuple(env_config.system.uav_fly_range_x)  # (min_x, max_x)
        self.uav_fly_range_y = tuple(env_config.system.uav_fly_range_y)  # (min_y, max_y)
        self.uav_fly_range_h = tuple(env_config.system.uav_fly_range_h)  # (min_height, max_height)
        self.num_das_per_slice = env_config.system.num_das_per_slice

        # Time scale parameters
        self.T_L = env_config.time.T_L  # Long-term period (seconds)
        self.T_S = env_config.time.T_S  # Short-term period (seconds)
        self.current_time = 0.0
        
        # Network slice QoS profiles
        qos = env_config.qos_profiles
        self.qos_profiles = {
            "embb": QoSProfile(min_rate=qos.embb.min_rate, 
                        max_latency=qos.embb.max_latency, 
                        min_reliability=qos.embb.min_reliability),   # eMBB
            "urllc": QoSProfile(min_rate=qos.urllc.min_rate, 
                        max_latency=qos.urllc.max_latency, 
                        min_reliability=qos.urllc.min_reliability),   # URLLC
            "mmtc": QoSProfile(min_rate=qos.mmtc.min_rate, 
                        max_latency=qos.mmtc.max_latency, 
                        min_reliability=qos.mmtc.min_reliability)     # mMTC
        }
        self.qos_weights = env_config.qos_weights  # Weights for throughput, delay, reliability
        
        # Slice-related parameters
        self.slice_weights = env_config.slicing.slice_weights  # Weights for each slice type
        self.slice_probs = env_config.slicing.slice_probabilities  # Probabilities for each slice type

        # Channel parameters
        self.noise_power = env_config.channel.noise_power  # Watts
        self.path_loss_exponent = env_config.channel.path_loss_exponent
        self.carrier_frequency = env_config.channel.carrier_frequency  # Hz

        # Resource block parameters
        self.rb_bandwidth = env_config.channel.rb_bandwidth  # Hz (LTE standard)
        self.total_bandwidth = env_config.channel.total_bandwidth  # Hz per UAV
        self.total_rbs = int(self.total_bandwidth / self.rb_bandwidth)
        self.carrier_frequency = env_config.channel.carrier_frequency  # Hz
        print(f"Total Resource Blocks per UAV: {self.total_rbs}")

        # Handover tracker
        self.handover_tracker = HandoverTracker()

        # UAV parameters
        self.uav_params = env_config.uav_params  # UAV parameters like max power, battery, etc.
        self.uav_beam_angle = getattr(env_config.uav_params, 'beam_angle', 60.0)
        self.energy_models = env_config.energy_models
        self.uav_buffer_size = env_config.uav_params.buffer_size  # bits
        
        # UE dynamics parameters
        self.ue_id_pool = set()  # Pool of available IDs for reuse
        self.next_ue_id = 0  # Only used when pool is empty
        self.ue_dynamics = env_config.ue_dynamics  # UE dynamics parameters
        self.ue_arrival_rate = self.ue_dynamics.arrival_rate  # New UEs per second
        self.ue_departure_rate = self.ue_dynamics.departure_rate  # Probability of UE leaving per minute
        self.ue_max_initial_velocity = self.ue_dynamics.max_initial_velocity  # m/s
        self.max_ues = int(self.num_ues * self.ue_dynamics.max_ues_multiplier)  # Maximum UEs allowed
        self.change_direction_prob = self.ue_dynamics.change_direction_prob  # Probability of changing direction

        self.hotspots = []
        self.hotspot_attraction_strength = 0.6
        self.hotspot_spawn_rate = 0.15  # Probability per T_L period (15% chance)
        self.hotspot_max_count = 4  # Maximum concurrent hotspots
        self.hotspot_min_lifetime = 300  # seconds
        self.hotspot_max_lifetime = 400.0  # seconds

        #Traffic model
        self.traffic_patterns = env_config.traffic_patterns

        for _ in range(2):
            self._spawn_new_hotspot()


        # Distance thresholds for forming Demand Areas
        self.da_distance_thresholds = env_config.da_distance_thresholds

        # SINR thresholds for DA classification (in dB)
        self.sinr_thresholds = env_config.sinr_thresholds  

        # Reward weights
        self.reward_weights = env_config.reward_weights

        # Statistics tracking
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'handovers': 0,
            'throughput_satisfaction': 0.0,
            'delay_satisfaction': 0.0,
            'reliability_satisfaction': 0,
        }
        
        # Initialize entities
        self.reset()
        self._print_statistics()

        # Add these new attributes for caching
        self.sinr_cache = {}
        self.sinr_cache_valid = False

        self.delay_cache = {}
        self.delay_cache_valid = False

        self.per_cache = {}
        self.per_cache_valid = False

        self.throughput_cache = {}
        self.throughput_cache_valid = False
        
        # Optional: track cache performance
        self.cache_hits = 0
        self.cache_misses = 0
        
    def reset(self) -> Dict[int, np.ndarray]:   
        """Reset environment to initial state"""
        # Initialize UAVs at random positions within service area
        self.uavs = {}
        for i in range(self.num_uavs):
            x = np.random.uniform(self.uav_fly_range_x[0], self.uav_fly_range_x[1])
            y = np.random.uniform(self.uav_fly_range_y[0], self.uav_fly_range_y[1])
            h = np.random.uniform(self.uav_fly_range_h[0], self.uav_fly_range_h[1])
            
            self.uavs[i] = UAV(
                id=i,
                position=np.array([x, y, h]),
                max_bandwidth=self.total_bandwidth,
                max_power=self.uav_params.max_power,  # Watts
                current_power=self.uav_params.initial_power,
                battery_capacity=self.uav_params.battery_capacity,  # Joules
                current_battery=self.uav_params.initial_battery,  # Start fully charged
                velocity_max=self.uav_params.velocity_max,  # m/s
                buffer_size=self.uav_buffer_size,  # bits
                energy_used={"movement": 0.0, "transmission": 0.0},
                RBs=[ResourceBlock(id=i, bandwidth = self.rb_bandwidth, frequency = self.carrier_frequency) for i in range(self.total_rbs)],
                beam_angle=self.uav_beam_angle,  # NEW
                beam_direction=np.array([0.0, 0.0, -1.0])  # NEW: Default pointing down
            )
        
        # Initialize UEs with random positions and slice types
        self.ues = {}
        self.next_ue_id = 0

        for i in range(self.num_ues):
            self._add_new_ue()
        
        # Initialize demand areas
        self.demand_areas = {}
        self.da_counter = 0
        
        # Perform initial UAV-UE association and DA formation
        self._associate_ues_to_uavs()
        self._form_demand_areas()
        
        # Get initial observations for each UAV
        observations = self._get_observations()
        
        self.current_time = 0.0
        return observations

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for each UAV agent
        Logic:
        Each UAV agent combines:
        - The UAV's state (5 dims):
            + position (x y h) 
            + power level (normalized)
            + battery level (normalized)
        - The UAV's demand area info: (9 DAs x 10 dims each = 90 dims)
            - Each DA's info: (10 dims)
                + number of users (normalized by dividing by total UEs): 1 dim
                + slice type (one hot encoded): 3 dims
                + connection quality level (normalized): 3 dims
                + buffer utilization (normalized): 1 dim
                + average delay (normalized by dividing by 100ms): 1 dim
                + drop rate (normalized): 1 dim
        - The UAV's handover state (4 dims):
            - recent handover rate (normalized)
            - ping-pong rate (normalized)
            - number of UEs currently in handover (normalized)
            - average handover penalty being experienced by UEs (normalized)
        - The UAV's load prediction features (5 dims):
            - UE arrival trend (arrivals in last T_L, normalized)
            - predicted number of UEs in next T_L (normalized)
            - predicted traffic load in next T_L (normalized)
            - predicted buffer utilization in next T_L (normalized)
            - predicted average delay in next T_L (normalized)
        - The UAV's surrounding condition (8 dims):
            - number of UEs to each direction (N, E, S, W)
            - interference from other UAVs to each direction (N, E, S, W)

        => Total dimension: 5 + 90 + 4 + 5 + 8 = 112 dims
        """
        observations = {}

        num_total_ues = max(1, len([ue for ue in self.ues.values() if ue.is_active]))
        slice_types = ["embb", "urllc", "mmtc"]
        distance_levels = ["Near", "Medium", "Far"]

        for uav_id, uav in self.uavs.items():
            obs = []

            # ============================================================
            # 1. UAV State (5 dims)
            # ============================================================
            norm_x = (uav.position[0] - self.uav_fly_range_x[0]) / (self.uav_fly_range_x[1] - self.uav_fly_range_x[0])
            norm_y = (uav.position[1] - self.uav_fly_range_y[0]) / (self.uav_fly_range_y[1] - self.uav_fly_range_y[0])
            norm_h = (uav.position[2] - self.uav_fly_range_h[0]) / (self.uav_fly_range_h[1] - self.uav_fly_range_h[0])
            obs.extend([norm_x, norm_y, norm_h])  # 3
            obs.append(uav.current_power / uav.max_power)  # 1
            obs.append(uav.current_battery / uav.battery_capacity)  # 1

            # ============================================================
            # 2. Demand Area Info (9 DAs × 7 features = 63 dims). So far 5 + 63 = 68
            # ============================================================
            # Fixed order: embb-Near, embb-Medium, embb-Far, urllc-Near, ...
            for slice_type in slice_types:  # 3 types
                for distance_level in distance_levels:  # 3 levels
                    da = next((da for da in self.demand_areas.values()
                            if da.uav_id == uav_id and 
                            da.slice_type == slice_type and 
                            da.distance_level == distance_level), None)
                    
                    # Number of users (normalized) (1 dim) ---
                    total_uav_ues = len([ue for ue in self.ues.values() 
                                    if ue.assigned_uav == uav_id and ue.is_active])
                    num_users = len(da.user_ids) / max(1, total_uav_ues) if da else 0.0
                    obs.append(num_users)
                    
                    # Slice type one-hot (3 dims)
                    obs.extend([1.0 if slice_type == st else 0.0 for st in slice_types])

                    # Distance level one-hot (3 dims)
                    obs.extend([1.0 if distance_level == st else 0.0 for st in distance_levels])

            # ============================================
            # 3. NEW: Handover State Features (4 dims). So far 5 + 63 + 4 = 72
            # ============================================
            # Recent handover rate (handovers per minute) (1 dim)
            handover_rate = self.handover_tracker.get_handover_rate(time_window=self.T_L)
            obs.append(handover_rate / 50.0)  # Normalize (assume max ~10/min)
            
            # Ping-pong rate (1 dim)
            total_ho = max(self.handover_tracker.total_handovers, 1)
            ping_pong_rate = self.handover_tracker.ping_pong_handovers / total_ho
            obs.append(ping_pong_rate)
            
            # Number of UEs currently in handover (1 dim)
            ues_in_handover = sum(1 for ue in self.ues.values()
                                if ue.assigned_uav == uav_id and 
                                getattr(ue, 'handover_penalty_ms', 0) > 0)
            obs.append(ues_in_handover / 10.0)  # Normalize
            
            # Average handover penalty being experienced by UEs (1 dim)
            # Average handover penalty being experienced by UEs (1 dim)
            ho_penalties = [getattr(ue, 'handover_penalty_ms', 0) 
                            for ue in self.ues.values()
                            if ue.assigned_uav == uav_id and ue.is_active]
            avg_ho_penalty = np.mean(ho_penalties) if ho_penalties else 0.0  # Fix: handle empty list
            obs.append(avg_ho_penalty / 50.0)  # Normalize by 50ms


            # ============================================================
            # 3. Surrounding Condition (8 dims). So far 5 + 63 + 4 + 8 = 80
            # ============================================================
            # UE distribution in 4 directions
            directions = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
            for ue in self.ues.values():
                if not ue.is_active:
                    continue
                dx = ue.position[0] - uav.position[0]
                dy = ue.position[1] - uav.position[1]
                
                # Quadrant assignment
                if dx > 0:
                    directions['E' if dy > 0 else 'S'] += 1
                else:
                    directions['W' if dy > 0 else 'N'] += 1
            
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(directions[dir] / num_total_ues)  # 4
            
            # Interference from other UAVs in 4 directions
            interference_dirs = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
            for other_id, other_uav in self.uavs.items():
                if other_id == uav_id:
                    continue
                dx = other_uav.position[0] - uav.position[0]
                dy = other_uav.position[1] - uav.position[1]
                
                if dx > 0:
                    interference_dirs['E' if dy > 0 else 'S'] += 1
                else:
                    interference_dirs['W' if dy > 0 else 'N'] += 1
            
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(interference_dirs[dir] / max(1, self.num_uavs - 1))  # 4

            observations[uav_id] = np.array(obs, dtype=np.float32)
            # print(f"UAV {uav_id} observations: {obs}")
            
            # Sanity check
            expected_dims = 80 
            assert len(obs) == expected_dims, f"Expected {expected_dims} dims, got {len(obs)}"

        return observations

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], float, bool, Dict]:
        """Execute actions and return next observations, reward, done, info"""

        # Parse and apply actions for each UAV
        for uav_id, action in actions.items():
            # print(f"UAV {uav_id} action: {action}")
            uav = self.uavs[uav_id]
            
            # Action format: [delta_x, delta_y, delta_h, delta_power, bandwidth_allocation_per_da...]
            # Position update (constrained by velocity)
            delta_pos = action[:3] * uav.velocity_max * self.T_L
            new_pos = uav.position + delta_pos
            # Constrain to service area
            new_pos[0] = np.clip(new_pos[0], self.uav_fly_range_x[0], self.uav_fly_range_x[1])
            new_pos[1] = np.clip(new_pos[1], self.uav_fly_range_y[0], self.uav_fly_range_y[1])
            new_pos[2] = np.clip(new_pos[2], self.uav_fly_range_h[0], self.uav_fly_range_h[1])
            
            # Calculate movement energy cost
            v_x = (new_pos[0] - uav.position[0]) / self.T_L
            v_y = (new_pos[1] - uav.position[1]) / self.T_L
            v_z = (new_pos[2] - uav.position[2]) / self.T_L
            
            h_factor = self.energy_models.get('horizontal_energy_factor')
            v_factor = self.energy_models.get('vertical_energy_factor')
            movement_energy = h_factor * (v_x**2 + v_y**2) * self.T_L + v_factor * (v_z**2) * self.T_L
            
            
            uav.position = new_pos
            
            # Power update
            uav.current_power = action[3] * uav.max_power
            
            # Energy consumption
            transmission_energy = uav.current_power * self.T_L
            uav.energy_used = {"movement": movement_energy, "transmission": transmission_energy}
            uav.current_battery -= sum(uav.energy_used.values())
            uav.current_battery = max(0, uav.current_battery)


        # Invalidate cache BEFORE making changes
        self._invalidate_sinr_cache()
        self._invalidate_throughput_cache()
        

        # Update UE dynamics
        self._update_ue_dynamics()

        # Update associations and DAs (every long-term period)
        self._associate_ues_to_uavs()
        self._form_demand_areas()

        # Update UEs traffic patterns
        self._update_ue_traffic_patterns()

        # Apply bandwidth allocation actions to DAs
        for uav_id, action in actions.items():
            uav_das = [da for da in self.demand_areas.values() if da.uav_id == uav_id]
            if len(uav_das) > 0:
                # Normalized bandwidth allocation
                bandwidth_actions = action[4:4+len(uav_das)]
                
                for i, da in enumerate(uav_das):
                    da.raw_allocated_action = bandwidth_actions[i]
                    da.allocated_bandwidth = bandwidth_actions[i] * uav.max_bandwidth

        # Bandwidth allocation to DAs
        self._allocate_rbs_fairly()

        # Update SINR, throughput, delay, reliability caches
        self._update_sinr_cache()
        self._update_throughput_cache()
        self._update_delay_cache()
        self._update_per_cache()

        # Update queuing statistics at UAVs
        self._update_uavs_queuing_stats()

        # Calculate rewards
        # print("Is throughput cache valid?", self.throughput_cache_valid)
        reward, qos_reward, energy_penalty, fairness_reward = self.calculate_global_reward()

        # Update time
        self.current_time += self.T_L
        
        # Check termination conditions
        done = any(uav.current_battery <= 0 for uav in self.uavs.values())
        done = False
        
        # Get new observations
        observations = self._get_observations()
        
        info = {
            'qos_satisfaction': qos_reward,
            'energy_usage_level': energy_penalty,
            'fairness_level': fairness_reward,
            'active_ues': len([ue for ue in self.ues.values() if ue.is_active]),
            'ue_arrivals': self.stats['arrivals'],
            'ue_departures': self.stats['departures']
        }
        
        return observations, reward, done, info

    # ============================================
    # UEs DYNAMIC METHODS
    # ============================================

    def _spawn_new_hotspot(self):
        """Create a new hotspot at random location"""
        # Random position in service area
        x = np.random.uniform(self.service_area[0] * 0.1, self.service_area[0] * 0.9)
        y = np.random.uniform(self.service_area[1] * 0.1, self.service_area[1] * 0.9)
        
        # Random properties
        hotspot = {
            'position': np.array([x, y]),
            'radius': np.random.uniform(100.0, 500.0),  # Random size
            'max_strength': np.random.uniform(0.5, 1.0),  # Peak strength
            'current_strength': 0.0,  # Start at 0, will grow
            'lifetime': np.random.uniform(self.hotspot_min_lifetime, self.hotspot_max_lifetime),
            'age': 0.0,  # Current age in seconds
            'state': 'growing'  # States: 'growing', 'active', 'fading'
        }
        
        self.hotspots.append(hotspot)
        return hotspot

    def _update_hotspots(self):
        """Update hotspot lifecycles - called every T_L"""
        
        # 1. Update existing hotspots
        hotspots_to_remove = []
        
        for i, hotspot in enumerate(self.hotspots):
            hotspot['age'] += self.T_L
            
            # Lifecycle stages with smooth transitions
            growth_duration = hotspot['lifetime'] * 0.2  # 20% of life spent growing
            fade_start = hotspot['lifetime'] * 0.7  # Start fading at 70% of lifetime
            
            if hotspot['age'] < growth_duration:
                # Growing phase: strength increases from 0 to max_strength
                hotspot['state'] = 'growing'
                progress = hotspot['age'] / growth_duration
                hotspot['current_strength'] = hotspot['max_strength'] * progress
                
            elif hotspot['age'] < fade_start:
                # Active phase: maintain full strength
                hotspot['state'] = 'active'
                hotspot['current_strength'] = hotspot['max_strength']
                
            elif hotspot['age'] < hotspot['lifetime']:
                # Fading phase: strength decreases to 0
                hotspot['state'] = 'fading'
                fade_duration = hotspot['lifetime'] - fade_start
                fade_progress = (hotspot['age'] - fade_start) / fade_duration
                hotspot['current_strength'] = hotspot['max_strength'] * (1 - fade_progress)
                
            else:
                # Hotspot expired
                hotspots_to_remove.append(i)
        
        # 2. Remove expired hotspots
        for i in reversed(hotspots_to_remove):
            del self.hotspots[i]
        
        # 3. Randomly spawn new hotspots
        if len(self.hotspots) < self.hotspot_max_count:
            if np.random.random() < self.hotspot_spawn_rate:
                self._spawn_new_hotspot()
    
    def _get_hotspot_influence(self, ue_position):
        """Calculate attraction vector from all active hotspots"""
        total_attraction = np.zeros(2)
        
        for hotspot in self.hotspots:
            # Skip if hotspot is too weak
            if hotspot['current_strength'] < 0.1:
                continue
            
            # Calculate distance to this hotspot
            direction = hotspot['position'] - ue_position[:2]
            distance = np.linalg.norm(direction)
            
            # Only attract if within radius
            if distance < hotspot['radius'] and distance > 1.0:
                # Normalize direction
                direction = direction / distance
                
                # Attraction decreases with distance
                distance_factor = (1 - distance / hotspot['radius'])
                attraction_magnitude = hotspot['current_strength'] * distance_factor
                
                total_attraction += direction * attraction_magnitude
        
        return total_attraction

    def _update_ue_dynamics(self):
        """OPTIMIZED: Update UE positions, handle arrivals/departures"""
        
        # Reset statistics
        self.stats['arrivals'] = 0
        self.stats['departures'] = 0
        
        # Update hotspot lifecycles
        self._update_hotspots()
        
        # ============================================
        # OPTIMIZATION 1: Get active UEs as arrays
        # ============================================
        active_ues = [ue for ue in self.ues.values() if ue.is_active]
        
        if not active_ues:
            # No active UEs, just handle arrivals
            self._handle_arrivals()
            return
        
        num_active = len(active_ues)
        
        # ============================================
        # OPTIMIZATION 2: Vectorized hotspot influence
        # ============================================
        if self.hotspots:
            # Get all UE positions at once
            ue_positions = np.array([ue.position[:2] for ue in active_ues], dtype=np.float32)
            
            # Vectorized hotspot influence calculation
            hotspot_pulls = np.zeros((num_active, 2), dtype=np.float32)
            
            for hotspot in self.hotspots:
                if hotspot['current_strength'] < 0.1:
                    continue
                
                # Vectorized distance calculation for all UEs
                directions = hotspot['position'] - ue_positions  # (num_ues, 2)
                distances = np.linalg.norm(directions, axis=1)  # (num_ues,)
                
                # Mask for UEs within radius
                in_radius = (distances < hotspot['radius']) & (distances > 1.0)
                
                if not np.any(in_radius):
                    continue
                
                # Vectorized attraction calculation
                normalized_dirs = directions[in_radius] / distances[in_radius, np.newaxis]
                distance_factors = 1 - distances[in_radius] / hotspot['radius']
                attraction = (hotspot['current_strength'] * distance_factors)[:, np.newaxis] * normalized_dirs
                
                hotspot_pulls[in_radius] += attraction
        else:
            hotspot_pulls = np.zeros((num_active, 2), dtype=np.float32)
        
        # ============================================
        # OPTIMIZATION 3: Vectorized velocity updates
        # ============================================
        for i, ue in enumerate(active_ues):
            current_speed = np.linalg.norm(ue.velocity[:2])
            
            if current_speed > 0 and np.any(hotspot_pulls[i]):
                # Blend velocity with hotspot pull
                ue.velocity[:2] = (
                    (1 - self.hotspot_attraction_strength) * ue.velocity[:2] +
                    self.hotspot_attraction_strength * hotspot_pulls[i] * current_speed
                )
        
        # ============================================
        # OPTIMIZATION 4: Vectorized position updates
        # ============================================
        for ue in active_ues:
            # Update position
            new_position = ue.position + ue.velocity * self.T_L
            
            # Boundary handling (vectorized per UE)
            for dim in range(2):
                if new_position[dim] < 0 or new_position[dim] > self.service_area[dim]:
                    ue.velocity[dim] = -ue.velocity[dim]
                    new_position[dim] = np.clip(new_position[dim], 0, self.service_area[dim])
            
            ue.position = new_position
        
        # ============================================
        # OPTIMIZATION 5: Batch random number generation
        # ============================================
        # Generate all random numbers at once (much faster)
        direction_changes = np.random.random(num_active) < self.change_direction_prob
        departures = np.random.random(num_active) < self.ue_departure_rate
        
        # Apply direction changes
        for i, ue in enumerate(active_ues):
            if direction_changes[i]:
                speed = np.linalg.norm(ue.velocity[:2])
                if speed > 0:
                    new_direction = np.random.uniform(0, 2 * np.pi)
                    ue.velocity[0] = speed * np.cos(new_direction)
                    ue.velocity[1] = speed * np.sin(new_direction)
        
        # ============================================
        # OPTIMIZATION 6: Efficient departure handling
        # ============================================
        departed_ues = []
        for i, ue in enumerate(active_ues):
            if departures[i]:
                ue.is_active = False
                self.stats['departures'] += 1
                departed_ues.append(ue.id)
        
        # Batch cleanup
        for ue_id in departed_ues:
            if hasattr(self, 'ue_id_pool'):
                self.ue_id_pool.add(ue_id)
            del self.ues[ue_id]
        
        # ============================================
        # 7. Handle arrivals
        # ============================================
        self._handle_arrivals()

    def _update_ue_traffic_patterns(self):

        for ue_id, ue in self.ues.items():
            if not ue.is_active:
                continue

            if ue.assigned_uav is None: # UE not assigned to any UAV
                ue.traffic_pattern = [0 for _ in range(int(self.T_L))]
                continue
            
            avg_arrival_rate = self.traffic_patterns[ue.slice_type]['avg_arrival_rate']
            ue_traffic_pattern = []
            for _ in range(int(self.T_L)):
                ue_traffic_pattern.append(np.random.poisson(avg_arrival_rate))
            ue.traffic_pattern = ue_traffic_pattern



    def _handle_arrivals(self):
        """Separate function for handling arrivals (cleaner code)"""
        active_ue_count = len([ue for ue in self.ues.values() if ue.is_active])
        expected_arrivals = self.ue_arrival_rate * self.T_L
        num_arrivals = min(
            np.random.poisson(expected_arrivals),
            self.max_ues - active_ue_count
        )
        
        for _ in range(num_arrivals):
            self._add_new_ue()
            self.stats['arrivals'] += 1

    def _add_new_ue(self):
        """Add a new UE to the network - spawn near active hotspots WITH ID REUSE"""
        
        # ============================================
        # Get UE ID (reuse from pool if available)
        # ============================================
        if self.ue_id_pool:
            # Reuse an ID from the pool
            new_ue_id = self.ue_id_pool.pop()
        else:
            # Generate new ID
            new_ue_id = self.next_ue_id
            self.next_ue_id += 1
        
        # ============================================
        # Position selection (existing logic)
        # ============================================
        # Find active hotspots (strength > 0.5)
        active_hotspots = [h for h in self.hotspots if h['current_strength'] > 0.5]
        
        # 70% chance to spawn near an active hotspot
        if active_hotspots and np.random.random() < 0.7:
            # Choose hotspot weighted by strength
            weights = [h['current_strength'] for h in active_hotspots]
            weights = np.array(weights) / sum(weights)
            hotspot = active_hotspots[np.random.choice(len(active_hotspots), p=weights)]
            
            # Spawn within hotspot radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, hotspot['radius'] * 0.6)
            
            x = hotspot['position'][0] + distance * np.cos(angle)
            y = hotspot['position'][1] + distance * np.sin(angle)
            
            # Clip to service area
            x = np.clip(x, 0, self.service_area[0])
            y = np.clip(y, 0, self.service_area[1])
        else:
            # Random position
            x = np.random.uniform(0, self.service_area[0])
            y = np.random.uniform(0, self.service_area[1])
        
        # ============================================
        # Velocity and slice type (existing logic)
        # ============================================
        speed = np.random.uniform(0, self.ue_max_initial_velocity)
        direction = np.random.uniform(0, 2 * np.pi)
        velocity_x = speed * np.cos(direction)
        velocity_y = speed * np.sin(direction)
        
        slice_type = np.random.choice(
            ["embb", "urllc", "mmtc"],
            p=[self.slice_probs["embb"], self.slice_probs["urllc"], self.slice_probs["mmtc"]]
        )
        
        # ============================================
        # Create UE with reused/new ID
        # ============================================
        new_ue = UE(
            id=new_ue_id,  # Reused or new ID
            position=np.array([x, y, 0.0]),
            slice_type=slice_type,
            velocity=np.array([velocity_x, velocity_y, 0.0]),
            is_active=True
        )
        
        self.ues[new_ue_id] = new_ue

    def _update_uavs_queuing_stats(self):
        '''Calculate the average queuing delay that UEs experience at their associated UAVs at each T_L period'''
        for uav in self.uavs.values():

            # Average packet size (bytes)
            packet_sizes = [self.traffic_patterns[ue.slice_type]['packet_size'] for ue in self.ues.values() if ue.assigned_uav == uav.id and ue.is_active]
            avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0.0

            
            # Cummulative arrival rate (packets/sec)
            arrival_rates = [self.traffic_patterns[ue.slice_type]['avg_arrival_rate'] for ue in self.ues.values() if ue.assigned_uav == uav.id and ue.is_active]
            uav_arrival_rate = np.sum(arrival_rates) if arrival_rates else 0.0
            
            # Cummulative service rate (packets/sec)
            througputs_of_ues_in_uav = [self._get_cached_throughput(ue) for ue in self.ues.values() if ue.assigned_uav == uav.id and ue.is_active]  # in bps
            uav_throughput_bps = np.sum(througputs_of_ues_in_uav) if througputs_of_ues_in_uav else 0.0
            uav_service_rate = uav_throughput_bps / (1e3 + avg_packet_size * 8)  # packets per second

            # Buffer size in packets
            uav_buffer_size = uav.buffer_size / (1e3 + avg_packet_size * 8) # in packets

            # CALCULATE QUEUING DELAY
            rho = uav_arrival_rate / max(uav_service_rate, 1)  # Utilization
            if rho >= 1:
                uav.avg_queuing_delay_ms = 99
            elif uav_service_rate == 0:
                uav.avg_queuing_delay_ms = 999.0
            else:
                uav.avg_queuing_delay_ms = rho / (2 * uav_service_rate * (1 - rho)) * 1000  # in ms, M/D/1 formula

            # CALCULATE DROP RATE ()
            if rho != 1:
                uav.avg_drop_rate = ( (rho**(uav_buffer_size + 1)) * (1 - rho) ) / (1 - rho**(uav_buffer_size + 1))
            elif uav_service_rate == 0:
                uav.avg_drop_rate = 1.0
            else:
                uav.avg_drop_rate = 1 / (uav_buffer_size + 1)
            
            if np.isnan(uav.avg_drop_rate):
                uav.avg_drop_rate = 1.0
            uav.avg_drop_rate = np.clip(uav.avg_drop_rate, 0.0, 1.0)

            # print(f"UAV {uav.id} - Arrival rate: {uav_arrival_rate:<10.2f} pkt/s | Service rate: {uav_service_rate:<12.2f} pkt/s | Rho: {rho:<10.4f} | Buffer size: {uav_buffer_size:<5.0f} pkts | Avg Delay: {uav.avg_queuing_delay_ms:<10.6f} ms | Drop Rate: {uav.avg_drop_rate:<10.4f}")
    # ============================================
    # ASSOCIATION METHODS
    # ============================================

    def _associate_ues_to_uavs(self):
        """VECTORIZED: Associate UEs to nearest UAV with beam constraint"""
        self.stats['handovers'] = 0
        self.stats['uncovered_ues'] = 0
        self.stats['covered_ues'] = 0
        
        # Store previous assignments
        previous_assignments = {ue.id: ue.assigned_uav for ue in self.ues.values() if ue.is_active}
        
        # Get all active UEs
        active_ues = [ue for ue in self.ues.values() if ue.is_active]
        if not active_ues:
            return
        
        # Prepare arrays
        ue_positions = np.array([ue.position for ue in active_ues], dtype=np.float32)
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float32)
        uav_beam_dirs = np.array([uav.beam_direction for uav in self.uavs.values()], dtype=np.float32)
        uav_beam_angles = np.array([uav.beam_angle for uav in self.uavs.values()], dtype=np.float32)
        
        # Compute ALL distances at once (num_ues × num_uavs matrix)
        # Shape: (num_ues, 1, 3) - (1, num_uavs, 3) = (num_ues, num_uavs, 3)
        diff_vectors = ue_positions[:, np.newaxis, :] - uav_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff_vectors, axis=2)  # (num_ues, num_uavs)
        
        # Compute ALL beam angles at once
        # Normalize direction vectors
        distances_safe = np.maximum(distances, 1e-6)
        normalized_dirs = diff_vectors / distances_safe[:, :, np.newaxis]
        
        # Dot product for all UE-UAV pairs
        # (num_ues, num_uavs, 3) · (1, num_uavs, 3) -> (num_ues, num_uavs)
        dot_products = np.sum(normalized_dirs * uav_beam_dirs[np.newaxis, :, :], axis=2)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Calculate angles in degrees
        angles = np.degrees(np.arccos(dot_products))  # (num_ues, num_uavs)
        
        # Check beam constraints
        within_beam = angles <= uav_beam_angles[np.newaxis, :]  # (num_ues, num_uavs)
        
        # Set invalid distances to infinity
        distances_masked = distances.copy()
        distances_masked[~within_beam] = np.inf
        
        # Find best UAV for each UE
        best_uav_indices = np.argmin(distances_masked, axis=1)
        
        # Assign UEs to UAVs
        uav_ids_list = list(self.uavs.keys())
        for i, ue in enumerate(active_ues):
            min_dist = distances_masked[i, best_uav_indices[i]]
            if min_dist < np.inf:
                ue.assigned_uav = uav_ids_list[best_uav_indices[i]]
            else:
                ue.assigned_uav = None
        
        # Handover tracking (existing code)
        for ue in active_ues:
            old_uav = previous_assignments.get(ue.id)
            new_uav = ue.assigned_uav
            if new_uav is None:
                self.stats['uncovered_ues'] += 1
            else:
                self.stats['covered_ues'] += 1

            handover_info = self.handover_tracker.check_handover(
                ue.id, old_uav, new_uav, self.current_time
            )
            
            if not hasattr(ue, 'handover_penalty_ms'):
                ue.handover_penalty_ms = 0.0
            
            if handover_info['handover']:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']
                self.stats['handovers'] += 1
            else:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']

    def _form_demand_areas(self):
        """OPTIMIZED: Cache UE-to-DA mapping to avoid repeated lookups"""
        self.demand_areas.clear()
        self.da_counter = 0

        # Pre-compute distance thresholds
        near_max = self.da_distance_thresholds['near_max']
        medium_max = self.da_distance_thresholds['medium_max']
        
        # Build UE-to-DA mapping dictionary for fast lookup
        self.ue_to_da_map = {}  # NEW: Cache this!
        
        for uav_id in range(self.num_uavs):
            uav_pos = self.uavs[uav_id].position
            
            # Get active UEs for this UAV (vectorize distance calc)
            uav_ues = [ue for ue in self.ues.values() 
                    if ue.assigned_uav == uav_id and ue.is_active]
            
            if not uav_ues:
                # Still create empty DAs
                for slice_type in ["embb", "urllc", "mmtc"]:
                    for level_name in ['Near', 'Medium', 'Far']:
                        da = DemandArea(
                            id=self.da_counter,
                            uav_id=uav_id,
                            slice_type=slice_type,
                            user_ids=[],
                            distance_level=level_name
                        )
                        da.center_position = uav_pos.copy()
                        self.demand_areas[self.da_counter] = da
                        self.da_counter += 1
                continue
            
            # Vectorized distance calculation for all UEs
            ue_positions = np.array([ue.position for ue in uav_ues])
            distances = np.linalg.norm(ue_positions - uav_pos, axis=1)
            
            # Group by slice type
            for slice_type in ["embb", "urllc", "mmtc"]:
                # Boolean mask for this slice
                slice_mask = np.array([ue.slice_type == slice_type for ue in uav_ues])
                slice_ues = [ue for ue, m in zip(uav_ues, slice_mask) if m]
                slice_distances = distances[slice_mask]
                
                # Categorize by distance (vectorized)
                distance_groups = {
                    'Near': [ue for ue, d in zip(slice_ues, slice_distances) if d <= near_max],
                    'Medium': [ue for ue, d in zip(slice_ues, slice_distances) if near_max < d <= medium_max],
                    'Far': [ue for ue, d in zip(slice_ues, slice_distances) if d > medium_max]
                }
                
                # Create DAs
                for level_name in ['Near', 'Medium', 'Far']:
                    ue_group = distance_groups[level_name]
                    
                    da = DemandArea(
                        id=self.da_counter,
                        uav_id=uav_id,
                        slice_type=slice_type,
                        user_ids=[ue.id for ue in ue_group],
                        distance_level=level_name
                    )
                    
                    if ue_group:
                        positions = np.array([ue.position for ue in ue_group])
                        da.center_position = np.mean(positions, axis=0)
                        
                        # Update mapping
                        for ue in ue_group:
                            ue.assigned_da = da.id
                            self.ue_to_da_map[ue.id] = da.id
                    else:
                        da.center_position = uav_pos.copy()
                    
                    self.demand_areas[self.da_counter] = da
                    self.da_counter += 1

    # ============================================
    # BEAM CALCULATION METHODS
    # ============================================

    def _calculate_beam_angle(self, uav: UAV, ue: UE) -> float:
        """
        Calculate the angle between UAV's beam direction and the direction to UE.
        Returns angle in degrees.
        
        Args:
            uav: UAV object
            ue: UE object
            
        Returns:
            Angle in degrees from UAV's beam center to UE
        """
        # Vector from UAV to UE
        vec_to_ue = ue.position - uav.position
        distance = np.linalg.norm(vec_to_ue)
        
        if distance < 1e-6:  # Avoid division by zero
            return 0.0
        
        # Normalize
        vec_to_ue_norm = vec_to_ue / distance
        
        # Calculate angle using dot product
        # angle = arccos(a · b) where a and b are unit vectors
        dot_product = np.dot(uav.beam_direction, vec_to_ue_norm)
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
  
    def _is_ue_in_beam(self, uav: UAV, ue: UE) -> bool:
        """
        Check if UE is within UAV's beam coverage.
        
        Args:
            uav: UAV object
            ue: UE object
            
        Returns:
            True if UE is within beam angle, False otherwise
        """
        angle = self._calculate_beam_angle(uav, ue)
        return angle <= uav.beam_angle

    def get_uav_beam_info(self, uav_id: int) -> Dict:
        """
        Get beam information for visualization.
        
        Args:
            uav_id: UAV ID
            
        Returns:
            Dictionary with beam parameters
        """
        if uav_id not in self.uavs:
            return None
        
        uav = self.uavs[uav_id]
        
        # Calculate beam radius at ground level
        height = uav.position[2]
        if height > 0:
            beam_radius = height * np.tan(np.radians(uav.beam_angle))
        else:
            beam_radius = 0
        
        return {
            'position': uav.position.copy(),
            'beam_angle': uav.beam_angle,
            'beam_direction': uav.beam_direction.copy(),
            'beam_radius_ground': beam_radius,
            'height': height
        }
    
    def get_ue_beam_status(self, ue_id: int) -> Dict:
        """
        Check if UE is within its assigned UAV's beam.
        
        Args:
            ue_id: UE ID
            
        Returns:
            Dictionary with coverage status
        """
        if ue_id not in self.ues:
            return None
        
        ue = self.ues[ue_id]
        if not ue.is_active or ue.assigned_uav is None:
            return {'covered': False, 'angle': None}
        
        uav = self.uavs[ue.assigned_uav]
        angle = self._calculate_beam_angle(uav, ue)
        covered = self._is_ue_in_beam(uav, ue)
        
        return {
            'covered': covered,
            'angle': angle,
            'max_angle': uav.beam_angle,
            'margin': uav.beam_angle - angle  # Positive = inside, negative = outside
        }

    # ============================================
    # CACHING METHODS
    # ============================================

    def _update_sinr_cache(self):
        if self.sinr_cache_valid:
            return
        
        self.sinr_cache.clear()
        
        # Collect all UE-RB pairs
        calculation_list = []
        ue_ids, serving_uav_ids, rb_ids = [], [], []
        
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_uav is None or not ue.assigned_rb:
                continue
            for rb in ue.assigned_rb:
                calculation_list.append((ue, rb))
                ue_ids.append(ue.id)
                serving_uav_ids.append(ue.assigned_uav)
                rb_ids.append(rb.id)
        
        if not calculation_list:
            self.sinr_cache_valid = True
            return
        
        # Prepare arrays for Numba
        ue_positions = np.array([ue.position for ue, _ in calculation_list], dtype=np.float32)
        rb_frequencies = np.array([rb.frequency for _, rb in calculation_list], dtype=np.float32)
        uav_positions = np.array([uav.position for uav in self.uavs.values()], dtype=np.float32)
        uav_powers = np.array([uav.current_power for uav in self.uavs.values()], dtype=np.float32)
        
        # Map serving UAV IDs to indices
        uav_ids_list = list(self.uavs.keys())
        uav_index_map = {uid: idx for idx, uid in enumerate(uav_ids_list)}
        serving_indices = np.array([uav_index_map[uid] for uid in serving_uav_ids], dtype=np.int32)
        
        # Call Numba-accelerated function
        sinr_db = calculate_sinr_batch_numba(
            ue_positions, rb_frequencies, uav_positions, uav_powers,
            serving_indices, self.noise_power, self.path_loss_exponent
        )
        
        # Fill cache
        for i, (ue_id, uav_id, rb_id) in enumerate(zip(ue_ids, serving_uav_ids, rb_ids)):
            self.sinr_cache[(ue_id, uav_id, rb_id)] = sinr_db[i]
        
        self.sinr_cache_valid = True
    
    def _update_delay_cache(self):
        """
        Calculate ALL delays at once
        """
        self.delay_cache.clear()
        
        # Collect all active UEs with RBs
        active_ues = [
            (ue_id, ue) for ue_id, ue in self.ues.items()
            if ue.is_active and ue.assigned_rb and ue.assigned_uav is not None
        ]
        
        if not active_ues:
            self.delay_cache_valid = True
            return
        # ========================
        # Propagation delays
        # ========================
        ue_positions = np.array([ue.position for _, ue in active_ues], dtype=np.float32)
        uav_positions = np.array([
            self.uavs[ue.assigned_uav].position for _, ue in active_ues
        ], dtype=np.float32)
        
        distances = np.linalg.norm(ue_positions - uav_positions, axis=1)
        t_prop = (distances / 3e8) * 1000  # ms

        # ========================
        # Transmission delays
        # ========================

        packet_size_bits = np.array([self.traffic_patterns[ue.slice_type]['packet_size'] * 8 for _, ue in active_ues], dtype=np.float32)
        throughputs = np.array([np.clip(self._get_cached_throughput(ue), 0.1, None) for _, ue in active_ues], dtype=np.float32)
        t_trans = packet_size_bits / throughputs  # Initial placeholder

        # ========================
        # Retransmission delays
        # ========================

        per_list = np.array([self._calculate_packet_error_rate(ue) for _, ue in active_ues], dtype=np.float32)
        t_retx = per_list * 4 * 8.0  # ms

        # ========================
        # Queuing delays
        # ========================

        t_queue = np.array([self.uavs[ue.assigned_uav].avg_queuing_delay_ms for _, ue in active_ues], dtype=np.float32)

        # ========================
        # Scheduling delays
        # ========================

        t_sched = np.zeros(len(active_ues), dtype=np.float32)

        # ========================
        # Handover delays
        # ========================

        t_handover = np.array([getattr(ue, 'handover_penalty_ms', 0.0) for _, ue in active_ues], dtype=np.float32)

        # ========================
        # Processing delays
        # ========================

        t_proc = np.full(len(active_ues), 3.0, dtype=np.float32)

        # Process each UE's delay components
        for i, (ue_id, ue) in enumerate(active_ues):
            
            # Total delay
            total_delay = (
                t_prop[i] + t_trans[i] + t_retx[i] +
                t_queue[i] + t_sched[i] +
                t_handover[i] +
                t_proc[i]
            )
            
            # Cache result
            self.delay_cache[ue_id] = {
                'total': total_delay,
                'breakdown': {
                    'propagation': t_prop[i],
                    'transmission': t_trans[i],
                    'retransmission': t_retx[i],
                    'queuing': t_queue[i],
                    'scheduling': t_sched[i],
                    'handover': t_handover[i],
                    'processing': t_proc[i],

                }
            }
        
        self.delay_cache_valid = True

    def _update_throughput_cache(self):
        
        self.throughput_cache.clear()
        
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_uav is None or len(ue.assigned_rb) == 0:
                self.throughput_cache[ue.id] = 0.0
                continue
            # Calculate total throughput across all RBs
            ue_throughput = 0.0
            for rb in ue.assigned_rb:
                sinr_db = self.sinr_cache.get((ue.id, ue.assigned_uav, rb.id), -10.0)
                sinr_linear = 10.0 ** (sinr_db * 0.1)
                ue_throughput += rb.bandwidth * np.log2(1.0 + sinr_linear)
            
            self.throughput_cache[ue.id] = ue_throughput

        self.throughput_cache_valid = True

    def _update_per_cache(self):
        """Update PER cache for all UE-RB pairs"""
        self.per_cache.clear()

        # Collect all active UEs with RBs
        active_ues = [
            (ue_id, ue) for ue_id, ue in self.ues.items()
            if ue.is_active and ue.assigned_rb and ue.assigned_uav is not None
        ]
        
        if not active_ues:
            self.per_cache_valid = True
            return

        for ue_id, ue in active_ues:
            per = self._calculate_packet_error_rate(ue)
            self.per_cache[ue.id] = per

    def _invalidate_sinr_cache(self):
        """Invalidate cache when network state changes"""
        self.sinr_cache_valid = False
    
    def _invalidate_throughput_cache(self):
        """Invalidate throughput cache when allocations change"""
        self.throughput_cache_valid = False 

    def _get_cached_sinr(self, ue: UE, uav: UAV, rb: ResourceBlock) -> float:
        """Get SINR from cache, updating if necessary"""
        if not self.sinr_cache_valid:
            self.cache_misses += 1
            self._update_sinr_cache()
        
        cache_key = (ue.id, uav.id, rb.id)
        if cache_key in self.sinr_cache:
            self.cache_hits += 1
            return self.sinr_cache[cache_key]
        else:
            # Fallback to single calculation if not in cache
            # This shouldn't happen often if cache is properly maintained
            # pass
            # print(f"Cache miss for UE {ue.id}, UAV {uav.id}, RB {rb.id}")
            self.cache_misses += 1
            self._invalidate_sinr_cache()
            self._update_sinr_cache()
            return self.sinr_cache[cache_key]

    def _get_cached_delay(self, ue: UE) -> Dict[str, float]:
        # Return cached value
        if ue.id in self.delay_cache:
            return self.delay_cache[ue.id]
        else:
            self._update_delay_cache()
            return self.delay_cache[ue.id]
        
        # # Fallback for edge cases
        # return {'total': float('inf'), 'breakdown': {}}

    def _get_cached_per(self, ue: UE) -> float:
        # Return cached value
        if ue.id in self.per_cache:
            return self.per_cache[ue.id]
        else:
            self._update_per_cache()
            return self.per_cache[ue.id]
    
    def _get_cached_throughput(self, ue: UE) -> float:
        # Return cached value
        if ue.id in self.throughput_cache and self.throughput_cache_valid:
            return self.throughput_cache[ue.id]
        else:
            self._update_throughput_cache()
            return self.throughput_cache[ue.id]

    def _calculate_packet_error_rate(self, ue: UE) -> float:
        """Calculate PER based on SINR (existing - keep as is or use this)"""

        uav = self.uavs[ue.assigned_uav]

        sinr_values = []
        for rb in ue.assigned_rb:
            sinr_db = self._get_cached_sinr(ue, uav, rb)
            sinr_values.append(sinr_db)

        avg_sinr_db = np.mean(sinr_values)
        packet_size_bytes = self.traffic_patterns[ue.slice_type]['packet_size']


        # Approximate BLER curve
        if avg_sinr_db < -5:
            per = 0.5
        elif avg_sinr_db < 0:
            per = 0.2 * np.exp(-avg_sinr_db / 3)
        elif avg_sinr_db < 10:
            per = 0.1 * np.exp(-avg_sinr_db / 5)
        elif avg_sinr_db < 20:
            per = 0.01 * np.exp(-avg_sinr_db / 10)
        else:
            per = 1e-5
        
        # Adjust for packet size
        size_factor = packet_size_bytes / 1500.0
        return min(1.0, per * size_factor)

    def _allocate_rbs_fairly(self):
        """
        GLOBALLY OPTIMIZED: Allocate RBs for ALL UAVs/DAs/UEs at once
        - Processes all UAVs in parallel using vectorization
        - Single pass through all entities
        - Minimal object lookups
        """
        
        # ============================================
        # STEP 1: Reset all allocations (vectorized)
        # ============================================
        # Reset all RBs across all UAVs
        for uav in self.uavs.values():
            for rb in uav.RBs:
                rb.allocated_da_id = -1
                rb.allocated_ue_id = -1
        
        # Reset all DA RB lists
        for da in self.demand_areas.values():
            da.RB_ids_list = []
        
        # Reset all UE RB assignments
        for ue in self.ues.values():
            if ue.is_active:
                ue.assigned_rb = []
        
        # ============================================
        # STEP 2: Group DAs by UAV (pre-compute)
        # ============================================
        uav_das_map = defaultdict(list)  # {uav_id: [da1, da2, ...]}
        
        for da in self.demand_areas.values():
            uav_das_map[da.uav_id].append(da)
        
        # ============================================
        # STEP 3: Process each UAV (can be parallelized)
        # ============================================
        for uav_id, uav in self.uavs.items():
            uav_das = uav_das_map.get(uav_id, [])
            
            if not uav_das:
                continue
            
            total_rbs = len(uav.RBs)
            
            # Vectorized bandwidth to RB conversion
            allocated_bandwidths = np.array(
                [da.allocated_bandwidth for da in uav_das], 
                dtype=np.float32
            )
            
            target_rbs = (allocated_bandwidths / self.rb_bandwidth).astype(np.int32)
            
            # Ensure we don't exceed total RBs
            total_requested = target_rbs.sum()
            
            if total_requested > total_rbs:
                # Scale down proportionally
                target_rbs = ((target_rbs * total_rbs) / total_requested).astype(np.int32)
            
            # Allocate using cumulative sum
            cumsum = np.concatenate([[0], np.cumsum(target_rbs)])
            
            # ============================================
            # STEP 4: Assign RBs to DAs (vectorized)
            # ============================================
            for i, da in enumerate(uav_das):
                start_idx = cumsum[i]
                end_idx = min(cumsum[i + 1], total_rbs)
                
                # Batch assign RBs to this DA
                for rb_id in range(start_idx, end_idx):
                    rb = uav.RBs[rb_id]
                    rb.allocated_da_id = da.id
                    da.RB_ids_list.append(rb_id)
        
        # ============================================
        # STEP 5: Assign RBs to UEs (global pass)
        # ============================================
        # Build UE lookup for faster access
        active_ues = {ue_id: ue for ue_id, ue in self.ues.items() if ue.is_active}
        
        for da in self.demand_areas.values():
            if not da.user_ids or not da.RB_ids_list:
                continue
            
            num_users = len(da.user_ids)
            uav = self.uavs[da.uav_id]
            
            # Round-robin assignment
            for i, rb_id in enumerate(da.RB_ids_list):
                ue_id = da.user_ids[i % num_users]
                rb = uav.RBs[rb_id]
                rb.allocated_ue_id = ue_id
                
                # Use pre-built lookup
                if ue_id in active_ues:
                    active_ues[ue_id].assigned_rb.append(rb)
        
        # ============================================
        # STEP 6: Invalidate cache once at the end
        # ============================================
        self._invalidate_sinr_cache()

    # ============================================
    # REWARD CALCULATION METHODS
    # ============================================

    def calculate_global_reward(self) -> Tuple[float, float, float, float]:
        """
        REFACTORED: Calculate global reward with separated QoS calculation
        
        Returns:
            total_reward: Combined reward
            qos_reward: QoS satisfaction component
            energy_penalty: Energy consumption component
            fairness_reward: Fairness component
        """
        # ============================================
        # 1. Calculate QoS reward (now separated!)
        # ============================================
        qos_reward, da_metrics = self._calculate_qos_reward()
        urllc_penalty = self._calculate_urllc_penalty(da_metrics)
        energy_penalty = self._calculate_energy_consumption_penalty()
        fairness_reward = self._calculate_fairness_index()
        
        # ============================================
        # 4. Combine with weights
        # ============================================
        alpha = self.reward_weights.qos
        beta = self.reward_weights.energy
        gamma = self.reward_weights.fairness
        delta = 0.4  # URLLC penalty weight
        
        total_reward = (
            alpha * qos_reward -
            beta * energy_penalty +
            gamma * fairness_reward -
            delta * urllc_penalty
        )

        return total_reward, qos_reward, energy_penalty, fairness_reward

    def _calculate_qos_reward(self) -> Tuple[float, Dict]:
        """
        Calculate QoS reward for all UEs calculated based on throughput, delay, reliability.
        
        Returns:
            qos_reward: Aggregated QoS satisfaction score
            da_metrics: Detailed metrics per demand area
        """
        # Ensure SINR cache is valid
        if not self.sinr_cache_valid:
            self._update_sinr_cache()
        
        # Track metrics per DA
        da_metrics = {}  # {da_id: {'satisfactions': [...], 'delays': [...], 'slice_type': str}}
        
        # ============================================
        # Collect metrics for all active UEs
        # ============================================
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            
            # Calculate individual QoS components
            throughput_sat, ue_throughput = self._calculate_throughput_satisfaction(ue)
            delay_sat, total_delay_ms = self._calculate_delay_satisfaction(ue)
            reliability_sat, ue_reliability = self._calculate_reliability_satisfaction(ue)

            ue.latency_ms = total_delay_ms  # Store for potential logging
            ue.throughput = ue_throughput  # Store for potential logging
            ue.reliability = ue_reliability  # Store for potential logging

            ue.throughput_satisfaction = throughput_sat
            ue.delay_satisfaction = delay_sat
            ue.reliability_satisfaction = reliability_sat

            # Combine using slice-specific weights
            ue_qos_weights = self.qos_weights[ue.slice_type]
            overall_satisfaction = (
                ue_qos_weights['throughput'] * throughput_sat +
                ue_qos_weights['delay'] * delay_sat +
                ue_qos_weights['reliability'] * reliability_sat
            )
            
            # Store in DA metrics
            da_id = ue.assigned_da
            if da_id not in da_metrics:
                da_metrics[da_id] = {
                    'satisfactions': [],
                    'delays': [],
                    'throughputs': [],
                    'reliabilities': [],
                    'slice_type': ue.slice_type
                }
            
            da_metrics[da_id]['satisfactions'].append(overall_satisfaction)
            da_metrics[da_id]['delays'].append(total_delay_ms)
            da_metrics[da_id]['throughputs'].append(ue_throughput)
            da_metrics[da_id]['reliabilities'].append(ue_reliability)
        
        # ============================================
        # Aggregate QoS reward across all DAs
        # ============================================
        aggregate_satisfaction = 0.0
        total_weight = 0.0
        
        for da_id, metrics in da_metrics.items():
            avg_satisfaction = np.mean(metrics['satisfactions'])
            slice_type = metrics['slice_type']
            weight = self.slice_weights[slice_type] * len(metrics['satisfactions'])
            
            aggregate_satisfaction += avg_satisfaction * weight
            total_weight += weight
        
        qos_reward = aggregate_satisfaction / total_weight if total_weight > 0 else 0.0

        return qos_reward, da_metrics

    def _calculate_throughput_satisfaction(self, ue) -> Tuple[float, float]:
        """
        Calculate throughput satisfaction for a single UE
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
            throughput: Actual throughput in bps
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 0.0
        
        # Calculate total throughput across all RBs
        ue_throughput = self._get_cached_throughput(ue)
        
        # Normalize by minimum required rate
        min_rate = self.qos_profiles[ue.slice_type].min_rate
        satisfaction = min(ue_throughput / min_rate, 1.0)
        
        return satisfaction, ue_throughput

    def _calculate_delay_satisfaction(self, ue) -> Tuple[float, float]:
        """
        Calculate delay satisfaction for a single UE
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
            delay_ms: Actual end-to-end delay in milliseconds
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 99999.0
        
        # Get total delay
        delay_info = self._get_cached_delay(ue)
        total_delay_ms = delay_info['total']
        
        # Calculate satisfaction with exponential penalty
        max_latency = self.qos_profiles[ue.slice_type].max_latency
        
        if total_delay_ms <= max_latency:
            satisfaction = 1.0
        else:
            excess = total_delay_ms - max_latency
            satisfaction = np.exp(-excess / max_latency)
        
        return satisfaction, total_delay_ms

    def _calculate_reliability_satisfaction(self, ue) -> float:
        """
        Calculate reliability satisfaction for a single UE
        
        Based on:
        - Buffer drops (from queuing model)
        - Handover losses
        - Packet error rate (from SINR)
        
        Returns:
            satisfaction: Normalized satisfaction score [0, 1]
        """
        if not ue.assigned_rb or len(ue.assigned_rb) == 0:
            return 0.0, 0.0
        
        # Drop rate (at assigned UAV)
        drop_rate = self.uavs[ue.assigned_uav].avg_drop_rate if ue.assigned_uav in self.uavs else 0.0
        
        # Handover loss
        handover_loss = 0.01 if getattr(ue, 'handover_penalty_ms', 0) > 0 else 0.0
        
        # Transmission success based on PER
        per = self._get_cached_per(ue)
        transmission_success = (1 - per) ** 4  # With 4 retransmissions
        
        # Overall reliability
        reliability = transmission_success * (1 - drop_rate) * (1 - handover_loss)
        
        # Normalize
        min_reliability = self.qos_profiles[ue.slice_type].min_reliability
        satisfaction = min(reliability / min_reliability, 1.0)
        
        return satisfaction, reliability

    def _calculate_urllc_penalty(self, da_metrics: Dict) -> float:
        """
        Calculate penalty for URLLC violations
        
        Args:
            da_metrics: Metrics dictionary from _calculate_qos_reward
        
        Returns:
            penalty: Penalty score for URLLC violations
        """
        urllc_penalty = 0.0
        
        for da_id, metrics in da_metrics.items():
            if metrics['slice_type'] == 'urllc':
                # Penalize if any URLLC UE exceeds 1ms
                delays_over_threshold = [d for d in metrics['delays'] if d > 1.0]
                if delays_over_threshold:
                    violation_rate = len(delays_over_threshold) / len(metrics['delays'])
                    urllc_penalty += 0.5 * violation_rate
        
        return urllc_penalty

    def _get_qos_weights(self, slice_type: str) -> Dict[str, float]:
        """Get QoS metric weights per slice"""
        weights = {
            'embb': {
                'throughput': 0.7,
                'delay': 0.2,
                'reliability': 0.1
            },
            'urllc': {
                'throughput': 0.1,
                'delay': 0.5,      # CRITICAL
                'reliability': 0.4  # CRITICAL
            },
            'mmtc': {
                'throughput': 0.2,
                'delay': 0.1,
                'reliability': 0.7
            }
        }
        return weights[slice_type]

    def _calculate_energy_consumption_penalty(self) -> float:
        """Calculate normalized energy consumption"""
        energy_consumption_penalty = 0.0
        total_energy_ratio = 0.0
        
        
        for uav in self.uavs.values():
            max_transmission_energy = uav.max_power * self.T_L  # Max possible energy use in T_L
            max_movement_energy = self.energy_models.get('vertical_energy_factor', 0.15) * (uav.velocity_max)**2 * self.T_L  # Max possible movement energy in T_L
            max_total_energy = max_transmission_energy + max_movement_energy # Max possible energy used in T_L
            total_energy_ratio += sum(uav.energy_used.values()) / max_total_energy  # Normalize by max possible energy use in T_L

        energy_consumption_penalty = total_energy_ratio / self.num_uavs
        # print("Total energy used by all UAVs:", total_energy_ratio)

        # Extra penalty if any UAV's power is near zero
        zero_power_penalty = sum(1 for uav in self.uavs.values() if uav.current_power < 0.03 * uav.max_power)
        if zero_power_penalty > 0:
            energy_consumption_penalty += zero_power_penalty * 0.2  # Add 0.2 penalty per zero-power UAV
        return energy_consumption_penalty
 
    def _calculate_fairness_index(self) -> float:
        """PRODUCTION VERSION: Fast and clean"""
        if not self.sinr_cache_valid:
            self._update_sinr_cache()
        
        # Pre-compute min rates
        min_rates = {
            'embb': self.qos_profiles['embb'].min_rate,
            'urllc': self.qos_profiles['urllc'].min_rate,
            'mmtc': self.qos_profiles['mmtc'].min_rate
        }
        
        # Collect normalized throughputs
        throughputs = []
        
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_uav is None:
                continue
            
            ue_throughput = self._get_cached_throughput(ue)
            
            # Normalize
            throughputs.append(ue_throughput / min_rates[ue.slice_type])
        
        if not throughputs:
            return 0.0
        
        # NumPy Jain's fairness
        x = np.array(throughputs, dtype=np.float32)
        n = len(x)
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x * x)
        
        if sum_x_sq == 0:
            return 0.0
        
        return float((sum_x * sum_x) / (n * sum_x_sq))

    # ============================================
    # STATISTICS METHODS
    # ============================================

    def print_cache_stats(self):
        """Print cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 0:
            hit_rate = self.cache_hits / total_accesses * 100
            print(f"SINR Cache Stats: Hits={self.cache_hits}, Misses={self.cache_misses}, Hit Rate={hit_rate:.1f}%")
            print(f"Cache Size: {len(self.sinr_cache)} entries")

    def get_da_details(self):
        """Return detailed table lines of each Demand Area's performance"""
        lines = []

        if not self.demand_areas:
            lines.append("No Demand Areas to display")
            return lines

        lines.append("=" * 60)
        lines.append("DEMAND AREA PERFORMANCE DETAILS")
        lines.append("=" * 60)

        # Header
        header = f"{'ID':<4} {'UAV':<4} {'Slice':<6} {'Distance':<8} {'#UEs':<5} {'Allocated BW':<13} {'RBs':<4} {'Raw Alloc':<10}"
        lines.append(header)
        lines.append("-" * 60)

        # Body
        for da in self.demand_areas.values():
            allocated_bw_mhz = da.allocated_bandwidth  # Convert to MHz
            raw_allocated_action = da.raw_allocated_action
            row = f"{da.id:<4} {da.uav_id:<4} {da.slice_type.upper():<6} {da.distance_level:<8} {len(da.user_ids):<5} {allocated_bw_mhz:<13.2f} {len(da.RB_ids_list):<4} {raw_allocated_action:<10.3f}"
            lines.append(row)

        lines.append("-" * 60)
        total_ues = sum(len(da.user_ids) for da in self.demand_areas.values())
        total_allocated_bw = sum(da.allocated_bandwidth for da in self.demand_areas.values()) / 1e6  # MHz
        summary = f"SUMMARY: {len(self.demand_areas)} DAs | Total UEs: {total_ues} | Total Allocated BW: {total_allocated_bw:.2f} MHz"
        lines.append(summary)
        lines.append("=" * 60)

        return lines

    def _print_statistics(self):
        """Print current statistics of the environment"""

        print("Environment Initialization Parameters:")
        print(f"  Number of UAVs: {self.num_uavs}")
        print(f"  Number of UEs: {self.num_ues}")
        print(f"  Service Area: {self.service_area} meters")
        print(f"  UAV Flight Range X: {self.uav_fly_range_x} meters")
        print(f"  UAV Flight Range Y: {self.uav_fly_range_y} meters")
        print(f"  UAV Flight Range H: {self.uav_fly_range_h} meters")
        print(f"  UAV Beam Angle: {self.uav_beam_angle} degrees")
        print(f"  Number of DAs per UAV: {self.num_das_per_slice}")
        print(f"  Long-term period T_L: {self.T_L} seconds")
        print(f"  Short-term period T_S: {self.T_S} seconds")
        print(f"  QoS Profiles: {self.qos_profiles}")
        print(f"  Slice Weights: {self.slice_weights}")
        print(f"  Noise Power: {self.noise_power} Watts")
        print(f"  Path Loss Exponent: {self.path_loss_exponent}")
        print(f"  Carrier Frequency: {self.carrier_frequency} Hz")
        print(f"  RB Bandwidth: {self.rb_bandwidth} Hz")
        print(f"  Total Bandwidth per UAV: {self.total_bandwidth} Hz")
        print(f"  Total Resource Blocks: {self.total_rbs}")
        print(f"  UE Arrival Rate: {self.ue_arrival_rate} per second")
        print(f"  UE Departure Rate: {self.ue_departure_rate} per minute")
        print(f"  UE dynamics Params: {self.ue_dynamics}")
        print(f"  Max UEs Allowed: {self.max_ues}")

    def get_uavs_power_usage(self) -> Dict[int, float]:
        """Get current power usage of each UAV"""
        uav_power_usage = {}
        for uav in self.uavs.values():
            uav_power_usage[uav.id] = uav.current_power
        return uav_power_usage

    def get_UEs_throughput_demand_and_satisfaction(self) -> Dict[int, Tuple[float, float, float]]:
        """Get current throughput, demand, and satisfaction of each UE"""
        ue_info = {}
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            ue_throughput = self._get_cached_throughput(ue)

            min_rate = self.qos_profiles[ue.slice_type].min_rate
            satisfaction = min(ue_throughput / min_rate, 1.0) if min_rate > 0 else 1.0
            
            ue_info[ue.id] = (ue_throughput, min_rate, satisfaction)
        
        return ue_info