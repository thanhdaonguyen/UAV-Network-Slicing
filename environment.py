
import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import Configuration
from numba import jit, prange
from collections import deque

@dataclass
class Packet:
    """Individual packet in queue"""
    ue_id: int
    size: int  # bytes
    enqueue_time: float  # seconds
    slice_type: str
    deadline: Optional[float] = None  # For URLLC

class QueueingModel:
    """
    Per-DA queuing with M/M/1 approximation
    Makes bandwidth allocation strategic!
    """
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.queues = {}  # {da_id: deque of Packets}
        self.dropped_packets = {}  # Track drops per DA
        self.serviced_packets = {}  # Track successful transmissions
        
    def enqueue_packet(self, da_id: int, packet: Packet) -> bool:
        """
        Try to enqueue packet
        Returns: True if enqueued, False if dropped (buffer overflow)
        """
        if da_id not in self.queues:
            self.queues[da_id] = deque(maxlen=self.buffer_size)
            self.dropped_packets[da_id] = 0
            self.serviced_packets[da_id] = 0
        
        if len(self.queues[da_id]) >= self.buffer_size:
            # Buffer overflow - packet dropped
            self.dropped_packets[da_id] += 1
            return False
        
        self.queues[da_id].append(packet)
        return True
    
    def service_packets(self, da_id: int, service_rate: float, 
                    timestep: float, current_time: float) -> List[Tuple[Packet, float]]:
        """
        Service packets from queue
        
        Args:
            da_id: Demand area ID
            service_rate: packets/second (from allocated bandwidth + SINR)
            timestep: seconds (T_L)
            current_time: current simulation time
        
        Returns:
            List of (packet, queuing_delay_ms) tuples
        """
        if da_id not in self.queues:
            return []
        
        # Number of packets we can serve
        num_serviced = int(service_rate * timestep)
        
        serviced = []
        for _ in range(min(num_serviced, len(self.queues[da_id]))):
            packet = self.queues[da_id].popleft()
            queuing_delay_ms = (current_time - packet.enqueue_time) * 1000
            serviced.append((packet, queuing_delay_ms))
            self.serviced_packets[da_id] += 1
        
        return serviced
    
    def get_queue_stats(self, da_id: int, current_time: float) -> Dict:
        """Get queue statistics for observation/reward"""
        if da_id not in self.queues or not self.queues[da_id]:
            return {
                'length': 0,
                'utilization': 0.0,
                'avg_delay_ms': 0.0,
                'max_delay_ms': 0.0,
                'drop_rate': 0.0
            }
        
        queue = self.queues[da_id]
        delays_ms = [(current_time - p.enqueue_time) * 1000 for p in queue]
        
        total_packets = self.serviced_packets[da_id] + self.dropped_packets[da_id]
        drop_rate = self.dropped_packets[da_id] / max(total_packets, 1)
        
        return {
            'length': len(queue),
            'utilization': len(queue) / self.buffer_size,
            'avg_delay_ms': np.mean(delays_ms) if delays_ms else 0.0,
            'max_delay_ms': np.max(delays_ms) if delays_ms else 0.0,
            'drop_rate': drop_rate
        }

class HandoverTracker:
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
        if old_uav is None or old_uav == new_uav:
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
        
        delay_penalty = self.handover_delay_ms
        if is_ping_pong:
            delay_penalty *= 1.5
        
        return {
            'handover': True,
            'delay_penalty_ms': delay_penalty,
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
    energy_used: float = 0.0  # Joules used in last step
    RBs: List['ResourceBlock'] = None  # List of ResourceBlocks
    beam_angle: float = 60.0  # NEW: Maximum beam angle in degrees (half-angle from vertical)
    beam_direction: np.ndarray = None  # NEW: Beam pointing direction (default: straight down)

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
    is_active: bool = True  # NEW: whether UE is currently active
    velocity: np.ndarray = None  # NEW: velocity vector for movement
    channel_gain: float = 0.0  # Channel gain to serving UAV
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

        # Queuing model
        self.queuing_model = QueueingModel(buffer_size=100)
        # Packet generation rates (packets/second per UE)
        self.packet_rates = env_config.packet_rates
        # Handover tracker
        self.handover_tracker = HandoverTracker()

        # UAV parameters
        self.uav_params = env_config.uav_params  # UAV parameters like max power, battery, etc.
        self.uav_beam_angle = getattr(env_config.uav_params, 'beam_angle', 60.0)
        
        # UE dynamics parameters
        self.ue_dynamics = env_config.ue_dynamics  # UE dynamics parameters
        self.ue_arrival_rate = self.ue_dynamics.arrival_rate  # New UEs per second
        self.ue_departure_rate = self.ue_dynamics.departure_rate  # Probability of UE leaving per minute
        self.ue_max_initial_velocity = self.ue_dynamics.max_initial_velocity  # m/s
        self.max_ues = int(self.num_ues * self.ue_dynamics.max_ues_multiplier)  # Maximum UEs allowed
        self.next_ue_id = self.num_ues  # For generating new UE IDs
        self.change_direction_prob = self.ue_dynamics.change_direction_prob  # Probability of changing direction

        self.hotspots = []
        self.hotspot_attraction_strength = 0.6
        self.hotspot_spawn_rate = 0.15  # Probability per T_L period (15% chance)
        self.hotspot_max_count = 4  # Maximum concurrent hotspots
        self.hotspot_min_lifetime = 300  # seconds
        self.hotspot_max_lifetime = 400.0  # seconds

        for _ in range(2):
            self._spawn_new_hotspot()


        # Distance thresholds for forming Demand Areas
        self.da_distance_thresholds = env_config.da_distance_thresholds

        # SINR thresholds for DA classification (in dB)
        self.sinr_thresholds = env_config.sinr_thresholds  

        # Reward weights
        self.reward_weights = env_config.reward_weights

        # Energy consumption parameters
        self.movement_energy_factor = env_config.energy_models.movement_energy_factor  # Joules per meter
        self.transmission_energy_factor = env_config.energy_models.transmission_energy_factor  # Joules per

        # Statistics tracking
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'handovers': 0
        }
        
        # Initialize entities
        self.reset()
        self._print_statistics()

        # Add these new attributes for caching
        self.sinr_cache = {}
        self.sinr_cache_valid = False

        self.delay_batch_cache = {}
        self.delay_batch_valid = False
        
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
                energy_used=0.0,
                RBs=self._create_RBs(),
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

    def _spawn_new_hotspot(self):
        """Create a new hotspot at random location"""
        # Random position in service area
        x = np.random.uniform(self.service_area[0] * 0.1, self.service_area[0] * 0.9)
        y = np.random.uniform(self.service_area[1] * 0.1, self.service_area[1] * 0.9)
        
        # Random properties
        hotspot = {
            'position': np.array([x, y]),
            'radius': np.random.uniform(100.0, 300.0),  # Random size
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
        """Update UE positions, handle arrivals/departures - WITH DYNAMIC HOTSPOTS"""
        # Reset statistics for this step
        self.stats['arrivals'] = 0
        self.stats['departures'] = 0
        
        # NEW: Update hotspot lifecycles first
        self._update_hotspots()

        # 1. Update existing UE positions
        for ue in list(self.ues.values()):
            if not ue.is_active:
                continue
            
            # NEW: Get hotspot attraction
            hotspot_pull = self._get_hotspot_influence(ue.position)
            current_speed = np.linalg.norm(ue.velocity[:2])
            
            if current_speed > 0:
                # Blend current direction with hotspot attraction
                ue.velocity[:2] = (
                    (1 - self.hotspot_attraction_strength) * ue.velocity[:2] + 
                    self.hotspot_attraction_strength * hotspot_pull * current_speed
                )
            
            # Update position based on velocity (SAME AS BEFORE)
            new_position = ue.position + ue.velocity * self.T_L
            
            # Boundary handling - bounce off walls (SAME AS BEFORE)
            for dim in range(2):  # Only x and y
                if new_position[dim] < 0 or new_position[dim] > self.service_area[dim]:
                    ue.velocity[dim] = -ue.velocity[dim]
                    new_position[dim] = np.clip(new_position[dim], 0, self.service_area[dim])
            
            ue.position = new_position
            
            # Randomly change direction occasionally (SAME AS BEFORE)
            if np.random.random() < self.change_direction_prob:
                speed = np.linalg.norm(ue.velocity[:2])
                if speed > 0:
                    new_direction = np.random.uniform(0, 2 * np.pi)
                    ue.velocity[0] = speed * np.cos(new_direction)
                    ue.velocity[1] = speed * np.sin(new_direction)
            
            # Check if UE should leave (SAME AS BEFORE)
            if np.random.random() < self.ue_departure_rate:
                ue.is_active = False
                self.stats['departures'] += 1
        
        # 2. Handle new UE arrivals (SAME AS BEFORE)
        active_ue_count = len([ue for ue in self.ues.values() if ue.is_active])
        expected_arrivals = self.ue_arrival_rate * self.T_L
        num_arrivals = min(
            np.random.poisson(expected_arrivals),
            self.max_ues - active_ue_count
        )
        
        for _ in range(num_arrivals):
            self._add_new_ue()
            self.stats['arrivals'] += 1
        
        # 3. Clean up inactive UEs periodically (SAME AS BEFORE)
        if len(self.ues) > self.max_ues:
            inactive_ues = [ue_id for ue_id, ue in self.ues.items() if not ue.is_active]
            for ue_id in inactive_ues[:len(inactive_ues)//2]:
                del self.ues[ue_id]

    def _add_new_ue(self):
        """Add a new UE to the network - spawn near active hotspots"""
        
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
        
        # Rest of your original code...
        speed = np.random.uniform(0, self.ue_max_initial_velocity)
        direction = np.random.uniform(0, 2 * np.pi)
        velocity_x = speed * np.cos(direction)
        velocity_y = speed * np.sin(direction)
        
        slice_type = np.random.choice(
            ["embb", "urllc", "mmtc"],
            p=[self.slice_probs["embb"], self.slice_probs["urllc"], self.slice_probs["mmtc"]]
        )
        
        new_ue = UE(
            id=self.next_ue_id,
            position=np.array([x, y, 0.0]),
            slice_type=slice_type,
            velocity=np.array([velocity_x, velocity_y, 0.0]),
            is_active=True
        )
        
        self.ues[self.next_ue_id] = new_ue
        self.next_ue_id += 1

    def _create_RBs(self) -> List[ResourceBlock]:
        """Create Resource Blocks for a UAV"""
        rbs = []
        for i in range(self.total_rbs):
            rb = ResourceBlock(
                id=i, 
                bandwidth = self.rb_bandwidth,
                frequency = self.carrier_frequency
            )
            rbs.append(rb)
        return rbs

    def get_hotspot_stats(self):
        """Get current hotspot statistics for monitoring"""
        if not self.hotspots:
            return "No active hotspots"
        
        stats = []
        for i, h in enumerate(self.hotspots):
            stats.append(
                f"Hotspot {i}: pos=({h['position'][0]:.0f},{h['position'][1]:.0f}) "
                f"strength={h['current_strength']:.2f} state={h['state']} "
                f"age={h['age']:.0f}s/{h['lifetime']:.0f}s"
            )
        return "\n".join(stats)
        
    def _generate_packets(self):
        """Generate packets for all active UEs"""
        for ue in self.ues.values():
            if not ue.is_active or ue.assigned_da is None:
                continue
            
            # Poisson arrivals
            rate = self.packet_rates[ue.slice_type]
            num_packets = np.random.poisson(rate * self.T_L)
            
            for _ in range(num_packets):
                packet = Packet(
                    ue_id=ue.id,
                    size=1500 if ue.slice_type == 'embb' else 100,  # bytes
                    enqueue_time=self.current_time,
                    slice_type=ue.slice_type,
                    deadline=self.current_time + 0.001 if ue.slice_type == 'urllc' else None
                )
                
                # Try to enqueue
                success = self.queuing_model.enqueue_packet(ue.assigned_da, packet)
                
                if not success:
                    # Buffer overflow - affects reliability!
                    pass  # Will be tracked in queue stats

    def _generate_packets(self):
        """Generate packets for all active UEs - VECTORIZED VERSION"""
        
        # Collect all active UEs with assigned DAs
        active_ues = [(ue_id, ue) for ue_id, ue in self.ues.items() 
                    if ue.is_active and ue.assigned_da is not None]
        
        if not active_ues:
            return
        
        # Vectorized Poisson sampling for all UEs at once
        ue_ids = [ue_id for ue_id, _ in active_ues]
        ues = [ue for _, ue in active_ues]
        rates = np.array([self.packet_rates[ue.slice_type] for ue in ues])
        
        # Generate packet counts for all UEs simultaneously
        num_packets_per_ue = np.random.poisson(rates * self.T_L)
        
        # Batch enqueue packets per DA (critical optimization!)
        da_packet_batches = {}  # {da_id: [(ue_id, num_packets, slice_type), ...]}
        
        for ue_id, ue, num_packets in zip(ue_ids, ues, num_packets_per_ue):
            if num_packets == 0:
                continue
            
            da_id = ue.assigned_da
            if da_id not in da_packet_batches:
                da_packet_batches[da_id] = []
            
            da_packet_batches[da_id].append((ue_id, num_packets, ue.slice_type))
        
        # Enqueue in batches
        for da_id, batch in da_packet_batches.items():
            for ue_id, num_packets, slice_type in batch:
                packet_size = 1500 if slice_type == 'embb' else 100
                deadline = self.current_time + 0.001 if slice_type == 'urllc' else None
                
                # Enqueue multiple packets at once (still need loop, but fewer iterations)
                for _ in range(num_packets):
                    packet = Packet(
                        ue_id=ue_id,
                        size=packet_size,
                        enqueue_time=self.current_time,
                        slice_type=slice_type,
                        deadline=deadline
                    )
                    
                    success = self.queuing_model.enqueue_packet(da_id, packet)
                    
                    if not success:
                        break  # Stop if buffer full for this UE

    def _service_packets(self):
        """Service packets from all DAs"""
        for da in self.demand_areas.values():
            if not da.user_ids:
                continue
            
            # Calculate service rate for this DA
            # Depends on: allocated bandwidth + average SINR
            uav = self.uavs[da.uav_id]
            
            # Total data rate for this DA (aggregate RBs)
            allocated_bw = len(da.RB_ids_list) * self.rb_bandwidth  # Hz
            
            # Average SINR for UEs in this DA
            avg_sinr = self._get_avg_da_sinr(da, uav)
            sinr_linear = 10 ** (avg_sinr / 10)
            
            # Shannon capacity
            total_data_rate_bps = allocated_bw * np.log2(1 + sinr_linear)
            
            # Service rate (packets/second)
            avg_packet_size = 1500 if da.slice_type == 'embb' else 100
            service_rate = total_data_rate_bps / (avg_packet_size * 8)
            
            # Service packets
            serviced = self.queuing_model.service_packets(
                da.id, 
                service_rate, 
                self.T_L, 
                self.current_time
            )
            
            # Store queuing delays for each UE (for reward calculation)
            for packet, delay_ms in serviced:
                if packet.ue_id not in self.ues:
                    continue
                
                # Track this for delay statistics
                if not hasattr(self, 'ue_queuing_delays'):
                    self.ue_queuing_delays = {}
                
                if packet.ue_id not in self.ue_queuing_delays:
                    self.ue_queuing_delays[packet.ue_id] = []
                
                self.ue_queuing_delays[packet.ue_id].append(delay_ms)
                
                # Keep only recent history
                if len(self.ue_queuing_delays[packet.ue_id]) > 100:
                    self.ue_queuing_delays[packet.ue_id].pop(0)

    def _get_avg_da_sinr(self, da: DemandArea, uav: UAV) -> float:
        """Calculate average SINR for DA (existing method - keep as is)"""
        if not da.user_ids:
            return 0.0
        
        sinrs = []
        for ue_id in da.user_ids:
            if ue_id in self.ues and self.ues[ue_id].is_active:
                ue = self.ues[ue_id]
                if ue.assigned_rb:
                    for rb in ue.assigned_rb:
                        sinr = self._get_cached_sinr(ue, uav, rb)
                        sinrs.append(sinr)
        
        return np.mean(sinrs) if sinrs else 0.0

    def _associate_ues_to_uavs(self):
        """Associate UEs to nearest UAV based on distance AND beam angle constraint"""
        self.stats['handovers'] = 0
        
        # Store previous assignments for handover tracking
        previous_assignments = {ue.id: ue.assigned_uav for ue in self.ues.values() if ue.is_active}
        
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            
            min_distance = float('inf')
            best_uav = None
            
            for uav in self.uavs.values():
                # Check beam angle constraint first
                if not self._is_ue_in_beam(uav, ue):
                    continue  # Skip this UAV if UE is outside beam
                
                distance = np.linalg.norm(uav.position - ue.position)
                if distance < min_distance:
                    min_distance = distance
                    best_uav = uav.id
            
            ue.assigned_uav = best_uav

        # After association, check for handovers
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            
            old_uav = previous_assignments.get(ue.id)
            new_uav = ue.assigned_uav
            
            handover_info = self.handover_tracker.check_handover(
                ue.id, old_uav, new_uav, self.current_time
            )
            
            # Store handover penalty for this UE
            if not hasattr(ue, 'handover_penalty_ms'):
                ue.handover_penalty_ms = 0.0
            
            if handover_info['handover']:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']
                self.stats['handovers'] += 1
            else:
                ue.handover_penalty_ms = 0.0

    def _associate_ues_to_uavs(self):
        """VECTORIZED: Associate UEs to nearest UAV with beam constraint"""
        self.stats['handovers'] = 0
        
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
            
            handover_info = self.handover_tracker.check_handover(
                ue.id, old_uav, new_uav, self.current_time
            )
            
            if not hasattr(ue, 'handover_penalty_ms'):
                ue.handover_penalty_ms = 0.0
            
            if handover_info['handover']:
                ue.handover_penalty_ms = handover_info['delay_penalty_ms']
                self.stats['handovers'] += 1
            else:
                ue.handover_penalty_ms = 0.0

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
    
    def _update_delay_batch_cache(self):
        """
        Calculate ALL delays at once instead of 1.2M individual calls
        This is the #2 bottleneck (175s)
        """
        self.delay_batch_cache.clear()
        
        # Collect all active UEs with RBs
        active_ues = [
            (ue_id, ue) for ue_id, ue in self.ues.items() 
            if ue.is_active and ue.assigned_rb and ue.assigned_uav is not None
        ]
        
        if not active_ues:
            self.delay_batch_valid = True
            return
        
        # Vectorized propagation delays
        ue_positions = np.array([ue.position for _, ue in active_ues], dtype=np.float32)
        uav_positions = np.array([
            self.uavs[ue.assigned_uav].position for _, ue in active_ues
        ], dtype=np.float32)
        
        distances = np.linalg.norm(ue_positions - uav_positions, axis=1)
        t_prop = (distances / 3e8) * 1000  # ms
        
        # Process each UE's delay components
        for i, (ue_id, ue) in enumerate(active_ues):
            uav = self.uavs[ue.assigned_uav]
            
            # Aggregate throughput across RBs (vectorized where possible)
            total_data_rate = 0.0
            sinr_values = []
            
            for rb in ue.assigned_rb:
                sinr_db = self._get_cached_sinr(ue, uav, rb)
                sinr_values.append(sinr_db)
                sinr_linear = 10 ** (sinr_db / 10)
                total_data_rate += rb.bandwidth * np.log2(1 + sinr_linear)
            
            # Transmission delay
            packet_size_bits = 1500 * 8 if ue.slice_type == 'embb' else 100 * 8
            t_trans = (packet_size_bits / total_data_rate) * 1000 if total_data_rate > 0 else 100.0
            t_trans = np.ceil(t_trans / 1.0) * 1.0
            
            # Retransmission
            avg_sinr = np.mean(sinr_values)
            per = self._calculate_packet_error_rate(avg_sinr, packet_size_bits // 8)
            t_retx = per * 4 * 8.0
            
            # Queuing delay
            if hasattr(self, 'ue_queuing_delays') and ue_id in self.ue_queuing_delays:
                t_queue = np.percentile(self.ue_queuing_delays[ue_id], 95) if self.ue_queuing_delays[ue_id] else 0.0
            else:
                t_queue = 0.0
            
            # Scheduling delay
            da = self.demand_areas.get(ue.assigned_da)
            if da and da.RB_ids_list:
                num_ues = len(da.user_ids)
                num_rbs = len(da.RB_ids_list)
                ues_per_rb = num_ues / num_rbs
                t_sched = max((ues_per_rb - 1) / 2, 0) * 1.0
            else:
                t_sched = 0.0
            
            # Other delays
            t_handover = getattr(ue, 'handover_penalty_ms', 0.0)
            t_proc = 3.0
            
            gateway_pos = np.array([self.service_area[0]/2, self.service_area[1]/2, 0])
            backhaul_distance_km = np.linalg.norm(uav.position[:2] - gateway_pos[:2]) / 1000.0
            t_backhaul = 10.0 + backhaul_distance_km * 0.01
            
            # Total delay
            total_delay = (
                t_prop[i] + t_trans + t_retx +
                t_queue + t_sched +
                t_handover +
                t_proc + t_backhaul
            )
            
            # Cache result
            self.delay_batch_cache[ue_id] = {
                'total': min(total_delay, 1000.0),
                'breakdown': {
                    'propagation': t_prop[i],
                    'transmission': t_trans,
                    'retransmission': t_retx,
                    'queuing': t_queue,
                    'scheduling': t_sched,
                    'handover': t_handover,
                    'processing': t_proc,
                    'backhaul': t_backhaul
                }
            }
        
        self.delay_batch_valid = True

    def calculate_total_delay_ms(self, ue: UE) -> Dict[str, float]:
        """OPTIMIZED: Use batch cache"""
        # Check if batch cache is valid
        if not self.delay_batch_valid:
            self._update_delay_batch_cache()
        
        # Return cached value
        if ue.id in self.delay_batch_cache:
            return self.delay_batch_cache[ue.id]
        
        # Fallback for edge cases
        return {'total': float('inf'), 'breakdown': {}}
    
    def _invalidate_delay_cache(self):
        """Call when network state changes"""
        self.delay_batch_valid = False

    def _calculate_packet_error_rate(self, sinr_db: float, packet_size_bytes: int) -> float:
        """Calculate PER based on SINR (existing - keep as is or use this)"""
        # Approximate BLER curve
        if sinr_db < -5:
            per = 0.5
        elif sinr_db < 0:
            per = 0.2 * np.exp(-sinr_db / 3)
        elif sinr_db < 10:
            per = 0.1 * np.exp(-sinr_db / 5)
        elif sinr_db < 20:
            per = 0.01 * np.exp(-sinr_db / 10)
        else:
            per = 1e-5
        
        # Adjust for packet size
        size_factor = packet_size_bytes / 1500.0
        return min(1.0, per * size_factor)

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

                    # Queue stats (3 dims) ---
                    queue_stats = self.queuing_model.get_queue_stats(da.id, self.current_time)
                    # print(f"UAV {uav_id} DA {da.id} queue stats: {queue_stats}")
                    # obs.append(queue_stats['utilization'])      # Buffer utilization [0-1]
                    # obs.append(queue_stats['avg_delay_ms'] / 100.0)  # Normalized by 100ms
                    # obs.append(queue_stats['drop_rate'])        # Packet drop rate [0-1]

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

            # ============================================
            # 4. Load Prediction Features (5 dims). So far 5 + 90 + 4 + 5 = 104
            # ============================================
            
            # UE arrival trend (arrivals in last timestep)
            # obs.append(self.stats['arrivals'] / 10.0)  # Normalize
            
            # # UE departure trend
            # obs.append(self.stats['departures'] / 10.0)
            
            # # Hotspot proximity (distance to nearest active hotspot)
            # if self.hotspots:
            #     distances = [np.linalg.norm(uav.position[:2] - h['position']) 
            #                 for h in self.hotspots if h['current_strength'] > 0.3]
            #     min_dist = min(distances) if distances else 2000.0
            #     obs.append(min_dist / 2000.0)  # Normalize by service area
                
            #     # Hotspot strength
            #     strongest = max([h['current_strength'] for h in self.hotspots], default=0.0)
            #     obs.append(strongest)
                
            #     # Hotspot state (growing/active/fading)
            #     nearest_hotspot = min(self.hotspots, 
            #                         key=lambda h: np.linalg.norm(uav.position[:2] - h['position']))
            #     state_encoding = {'growing': 0.33, 'active': 0.66, 'fading': 1.0}
            #     obs.append(state_encoding.get(nearest_hotspot['state'], 0.0))
            # else:
            #     obs.extend([1.0, 0.0, 0.0])  # No hotspots
            

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

        # Invalidate cache BEFORE making changes
        self._invalidate_sinr_cache()

        # Update UE dynamics (NEW)
        self._update_ue_dynamics()

        # NEW: Generate and enqueue packets (BEFORE calculating reward)
        self._generate_packets()
        
        # NEW: Service packets from queues
        self._service_packets()
        
        # Update associations and DAs (every long-term period)
        self._associate_ues_to_uavs()
        self._form_demand_areas()

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
            movement_distance = np.linalg.norm(delta_pos)
            movement_energy = self.movement_energy_factor * movement_distance  # Simple linear model

            uav.position = new_pos
            
            # Power update
            uav.current_power = np.clip(action[3] * uav.max_power, 0.3 * uav.max_power, uav.max_power)
            
            # Energy consumption
            transmission_energy = uav.current_power * self.T_L
            uav.energy_used = (transmission_energy + movement_energy)
            uav.current_battery -= uav.energy_used
            uav.current_battery = max(0, uav.current_battery)
            
            # Bandwidth allocation to DAs
            uav_das = [da for da in self.demand_areas.values() if da.uav_id == uav_id]
            if len(uav_das) > 0:
                # Normalized bandwidth allocation
                bandwidth_actions = action[4:4+len(uav_das)]
                
                for i, da in enumerate(uav_das):
                    da.raw_allocated_action = bandwidth_actions[i]
                    da.allocated_bandwidth = bandwidth_actions[i] * uav.max_bandwidth

                

            self._allocate_rbs_fairly(uav, uav_das)

        # Calculate rewards

        reward, qos_reward, energy_penalty, fairness_reward = self._calculate_global_reward()

        # Update time
        self.current_time += self.T_L
        
        # Check termination conditions
        done = any(uav.current_battery <= 0 for uav in self.uavs.values())
        done = False
        
        # Get new observations
        observations = self._get_observations()
        
        info = {
            'qos_satisfaction': qos_reward,
            'energy_efficiency': energy_penalty,
            'fairness_level': fairness_reward,
            'active_ues': len([ue for ue in self.ues.values() if ue.is_active]),
            'ue_arrivals': self.stats['arrivals'],
            'ue_departures': self.stats['departures']
        }
        
        return observations, reward, done, info

    def _allocate_rbs_fairly(self, uav, uav_das):
        """OPTIMIZED: Use NumPy for faster allocation"""
        
        total_rbs = len(uav.RBs)
        num_das = len(uav_das)
        
        if num_das == 0:
            return
        
        # Reset (faster with list comprehension)
        for rb in uav.RBs:
            rb.allocated_da_id = -1
        for da in uav_das:
            da.RB_ids_list = []
        
        # Calculate targets using NumPy (vectorized)
        allocated_bandwidths = np.array([da.allocated_bandwidth for da in uav_das], dtype=np.float32)
        target_rbs = (allocated_bandwidths / self.rb_bandwidth).astype(np.int32)
        
        # Ensure we don't exceed total RBs
        total_requested = target_rbs.sum()
        if total_requested > total_rbs:
            # Scale down proportionally
            target_rbs = ((target_rbs * total_rbs) / total_requested).astype(np.int32)
        
        # Allocate using cumulative sum (vectorized)
        cumsum = np.concatenate([[0], np.cumsum(target_rbs)])
        
        for i, da in enumerate(uav_das):
            start_idx = cumsum[i]
            end_idx = min(cumsum[i + 1], total_rbs)
            
            # Assign RBs to this DA
            for rb_id in range(start_idx, end_idx):
                rb = uav.RBs[rb_id]
                rb.allocated_da_id = da.id
                da.RB_ids_list.append(rb_id)
        
        # RB allocation to UEs (vectorized where possible)
        # Reset all UE assignments for this UAV
        for ue in self.ues.values():
            if ue.assigned_uav == uav.id and ue.is_active:
                ue.assigned_rb = []
        
        # Round-robin within each DA
        for da in uav_das:
            if not da.user_ids or not da.RB_ids_list:
                continue
            
            num_users = len(da.user_ids)
            num_rbs = len(da.RB_ids_list)
            
            # Vectorized assignment
            for i, rb_id in enumerate(da.RB_ids_list):
                ue_id = da.user_ids[i % num_users]
                rb = uav.RBs[rb_id]
                rb.allocated_ue_id = ue_id
                
                if ue_id in self.ues and self.ues[ue_id].is_active:
                    self.ues[ue_id].assigned_rb.append(rb)

        self._invalidate_sinr_cache()
    
    # ============================================
    # REWARD CALCULATION METHODS
    # ============================================
    def _calculate_global_reward(self) -> Tuple[float, float, float, float]:
        """
        ENHANCED: Multi-factor QoS (throughput + delay + reliability)
        """
        if not self.sinr_cache_valid:
            self._update_sinr_cache_vectorized()
        
        # Track metrics per DA
        da_metrics = {}  # {da_id: {'satisfactions': [...], 'delays': [...], 'slice_type': str}}
        
        # Collect metrics for all UEs
        for ue in self.ues.values():
            if not ue.is_active:
                continue
            # ===============================
            # Calculate throughput
            # ===============================
            ue_throughput = 0.0
            if ue.assigned_rb is None or len(ue.assigned_rb) == 0:
                throughput_satisfaction = 0.0
            else:
                for rb in ue.assigned_rb:
                    sinr_db = self.sinr_cache.get((ue.id, ue.assigned_uav, rb.id), -10.0)
                    sinr_linear = 10.0 ** (sinr_db * 0.1)
                    ue_throughput += rb.bandwidth * np.log2(1.0 + sinr_linear)
                
                min_rate = self.qos_profiles[ue.slice_type].min_rate
                throughput_satisfaction = min(ue_throughput / min_rate, 1.0)
            
            # ===============================
            # Calculate delay
            # ===============================
            if ue.assigned_rb is None or len(ue.assigned_rb) == 0:
                total_delay_ms = 99999
                delay_satisfaction = 0.0
            else:
                delay_info = self.calculate_total_delay_ms(ue)
                total_delay_ms = delay_info['total']
                max_latency = self.qos_profiles[ue.slice_type].max_latency

                # Delay satisfaction (exponential penalty for exceeding)
                if total_delay_ms <= max_latency:
                    delay_satisfaction = 1.0
                else:
                    excess = total_delay_ms - max_latency
                    delay_satisfaction = np.exp(-excess / max_latency)
                
            # ===============================
            # Calculate reliability
            # (Based on: buffer drops + handover losses + PER)
            # ===============================
            if ue.assigned_rb is None or len(ue.assigned_rb) == 0:
                reliability_satisfaction = 0.0
            else:
                da = self.demand_areas[ue.assigned_da]
                queue_stats = self.queuing_model.get_queue_stats(da.id, self.current_time)
                drop_rate = queue_stats['drop_rate']
                
                # Handover loss
                handover_loss = 0.01 if getattr(ue, 'handover_penalty_ms', 0) > 0 else 0.0
                
                # Transmission reliability (from SINR)
                avg_sinr = np.mean([self.sinr_cache.get((ue.id, ue.assigned_uav, rb.id), -10.0) 
                                for rb in ue.assigned_rb]) if ue.assigned_rb else -10.0
                per = self._calculate_packet_error_rate(avg_sinr, 1500)
                transmission_success = (1 - per) ** 4  # With 4 retransmissions
                
                # Overall reliability
                reliability = transmission_success * (1 - drop_rate) * (1 - handover_loss)
                min_reliability = self.qos_profiles[ue.slice_type].min_reliability
                reliability_satisfaction = min(reliability / min_reliability, 1.0)
            
            # Combined QoS satisfaction (weighted by slice type)
            qos_weights = self._get_qos_weights(ue.slice_type)
            overall_satisfaction = (
                qos_weights['throughput'] * throughput_satisfaction +
                qos_weights['delay'] * delay_satisfaction +
                qos_weights['reliability'] * reliability_satisfaction
            )
            
            # Store in DA metrics
            da_id = ue.assigned_da
            if da_id not in da_metrics:
                da_metrics[da_id] = {
                    'satisfactions': [],
                    'delays': [],
                    'slice_type': ue.slice_type
                }
            
            da_metrics[da_id]['satisfactions'].append(overall_satisfaction)
            da_metrics[da_id]['delays'].append(total_delay_ms)
        
        # ===============================
        # Aggregate QoS reward
        # ===============================
        aggregate_satisfaction = 0.0
        total_weight = 0.0
        
        for da_id, metrics in da_metrics.items():
            avg_satisfaction = np.mean(metrics['satisfactions'])
            slice_type = metrics['slice_type']
            weight = self.slice_weights[slice_type] * len(metrics['satisfactions'])
            
            aggregate_satisfaction += avg_satisfaction * weight
            total_weight += weight
        
        qos_reward = aggregate_satisfaction / total_weight if total_weight > 0 else 0.0
        
        # URLLC penalty (harsh for failures)
        urllc_penalty = 0.0
        for da_id, metrics in da_metrics.items():
            if metrics['slice_type'] == 'urllc':
                # Penalize if any URLLC UE exceeds 1ms
                delays_over_threshold = [d for d in metrics['delays'] if d > 1.0]
                if delays_over_threshold:
                    urllc_penalty += 0.5 * len(delays_over_threshold) / len(metrics['delays'])
        
        # Energy and fairness (existing)
        energy_penalty = self._calculate_energy_consumption_penalty()
        fairness_reward = self._calculate_fairness_index()
        
        # Combined reward
        alpha = self.reward_weights.qos
        beta = self.reward_weights.energy
        gamma = self.reward_weights.fairness
        delta = 0.4  # NEW: URLLC penalty weight
        
        total_reward = (
            alpha * qos_reward -
            beta * energy_penalty +
            gamma * fairness_reward -
            delta * urllc_penalty
        )
        
        return total_reward, qos_reward, energy_penalty, fairness_reward

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
            total_energy_ratio += uav.energy_used / (uav.max_power * self.T_L)  # Normalize by max possible energy use in T_L

        energy_consumption_penalty = total_energy_ratio / self.num_uavs
        # print("Total energy used by all UAVs:", total_energy_ratio)
        # Extra penalty if any UAV's power is near zero
        zero_power_penalty = sum(1 for uav in self.uavs.values() if uav.current_power < 0.01 * uav.max_power)
        if zero_power_penalty > 0:
            energy_consumption_penalty += zero_power_penalty * 0.2  # Add 0.2 penalty per zero-power UAV
        return energy_consumption_penalty

    def _calculate_channel_gain(self, receiver: UE, transmitter: UAV, rb: ResourceBlock) -> float:
        """Calculate channel gain between two devices"""
        distance = np.linalg.norm(receiver.position - transmitter.position)
        wavelength = 3e8 / rb.frequency  # Speed of light / frequency
        path_loss = (wavelength / (4 * np.pi * distance)) ** self.path_loss_exponent

        return path_loss

    def _calculate_sinr(self, receiver: UE, transmitter: UAV, rb: ResourceBlock) -> float:
        """Calculate SINR for UE from serving UAV"""
        if rb is None:
            return 0.0
        # Signal power from serving UAV
        signal_power = self._calculate_channel_gain(receiver, transmitter, rb) * transmitter.current_power # in Watts


        # Interference from other UAVs on the same RB
        interference_power = 0.0
        for other_uav in self.uavs.values():
            if other_uav.id != transmitter.id and self._is_ue_in_beam(other_uav, receiver):
                interference_power += self._calculate_channel_gain(receiver, other_uav, rb) * other_uav.current_power
        
        # SINR calculation (in dB)
        sinr = signal_power / (interference_power + self.noise_power)
        sinr_db = 10 * np.log10(sinr + 1e-10)  # in dB

        if sinr_db > 100:
            print(f"Debug SINR Calculation:")
            print(f"  UE ID: {receiver.id}")
            print(f"  Transmitter UAV ID: {transmitter.id}")
            print(f"  Resource Block: {rb.id}")
            print(f"  Gain: {self._calculate_channel_gain(receiver, transmitter, rb)}")
            print(f"  Transmitter Power: {transmitter.current_power} W")
            print(f"  Signal Power: {signal_power} W")
            print(f"  Interference Power: {interference_power} W") 
            print(f"  Noise Power: {self.noise_power} W")
            print(f"  SINR (dB): {sinr_db}")

        if sinr_db < -10:
            sinr_db = -10
        elif sinr_db > 50:
            sinr_db = 50

        return sinr_db
    
    def _calculate_power_fairness(self) -> float:
        """Calculate fairness of power distribution across UAVs"""
        power_ratios = []
        
        for uav in self.uavs.values():
            # Normalized power usage
            power_ratio = uav.current_power / uav.max_power
            power_ratios.append(power_ratio)
        
        # Jain's fairness on power distribution
        sum_x = sum(power_ratios)
        sum_x_squared = sum(x**2 for x in power_ratios)
        n = len(power_ratios)
        
        if sum_x_squared == 0:
            return 0.0
        
        fairness = (sum_x ** 2) / (n * sum_x_squared)
        
        # Extra penalty for any UAV at zero power
        zero_power_penalty = sum(1 for p in power_ratios if p < 0.01)
        fairness *= (1 - 0.3 * zero_power_penalty / n)  # Reduce fairness if UAVs are off
        
        return max(0, fairness)

    def _calculate_fairness_index(self) -> float:
        """
        OPTIMIZED: Reduce from 92s to ~20s by:
        1. Removing variance calculation (not critical)
        2. Using faster accumulation
        3. Early exit for empty cases
        """
        if not self.sinr_cache_valid:
            self._update_sinr_cache_vectorized()
        
        # Collect throughputs with minimal overhead
        throughputs = []
        min_rates = []
        
        for ue in self.ues.values():
            if not ue.is_active or not ue.assigned_rb:
                continue
            
            # Calculate throughput (can't avoid this)
            ue_throughput = 0.0
            for rb in ue.assigned_rb:
                sinr_db = self.sinr_cache.get((ue.id, ue.assigned_uav, rb.id), -10.0)
                sinr_linear = 10.0 ** (sinr_db * 0.1)
                ue_throughput += rb.bandwidth * np.log2(1.0 + sinr_linear)
            
            min_rate = self.qos_profiles[ue.slice_type].min_rate
            normalized = min(ue_throughput / min_rate, 2.0)
            
            throughputs.append(normalized)
            min_rates.append(min_rate)
        
        if not throughputs:
            return 0.0
        
        # Fast Jain's fairness (no variance calculation!)
        n = len(throughputs)
        sum_x = sum(throughputs)
        sum_x_sq = sum(x*x for x in throughputs)
        
        if sum_x_sq == 0:
            return 0.0
        
        fairness = (sum_x * sum_x) / (n * sum_x_sq)
        
        # REMOVE: Variance penalty - costs 30s for marginal benefit!
        # if n > 1:
        #     mean_throughput = sum_x / n
        #     variance = sum((x - mean_throughput)**2 for x in throughputs) / n
        #     cv = (variance ** 0.5) / (mean_throughput + 1e-6)
        #     variance_penalty = min(cv / 0.5, 1.0) * 0.3
        #     fairness *= (1.0 - variance_penalty)
        
        return fairness

    def _calculate_rb_utilization(self) -> Dict[int, float]:
        """Calculate RB utilization across all UAVs, based on allocated RBs and power levels"""
        uavs_rb_utilizations = {}
        for uav in self.uavs.values():
            allocated_rbs = [rb for rb in uav.RBs if rb.allocated_ue_id != -1]
            if len(uav.RBs) == 0:
                utilization = 0.0
            else:
                utilization = (len(allocated_rbs) / len(uav.RBs)) * (uav.current_power / uav.max_power)
            uavs_rb_utilizations[uav.id] = utilization

        return uavs_rb_utilizations

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
            ue_throughput = 0.0
            if ue.assigned_uav is not None and ue.assigned_rb:
                for rb in ue.assigned_rb:
                    sinr_db = self._get_cached_sinr(ue, self.uavs[ue.assigned_uav], rb)
                    sinr_linear = 10 ** (sinr_db / 10)
                    throughput = rb.bandwidth * np.log2(1 + sinr_linear)  # in bps
                    ue_throughput += throughput

            min_rate = self.qos_profiles[ue.slice_type].min_rate
            satisfaction = min(ue_throughput / min_rate, 1.0) if min_rate > 0 else 1.0
            
            ue_info[ue.id] = (ue_throughput, min_rate, satisfaction)
        
        return ue_info

    def _update_sinr_cache_vectorized(self):
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

    def _calculate_qos_with_cache(self, ue_sinr_map: dict) -> float:
        """QoS calculation using pre-extracted SINR values"""
        
        aggregate_satisfaction = 0.0
        total_weight = 0.0
        
        for da in self.demand_areas.values():
            if not da.user_ids:
                continue
            
            satisfactions = []
            
            for ue_id in da.user_ids:
                if ue_id not in ue_sinr_map:
                    continue
                
                ue = self.ues[ue_id]
                rb_sinr_pairs = ue_sinr_map[ue_id]
                
                # Calculate throughput using pre-extracted values
                ue_throughput = 0.0
                for rb, sinr_db in rb_sinr_pairs:
                    sinr_linear = 10.0 ** (sinr_db * 0.1)
                    ue_throughput += rb.bandwidth * np.log2(1.0 + sinr_linear)
                
                min_rate = self.qos_profiles[ue.slice_type].min_rate
                satisfaction = min(ue_throughput / min_rate, 1.0)
                satisfactions.append(satisfaction)
            
            if satisfactions:
                avg_satisfaction = sum(satisfactions) / len(satisfactions)
                slice_weight = self.slice_weights[da.slice_type] * len(da.user_ids)
                aggregate_satisfaction += avg_satisfaction * slice_weight
                total_weight += slice_weight
        
        return aggregate_satisfaction / total_weight if total_weight > 0 else 0.0

    def _calculate_fairness_with_cache(self, ue_sinr_map: dict) -> float:
        """Fairness calculation using pre-extracted SINR values"""
        
        throughputs = []
        
        for ue_id, rb_sinr_pairs in ue_sinr_map.items():
            ue = self.ues[ue_id]
            
            # Calculate throughput
            ue_throughput = 0.0
            for rb, sinr_db in rb_sinr_pairs:
                sinr_linear = 10.0 ** (sinr_db * 0.1)
                ue_throughput += rb.bandwidth * np.log2(1.0 + sinr_linear)
            
            min_rate = self.qos_profiles[ue.slice_type].min_rate
            normalized = min(ue_throughput / min_rate, 2.0)
            throughputs.append(normalized)
        
        if not throughputs:
            return 0.0
        
        n = len(throughputs)
        sum_x = sum(throughputs)
        sum_x_sq = sum(x*x for x in throughputs)
        
        if sum_x_sq == 0:
            return 0.0
        
        fairness = (sum_x * sum_x) / (n * sum_x_sq)
        
        if n > 1:
            mean_throughput = sum_x / n
            variance = sum((x - mean_throughput)**2 for x in throughputs) / n
            cv = (variance ** 0.5) / (mean_throughput + 1e-6)
            variance_penalty = min(cv / 0.5, 1.0) * 0.3
            fairness *= (1.0 - variance_penalty)
        
        return fairness

    def _invalidate_sinr_cache(self):
        """Invalidate cache when network state changes"""
        self.sinr_cache_valid = False
    
    def _get_cached_sinr(self, ue: UE, uav: UAV, rb: ResourceBlock) -> float:
        """Get SINR from cache, updating if necessary"""
        if not self.sinr_cache_valid:
            self.cache_misses += 1
            self._update_sinr_cache_vectorized()
        
        cache_key = (ue.id, uav.id, rb.id)
        if cache_key in self.sinr_cache:
            self.cache_hits += 1
            return self.sinr_cache[cache_key]
        else:
            # Fallback to single calculation if not in cache
            # This shouldn't happen often if cache is properly maintained
            # pass
            print(f"Cache miss for UE {ue.id}, UAV {uav.id}, RB {rb.id}")
            self.cache_misses += 1
            return self._calculate_sinr(ue, uav, rb)
    
    def print_cache_stats(self):
        """Print cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 0:
            hit_rate = self.cache_hits / total_accesses * 100
            print(f"SINR Cache Stats: Hits={self.cache_hits}, Misses={self.cache_misses}, Hit Rate={hit_rate:.1f}%")
            print(f"Cache Size: {len(self.sinr_cache)} entries")

    def print_individual_ue_details(self):
        """Print detailed table of each UE's performance"""
        ue_info = self.get_UEs_throughput_demand_and_satisfaction()
        
        if not ue_info:
            print("No active UEs to display")
            return
        
        print("\n" + "="*120)
        print("INDIVIDUAL UE PERFORMANCE DETAILS")
        print("="*120)
        
        # Table header
        header = f"{'UE ID':<6} {'Slice':<6} {'DA': <6} {'Position (x,y)':<15} {'UAV':<4} {'RBs':<4} {'Throughput':<12} {'Demand':<12} {'Satisfaction':<12} {'Status':<12}"
        print(header)
        print("-" * 120)
        
        # Sort UEs by satisfaction (worst first)
        sorted_ues = sorted(ue_info.items(), key=lambda x: x[1][2])
        
        for ue_id, (throughput, demand, satisfaction) in sorted_ues:
            ue = self.ues[ue_id]
            
            # Status determination
            if satisfaction >= 0.9:
                status = "✓ Excellent"
            elif satisfaction >= 0.7:
                status = "○ Good"
            elif satisfaction >= 0.5:
                status = "△ Fair"
            else:
                status = "✗ Poor"
            
            # Format values
            pos_str = f"({ue.position[0]:.0f},{ue.position[1]:.0f})"
            throughput_str = f"{throughput/1e6:.2f} Mbps"
            demand_str = f"{demand/1e6:.2f} Mbps"
            satisfaction_str = f"{satisfaction:.3f} ({satisfaction*100:.1f}%)"
            
            row = f"{ue_id:<6} {ue.slice_type.upper():<6} {ue.assigned_da:<6} {pos_str:<15} {ue.assigned_uav:<4} {len(ue.assigned_rb):<4} {throughput_str:<12} {demand_str:<12} {satisfaction_str:<12} {status:<12}"
            print(row)
        
        # Summary statistics
        satisfactions = [info[2] for info in ue_info.values()]
        # satisfactions = [1 for info in ue_info.values()]
        print("-" * 120)
        print(f"SUMMARY: {len(ue_info)} UEs | Avg Satisfaction: {np.mean(satisfactions):.3f} | "
            f"Satisfied (>90%): {sum(1 for s in satisfactions if s > 0.9)} | "
            f"Unsatisfied (<50%): {sum(1 for s in satisfactions if s < 0.5)}")
        print("="*120)

    def print_da_details(self):
        """Print detailed table of each Demand Area's performance"""
        if not self.demand_areas:
            print("No Demand Areas to display")
            return
        
        print("\n" + "="*100)
        print("DEMAND AREA PERFORMANCE DETAILS")
        print("="*100)
        
        # Table header
        header = f"{'DA ID':<6} {'UAV':<4} {'Slice':<6} {'SINR Level':<10} {'# UEs':<6} {'Allocated BW (MHz)':<18} {'RBs':<4} {'Raw Allocated Action':<20}"
        print(header)
        print("-" * 100)
        
        for da in self.demand_areas.values():
            allocated_bw_mhz = da.allocated_bandwidth  # Convert to MHz
            raw_allocated_action = da.raw_allocated_action
            row = f"{da.id:<6} {da.uav_id:<4} {da.slice_type.upper():<6} {da.distance_level:<10} {len(da.user_ids):<6} {allocated_bw_mhz:<18.2f} {len(da.RB_ids_list):<4} {raw_allocated_action:<20.3f}"
            print(row)
        
        print("-" * 100)
        total_ues = sum(len(da.user_ids) for da in self.demand_areas.values())
        total_allocated_bw = sum(da.allocated_bandwidth for da in self.demand_areas.values()) / 1e6  # MHz
        print(f"SUMMARY: {len(self.demand_areas)} DAs | Total UEs: {total_ues} | Total Allocated BW: {total_allocated_bw:.2f} MHz")
        print("="*100)

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

