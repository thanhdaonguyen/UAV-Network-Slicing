# utils.py
import yaml
from typing import Any, Dict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# ============================================================================
# APPROACH 1: PROCEDURAL DRONE MODEL (Recommended - No External Files)
# ============================================================================

class ProceduralDroneModel:
    """
    Creates a detailed drone model using OpenGL primitives.
    No external files needed - all geometry is generated procedurally.
    """
    
    def __init__(self):
        self.body_size = 10
        self.arm_length = 15
        self.arm_thickness = 1.5
        self.rotor_radius = 5
        self.rotor_thickness = 0.3
        
    def draw_box(self, width, height, depth):
        """Draw a box centered at origin"""
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-width/2, -height/2, depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        glVertex3f(width/2, height/2, depth/2)
        glVertex3f(-width/2, height/2, depth/2)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-width/2, -height/2, -depth/2)
        glVertex3f(-width/2, height/2, -depth/2)
        glVertex3f(width/2, height/2, -depth/2)
        glVertex3f(width/2, -height/2, -depth/2)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-width/2, height/2, -depth/2)
        glVertex3f(-width/2, height/2, depth/2)
        glVertex3f(width/2, height/2, depth/2)
        glVertex3f(width/2, height/2, -depth/2)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-width/2, -height/2, -depth/2)
        glVertex3f(width/2, -height/2, -depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        glVertex3f(-width/2, -height/2, depth/2)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(width/2, -height/2, -depth/2)
        glVertex3f(width/2, height/2, -depth/2)
        glVertex3f(width/2, height/2, depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-width/2, -height/2, -depth/2)
        glVertex3f(-width/2, -height/2, depth/2)
        glVertex3f(-width/2, height/2, depth/2)
        glVertex3f(-width/2, height/2, -depth/2)
        
        glEnd()
    
    def draw_cylinder(self, radius, height, slices=16):
        """Draw a cylinder"""
        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, height, slices, 1)
        
        # Draw caps
        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(0, 0, height)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()
        
        gluDeleteQuadric(quad)
    
    def draw_sphere(self, radius, slices=16, stacks=16):
        """Draw a sphere"""
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
    
    def draw_rotor(self, animation_angle):
        """Draw a single rotor with spinning blades"""
        # Rotor motor housing
        glColor3f(0.2, 0.2, 0.2)
        self.draw_cylinder(self.rotor_radius * 0.3, 2, 12)
        
        # Rotor blades (spinning)
        glPushMatrix()
        glTranslatef(0, 0, 2)
        glRotatef(animation_angle, 0, 0, 1)
        
        glColor3f(0.1, 0.1, 0.15)
        for i in range(2):  # 2 blades
            glPushMatrix()
            glRotatef(i * 180, 0, 0, 1)
            glTranslatef(self.rotor_radius * 0.5, 0, 0)
            
            # Blade shape (elongated box)
            self.draw_box(self.rotor_radius * 1.2, 1, self.rotor_thickness)
            
            glPopMatrix()
        
        glPopMatrix()
    
    def draw_arm(self):
        """Draw a single drone arm"""
        glColor3f(0.3, 0.3, 0.35)
        
        # Main arm tube
        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        self.draw_cylinder(self.arm_thickness, self.arm_length, 8)
        glPopMatrix()
        
        # Arm tip connector
        glPushMatrix()
        glTranslatef(self.arm_length, 0, 0)
        glColor3f(0.25, 0.25, 0.3)
        self.draw_sphere(self.arm_thickness * 1.3, 8, 8)
        glPopMatrix()
    
    def draw_body(self):
        """Draw the main drone body"""
        # Main body (slightly rounded box)
        glColor3f(0.4, 0.4, 0.45)
        self.draw_box(self.body_size, self.body_size * 0.6, self.body_size)
        
        # Top camera/sensor dome
        glPushMatrix()
        glTranslatef(0, self.body_size * 0.35, 0)
        glColor3f(0.2, 0.2, 0.25)
        glScalef(1, 0.5, 1)
        self.draw_sphere(self.body_size * 0.3, 12, 12)
        glPopMatrix()
        
        # Bottom landing gear
        glColor3f(0.3, 0.3, 0.3)
        for i in range(4):
            angle = i * 90 + 45
            rad = math.radians(angle)
            x = math.cos(rad) * self.body_size * 0.4
            z = math.sin(rad) * self.body_size * 0.4
            
            glPushMatrix()
            glTranslatef(x, -self.body_size * 0.4, z)
            glRotatef(90, 1, 0, 0)
            self.draw_cylinder(0.5, 2, 6)
            glPopMatrix()
    
    def draw(self, animation_time, uav_id, is_selected=False):
        """
        Draw the complete drone model
        
        Args:
            animation_time: Time value for animations
            uav_id: UAV ID for unique rotations
            is_selected: Whether this UAV is selected
        """
        # Rotation for rotor animation (faster spin)
        rotor_angle = (animation_time * 720 + uav_id * 120) % 360
        
        # Slight hovering animation
        hover_offset = math.sin(animation_time * 2 + uav_id) * 1.5
        glTranslatef(0, hover_offset, 0)
        
        # Draw main body with selection highlight
        if is_selected:
            # Pulsing glow effect for selected UAV
            pulse = 0.5 + 0.5 * math.sin(animation_time * 3)
            glColor3f(1.0, 0.8 + pulse * 0.2, 0.4)
            
            # # Draw larger translucent sphere as selection indicator
            # glDisable(GL_LIGHTING)
            # glEnable(GL_BLEND)
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # glColor4f(1.0, 0.8, 0.4, 0.3)
            # self.draw_sphere(self.body_size * 1.3, 16, 16)
            # glEnable(GL_LIGHTING)
        
        self.draw_body()
        
        # Draw 4 arms with rotors at cardinal directions
        for i in range(4):
            angle = i * 90 + 45  # Offset by 45 degrees
            
            glPushMatrix()
            glRotatef(angle, 0, 1, 0)
            
            # Draw arm
            self.draw_arm()
            
            # Draw rotor at end of arm
            glTranslatef(self.arm_length, 3, 0)
            # Alternate rotor spin directions for realistic physics
            rotor_direction = 1 if i % 2 == 0 else -1
            self.draw_rotor(rotor_angle * rotor_direction)
            
            glPopMatrix()


# ============================================================================
# APPROACH 2: OBJ MODEL LOADER (Optional - Better Quality)
# ============================================================================

class OBJModelLoader:
    """
    Load and render .obj 3D model files.
    Use this if you have a high-quality drone model.
    """
    
    def __init__(self, filename=None):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.display_list = None
        
        if filename:
            self.load(filename)
    
    def load(self, filename):
        """Load an OBJ file"""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    values = line.split()
                    if not values:
                        continue
                    
                    if values[0] == 'v':
                        # Vertex
                        v = [float(x) for x in values[1:4]]
                        self.vertices.append(v)
                    
                    elif values[0] == 'vn':
                        # Normal
                        n = [float(x) for x in values[1:4]]
                        self.normals.append(n)
                    
                    elif values[0] == 'vt':
                        # Texture coordinate
                        t = [float(x) for x in values[1:3]]
                        self.texcoords.append(t)
                    
                    elif values[0] == 'f':
                        # Face
                        face = []
                        for v in values[1:]:
                            w = v.split('/')
                            face.append([int(w[0]) - 1,  # vertex index
                                       int(w[1]) - 1 if len(w) >= 2 and w[1] else -1,  # texcoord
                                       int(w[2]) - 1 if len(w) >= 3 and w[2] else -1])  # normal
                        self.faces.append(face)
            
            print(f"Loaded {len(self.vertices)} vertices, {len(self.faces)} faces from {filename}")
            
            # Create display list for faster rendering
            self.create_display_list()
            
        except FileNotFoundError:
            print(f"Error: Model file {filename} not found!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def create_display_list(self):
        """Create OpenGL display list for faster rendering"""
        if not self.vertices:
            return
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex in face:
                v_idx, t_idx, n_idx = vertex
                
                if n_idx >= 0 and n_idx < len(self.normals):
                    glNormal3fv(self.normals[n_idx])
                
                if v_idx >= 0 and v_idx < len(self.vertices):
                    glVertex3fv(self.vertices[v_idx])
        glEnd()
        
        glEndList()
    
    def draw(self, animation_time, uav_id, is_selected=False, scale=1.0):
        """Draw the loaded model"""
        if self.display_list:
            glPushMatrix()
            
            # Scale the model
            glScalef(scale, scale, scale)
            
            # Selection highlight
            if is_selected:
                glColor3f(1.0, 0.8, 0.4)
            else:
                glColor3f(0.8, 0.8, 0.8)
            
            # Draw the model
            glCallList(self.display_list)
            
            glPopMatrix()


class NestedDict:
    """Convert nested dict to object with dot notation access"""
    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, NestedDict(value))
            else:
                setattr(self, key, self._convert_value(value))
    
    def _convert_value(self, value):
        """Convert string numbers to appropriate types"""
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or '.' in value:
                    return float(value)
                elif value.isdigit():
                    return int(value)
            except ValueError:
                pass
        return value
    
    def __getitem__(self, key):
        """Enable subscript access like dict[key]"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Enable subscript assignment like dict[key] = value"""
        setattr(self, key, value)
    
    def __contains__(self, key):
        """Enable 'in' operator"""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dict-like get method"""
        return getattr(self, key, default)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"NestedDict({attrs})"

class Configuration:
    """Simple configuration class with nested YAML support and type conversion"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load YAML config and set nested attributes"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Set all config keys as attributes (with nested support)
            for key, value in config_data.items():
                if isinstance(value, dict):
                    setattr(self, key, NestedDict(value))
                else:
                    setattr(self, key, self._convert_value(value))
            
            print(f"✓ Loaded config from {self.config_path}")
            
        except FileNotFoundError:
            print(f"❌ Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
    
    def _convert_value(self, value):
        """Convert string numbers to appropriate types"""
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or '.' in value:
                    return float(value)
                elif value.isdigit():
                    return int(value)
            except ValueError:
                pass
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default value (supports dot notation)"""
        if '.' in key:
            # Handle nested access like 'qos.embb.min_rate'
            keys = key.split('.')
            obj = self
            for k in keys:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    return default
            return obj
        else:
            return getattr(self, key, default)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Configuration({attrs})"

@dataclass
class Packet:
    """Individual packet in queue"""
    ue_id: int
    size: int  # bytes
    enqueue_time: float  # seconds
    slice_type: str
    deadline: Optional[float] = None  # For URLLC


from collections import deque
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
    

    def _service_downlink_packets(self):
        """
        Service downlink packets: UAV transmits to UEs
        - UAV serves packets from its downlink queue
        - Transmission rate depends on SINR and allocated bandwidth
        - Packets successfully delivered to UEs
        """
        
        for da in self.demand_areas.values():
            if not da.user_ids:
                continue
            
            uav = self.uavs[da.uav_id]
            
            # ============================================
            # Calculate DOWNLINK transmission rate
            # ============================================
            # Total allocated bandwidth for this DA
            allocated_bw = len(da.RB_ids_list) * self.rb_bandwidth  # Hz
            
            # Average SINR for UEs in this DA (downlink: UAV → UE)
            avg_sinr = self._get_avg_da_sinr(da, uav)
            sinr_linear = 10 ** (avg_sinr / 10)
            
            # Shannon capacity (downlink)
            total_data_rate_bps = allocated_bw * np.log2(1 + sinr_linear)
            
            # Service rate (packets/second that UAV can transmit)
            avg_packet_size = 1500 if da.slice_type == 'embb' else 100
            service_rate = total_data_rate_bps / (avg_packet_size * 8)
            
            # ============================================
            # Transmit packets from UAV to UEs
            # ============================================
            serviced = self.queuing_model.service_packets(
                da.id, 
                service_rate, 
                self.T_L, 
                self.current_time
            )
            
            # ============================================
            # Track delivery statistics
            # ============================================
            for packet, queuing_delay_ms in serviced:
                dest_ue_id = packet.ue_id  # DESTINATION UE
                
                if dest_ue_id not in self.ues:
                    continue
                
                # Track downlink queuing delays
                if not hasattr(self, 'downlink_queuing_delays'):
                    self.downlink_queuing_delays = {}
                
                if dest_ue_id not in self.downlink_queuing_delays:
                    self.downlink_queuing_delays[dest_ue_id] = []
                
                self.downlink_queuing_delays[dest_ue_id].append(queuing_delay_ms)
                
                # Keep recent history
                if len(self.downlink_queuing_delays[dest_ue_id]) > 100:
                    self.downlink_queuing_delays[dest_ue_id].pop(0)


import time
import psutil
import os
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

class PerformanceProfiler:
    """Lightweight performance profiler"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        if not enabled:
            return
            
        self.process = psutil.Process(os.getpid())
        self.timings = defaultdict(list)
        self.current_timers = {}
        self.memory_snapshots = []
        self.step_counters = defaultdict(list)
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def start_timer(self, name: str):
        if self.enabled:
            self.current_timers[name] = time.time()
    
    def end_timer(self, name: str):
        if self.enabled and name in self.current_timers:
            duration = time.time() - self.current_timers[name]
            self.timings[name].append(duration * 1000)
            del self.current_timers[name]
    
    def record_memory(self, label: str = ""):
        if not self.enabled:
            return 0
        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        self.memory_snapshots.append({
            'time': time.time() - self.start_time,
            'memory_mb': mem_mb,
            'label': label
        })
        return mem_mb
    
    def record_step(self, step: int, env, info: dict):
        if not self.enabled:
            return
            
        self.step_counters['active_ues'].append(len([ue for ue in env.ues.values() if ue.is_active]))
        self.step_counters['qos'].append(info.get('qos_satisfaction', 0))
        self.step_counters['energy'].append(info.get('energy_efficiency', 0))
        self.step_counters['fairness'].append(info.get('fairness_level', 0))
        
        if hasattr(env, 'cache_hits'):
            total = env.cache_hits + env.cache_misses
            self.step_counters['cache_hit_rate'].append(env.cache_hits / total if total > 0 else 0)
    
    def log_summary(self, step: int, interval: int = 100):
        if not self.enabled or step % interval != 0:
            return
            
        elapsed = time.time() - self.last_log_time
        print(f"\n{'='*80}")
        print(f"PROFILING - Step {step} ({elapsed:.1f}s)")
        print(f"{'='*80}")
        
        if self.memory_snapshots:
            recent = [s['memory_mb'] for s in self.memory_snapshots[-interval:]]
            print(f"📊 MEMORY: Current={recent[-1]:.1f}MB  Peak={max(recent):.1f}MB  Growth={recent[-1]-recent[0]:+.1f}MB")
        
        if self.timings:
            print(f"⏱️  TIMING:")
            for name, times in sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)[:5]:
                recent = times[-interval:]
                print(f"  {name:25s}: avg={np.mean(recent):5.1f}ms  total={sum(recent)/1000:5.1f}s")
        
        print(f"{'='*80}\n")
        self.last_log_time = time.time()
    
    def generate_report(self, save_path: str = None):
        if not self.enabled:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Performance Report', fontweight='bold')
        
        # Memory
        if self.memory_snapshots:
            times = [s['time'] for s in self.memory_snapshots]
            memory = [s['memory_mb'] for s in self.memory_snapshots]
            axes[0,0].plot(times, memory)
            axes[0,0].set_title('Memory Usage')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Memory (MB)')
            axes[0,0].grid(alpha=0.3)
        
        # Timing breakdown
        if self.timings:
            names = list(self.timings.keys())[:10]
            totals = [sum(self.timings[n])/1000 for n in names]
            axes[0,1].barh(range(len(names)), totals)
            axes[0,1].set_yticks(range(len(names)))
            axes[0,1].set_yticklabels([n[:20] for n in names])
            axes[0,1].set_title('Time Breakdown')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].grid(alpha=0.3)
        
        # QoS over time
        if 'qos' in self.step_counters:
            axes[1,0].plot(self.step_counters['qos'], label='QoS')
            if 'energy' in self.step_counters:
                axes[1,0].plot(self.step_counters['energy'], label='Energy')
            if 'fairness' in self.step_counters:
                axes[1,0].plot(self.step_counters['fairness'], label='Fairness')
            axes[1,0].set_title('Metrics Over Time')
            axes[1,0].legend()
            axes[1,0].grid(alpha=0.3)
        
        # Active UEs
        if 'active_ues' in self.step_counters:
            axes[1,1].plot(self.step_counters['active_ues'])
            axes[1,1].set_title('Active UEs')
            axes[1,1].set_xlabel('Step')
            axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Report saved: {save_path}")
        
        plt.show()
        
        # Text summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        total_time = time.time() - self.start_time
        print(f"Runtime: {total_time:.1f}s ({total_time/60:.1f}min)")
        
        if self.memory_snapshots:
            mem = [s['memory_mb'] for s in self.memory_snapshots]
            print(f"Memory: Initial={mem[0]:.1f}MB  Final={mem[-1]:.1f}MB  Peak={max(mem):.1f}MB")
        
        if self.timings:
            print("\nTop Time Consumers:")
            for name, times in sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)[:5]:
                print(f"  {name:30s}: {sum(times)/1000:6.1f}s")
        
        print(f"{'='*80}\n")