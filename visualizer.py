import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from environment import NetworkSlicingEnv, UAV, UE, DemandArea
from agents import MADRLAgent
from utils import *
import argparse
from baseline import *
import io
import pstats
import cProfile
from pstats import SortKey



class Camera3D:
    """3D Camera controller"""
    def __init__(self):
        self.distance = 2000
        self.theta = math.pi / 4
        self.phi = math.pi / 6
        self.target = np.array([500, 0.0, 500.0])
        self.position = self.calculate_position()
        
    def calculate_position(self):
        x = self.target[0] + self.distance * math.cos(self.phi) * math.cos(self.theta)
        y = self.target[1] + self.distance * math.sin(self.phi)
        z = self.target[2] + self.distance * math.cos(self.phi) * math.sin(self.theta)
        return np.array([x, y, z])
    
    def rotate(self, delta_theta, delta_phi):
        self.theta += delta_theta
        self.phi = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.phi + delta_phi))
        self.position = self.calculate_position()
    
    def zoom(self, delta):
        self.distance = max(500, min(8000, self.distance + delta))
        self.position = self.calculate_position()
    
    def pan(self, dx, dy):
        right = np.array([math.sin(self.theta), 0, -math.cos(self.theta)])
        up = np.array([0, 1, 0])
        self.target += right * dx + up * dy
        self.position = self.calculate_position()
    
    def apply(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.position[0], self.position[1], self.position[2],
            self.target[0], self.target[1], self.target[2],
            0, 1, 0
        )
    
    def get_ray_from_screen(self, screen_x, screen_y, width, height):
        """Convert screen coordinates to world ray for picking"""
        # Get matrices
        glMatrixMode(GL_MODELVIEW)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        glMatrixMode(GL_PROJECTION)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Convert screen to normalized device coordinates
        win_y = viewport[3] - screen_y  # Flip Y
        
        # Unproject near and far points
        near_point = gluUnProject(screen_x, win_y, 0.0, modelview, projection, viewport)
        far_point = gluUnProject(screen_x, win_y, 1.0, modelview, projection, viewport)
        
        # Create ray
        origin = np.array(near_point)
        direction = np.array(far_point) - origin
        direction = direction / np.linalg.norm(direction)
        
        return origin, direction


class UISlider:
    """Simple slider widget for 2D overlay"""
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
    
    def contains_point(self, px, py):
        """Check if point is inside slider"""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def update(self, mouse_x):
        """Update slider value based on mouse position"""
        relative_x = max(0, min(self.width, mouse_x - self.x))
        ratio = relative_x / self.width
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
    
    def get_normalized_value(self):
        """Get value normalized to [0, 1]"""
        return (self.value - self.min_val) / (self.max_val - self.min_val)
    
    def draw(self, font):
        """Draw slider (returns list of draw commands)"""
        commands = []
        
        # Background
        commands.append(('rect', (0.2, 0.2, 0.25, 0.8), 
                        (self.x, self.y, self.width, self.height)))
        
        # Filled portion
        fill_width = self.width * self.get_normalized_value()
        commands.append(('rect', (0.4, 0.6, 0.8, 0.8),
                        (self.x, self.y, fill_width, self.height)))
        
        # Label
        commands.append(('text', f"{self.label}: {self.value:.1f}",
                        (self.x, self.y - 20), font))
        
        return commands


class Network3DVisualizer:
    """3D Visualizer for UAV Network Slicing with enhanced controls"""
    
    def __init__(self, env: NetworkSlicingEnv, agent: Optional[MADRLAgent] = None,
                 window_width: int = 1600, window_height: int = 900, enable_profiling: bool = False):
        pygame.init()
        
        self.env = env
        self.agent = agent

        self.window_width = window_width
        self.window_height = window_height
        self.enable_profiling = enable_profiling

        
        # OpenGL setup
        self.display = pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("UAV Network 3D Visualizer")
        self.drone_model = ProceduralDroneModel()
        
        # Camera
        self.camera = Camera3D()
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [2000, 1500, 2000, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, window_width / window_height, 1, 12000)
        
        # View options
        self.show_connections = True
        self.show_das = False
        self.show_paths = True
        self.show_grid = True
        self.show_beam_cones = True  # NEW: Toggle beam cone visualization
        self.selected_uav = 0
        self.selected_ue = None  # NEW: Selected UE
        self.show_control_panel = True  # NEW: Control panel toggle
        
        # Mouse state
        self.mouse_dragging = False
        self.mouse_button = None
        self.last_mouse_pos = (0, 0)
        
        # Animation
        self.animation_time = 0
        self.clock = pygame.time.Clock()
        self.paused = False  # NEW: Pause state
        
        # UAV paths
        self.uav_paths = {uav_id: [] for uav_id in self.env.uavs.keys()}
        self.max_path_length = 100
        
        # Fonts
        self.font_small = pygame.font.SysFont("Consolas", 11)
        self.font_medium = pygame.font.SysFont("Consolas", 15)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # Performance metrics
        self.env.stats = {'reward': 0.0, 'qos': 0.0, 'energy': 0.0, 'fairness': 0.0}
        
        # NEW: UAV Control Panel
        self.control_panel_x = window_width - 280
        self.control_panel_y = 10
        self.control_panel_width = 280
        self.sliders = {}
        self.init_control_sliders()
    
    def init_control_sliders(self):
        """Initialize control sliders for UAV adjustment"""
        base_y = self.control_panel_y + 140
        slider_height = 16  # Slightly smaller
        spacing = 42  # More compact spacing
        
        self.sliders = {
            'pos_x': UISlider(self.control_panel_x + 10, base_y, 250, slider_height,
                             self.env.uav_fly_range_x[0], self.env.uav_fly_range_x[1], 
                             500, "Position X"),
            'pos_y': UISlider(self.control_panel_x + 10, base_y + spacing, 250, slider_height,
                             self.env.uav_fly_range_y[0], self.env.uav_fly_range_y[1],
                             500, "Position Y"),
            'pos_z': UISlider(self.control_panel_x + 10, base_y + spacing*2, 250, slider_height,
                             self.env.uav_fly_range_h[0], self.env.uav_fly_range_h[1],
                             50, "Height Z"),
            'power': UISlider(self.control_panel_x + 10, base_y + spacing*3, 250, slider_height,
                             0.0, self.env.uav_params.max_power, 
                             self.env.uav_params.max_power/2, "Power (W)"),
            'battery': UISlider(self.control_panel_x + 10, base_y + spacing*4, 250, slider_height,
                               0.0, self.env.uav_params.battery_capacity,
                               self.env.uav_params.battery_capacity, "Battery (J)"),
            'beam_angle': UISlider(self.control_panel_x + 10, base_y + spacing*5, 250, slider_height,
                                   10.0, 90.0, self.env.uav_beam_angle, "Beam Angle (°)")
        }
        
        # Add bandwidth allocation sliders for each demand area (9 DAs)
        bw_base_y = base_y + spacing * 6 + 40  # After basic sliders
        slice_types = ["embb", "urllc", "mmtc"]
        distance_levels = ["Near", "Medium", "Far"]
        
        for i, slice_type in enumerate(slice_types):
            for j, distance_level in enumerate(distance_levels):
                da_idx = i * 3 + j
                label = f"{slice_type.upper()}-{distance_level}"  # Shorter labels
                self.sliders[f'bw_{da_idx}'] = UISlider(
                    self.control_panel_x + 10, 
                    bw_base_y + da_idx * spacing,
                    250, 
                    slider_height,
                    0.0, 
                    self.env.total_bandwidth,
                    self.env.total_bandwidth / 9,  # Equal initial distribution
                    label
                )
    
    def update_sliders_from_uav(self):
        """Update slider values from selected UAV"""
        if self.selected_uav is not None and self.selected_uav in self.env.uavs:
            uav = self.env.uavs[self.selected_uav]
            self.sliders['pos_x'].value = uav.position[0]
            self.sliders['pos_y'].value = uav.position[1]
            self.sliders['pos_z'].value = uav.position[2]
            self.sliders['power'].value = uav.current_power
            self.sliders['battery'].value = uav.current_battery
            self.sliders['beam_angle'].value = uav.beam_angle  # NEW
        
            uav_das = sorted([da for da in self.env.demand_areas.values() if da.uav_id == self.selected_uav],
                           key=lambda da: da.id)
            
            for idx, da in enumerate(uav_das[:9]):  # Should be exactly 9 DAs
                slider_key = f'bw_{idx}'
                if slider_key in self.sliders:
                    self.sliders[slider_key].value = da.allocated_bandwidth
    
    def apply_sliders_to_uav(self):
        """Apply slider values to selected UAV and normalize bandwidth"""

        if self.selected_uav is not None and self.selected_uav in self.env.uavs:
            uav = self.env.uavs[self.selected_uav]
            
            # Apply position, power, battery
            uav.position[0] = self.sliders['pos_x'].value
            uav.position[1] = self.sliders['pos_y'].value
            uav.position[2] = self.sliders['pos_z'].value
            uav.current_power = self.sliders['power'].value
            uav.current_battery = self.sliders['battery'].value
            uav.beam_angle = self.sliders['beam_angle'].value  # NEW

            # Reassociate UEs based on new position
            self.env._associate_ues_to_uavs()
            
            # Apply bandwidth allocation
            # Get bandwidth values from sliders
            bw_values = []
            for i in range(9):
                slider_key = f'bw_{i}'
                if slider_key in self.sliders:
                    bw_values.append(self.sliders[slider_key].value)
            
            # Normalize to total available bandwidth
            total_requested = sum(bw_values)
            if total_requested > 0:
                scale_factor = uav.max_bandwidth / total_requested
                bw_values = [bw * scale_factor for bw in bw_values]
            
            # Apply to demand areas
            uav_das = sorted([da for da in self.env.demand_areas.values() if da.uav_id == self.selected_uav],
                           key=lambda da: da.id)
            
            for idx, da in enumerate(uav_das[:9]):
                if idx < len(bw_values):
                    da.allocated_bandwidth = bw_values[idx]
            
            # Re-allocate resource blocks based on new bandwidth
            self.env._allocate_rbs_fairly()

    def normalize_bandwidth_sliders(self):
        """Normalize bandwidth sliders to sum to max bandwidth"""
        if self.selected_uav is not None and self.selected_uav in self.env.uavs:
            uav = self.env.uavs[self.selected_uav]
            
            # Get current values
            bw_values = []
            for i in range(9):
                slider_key = f'bw_{i}'
                if slider_key in self.sliders:
                    bw_values.append(self.sliders[slider_key].value)
            
            # Normalize
            total = sum(bw_values)
            if total > 0:
                scale_factor = uav.max_bandwidth / total
                for i in range(9):
                    slider_key = f'bw_{i}'
                    if slider_key in self.sliders:
                        self.sliders[slider_key].value = bw_values[i] * scale_factor
            else:
                # If all zero, distribute equally
                equal_share = uav.max_bandwidth / 9
                for i in range(9):
                    slider_key = f'bw_{i}'
                    if slider_key in self.sliders:
                        self.sliders[slider_key].value = equal_share

        # self.apply_sliders_to_uav()
    
    def find_clicked_ue(self, mouse_x, mouse_y):
        """Find UE closest to mouse click using ray casting"""
        origin, direction = self.camera.get_ray_from_screen(
            mouse_x, mouse_y, self.window_width, self.window_height
        )
        
        min_distance = float('inf')
        closest_ue = None
        
        for ue_id, ue in self.env.ues.items():
            if not ue.is_active:
                continue
            
            # UE position in world space (OpenGL coords: x, z, y swapped)
            ue_pos = np.array([ue.position[0], 5, ue.position[1]])
            
            # Distance from ray to point
            # d = ||(point - origin) - ((point - origin) · direction) * direction||
            to_point = ue_pos - origin
            projection_length = np.dot(to_point, direction)
            
            if projection_length < 0:  # Behind camera
                continue
            
            closest_point_on_ray = origin + projection_length * direction
            distance = np.linalg.norm(ue_pos - closest_point_on_ray)
            
            # Threshold based on camera distance (adaptive)
            threshold = 20 + self.camera.distance * 0.01
            
            if distance < threshold and distance < min_distance:
                min_distance = distance
                closest_ue = ue_id
        
        return closest_ue
    
    def draw_ground(self):
        """Draw ground plane"""
        glDisable(GL_LIGHTING)
        
        # Area of interest
        glColor4f(0.1, 0.18, 0.18, 0.7)
        glBegin(GL_QUADS)
        glVertex3f(0, -1, 0)
        glVertex3f(self.env.service_area[1], -1, 0)
        glVertex3f(self.env.service_area[0], -1, self.env.service_area[1])
        glVertex3f(0, -1, self.env.service_area[1])
        glEnd()

        # UAV fly area
        glColor4f(0.1, 0.18, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(self.env.uav_fly_range_x[0], 0, self.env.uav_fly_range_y[0])
        glVertex3f(self.env.uav_fly_range_x[1], 0, self.env.uav_fly_range_y[0])
        glVertex3f(self.env.uav_fly_range_x[1], 0, self.env.uav_fly_range_y[1])
        glVertex3f(self.env.uav_fly_range_x[0], 0, self.env.uav_fly_range_y[1])
        glEnd()
        
        # Axes
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_cylinder(self, base_radius, top_radius, height, slices=16):
        """Draw a cylinder"""
        quad = gluNewQuadric()
        gluCylinder(quad, base_radius, top_radius, height, slices, 1)
        gluDeleteQuadric(quad)
    
    def draw_cone(self, base_radius, height, slices=16):
        """Draw a cone"""
        self.draw_cylinder(base_radius, 0, height, slices)

    def draw_ue_beam_status(self, ue_id: int):
        """
        Visually indicate if UE is within beam coverage.
        Draws a colored ring around UE.
        
        Args:
            ue_id: UE ID
        """
        if ue_id not in self.env.ues:
            return
        
        ue = self.env.ues[ue_id]
        if not ue.is_active:
            return
        
        beam_status = self.env.get_ue_beam_status(ue_id)
        if beam_status is None:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        # Color code based on coverage status
        if beam_status['covered']:
            # Green for covered
            margin = beam_status['margin']
            # Gradient from yellow (near edge) to green (well inside)
            green_intensity = min(margin / 10.0, 1.0)  # 10 degrees margin = full green
            glColor4f(1.0 - green_intensity, 1.0, 0.0, 0.8)
        else:
            # Red for not covered
            glColor4f(1.0, 0.2, 0.2, 0.8)
        
        # Draw ring around UE
        num_segments = 16
        ring_radius = 5
        
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = (i / num_segments) * 2 * np.pi
            x = ue.position[0] + ring_radius * np.cos(angle)
            z = ue.position[1] + ring_radius * np.sin(angle)
            glVertex3f(x, 6, z)  # Slightly above UE
        glEnd()
        
        glEnable(GL_LIGHTING)

    def draw_uav_beam_cone(self, uav_id: int):
        """
        Draw the UAV's beam coverage cone with wireframe style for better visibility.
        
        Args:
            uav_id: UAV ID
        """
        if not self.show_beam_cones or uav_id not in self.env.uavs:
            return
        
        beam_info = self.env.get_uav_beam_info(uav_id)
        if beam_info is None:
            return
        
        uav_pos = beam_info['position']
        beam_angle = beam_info['beam_angle']
        height = beam_info['height']
        beam_radius = beam_info['beam_radius_ground']
        
        if height <= 0:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Disable depth writes so UEs always show through
        glDepthMask(GL_FALSE)
        
        num_segments = 32
        
        # Highlight selected UAV's cone
        if uav_id == self.selected_uav:
            edge_color = (1.0, 0.85, 0.4, 0.7)  # Golden edges
            fill_color = (1.0, 0.8, 0.4, 0.1)  # Very light golden fill
        else:
            edge_color = (0.4, 0.7, 1.0, 0.5)  # Blue edges
            fill_color = (0.3, 0.6, 1.0, 0.1)  # Very light blue fill

        # Cone surface
        glColor4f(*fill_color)
        glBegin(GL_TRIANGLE_FAN)
        # Apex (UAV position)
        glVertex3f(uav_pos[0], uav_pos[2], uav_pos[1])
        
        # Base circle on ground
        for i in range(num_segments + 1):
            angle = (i / num_segments) * 2 * np.pi
            x = uav_pos[0] + beam_radius * np.cos(angle)
            z = uav_pos[1] + beam_radius * np.sin(angle)
            glVertex3f(x, 0, z)  # Ground level (y=0)
        glEnd()
        
        # Draw ONLY the base circle on ground (most important reference)
        glColor4f(*edge_color)
        glLineWidth(2.5)
        glBegin(GL_LINE_LOOP)
        for i in range(num_segments):
            angle = (i / num_segments) * 2 * np.pi
            x = uav_pos[0] + beam_radius * np.cos(angle)
            z = uav_pos[1] + beam_radius * np.sin(angle)
            glVertex3f(x, 0, z)
        glEnd()
        

        
        # Re-enable depth writes
        glDepthMask(GL_TRUE)
        
        glEnable(GL_LIGHTING)
    
    def draw_uav_enhanced(self, uav):
        """
        Enhanced UAV rendering with 3D drone model.
        Replace your existing draw_uav() method with this.
        """
        glPushMatrix()
        
        # Position the drone
        glTranslatef(uav.position[0], uav.position[2], uav.position[1])
        
        # Orientation (rotate to face movement direction)
        # Optional: calculate heading from velocity
        glRotatef(self.animation_time * 20 + uav.id * 45, 0, 1, 0)
        
        # Draw the drone model
        is_selected = (uav.id == self.selected_uav)
        self.drone_model.draw(self.animation_time, uav.id, is_selected)
        
        # Draw ground connection line
        glDisable(GL_LIGHTING)
        if is_selected:
            glColor4f(1.0, 0.8, 0.4, 0.4)
        else:
            glColor4f(1.0, 0.4, 0.4, 0.3)
        glLineWidth(1)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, -uav.position[2], 0)
        glEnd()
        glEnable(GL_LIGHTING)
        
        glPopMatrix()

    def draw_ues(self):
        """Draw user equipment with beam coverage indication"""
        ue_info = self.env.get_UEs_throughput_demand_and_satisfaction()
        
        for ue_id, (throughput, demand, satisfaction) in ue_info.items():
            ue = self.env.ues[ue_id]
            
            # NEW: Draw beam coverage status ring
            self.draw_ue_beam_status(ue_id)
            
            glPushMatrix()
            glTranslatef(ue.position[0], 5, ue.position[1])
            
            # Highlight selected UE
            if ue_id == self.selected_ue:
                glColor3f(1.0, 1.0, 0.0)  # Yellow highlight
            elif ue.slice_type == "embb":
                glColor3f(0.4, 0.6, 1.0)
            elif ue.slice_type == "urllc":
                glColor3f(0.4, 1.0, 0.6)
            else:
                glColor3f(1.0, 0.7, 0.4)
            
            glRotatef(-90, 1, 0, 0)
            self.draw_cone(3, 8, 8)
            
            glPopMatrix()
            
            # # Draw connection (existing code)
            # if self.show_connections and ue.assigned_uav in self.env.uavs:
            #     uav = self.env.uavs[ue.assigned_uav]
                
            #     glDisable(GL_LIGHTING)
            #     glLineWidth(2)
                
            #     quality = satisfaction
            #     glColor4f(1.0 - quality, quality, 0.2, 0.6)
                
            #     glBegin(GL_LINES)
            #     glVertex3f(ue.position[0], 5, ue.position[1])
            #     glVertex3f(uav.position[0], uav.position[2], uav.position[1])
            #     glEnd()
                
            #     glEnable(GL_LIGHTING)
    
    def draw_uav_path(self, uav_id: int):
        """Draw UAV path"""
        if not self.show_paths or uav_id not in self.uav_paths:
            return
        
        path = self.uav_paths[uav_id]
        if len(path) < 2:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(3)
        
        for i in range(len(path) - 1):
            alpha = (i + 1) / len(path)
            if uav_id == self.selected_uav:
                glColor4f(1.0, 0.8, 0.4, alpha * 0.8)
            else:
                glColor4f(1.0, 0.4, 0.4, alpha * 0.5)
            
            glBegin(GL_LINES)
            glVertex3f(path[i][0], path[i][2], path[i][1])
            glVertex3f(path[i+1][0], path[i+1][2], path[i+1][1])
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_demand_area(self, da: DemandArea):
        """Draw demand area (unchanged from original)"""
        if not self.show_das or da.center_position is None:
            return
        
        ue_positions = []
        for ue_id in da.user_ids:
            if ue_id in self.env.ues:
                ue = self.env.ues[ue_id]
                ue_positions.append((ue.position[0], ue.position[1]))
        
        if len(ue_positions) < 2:
            return
        
        base_height = 20
        height_increment = 10
        height = base_height + da.uav_id * 100 + da.id % 9 * height_increment
        
        glDisable(GL_LIGHTING)
        
        if da.slice_type == "embb":
            line_color = (0.4, 0.6, 1.0, 0.6)
        elif da.slice_type == "urllc":
            line_color = (0.4, 1.0, 0.6, 0.6)
        else:
            line_color = (1.0, 0.7, 0.4, 0.6)
        
        if len(ue_positions) >= 3:
            hull = self.compute_convex_hull_2d(ue_positions)
            
            if len(hull) >= 3:
                glLineWidth(3)
                glColor4f(*line_color)
                glBegin(GL_LINE_LOOP)
                for pos in hull:
                    glVertex3f(pos[0], height, pos[1])
                glEnd()
                
                glColor4f(line_color[0], line_color[1], line_color[2], 0.3)
                glLineWidth(2)
                for pos in hull:
                    glBegin(GL_LINES)
                    glVertex3f(pos[0], 0, pos[1])
                    glVertex3f(pos[0], height, pos[1])
                    glEnd()
                
                glColor4f(line_color[0], line_color[1], line_color[2], 0.2)
                glBegin(GL_POLYGON)
                for pos in hull:
                    glVertex3f(pos[0], height, pos[1])
                glEnd()
        
        glEnable(GL_LIGHTING)
    
    def compute_convex_hull_2d(self, points):
        """Compute 2D convex hull"""
        if len(points) < 3:
            return points
        
        points = list(set(points))
        if len(points) < 3:
            return points
        
        points = sorted(points)
        
        lower = []
        for p in points:
            while len(lower) >= 2 and self.cross_product_2d(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and self.cross_product_2d(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return lower[:-1] + upper[:-1]
    
    def cross_product_2d(self, o, a, b):
        """2D cross product"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def draw_3d_scene(self):
        """Draw the 3D scene with proper depth ordering"""


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.camera.apply()
        
        # Draw in back-to-front order for proper transparency
        
        # 1. Ground (furthest back)
        self.draw_ground()
        
        # 2. Beam cones (transparent, drawn first so everything else is on top)
        if self.show_beam_cones:
            for uav_id in self.env.uavs.keys():
                self.draw_uav_beam_cone(uav_id)
        
        # 3. UAV paths
        for uav_id in self.env.uavs.keys():
            self.draw_uav_path(uav_id)
        
        # 4. Demand areas
        # for da in self.env.demand_areas.values():
        #     self.draw_demand_area(da)
        
        # 5. Connections (semi-transparent lines)
        if self.show_connections:
            self.draw_all_connections()
        
        # 6. UEs (solid objects, high priority)
        self.draw_ues()
        
        # 7. UAVs (drawn last so they're always on top)
        for uav in self.env.uavs.values():
            self.draw_uav_enhanced(uav)


    def draw_all_connections(self):
        """Draw all UE-UAV connections separately for better control"""
        ue_info = self.env.get_UEs_throughput_demand_and_satisfaction()
        
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for ue_id, (throughput, demand, satisfaction) in ue_info.items():
            ue = self.env.ues[ue_id]
            
            if ue.assigned_uav in self.env.uavs:
                uav = self.env.uavs[ue.assigned_uav]
                
                quality = satisfaction
                glColor4f(1.0 - quality, quality, 0.2, 1)
                
                glBegin(GL_LINES)
                glVertex3f(ue.position[0], 5, ue.position[1])
                glVertex3f(uav.position[0], uav.position[2], uav.position[1])
                glEnd()
        
        glEnable(GL_LIGHTING)

    def draw_2d_overlay(self):
        """Draw 2D UI overlay with all information"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_width, self.window_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # print(self.env.stats)
        
        # Main info panel
        info_lines = [
            f"Overall Statistics: {'[PAUSED]' if self.paused else ''}",
            f"Time: {self.env.current_time:.1f}s | Step: {(self.env.current_time / self.env.T_L):.0f}",
            f"       Total Reward : {self.env.stats['reward']:.2f}",
            f"         QoS Reward : {self.env.stats['qos']:.2f}",
            f"     Energy Penalty : {self.env.stats['energy']:.2f}",
            f"    Fairness Reward : {self.env.stats['fairness']:.2f}",
            f" Avg Throughput Sat : {self.env.stats.get('avg_throughput_sat', 0):.2f}",
            f"      Avg Delay Sat : {self.env.stats.get('avg_delay_sat', np.inf):.2f}",
            f"Avg Reliability Sat : {self.env.stats.get('avg_reliability_sat', 0):.2f}",
            f"           Handover : {self.env.stats.get('handovers', 0)} ",
            f"               UAVs : {len(self.env.uavs):<3}",
            f"                UEs : {sum(1 for ue in self.env.ues.values() if ue.is_active)}",
            f"      Uncovered UEs : {self.env.stats.get('uncovered_ues', 0):<3}",
            f"        Covered UEs : {self.env.stats.get('covered_ues', 0)}",
            "",
            # "Controls:",
            # "P: Pause | Space: Step | R: Reset",
            # "Left Click: Select UE | Tab: Cycle UAV",
            # "U: UAV Panel | N: Normalize BW | Enter: Apply",
            # "C/D/B: Toggle Views | Mouse: Rotate/Pan/Zoom",
            # "ESC: Exit"
        ]

         # Add profiling info if enabled
        
        # Draw main info background
        glColor4f(0.1, 0.1, 0.15, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(350, 10)
        glVertex2f(350, len(info_lines) * 20)
        glVertex2f(10, len(info_lines) * 20)
        glEnd()
        
        # Render main info text
        y_offset = 20
        for line in info_lines:
            if line == info_lines[0]:
                text_surface = self.font_medium.render(line, True, (200, 200, 200))
                y_offset += 10
            else:
                text_surface = self.font_small.render(line, True, (180, 180, 180))
            
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(20, y_offset)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            y_offset += text_surface.get_height() + 5
        
        # Add UAV info to main panel if UAV is selected
        if self.selected_uav is not None and self.selected_uav in self.env.uavs:
            uav = self.env.uavs[self.selected_uav]
            uav_lines = [
                f"Selected UAV {self.selected_uav}:",
                f"Position: ({uav.position[0]:.0f}, {uav.position[1]:.0f}, {uav.position[2]:.0f})",
                f"Power: {uav.current_power:.2f}W / {uav.max_power:.2f}W",
                f"Battery: {uav.current_battery:.0f} / {uav.battery_capacity:.0f} J",
                f"Energy Used:", 
                f"   - Movement: {uav.energy_used.get('movement', 'N/A'):.2f} J",
                f"   - Transmission: {uav.energy_used.get('transmission', 'N/A'):.2f} J",
                f"Resource Blocks: {len([rb for rb in uav.RBs if rb.allocated_ue_id != -1])} / {len(uav.RBs)} allocated"
            ]

        # Draw main info background
        glColor4f(0.1, 0.15, 0.3, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(10, 320)
        glVertex2f(350, 320)
        glVertex2f(350, 320 + len(uav_lines) * 20)
        glVertex2f(10, 320 + len(uav_lines) * 20)
        glEnd()
        
        # Render main info text
        y_offset = 340
        for line in uav_lines:
            if line == uav_lines[0]:
                text_surface = self.font_medium.render(line, True, (200, 200, 240))
                y_offset += 10
            else:
                text_surface = self.font_small.render(line, True, (200, 200, 240))
            
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(20, y_offset)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            y_offset += text_surface.get_height() + 5
        

        # Draw info panel for queueing stats of all DAs
        da_lines = [f"Demand Area Queue Data:"]
        for uav in self.env.uavs.values():
            da_lines.append("-"*120)  # Blank line between UAVs
            for da in [da for da in self.env.demand_areas.values() if da.uav_id == uav.id]:
                
                avg_da_delay_satisfaction = np.mean([self.env.ues[ue.id].delay_satisfaction for ue in self.env.ues.values() if ue.assigned_da == da.id])
                avg_da_reliability = np.mean([self.env.ues[ue.id].reliability for ue in self.env.ues.values() if ue.assigned_da == da.id])
                avg_throughput = np.mean([self.env.ues[ue.id].throughput for ue in self.env.ues.values() if ue.assigned_da == da.id])
                avg_throughput_satisfaction = np.mean([self.env.ues[ue.id].throughput_satisfaction for ue in self.env.ues.values() if ue.assigned_da == da.id])

                da_lines.append(f"UAV {da.uav_id} {da.slice_type:<5} {da.distance_level:<6}: Num UEs: {len(da.user_ids):<3} | Avg Delay Sat: {avg_da_delay_satisfaction:<5.2f} |  Avg Reliability: {avg_da_reliability:<5.2f} | Avg Throughput: {avg_throughput / 1e6:<8.2f} Mbps | Avg TP Sat: {avg_throughput_satisfaction:<5.2f}")

        # Render DA text
        if self.show_das:
            # Draw DA info background
            panel_y = 10
            glColor4f(0.1, 0.15, 0.25, 0.85)
            glBegin(GL_QUADS)
            glVertex2f(370, panel_y)
            glVertex2f(1300, panel_y)
            glVertex2f(1300, panel_y + len(da_lines) * 14 + 20)
            glVertex2f(370, panel_y + len(da_lines) * 14 + 20)
            glEnd()

            y_offset = panel_y + 10
            for line in da_lines:            
                if line == da_lines[0]:
                    text_surface = self.font_medium.render(line, True, (200, 220, 240))
                    y_offset += 10
                else:
                    text_surface = self.font_small.render(line, True, (180, 220, 240))
                text_data = pygame.image.tostring(text_surface, "RGBA", True)
                glRasterPos2f(380, y_offset)
                glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                            GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                y_offset += text_surface.get_height() + 2
        
        # Selected UE info panel
        if self.selected_ue is not None and self.selected_ue in self.env.ues:
            ue = self.env.ues[self.selected_ue]
            beam_status = self.env.get_ue_beam_status(self.selected_ue)
                
            ue_lines = [
                f"Selected UE {self.selected_ue}:",
                f"Type: {ue.slice_type.upper()} {self.env.demand_areas[ue.assigned_da].distance_level}" if ue.assigned_uav is not None else "Type: N/A",
                f"Position: ({ue.position[0]:.0f}, {ue.position[1]:.0f})",
                f"Velocity: ({ue.velocity[0]:.1f}, {ue.velocity[1]:.1f}) m/s",
                f"Assigned UAV: {ue.assigned_uav}",
                f"Angle to UAV: {beam_status['angle']:.1f}° / {beam_status['max_angle']:.1f}°" if beam_status['covered'] else "Angle to UAV: N/A",
                f"Beam Coverage: {'YES' if beam_status['covered'] else 'NO'}",
                f"RBs Allocated: {len(ue.assigned_rb) if ue.assigned_rb else 0}",
                f"Throughput: {ue.throughput/1e6:.2f} / {self.env.qos_profiles[ue.slice_type].min_rate/1e6:.2f} Mbps",
                f"Reliability: {ue.reliability:.2%} / {self.env.qos_profiles[ue.slice_type].min_reliability:.2%}",
                f"Delay: {self.env.delay_cache.get(ue.id, {}).get('total', np.inf):.2f} / {self.env.qos_profiles[ue.slice_type].max_latency:.2f} ms"
            ]
            ue_lines.extend([f"  - {key:<15} : {value:.3f} ms" for key, value in self.env.delay_cache.get(ue.id, {}).get("breakdown", {}).items()])

            # print(self.env.delay_cache.get(ue.id, 'N/A'))

            # Draw UE info background
            panel_y = 600
            glColor4f(0.1, 0.15, 0.1, 0.85)
            glBegin(GL_QUADS)
            glVertex2f(10, panel_y)
            glVertex2f(350, panel_y)
            glVertex2f(350, panel_y + len(ue_lines) * 15 + 20)
            glVertex2f(10, panel_y + len(ue_lines) * 15 + 20)
            glEnd()
            
            # Render UE text
            y_offset = panel_y + 10
            for line in ue_lines:
                if line == ue_lines[0]:
                    text_surface = self.font_medium.render(line, True, (180, 220, 180))
                    y_offset += 10
                else:
                    text_surface = self.font_small.render(line, True, (180, 220, 180))
                text_data = pygame.image.tostring(text_surface, "RGBA", True)
                glRasterPos2f(20, y_offset)
                glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                            GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                y_offset += text_surface.get_height() + 2
        
        # UAV Control Panel
        if self.show_control_panel and self.selected_uav is not None:
            panel_x = self.control_panel_x
            panel_y = self.control_panel_y
            panel_w = self.control_panel_width
            panel_h = 800  # Increased to fit all 9 bandwidth sliders + 5 basic sliders
            
            # Draw control panel background
            glColor4f(0.15, 0.1, 0.15, 0.9)
            glBegin(GL_QUADS)
            glVertex2f(panel_x, panel_y)
            glVertex2f(panel_x + panel_w, panel_y)
            glVertex2f(panel_x + panel_w, panel_y + panel_h)
            glVertex2f(panel_x, panel_y + panel_h)
            glEnd()
            
            # Title
            title = f"UAV {self.selected_uav} Control Panel"
            text_surface = self.font_medium.render(title, True, (220, 200, 220))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(panel_x + 10, panel_y + 20)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Instructions
            instr = "N: Normalize BW | Enter: Apply"
            text_surface = self.font_small.render(instr, True, (180, 180, 180))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(panel_x + 10, panel_y + 45)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Calculate total bandwidth allocation
            total_bw = sum(self.sliders[f'bw_{i}'].value for i in range(9) if f'bw_{i}' in self.sliders)
            max_bw = self.env.total_bandwidth
            bw_usage_pct = (total_bw / max_bw) * 100 if max_bw > 0 else 0
            
            # Show bandwidth usage
            bw_color = (0.4, 0.8, 0.4) if bw_usage_pct <= 100.001 else (0.8, 0.4, 0.4)
            bw_text = f"Total BW: {total_bw/1e6:.1f}/{max_bw/1e6:.1f} MHz ({bw_usage_pct:.0f}%)"
            text_surface = self.font_small.render(bw_text, True, tuple(int(c*255) for c in bw_color))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(panel_x + 10, panel_y + 65)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Section dividers
            # # Draw "Basic Controls" header
            # basic_header = "--- UAV Controls ---"
            # text_surface = self.font_small.render(basic_header, True, (150, 150, 180))
            # text_data = pygame.image.tostring(text_surface, "RGBA", True)
            # glRasterPos2f(panel_x + 70, panel_y + 85)
            # glDrawPixels(text_surface.get_width(), text_surface.get_height(),
            #             GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            

            
            # Draw sliders
            for slider_name, slider in self.sliders.items():
                # # Skip if slider is outside visible area
                # if slider.y > panel_y + panel_h - 25:
                #     continue
                
                # Slider background
                glColor4f(0.2, 0.2, 0.25, 0.8)
                glBegin(GL_QUADS)
                glVertex2f(slider.x, slider.y)
                glVertex2f(slider.x + slider.width, slider.y)
                glVertex2f(slider.x + slider.width, slider.y + slider.height)
                glVertex2f(slider.x, slider.y + slider.height)
                glEnd()
                
                # Filled portion
                fill_width = slider.width * slider.get_normalized_value()
                
                # Color code bandwidth sliders by slice type
                if slider_name.startswith('bw_'):
                    idx = int(slider_name.split('_')[1])
                    if idx < 3:  # eMBB (0-2)
                        glColor4f(0.4, 0.6, 1.0, 0.8)
                    elif idx < 6:  # URLLC (3-5)
                        glColor4f(0.4, 1.0, 0.6, 0.8)
                    else:  # mMTC (6-8)
                        glColor4f(1.0, 0.7, 0.4, 0.8)
                else:
                    glColor4f(0.4, 0.6, 0.8, 0.8)
                
                glBegin(GL_QUADS)
                glVertex2f(slider.x, slider.y)
                glVertex2f(slider.x + fill_width, slider.y)
                glVertex2f(slider.x + fill_width, slider.y + slider.height)
                glVertex2f(slider.x, slider.y + slider.height)
                glEnd()
                
                # Label with value
                if slider_name.startswith('bw_'):
                    label_text = f"{slider.label}: {slider.value/1e6:.1f} MHz"
                else:
                    label_text = f"{slider.label}: {slider.value:.1f}"
                
                text_surface = self.font_small.render(label_text, True, (200, 200, 200))
                text_data = pygame.image.tostring(text_surface, "RGBA", True)
                glRasterPos2f(slider.x, slider.y - 4)
                glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                            GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore 3D
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def update_uav_paths(self):
        """Update UAV paths"""
        for uav_id, uav in self.env.uavs.items():
            current_pos = tuple(uav.position)
            
            if not self.uav_paths[uav_id] or \
               np.linalg.norm(np.array(current_pos) - np.array(self.uav_paths[uav_id][-1])) > 5.0:
                self.uav_paths[uav_id].append(current_pos)
                
                if len(self.uav_paths[uav_id]) > self.max_path_length:
                    self.uav_paths[uav_id].pop(0)
    
    def step_simulation(self):
        """Execute one step with profiling"""
        
        if self.agent:
            observation = self.env._get_observations()
            actions = self.agent.select_actions(observation, explore=False)

        self.env.step(actions)
        self.update_uav_paths()
    
    def reset_simulation(self):
        """Reset with profiling"""
        self.env.reset()
        self.uav_paths = {uav_id: [] for uav_id in self.env.uavs.keys()}
        self.animation_time = 0.0
        
    def handle_mouse(self):
        """Handle mouse input"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        
        # Check if clicking on control panel sliders
        if self.show_control_panel and mouse_buttons[0]:
            for slider in self.sliders.values():
                if slider.contains_point(mouse_pos[0], mouse_pos[1]):
                    slider.dragging = True
                    slider.update(mouse_pos[0])
                    self.normalize_bandwidth_sliders()
                    self.apply_sliders_to_uav()
                    return
        
        if not any(mouse_buttons):
            for slider in self.sliders.values():
                slider.dragging = False
        
        if mouse_buttons[0]:  # Left button - rotate
            if not self.mouse_dragging or self.mouse_button != 0:
                self.last_mouse_pos = mouse_pos
                self.mouse_dragging = True
                self.mouse_button = 0
            else:
                dx = (mouse_pos[0] - self.last_mouse_pos[0]) * 0.01
                dy = (mouse_pos[1] - self.last_mouse_pos[1]) * 0.01
                self.camera.rotate(dx, dy)
                self.last_mouse_pos = mouse_pos
        elif mouse_buttons[2]:  # Right button - pan
            if not self.mouse_dragging or self.mouse_button != 2:
                self.last_mouse_pos = mouse_pos
                self.mouse_dragging = True
                self.mouse_button = 2
            else:
                dx = (mouse_pos[0] - self.last_mouse_pos[0]) * 0.5
                dy = (mouse_pos[1] - self.last_mouse_pos[1]) * 0.5
                self.camera.pan(-dx, dy)
                self.last_mouse_pos = mouse_pos
        else:
            self.mouse_dragging = False
    
    def run(self):
        """Main loop to run the visualizer"""

        profiler = None
        if self.enable_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        running = True
        
        while running:

            # Step simulation if not paused
            if not self.paused:
                self.step_simulation()
                self.update_sliders_from_uav()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:  # NEW: Pause toggle
                        self.paused = not self.paused
                    elif event.key == pygame.K_SPACE:
                        self.step_simulation()  # Single step when paused
                        self.update_sliders_from_uav()
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_b:  # NEW: Toggle beam cones
                        self.show_beam_cones = not self.show_beam_cones
                    elif event.key == pygame.K_c:
                        self.show_connections = not self.show_connections
                    elif event.key == pygame.K_d:
                        self.show_das = not self.show_das
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_u:  # NEW: Toggle control panel
                        self.show_control_panel = not self.show_control_panel
                        if self.show_control_panel and self.selected_uav is not None:
                            self.update_sliders_from_uav()
                    elif event.key == pygame.K_RETURN:  # NEW: Apply slider values
                        if self.show_control_panel:
                            self.apply_sliders_to_uav()
                    elif event.key == pygame.K_TAB:
                        if self.selected_uav is None:
                            self.selected_uav = 0
                        else:
                            self.selected_uav = (self.selected_uav + 1) % len(self.env.uavs)
                        if self.show_control_panel:
                            self.update_sliders_from_uav()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click - NEW: Select UE
                        clicked_ue = self.find_clicked_ue(event.pos[0], event.pos[1])
                        if clicked_ue is not None:
                            self.selected_ue = clicked_ue
                    elif event.button == 4:  # Scroll up
                        self.camera.zoom(-50)
                    elif event.button == 5:  # Scroll down
                        self.camera.zoom(50)
            
            self.handle_mouse()
            
            # Draw
            self.draw_3d_scene()
            self.draw_2d_overlay()
            pygame.display.flip()
            
            # Update animation
            if not self.paused:
                self.animation_time += 0.016
            
            self.clock.tick(60)

        if self.enable_profiling and profiler is not None:
            profiler.disable()
        
            # Print profiling results
            print("\n" + "="*80)
            print("PROFILING RESULTS - TOP 30 FUNCTIONS BY CUMULATIVE TIME")
            print("="*80)
            
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.strip_dirs()
            
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats('environment.py|agents.py|baseline.py|utils.py', 30)
            
            print(s.getvalue())
        pygame.quit()


def profile_training_step():
    """Profile a single training step to identify bottlenecks"""
    
    # Initialize environment and agent
    env = NetworkSlicingEnv(config_path="config/environment/default.yaml")
    obs = env._get_observations()
    
    agent = MADRLAgent(
        num_agents=env.num_uavs,
        obs_dim=len(list(obs.values())[0]),
        action_dim=4 + env.num_das_per_slice * 3
    )
    
    # Warm up (fill buffer)
    print("Warming up buffer...")
    progress_bar = tqdm(range(1000), desc="Warming Up Buffer")
    for _ in progress_bar:
        actions = agent.select_actions(obs, explore=True)
        next_obs, reward, done, info = env.step(actions)
        agent.store_transition(obs, actions, actions, reward, next_obs, done)
        obs = next_obs
    
    # Profile the actual training loop
    print("Starting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    

    progress_bar = tqdm(range(2000), desc="Training Steps")
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
    parser = argparse.ArgumentParser(description='Enhanced UAV Network 3D Visualizer')
    parser.add_argument('--checkpoint', type=str, default="./saved_models/model29",
                       help='Path to trained model checkpoint')
    
    # ============================================
    # PROFILING ARGUMENTS 
    # ============================================
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    
    args = parser.parse_args()
    
    # Create environment
    env = NetworkSlicingEnv(config_path="./config/environment/default.yaml")
    
    # Load agent
    agent = None
    if args.checkpoint:
        model_dir = args.checkpoint
        model_checkpoint = "./commit_models/model1/checkpoints/checkpoint_step_400000.pth"
        env_config = Configuration("./config/environment/default.yaml")
        num_agents = env_config.system.num_uavs
        obs_dim = 80
        action_dim = 13
        agent = MADRLAgent(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim, training=False)
        agent.load_models(model_checkpoint)
        print(f"✓ Loaded model from {model_checkpoint}")

    # agent = DynamicHeightGreedyAgent(len(env.uavs), 80, 13, env)
    
    # Create visualizer with profiling
    visualizer = Network3DVisualizer(
        env, 
        agent,
        enable_profiling=args.profile,
    )
    
    # Run
    visualizer.run()
    
    print("\n✓ Visualizer closed gracefully")

if __name__ == "__main__":
    main()