# utils.py
import yaml
from typing import Any, Dict
import torch
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


if __name__ == "__main__":
    # Example usage
    config = Configuration('config/environment/default.yaml')

    
    # Access nested attributes
    num_uavs = config.slicing.slice_weights
    print(num_uavs)
    
    # Pretty print the config
    # print(config)
    
    # Convert to dict
    # config_dict = config.__dict__
    # print("Config as dict:", config_dict)
    
    # Check if a key exists
    if hasattr(config, 'system'):
        print("UAVs configured")

    print(torch.backends.mps.is_available())  # should print True
