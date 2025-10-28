"""
3D Water Drop Physics Visualizer for Piano
Creates a 3D visual interpretation of music using OpenGL to show water droplets,
splashes, and ripple effects in three dimensions.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import colorsys
import queue


@dataclass
class WaterDrop3D:
    """Represents a 3D water droplet."""
    x: float
    y: float
    z: float  # depth dimension
    vx: float
    vy: float
    vz: float
    radius: float
    color: Tuple[float, float, float]  # RGB in 0-1 range for OpenGL
    lifetime: float
    max_lifetime: float
    note_name: str
    frequency: float
    audio_data: Optional['np.ndarray'] = None
    duration: float = 0.5
    velocity: float = 1.0
    rotation: float = 0.0
    rotation_speed: float = 0.0
    
    def update(self, dt: float, gravity: float = 15.0):
        """Update droplet position and velocity."""
        self.vy -= gravity * dt  # Gravity pulls down (negative y)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.rotation += self.rotation_speed * dt
        self.lifetime += dt
        
    def is_alive(self) -> bool:
        """Check if droplet should still exist."""
        return self.lifetime < self.max_lifetime


@dataclass
class Ripple3D:
    """Represents a 3D water ripple effect."""
    x: float
    y: float
    z: float
    radius: float
    max_radius: float
    lifetime: float
    max_lifetime: float
    color: Tuple[float, float, float]
    alpha: float
    height: float  # Wave height
    
    def update(self, dt: float):
        """Update ripple expansion."""
        self.lifetime += dt
        progress = self.lifetime / self.max_lifetime
        self.radius = self.max_radius * progress
        self.alpha = 1.0 - progress
        self.height = math.sin(progress * math.pi) * 0.5
        
    def is_alive(self) -> bool:
        """Check if ripple should still exist."""
        return self.lifetime < self.max_lifetime


@dataclass
class Splash3D:
    """Represents a 3D splash particle."""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    size: float
    color: Tuple[float, float, float]
    lifetime: float
    max_lifetime: float
    
    def update(self, dt: float, gravity: float = 15.0):
        """Update splash particle."""
        self.vy -= gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.lifetime += dt
        
    def is_alive(self) -> bool:
        """Check if splash particle should still exist."""
        return self.lifetime < self.max_lifetime and self.y > -10


class WaterPool3D:
    """Represents a 3D water surface with wave dynamics."""
    def __init__(self, size: float = 20.0, resolution: int = 50):
        self.size = size
        self.resolution = resolution
        self.y_position = 0.0  # Water surface at y=0
        
        # Create grid of water surface points
        self.grid = np.zeros((resolution, resolution))
        self.velocities = np.zeros((resolution, resolution))
        
        # Wave parameters
        self.tension = 0.025
        self.dampening = 0.025
        self.spread = 0.25
        
    def splash_at(self, x: float, z: float, force: float):
        """Create a splash at position (x, z) with given force."""
        # Convert world coordinates to grid coordinates
        grid_x = int((x + self.size / 2) / self.size * self.resolution)
        grid_z = int((z + self.size / 2) / self.size * self.resolution)
        
        if 0 <= grid_x < self.resolution and 0 <= grid_z < self.resolution:
            self.velocities[grid_z, grid_x] += force
            
    def update(self, dt: float = 1.0):
        """Update water surface physics."""
        # Update velocities based on displacement
        displacement = self.y_position - self.grid
        self.velocities += self.tension * displacement - self.dampening * self.velocities
        
        # Update positions
        self.grid += self.velocities * dt
        
        # Spread to neighbors
        for i in range(1, self.resolution - 1):
            for j in range(1, self.resolution - 1):
                avg = (self.grid[i-1, j] + self.grid[i+1, j] + 
                       self.grid[i, j-1] + self.grid[i, j+1]) / 4
                self.velocities[i, j] += self.spread * (avg - self.grid[i, j])


class Camera:
    """3D Camera with smooth controls."""
    def __init__(self):
        self.distance = 25.0
        self.angle_x = 30.0  # Look down slightly
        self.angle_y = 0.0
        self.target_distance = 25.0
        self.target_angle_x = 30.0
        self.target_angle_y = 0.0
        self.auto_rotate = True
        self.auto_rotate_speed = 5.0  # degrees per second
        
    def update(self, dt: float):
        """Smooth camera movement."""
        # Smooth interpolation
        self.distance += (self.target_distance - self.distance) * 5 * dt
        self.angle_x += (self.target_angle_x - self.angle_x) * 5 * dt
        self.angle_y += (self.target_angle_y - self.angle_y) * 5 * dt
        
        # Auto-rotate
        if self.auto_rotate:
            self.target_angle_y += self.auto_rotate_speed * dt
            if self.target_angle_y > 360:
                self.target_angle_y -= 360
                
    def apply(self):
        """Apply camera transformation."""
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.angle_x, 1, 0, 0)
        glRotatef(self.angle_y, 0, 1, 0)


class PianoWaterVisualizer3D:
    """Main 3D visualizer class."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        
        # Create OpenGL window
        self.screen = pygame.display.set_mode(
            (width, height), 
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Piano Water Drop Visualizer 3D")
        
        # Initialize OpenGL
        self._init_opengl()
        
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Physics objects
        self.drops: List[WaterDrop3D] = []
        self.ripples: List[Ripple3D] = []
        self.splashes: List[Splash3D] = []
        self.water_pool = WaterPool3D(size=20.0, resolution=50)
        
        # Camera
        self.camera = Camera()
        
        # Queues
        self.note_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # FPS
        self.target_fps = 60
        
        # Lighting
        self.setup_lighting()
        
    def _init_opengl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Background color (dark blue)
        glClearColor(0.05, 0.1, 0.2, 1.0)
        
    def setup_lighting(self):
        """Setup 3D lighting."""
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        # Main light (white, from above)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 10, 5, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 1.0, 1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
        
        # Fill light (colored, from side)
        glLightfv(GL_LIGHT1, GL_POSITION, (-10, 5, -5, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.3, 0.5, 0.7, 1))
        
    def frequency_to_color(self, frequency: float, velocity: float = 1.0) -> Tuple[float, float, float]:
        """Convert note frequency to RGB color (0-1 range for OpenGL)."""
        min_freq = 27.5
        max_freq = 4186.0
        
        log_freq = math.log(frequency)
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        
        normalized = (log_freq - log_min) / (log_max - log_min)
        normalized = max(0.0, min(1.0, normalized))
        
        hue = normalized * 0.75
        saturation = 0.8 + 0.2 * velocity
        lightness = 0.5 + 0.3 * velocity
        
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        return (r, g, b)
        
    def add_note(self, note_name: str, frequency: float, duration: float, velocity: float, audio_data=None):
        """Add a note to be visualized."""
        self.note_queue.put({
            'name': note_name,
            'frequency': frequency,
            'duration': duration,
            'velocity': velocity,
            'audio_data': audio_data
        })
        
    def create_drop_from_note(self, note_data: Dict):
        """Create 3D water drop from note data."""
        frequency = note_data['frequency']
        velocity = note_data['velocity']
        note_name = note_data['name']
        audio_data = note_data.get('audio_data', None)
        duration = note_data.get('duration', 0.5)
        
        # Random position in 3D space
        x = random.uniform(-8, 8)
        z = random.uniform(-8, 8)
        y = random.uniform(12, 15)
        
        # Initial velocity
        vx = random.uniform(-1, 1)
        vz = random.uniform(-1, 1)
        vy = random.uniform(-2, -1)
        
        # Size based on velocity
        radius = 0.2 + velocity * 0.5
        
        # Color based on frequency
        color = self.frequency_to_color(frequency, velocity)
        
        drop = WaterDrop3D(
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            radius=radius,
            color=color,
            lifetime=0.0,
            max_lifetime=5.0,
            note_name=note_name,
            frequency=frequency,
            audio_data=audio_data,
            duration=duration,
            velocity=velocity,
            rotation=random.uniform(0, 360),
            rotation_speed=random.uniform(-180, 180)
        )
        
        self.drops.append(drop)
        
    def create_splash(self, x: float, y: float, z: float, color: Tuple[float, float, float], intensity: float = 1.0):
        """Create 3D splash particles at impact point."""
        num_particles = int(20 + intensity * 30)
        
        for _ in range(num_particles):
            # Hemisphere of particles
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi / 2)
            speed = random.uniform(3, 8) * intensity
            
            vx = math.sin(phi) * math.cos(theta) * speed
            vy = math.cos(phi) * speed
            vz = math.sin(phi) * math.sin(theta) * speed
            
            size = random.uniform(0.05, 0.15) * intensity
            
            splash = Splash3D(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                size=size,
                color=color,
                lifetime=0.0,
                max_lifetime=random.uniform(0.5, 1.2)
            )
            self.splashes.append(splash)
            
    def create_ripple(self, x: float, y: float, z: float, color: Tuple[float, float, float], intensity: float = 1.0):
        """Create 3D ripple effect at impact point."""
        max_radius = 3 + intensity * 5
        
        ripple = Ripple3D(
            x=x, y=y, z=z,
            radius=0,
            max_radius=max_radius,
            lifetime=0.0,
            max_lifetime=2.0,
            color=color,
            alpha=1.0,
            height=0.0
        )
        self.ripples.append(ripple)
        
    def update(self, dt: float):
        """Update all physics objects."""
        # Process note queue
        while not self.note_queue.empty():
            try:
                note_data = self.note_queue.get_nowait()
                self.create_drop_from_note(note_data)
            except queue.Empty:
                break
                
        # Update camera
        self.camera.update(dt)
        
        # Update water drops
        for drop in self.drops[:]:
            drop.update(dt)
            
            # Check if drop hits water surface
            if drop.y <= self.water_pool.y_position:
                # Create splash and ripple
                self.create_splash(drop.x, drop.y, drop.z, drop.color, drop.radius)
                self.create_ripple(drop.x, drop.y, drop.z, drop.color, drop.radius)
                
                # Make wave in water pool
                force = -drop.vy * 0.5 * drop.radius
                self.water_pool.splash_at(drop.x, drop.z, force)
                
                # Play audio on impact
                if drop.audio_data is not None:
                    self.audio_queue.put(drop.audio_data)
                
                # Remove drop
                self.drops.remove(drop)
            elif not drop.is_alive():
                self.drops.remove(drop)
                
        # Update splashes
        self.splashes = [s for s in self.splashes if s.is_alive()]
        for splash in self.splashes:
            splash.update(dt)
            
        # Update ripples
        self.ripples = [r for r in self.ripples if r.is_alive()]
        for ripple in self.ripples:
            ripple.update(dt)
            
        # Update water pool
        self.water_pool.update()
        
    def draw_sphere(self, radius: float, slices: int = 16, stacks: int = 16):
        """Draw a sphere using GLU."""
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
        
    def draw_water_drop(self, drop: WaterDrop3D):
        """Draw a 3D water droplet."""
        glPushMatrix()
        glTranslatef(drop.x, drop.y, drop.z)
        glRotatef(drop.rotation, 0, 1, 0)
        
        # Set color with slight transparency
        glColor4f(*drop.color, 0.8)
        
        # Main sphere
        self.draw_sphere(drop.radius)
        
        # Highlight
        glPushMatrix()
        glTranslatef(drop.radius * 0.3, drop.radius * 0.3, drop.radius * 0.3)
        glColor4f(1, 1, 1, 0.6)
        self.draw_sphere(drop.radius * 0.3)
        glPopMatrix()
        
        glPopMatrix()
        
    def draw_splash_particle(self, splash: Splash3D):
        """Draw a splash particle."""
        alpha = 1.0 - (splash.lifetime / splash.max_lifetime)
        
        glPushMatrix()
        glTranslatef(splash.x, splash.y, splash.z)
        glColor4f(*splash.color, alpha * 0.7)
        self.draw_sphere(splash.size)
        glPopMatrix()
        
    def draw_ripple(self, ripple: Ripple3D):
        """Draw a 3D ripple as a torus."""
        if ripple.alpha <= 0:
            return
            
        glPushMatrix()
        glTranslatef(ripple.x, ripple.y, ripple.z)
        
        # Draw ripple as a ring
        glColor4f(*ripple.color, ripple.alpha * 0.5)
        glBegin(GL_QUAD_STRIP)
        
        segments = 32
        inner_radius = ripple.radius - 0.2
        outer_radius = ripple.radius + 0.2
        
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Inner vertex
            glVertex3f(inner_radius * cos_a, ripple.height, inner_radius * sin_a)
            # Outer vertex
            glVertex3f(outer_radius * cos_a, ripple.height, outer_radius * sin_a)
            
        glEnd()
        glPopMatrix()
        
    def draw_water_surface(self):
        """Draw the 3D water surface."""
        glEnable(GL_BLEND)
        
        size = self.water_pool.size
        res = self.water_pool.resolution
        grid = self.water_pool.grid
        
        # Water color
        glColor4f(0.2, 0.5, 0.8, 0.6)
        
        # Draw water surface as a mesh
        for i in range(res - 1):
            glBegin(GL_QUAD_STRIP)
            for j in range(res):
                x1 = (i / res) * size - size / 2
                z1 = (j / res) * size - size / 2
                y1 = grid[i, j]
                
                x2 = ((i + 1) / res) * size - size / 2
                z2 = (j / res) * size - size / 2
                y2 = grid[i + 1, j]
                
                # Calculate normals for lighting
                if j < res - 1 and i < res - 1:
                    dx = grid[i + 1, j] - grid[i, j]
                    dz = grid[i, j + 1] - grid[i, j]
                    normal = np.array([-dx, 1, -dz])
                    normal = normal / np.linalg.norm(normal)
                    glNormal3f(*normal)
                
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
            glEnd()
            
    def draw_grid(self):
        """Draw a reference grid."""
        glDisable(GL_LIGHTING)
        glColor4f(0.2, 0.3, 0.4, 0.3)
        
        size = 20
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i * 2, -5, -size)
            glVertex3f(i * 2, -5, size)
            glVertex3f(-size, -5, i * 2)
            glVertex3f(size, -5, i * 2)
        glEnd()
        
        glEnable(GL_LIGHTING)
        
    def draw(self):
        """Draw all 3D elements."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera
        self.camera.apply()
        
        # Draw scene
        self.draw_grid()
        
        # Draw water surface
        self.draw_water_surface()
        
        # Draw ripples (before drops for transparency)
        for ripple in self.ripples:
            self.draw_ripple(ripple)
            
        # Draw water drops
        for drop in self.drops:
            self.draw_water_drop(drop)
            
        # Draw splash particles
        for splash in self.splashes:
            self.draw_splash_particle(splash)
            
        # Draw UI in 2D
        self.draw_ui()
        
        pygame.display.flip()
        
    def draw_ui(self):
        """Draw 2D UI overlay."""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Draw text info (simplified - just show particle count)
        glColor4f(0.8, 0.9, 1.0, 0.8)
        
        # Reset matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
    def handle_input(self):
        """Handle keyboard and mouse input."""
        keys = pygame.key.get_pressed()
        
        # Camera controls
        if keys[K_UP]:
            self.camera.target_angle_x = max(0, self.camera.target_angle_x - 1)
        if keys[K_DOWN]:
            self.camera.target_angle_x = min(90, self.camera.target_angle_x + 1)
        if keys[K_LEFT]:
            self.camera.target_angle_y -= 2
        if keys[K_RIGHT]:
            self.camera.target_angle_y += 2
            
        # Zoom
        if keys[K_w]:
            self.camera.target_distance = max(10, self.camera.target_distance - 0.5)
        if keys[K_s]:
            self.camera.target_distance = min(50, self.camera.target_distance + 0.5)
            
        # Toggle auto-rotate
        if keys[K_r]:
            self.camera.auto_rotate = not self.camera.auto_rotate
            
    def run(self):
        """Main visualization loop."""
        self.running = True
        
        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == K_SPACE:
                        # Test drop
                        self.add_note("A4", 440.0, 0.5, 0.8)
                    elif event.key == K_r:
                        self.camera.auto_rotate = not self.camera.auto_rotate
                        
            self.handle_input()
            self.update(dt)
            self.draw()
            
        pygame.quit()


def demo_visualizer_3d():
    """Demo function to test the 3D visualizer."""
    visualizer = PianoWaterVisualizer3D()
    
    import threading
    import time
    
    def add_notes_periodically():
        time.sleep(1)
        test_notes = [
            ("C4", 261.63, 0.5, 0.8),
            ("E4", 329.63, 0.5, 0.9),
            ("G4", 392.00, 0.5, 0.85),
            ("C5", 523.25, 0.5, 0.95),
        ]
        
        for note_name, freq, dur, vel in test_notes:
            if not visualizer.running:
                break
            visualizer.add_note(note_name, freq, dur, vel)
            time.sleep(0.5)
            
        # Add random notes
        for _ in range(20):
            if not visualizer.running:
                break
            time.sleep(0.3)
            freq = random.uniform(200, 800)
            visualizer.add_note("Test", freq, 0.5, random.uniform(0.5, 1.0))
    
    note_thread = threading.Thread(target=add_notes_periodically, daemon=True)
    note_thread.start()
    
    print("3D Water Drop Visualizer")
    print("Controls:")
    print("  ARROW KEYS: Rotate camera")
    print("  W/S: Zoom in/out")
    print("  R: Toggle auto-rotate")
    print("  SPACE: Create test drop")
    print("  ESC: Exit")
    
    visualizer.run()


if __name__ == "__main__":
    demo_visualizer_3d()

