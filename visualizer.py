"""
Water Drop Physics Visualizer for Piano
Creates a visual interpretation of music using a physics engine to show water droplets,
splashes, and ripple effects that correspond to notes being played.
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import colorsys
import threading
import queue


@dataclass
class WaterDrop:
    """Represents a single water droplet."""
    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    radius: float
    color: Tuple[int, int, int]
    lifetime: float
    max_lifetime: float
    note_name: str
    frequency: float
    audio_data: Optional['np.ndarray'] = None  # Audio wave to play on impact
    duration: float = 0.5  # Note duration
    velocity: float = 1.0  # Note velocity

    def update(self, dt: float, gravity: float = 980.0):
        """Update droplet position and velocity."""
        self.vy += gravity * dt  # Apply gravity
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime += dt

    def is_alive(self) -> bool:
        """Check if droplet should still exist."""
        return self.lifetime < self.max_lifetime


@dataclass
class Ripple:
    """Represents a water ripple effect."""
    x: float
    y: float
    radius: float
    max_radius: float
    lifetime: float
    max_lifetime: float
    color: Tuple[int, int, int]
    alpha: int

    def update(self, dt: float):
        """Update ripple expansion."""
        self.lifetime += dt
        progress = self.lifetime / self.max_lifetime
        self.radius = self.max_radius * progress
        self.alpha = int(255 * (1 - progress))

    def is_alive(self) -> bool:
        """Check if ripple should still exist."""
        return self.lifetime < self.max_lifetime


@dataclass
class Splash:
    """Represents a splash particle."""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: Tuple[int, int, int]
    lifetime: float
    max_lifetime: float

    def update(self, dt: float, gravity: float = 980.0):
        """Update splash particle."""
        self.vy += gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime += dt

    def is_alive(self) -> bool:
        """Check if splash particle should still exist."""
        return self.lifetime < self.max_lifetime and self.y < 800


class WaterPool:
    """Represents the water surface with wave dynamics."""
    def __init__(self, width: int, y_position: int, num_points: int = 200):
        self.width = width
        self.y_position = y_position
        self.num_points = num_points

        # Water surface points
        self.points = [y_position for _ in range(num_points)]
        self.velocities = [0.0 for _ in range(num_points)]

        # Wave parameters
        self.tension = 0.025
        self.dampening = 0.025
        self.spread = 0.25

    def splash_at(self, x: float, force: float):
        """Create a splash at position x with given force."""
        index = int((x / self.width) * self.num_points)
        index = max(0, min(index, self.num_points - 1))
        self.velocities[index] += force

    def update(self, dt: float = 1.0):
        """Update water surface physics."""
        # Update velocities based on displacement
        for i in range(self.num_points):
            displacement = self.y_position - self.points[i]
            self.velocities[i] += self.tension * displacement - self.dampening * self.velocities[i]

        # Update positions
        for i in range(self.num_points):
            self.points[i] += self.velocities[i] * dt

        # Spread effect to neighboring points
        left_deltas = [0.0] * self.num_points
        right_deltas = [0.0] * self.num_points

        for i in range(self.num_points):
            if i > 0:
                left_deltas[i] = self.spread * (self.points[i] - self.points[i - 1])
                self.velocities[i - 1] += left_deltas[i]

            if i < self.num_points - 1:
                right_deltas[i] = self.spread * (self.points[i] - self.points[i + 1])
                self.velocities[i + 1] += right_deltas[i]

        for i in range(self.num_points):
            if i > 0:
                self.points[i - 1] += left_deltas[i]
            if i < self.num_points - 1:
                self.points[i + 1] += right_deltas[i]

    def get_surface_points(self) -> List[Tuple[float, float]]:
        """Get points for drawing the water surface."""
        points = []
        spacing = self.width / (self.num_points - 1)
        for i, y in enumerate(self.points):
            x = i * spacing
            points.append((x, y))
        return points


class PianoWaterVisualizer:
    """Main visualizer class that combines piano input with water physics."""

    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Piano Water Drop Visualizer")

        self.clock = pygame.time.Clock()
        self.running = False

        # Physics objects
        self.drops: List[WaterDrop] = []
        self.ripples: List[Ripple] = []
        self.splashes: List[Splash] = []
        self.water_pool = WaterPool(width, int(height * 0.75), num_points=150)

        # Visual settings
        self.background_color = (10, 15, 30)  # Dark blue background
        self.water_color = (30, 100, 180)  # Blue water

        # Note queue for receiving notes from piano
        self.note_queue = queue.Queue()

        # Audio playback queue for when drops hit water
        self.audio_queue = queue.Queue()

        # Audio callback (will be set by piano integration)
        self.audio_callback = None

        # FPS
        self.target_fps = 60

    def frequency_to_color(self, frequency: float, velocity: float = 1.0) -> Tuple[int, int, int]:
        """Convert note frequency to a color (lower freq = red, higher = violet)."""
        # Map frequency range (27.5 Hz - 4186 Hz for piano) to hue (0-0.8)
        min_freq = 27.5  # A0
        max_freq = 4186.0  # C8

        # Logarithmic mapping for better color distribution
        log_freq = math.log(frequency)
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)

        normalized = (log_freq - log_min) / (log_max - log_min)
        normalized = max(0.0, min(1.0, normalized))

        # Map to hue (red to violet: 0 to 0.75)
        hue = normalized * 0.75
        saturation = 0.8 + 0.2 * velocity
        lightness = 0.4 + 0.3 * velocity

        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        return (int(r * 255), int(g * 255), int(b * 255))

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
        """Create water drop(s) from note data."""
        frequency = note_data['frequency']
        velocity = note_data['velocity']
        note_name = note_data['name']
        audio_data = note_data.get('audio_data', None)
        duration = note_data.get('duration', 0.5)

        # Random x position with slight variation
        x = random.uniform(self.width * 0.2, self.width * 0.8)

        # Drop starts from top with slight randomness
        y = random.uniform(-50, 50)

        # Initial velocity (slightly random)
        vx = random.uniform(-30, 30)
        vy = random.uniform(50, 150)

        # Size based on velocity (volume)
        radius = 3 + velocity * 7

        # Color based on frequency
        color = self.frequency_to_color(frequency, velocity)

        # Lifetime until it hits water
        max_lifetime = 3.0

        drop = WaterDrop(
            x=x, y=y, vx=vx, vy=vy,
            radius=radius,
            color=color,
            lifetime=0.0,
            max_lifetime=max_lifetime,
            note_name=note_name,
            frequency=frequency,
            audio_data=audio_data,
            duration=duration,
            velocity=velocity
        )

        self.drops.append(drop)

    def create_splash(self, x: float, y: float, color: Tuple[int, int, int], intensity: float = 1.0):
        """Create splash particles at impact point."""
        num_particles = int(10 + intensity * 20)

        for _ in range(num_particles):
            angle = random.uniform(-math.pi, 0)  # Upward hemisphere
            speed = random.uniform(50, 200) * intensity

            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            size = random.uniform(1, 3) * intensity

            splash = Splash(
                x=x, y=y, vx=vx, vy=vy,
                size=size,
                color=color,
                lifetime=0.0,
                max_lifetime=random.uniform(0.3, 0.8)
            )
            self.splashes.append(splash)

    def create_ripple(self, x: float, y: float, color: Tuple[int, int, int], intensity: float = 1.0):
        """Create ripple effect at impact point."""
        max_radius = 50 + intensity * 100

        ripple = Ripple(
            x=x, y=y,
            radius=0,
            max_radius=max_radius,
            lifetime=0.0,
            max_lifetime=1.5,
            color=color,
            alpha=255
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

        # Update water drops
        for drop in self.drops[:]:
            drop.update(dt)

            # Check if drop hits water surface
            water_y = self.water_pool.y_position
            if drop.y >= water_y:
                # Create splash and ripple
                self.create_splash(drop.x, water_y, drop.color, drop.radius / 10)
                self.create_ripple(drop.x, water_y, drop.color, drop.radius / 10)

                # Make wave in water pool
                force = -drop.vy * 0.1 * (drop.radius / 5)
                self.water_pool.splash_at(drop.x, force)

                # Play audio on impact if available
                if drop.audio_data is not None:
                    self.audio_queue.put(drop.audio_data)
                elif self.audio_callback:
                    # Call audio callback with note info
                    self.audio_callback(drop.note_name, drop.frequency, drop.duration, drop.velocity)

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

    def draw(self):
        """Draw all visual elements."""
        # Clear screen
        self.screen.fill(self.background_color)

        # Draw ripples (behind everything)
        for ripple in self.ripples:
            if ripple.alpha > 0:
                # Create surface for alpha blending
                ripple_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.circle(
                    ripple_surface,
                    (*ripple.color, ripple.alpha),
                    (int(ripple.x), int(ripple.y)),
                    int(ripple.radius),
                    2
                )
                self.screen.blit(ripple_surface, (0, 0))

        # Draw water pool
        surface_points = self.water_pool.get_surface_points()
        if len(surface_points) > 2:
            # Draw filled water
            bottom_points = [
                (self.width, self.height),
                (0, self.height)
            ]
            all_points = surface_points + bottom_points
            pygame.draw.polygon(self.screen, self.water_color, all_points)

            # Draw water surface line (lighter blue)
            pygame.draw.lines(
                self.screen,
                (50, 150, 220),
                False,
                surface_points,
                3
            )

        # Draw water drops
        for drop in self.drops:
            # Main drop
            pygame.draw.circle(
                self.screen,
                drop.color,
                (int(drop.x), int(drop.y)),
                int(drop.radius)
            )

            # Highlight for 3D effect
            highlight_offset = drop.radius * 0.3
            pygame.draw.circle(
                self.screen,
                (min(255, drop.color[0] + 80),
                 min(255, drop.color[1] + 80),
                 min(255, drop.color[2] + 80)),
                (int(drop.x - highlight_offset), int(drop.y - highlight_offset)),
                int(drop.radius * 0.4)
            )

        # Draw splash particles
        for splash in self.splashes:
            alpha = int(255 * (1 - splash.lifetime / splash.max_lifetime))
            if alpha > 0:
                splash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.circle(
                    splash_surface,
                    (*splash.color, alpha),
                    (int(splash.x), int(splash.y)),
                    int(splash.size)
                )
                self.screen.blit(splash_surface, (0, 0))

        # Draw UI info
        self._draw_ui()

        pygame.display.flip()

    def _draw_ui(self):
        """Draw UI information."""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        title = font.render("Piano Water Drop Visualizer", True, (200, 220, 255))
        self.screen.blit(title, (20, 20))

        info = small_font.render(
            f"Drops: {len(self.drops)} | Splashes: {len(self.splashes)} | Ripples: {len(self.ripples)}",
            True,
            (150, 170, 200)
        )
        self.screen.blit(info, (20, 60))

    def run(self):
        """Main visualization loop."""
        self.running = True

        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0  # Convert to seconds

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        # Test drop
                        self.add_note("A4", 440.0, 0.5, 0.8)

            self.update(dt)
            self.draw()

        pygame.quit()

    def run_with_piano(self, piano_instance, notes: List):
        """Run visualizer synchronized with piano playback."""
        import threading
        import time

        # Start visualization in separate thread
        viz_thread = threading.Thread(target=self.run, daemon=True)
        viz_thread.start()

        # Play notes and send to visualizer
        for note in notes:
            # Add note to visualizer
            self.add_note(note.name, note.frequency, note.duration, note.velocity)

            # Wait for note duration
            time.sleep(note.duration)

        # Keep visualizer running for a bit after song ends
        time.sleep(3)
        self.running = False


def demo_visualizer():
    """Demo function to test the visualizer."""
    visualizer = PianoWaterVisualizer()

    # Add some test notes with different frequencies
    test_notes = [
        ("C4", 261.63, 0.5, 0.8),
        ("E4", 329.63, 0.5, 0.9),
        ("G4", 392.00, 0.5, 0.85),
        ("C5", 523.25, 0.5, 0.95),
    ]

    import threading
    import time

    def add_notes_periodically():
        time.sleep(1)  # Wait for window to open
        for note_name, freq, dur, vel in test_notes:
            visualizer.add_note(note_name, freq, dur, vel)
            time.sleep(0.5)

        # Add some more random notes
        for _ in range(10):
            time.sleep(0.3)
            freq = random.uniform(100, 1000)
            visualizer.add_note("Test", freq, 0.5, random.uniform(0.5, 1.0))

    note_thread = threading.Thread(target=add_notes_periodically, daemon=True)
    note_thread.start()

    visualizer.run()


if __name__ == "__main__":
    print("Water Drop Physics Visualizer")
    print("Press SPACE to create test drops")
    print("Press ESC to exit")
    demo_visualizer()

