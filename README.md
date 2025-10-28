# Piano Dropper - Water Drop Physics Visualizer

A beautiful visual interpretation of piano music using realistic water drop physics. Watch colorful water droplets fall, splash, and create ripples synchronized to musical notes!

## Features

### Physics Simulation
- **Realistic Water Drops**: Full gravity physics with velocity and acceleration
- **Dynamic Splashes**: Particle effects when drops hit the water surface
- **Ripple Effects**: Expanding waves that fade over time
- **Interactive Water Surface**: 150-point wave simulation with tension and dampening
- **Close-up View**: Focused on water interaction for detailed physics

### Visual Mapping
- **Frequency → Color**: Low notes (bass) appear red/orange, high notes (treble) appear blue/violet
- **Velocity → Size & Brightness**: Louder notes create larger, brighter drops
- **Duration → Drop Count**: Longer notes can create multiple drops

### Performance
- 60 FPS smooth animation
- Thread-safe note queue for real-time synchronization
- Efficient particle management with automatic cleanup

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.24.0
- scipy >= 1.10.0
- pygame >= 2.5.0
- pyaudio >= 0.2.11 (optional, for audio playback)

## Usage

### 1. Run with Sample Song
```bash
python run_piano_visualizer.py
```
Then select option 1 to play the included sample song with visualization.

### 2. Standalone Visualizer Demo
```bash
python visualizer.py
```
Press SPACE to create test drops, ESC to exit.

### 3. Custom Song
Create a JSON file with your notes:
```json
[
  {"name": "C4", "duration": 0.5, "velocity": 0.8},
  {"name": "E4", "duration": 0.5, "velocity": 0.9},
  {"name": "G4", "duration": 1.0, "velocity": 0.95}
]
```

Then run:
```bash
python run_piano_visualizer.py
```
Select option 3 and enter your filename.

### 4. Demo Mode (Random Notes)
```bash
python run_piano_visualizer.py
```
Select option 2 for continuous random note generation.

## Integration with Piano

The visualizer integrates seamlessly with the Piano class:

```python
from piano import Piano
from visualizer import PianoWaterVisualizer
import threading
import time

# Create instances
piano = Piano()
visualizer = PianoWaterVisualizer()

# Start visualizer in separate thread
viz_thread = threading.Thread(target=visualizer.run, daemon=True)
viz_thread.start()

# Load and play notes
notes = piano.parse_sheet_music("your_song.json")
for note in notes:
    visualizer.add_note(note.name, note.frequency, note.duration, note.velocity)
    time.sleep(note.duration)
```

## Controls

- **SPACE**: Create test water drop (in standalone mode)
- **ESC**: Exit visualizer

## Physics Parameters

You can customize the simulation by adjusting these parameters in `visualizer.py`:

### Water Drop
- `gravity`: 980.0 (pixels/s²)
- `max_lifetime`: 3.0 seconds
- `radius`: 3-10 pixels (based on velocity)

### Water Pool
- `num_points`: 150 (surface detail)
- `tension`: 0.025 (wave restoration force)
- `dampening`: 0.025 (wave energy loss)
- `spread`: 0.25 (wave propagation to neighbors)

### Splash
- `num_particles`: 10-30 (based on impact)
- `max_lifetime`: 0.3-0.8 seconds

### Ripple
- `max_radius`: 50-150 pixels
- `max_lifetime`: 1.5 seconds

## Color Mapping

The visualizer uses logarithmic frequency mapping to HSL color space:
- **A0 (27.5 Hz)**: Red
- **C4 (261.6 Hz)**: Yellow/Green
- **A4 (440 Hz)**: Cyan
- **C8 (4186 Hz)**: Violet

## File Structure

```
Piano_Dropper/
├── piano.py                 # Piano sound generation
├── visualizer.py            # Water drop physics simulation
├── run_piano_visualizer.py  # Integration script
├── sample_song.json         # Example song
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## How It Works

1. **Note Input**: Piano sends note data (name, frequency, duration, velocity)
2. **Drop Creation**: Visualizer creates water drops with:
   - Color based on frequency (using logarithmic mapping)
   - Size based on velocity
   - Random x-position for variety
3. **Physics Update**: Each frame (60 FPS):
   - Apply gravity to drops
   - Detect water surface collision
   - Generate splash particles and ripples
   - Update water surface waves
4. **Rendering**: Draw ripples → water → drops → splashes with smooth alpha blending

## Tips for Best Results

- **Tempo**: Works best with moderate tempo (60-120 BPM)
- **Note Density**: 2-5 simultaneous notes create beautiful patterns
- **Variety**: Mix high and low notes for colorful displays
- **Duration**: Notes between 0.2-1.0 seconds work well

## Troubleshooting

### No display window appears
- Make sure pygame is installed: `pip install pygame`
- Check if your system supports graphical display
- On Linux, ensure you have X11 or Wayland display server running

### Visualizer is laggy
- Reduce `num_points` in WaterPool (line 166)
- Reduce `target_fps` (line 189)
- Close other applications

### OpenGL/Threading Errors
- The visualizer MUST run in the main thread (pygame requirement)
- If integrating into your own code, ensure `visualizer.run()` is called from the main thread
- Put any note scheduling/piano logic in background threads instead

### No audio (optional)
- PyAudio may not be available on all systems
- The visualizer works without audio output
- Audio playback is commented out in `run_piano_visualizer.py` by default

### Running in headless environment
- The visualizer requires a display
- For headless servers, consider using virtual display (Xvfb) or disable visualization

## Future Enhancements

Potential additions:
- Multiple drop types (rain, stream, fountain)
- Camera zoom/pan controls
- Recording to video file
- MIDI file input support
- Real-time keyboard input
- Adjustable physics parameters via UI

## Credits

Created for the Piano Dropper project - bringing music to life through water physics!

