"""
Integration script to run piano playback with water drop visualizer.
Plays music while showing synchronized water drop physics visualization.
"""

import threading
import time
from piano import Piano, Note
from visualizer_3d import PianoWaterVisualizer3D


def run_piano_with_visualizer(song_file: str = "sample_song.json"):
    """Run piano playback with synchronized water drop visualizer."""
    print("Piano Dropper - Integrated Visualizer")
    print("=" * 50)

    # Create piano and visualizer instances
    print("\nInitializing piano...")
    piano = Piano()

    print("Initializing visualizer...")
    visualizer = PianoWaterVisualizer3D(width=1200, height=800)

    # Parse the song
    print(f"\nLoading song from {song_file}...")
    notes = piano.parse_sheet_music(song_file)
    print(f"Loaded {len(notes)} notes")

    print("\nPlaying song with visualization...")
    print("Press ESC in the visualizer window to stop")
    print("-" * 50)

    # Audio player thread that plays audio when drops hit water
    def audio_player():
        import pyaudio
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=piano.sample_rate,
                          output=True)

            while visualizer.running:
                try:
                    audio_wave = visualizer.audio_queue.get(timeout=0.1)
                    import numpy as np
                    wave_normalized = np.int16(audio_wave * 32767)
                    stream.write(wave_normalized.tobytes())
                except:
                    pass

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Audio player error: {e}")

    # Define a function to send notes in a background thread
    def send_notes():
        time.sleep(0.5)  # Give visualizer time to initialize

        start_time = time.time()
        note_index = 0

        while note_index < len(notes) and visualizer.running:
            current_time = time.time() - start_time

            # Send all notes that should start at or before current time
            while note_index < len(notes) and notes[note_index].timestamp <= current_time:
                note = notes[note_index]

                # Display note info
                print(f"Note {note_index + 1}/{len(notes)}: {note.name} ({note.frequency:.2f} Hz) - "
                      f"Duration: {note.duration}s, Velocity: {note.velocity:.2f}, Timestamp: {note.timestamp:.2f}s")

                # Generate audio wave but don't play it yet
                try:
                    audio_wave = piano.play_note(note, send_to_visualizer=False)
                    # Send note to visualizer WITH audio data - it will play when drop hits water
                    visualizer.add_note(note.name, note.frequency, note.duration, note.velocity, audio_wave)
                except Exception as e:
                    print(f"  (Audio generation error: {e})")
                    visualizer.add_note(note.name, note.frequency, note.duration, note.velocity)

                note_index += 1

            # Small sleep to prevent busy waiting
            time.sleep(0.01)

        # Keep visualizer running after song ends
        print("\nSong complete! Visualizer will continue for 5 more seconds...")
        time.sleep(5)

        # Stop visualizer
        visualizer.running = False

    # Start audio player thread
    audio_thread = threading.Thread(target=audio_player, daemon=True)
    audio_thread.start()

    # Start note sending in background thread
    note_thread = threading.Thread(target=send_notes, daemon=True)
    note_thread.start()

    # Run visualizer in main thread (required for pygame/OpenGL)
    visualizer.run()

    print("\nDone!")


def demo_mode():
    """Run a continuous demo with random notes."""
    import random

    print("Piano Dropper - Demo Mode")
    print("=" * 50)
    print("\nRunning continuous demo with random notes...")
    print("Press ESC in the visualizer window to stop\n")

    visualizer = PianoWaterVisualizer3D(width=1200, height=800)
    piano = Piano()

    # Generate random notes
    note_names = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5']

    # Audio player thread
    def audio_player():
        import pyaudio
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=piano.sample_rate,
                          output=True)

            while visualizer.running:
                try:
                    audio_wave = visualizer.audio_queue.get(timeout=0.1)
                    import numpy as np
                    wave_normalized = np.int16(audio_wave * 32767)
                    stream.write(wave_normalized.tobytes())
                except:
                    pass

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Audio player error: {e}")

    def send_random_notes():
        time.sleep(0.5)  # Give visualizer time to initialize

        while visualizer.running:
            note_name = random.choice(note_names)
            frequency = piano.get_frequency(note_name)
            duration = random.uniform(0.2, 0.8)
            velocity = random.uniform(0.6, 1.0)

            print(f"Playing: {note_name} ({frequency:.2f} Hz)")

            # Generate audio wave and send to visualizer
            try:
                note = Note(note_name, frequency, duration, velocity)
                audio_wave = piano.play_note(note, send_to_visualizer=False)
                visualizer.add_note(note_name, frequency, duration, velocity, audio_wave)
            except Exception as e:
                visualizer.add_note(note_name, frequency, duration, velocity)

            time.sleep(duration * 0.8)  # Slight overlap for continuous feel

        print("\nDemo stopped!")

    # Start audio player thread
    audio_thread = threading.Thread(target=audio_player, daemon=True)
    audio_thread.start()

    # Start random note generation in background thread
    note_thread = threading.Thread(target=send_random_notes, daemon=True)
    note_thread.start()

    # Run visualizer in main thread (required for pygame/OpenGL)
    visualizer.run()


if __name__ == "__main__":
    import sys

    print("\nWhat would you like to do?")
    print("1. Play sample song with visualization")
    print("2. Run demo mode (random notes)")
    print("3. Play custom song file")

    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"
        print("1")

    if choice == "1":
        run_piano_with_visualizer("sample_song.json")
    elif choice == "2":
        demo_mode()
    elif choice == "3":
        filename = input("Enter song filename: ").strip()
        if filename:
            run_piano_with_visualizer(filename)
        else:
            print("No filename provided. Using sample_song.json")
            run_piano_with_visualizer("sample_song.json")
    else:
        print("Invalid choice. Running sample song...")
        run_piano_with_visualizer("sample_song.json")

