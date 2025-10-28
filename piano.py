"""
Piano program that takes a sheet music file containing notes and their durations and plays them.
It passes this along to visualizer.py to show the notes being played with water drop effects.
"""

import numpy as np
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time


@dataclass
class Note:
    """Represents a musical note with its properties."""
    name: str
    frequency: float
    duration: float  # in seconds
    velocity: float = 1.0  # intensity/volume (0.0 to 1.0)


class Piano:
    """Piano class that maps notes to frequencies and generates  sound."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.note_frequencies = self._create_note_frequency_map()

    def _create_note_frequency_map(self) -> Dict[str, float]:
        """
        Create a mapping of note names to their frequencies.
        Uses A4 = 440 Hz as the reference pitch.
        Covers a full 88-key piano range (A0 to C8).
        """
        frequencies = {}

        # Note names in an octave
        note_names = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F',
                      'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']

        # Chromatic scale positions (C# and Db are the same, etc.)
        chromatic_positions = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

        # Generate frequencies for all 88 piano keys (A0 to C8)
        # A4 = 440 Hz is our reference (octave 4, position 9)
        a4_freq = 440.0

        for octave in range(0, 9):  # Octaves 0 through 8
            for note_name, position in chromatic_positions.items():
                # Calculate semitones from A4
                semitones_from_a4 = (octave - 4) * 12 + (position - 9)

                # Calculate frequency using equal temperament formula
                # f = 440 * 2^(n/12) where n is semitones from A4
                frequency = a4_freq * (2 ** (semitones_from_a4 / 12))

                note_key = f"{note_name}{octave}"
                frequencies[note_key] = frequency

        return frequencies

    def get_frequency(self, note_name: str) -> float:
        """Get the frequency for a given note name (e.g., 'A4', 'C#5')."""
        if note_name in self.note_frequencies:
            return self.note_frequencies[note_name]
        else:
            raise ValueError(f"Note '{note_name}' not found in frequency map")

    def generate_tone(self, frequency: float, duration: float, velocity: float = 1.0) -> np.ndarray:
        """
        Generate a tone with the given frequency and duration.
        Uses ADSR envelope for more realistic piano sound.
        """
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)

        # Generate the base sine wave
        wave = np.sin(2 * np.pi * frequency * t)

        # Add harmonics for richer piano-like sound
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
        wave += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
        wave += 0.125 * np.sin(2 * np.pi * frequency * 4 * t)  # 4th harmonic

        # Normalize
        wave = wave / np.max(np.abs(wave))

        # Apply ADSR envelope (Attack, Decay, Sustain, Release)
        envelope = self._create_adsr_envelope(num_samples, duration)
        wave = wave * envelope * velocity

        return wave

    def _create_adsr_envelope(self, num_samples: int, duration: float) -> np.ndarray:
        """Create an ADSR envelope for more realistic sound."""
        envelope = np.zeros(num_samples)

        # ADSR parameters (in seconds)
        attack_time = min(0.01, duration * 0.1)
        decay_time = min(0.1, duration * 0.2)
        release_time = min(0.2, duration * 0.3)
        sustain_level = 0.7

        attack_samples = int(self.sample_rate * attack_time)
        decay_samples = int(self.sample_rate * decay_time)
        release_samples = int(self.sample_rate * release_time)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples

        if sustain_samples < 0:
            sustain_samples = 0
            release_samples = num_samples - attack_samples - decay_samples

        # Attack: linear ramp up
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay: exponential decay to sustain level
        if decay_samples > 0:
            envelope[attack_samples:attack_samples + decay_samples] = \
                np.linspace(1, sustain_level, decay_samples)

        # Sustain: constant level
        if sustain_samples > 0:
            envelope[attack_samples + decay_samples:
                    attack_samples + decay_samples + sustain_samples] = sustain_level

        # Release: exponential decay to zero
        if release_samples > 0:
            envelope[-release_samples:] = \
                np.linspace(sustain_level, 0, release_samples)

        return envelope

    def play_note(self, note: Note, send_to_visualizer=True) -> np.ndarray:
        """
        Play a single note and optionally send to visualizer.
        Returns the generated audio wave.
        """
        frequency = self.get_frequency(note.name)
        wave = self.generate_tone(frequency, note.duration, note.velocity)

        if send_to_visualizer:
            self._send_to_visualizer(note, frequency)

        return wave

    def _send_to_visualizer(self, note: Note, frequency: float):
        """Send note data to the visualizer for water drop effects."""
        # This will be called by visualizer.py
        # For now, we'll prepare the data structure
        note_data = {
            'name': note.name,
            'frequency': frequency,
            'duration': note.duration,
            'velocity': note.velocity,
            'timestamp': time.time()
        }
        # The visualizer will receive this data
        return note_data

    def parse_sheet_music(self, sheet_music_file: str) -> List[Note]:
        """
        Parse a sheet music file and return a list of Notes.
        Expected format: JSON file with list of notes.
        Example:
        [
            {"name": "C4", "duration": 0.5, "velocity": 0.8},
            {"name": "E4", "duration": 0.5, "velocity": 0.8},
            {"name": "G4", "duration": 1.0, "velocity": 0.9}
        ]
        """
        with open(sheet_music_file, 'r') as f:
            data = json.load(f)

        notes = []
        for note_data in data:
            note = Note(
                name=note_data['name'],
                frequency=self.get_frequency(note_data['name']),
                duration=note_data.get('duration', 0.5),
                velocity=note_data.get('velocity', 1.0)
            )
            notes.append(note)

        return notes

    def play_song(self, notes: List[Note], send_to_visualizer=True) -> np.ndarray:
        """
        Play a sequence of notes (a song).
        Returns the combined audio wave.
        """
        waves = []

        for note in notes:
            wave = self.play_note(note, send_to_visualizer)
            waves.append(wave)

        # Concatenate all waves
        if waves:
            combined_wave = np.concatenate(waves)
            return combined_wave
        else:
            return np.array([])

    def create_chord(self, note_names: List[str], duration: float, velocity: float = 1.0) -> np.ndarray:
        """
        Create a chord by playing multiple notes simultaneously.
        """
        waves = []
        max_length = 0

        for note_name in note_names:
            frequency = self.get_frequency(note_name)
            wave = self.generate_tone(frequency, duration, velocity)
            waves.append(wave)
            max_length = max(max_length, len(wave))

        # Pad shorter waves and combine
        combined = np.zeros(max_length)
        for wave in waves:
            padded = np.pad(wave, (0, max_length - len(wave)), 'constant')
            combined += padded

        # Normalize to prevent clipping
        combined = combined / len(waves)

        return combined

    def play_audio_realtime(self, wave: np.ndarray):
        """Play the audio wave in real-time through speakers/headphones."""
        try:
            import pyaudio

            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Normalize to 16-bit range
            wave_normalized = np.int16(wave * 32767)

            # Open stream
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=self.sample_rate,
                          output=True)

            # Play the audio
            print("Playing audio...")
            stream.write(wave_normalized.tobytes())

            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Playback complete!")

        except ImportError:
            print("pyaudio not installed. Install with: pip install pyaudio")
            print("On Linux you may also need: sudo apt-get install portaudio19-dev")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def save_audio(self, wave: np.ndarray, filename: str):
        """Save the audio wave to a WAV file."""
        try:
            import scipy.io.wavfile as wavfile
            # Normalize to 16-bit range
            wave_normalized = np.int16(wave * 32767)
            wavfile.write(filename, self.sample_rate, wave_normalized)
            print(f"Audio saved to {filename}")
        except ImportError:
            print("scipy not installed. Install with: pip install scipy")

    def get_all_notes(self) -> List[str]:
        """Get a list of all available note names."""
        return sorted(self.note_frequencies.keys())

    def print_frequency_map(self):
        """Print the note-to-frequency mapping for reference."""
        print("Note-to-Frequency Mapping:")
        print("-" * 40)
        for octave in range(0, 9):
            octave_notes = {k: v for k, v in self.note_frequencies.items()
                          if k.endswith(str(octave))}
            if octave_notes:
                print(f"\nOctave {octave}:")
                for note, freq in sorted(octave_notes.items()):
                    print(f"  {note:6s}: {freq:8.2f} Hz")


def create_sample_song() -> str:
    """Create a sample song file for testing."""
    sample_song = [
        {"name": "C4", "duration": 0.5, "velocity": 0.8},
        {"name": "D4", "duration": 0.5, "velocity": 0.8},
        {"name": "E4", "duration": 0.5, "velocity": 0.8},
        {"name": "F4", "duration": 0.5, "velocity": 0.8},
        {"name": "G4", "duration": 0.5, "velocity": 0.9},
        {"name": "G4", "duration": 0.5, "velocity": 0.9},
        {"name": "A4", "duration": 0.25, "velocity": 0.85},
        {"name": "A4", "duration": 0.25, "velocity": 0.85},
        {"name": "A4", "duration": 0.25, "velocity": 0.85},
        {"name": "A4", "duration": 0.25, "velocity": 0.85},
        {"name": "G4", "duration": 1.0, "velocity": 0.9},
    ]

    filename = "sample_song.json"
    with open(filename, 'w') as f:
        json.dump(sample_song, f, indent=2)

    print(f"Sample song created: {filename}")
    return filename


def main():
    """Main function to demonstrate the piano functionality."""
    print("Piano Dropper - Piano Module")
    print("=" * 50)

    # Create piano instance
    piano = Piano()

    # Display some note frequencies
    print("\nSample Note Frequencies:")
    sample_notes = ['A0', 'C4', 'A4', 'C5', 'C8']
    for note_name in sample_notes:
        try:
            freq = piano.get_frequency(note_name)
            print(f"  {note_name}: {freq:.2f} Hz")
        except ValueError as e:
            print(f"  {e}")

    # Create and play a sample song
    print("\nCreating sample song...")
    song_file = create_sample_song()

    print("\nParsing sheet music...")
    notes = piano.parse_sheet_music(song_file)
    print(f"Loaded {len(notes)} notes")

    print("\nGenerating audio...")
    audio_wave = piano.play_song(notes, send_to_visualizer=False)

    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Play audio through speakers")
    print("2. Save to WAV file")
    print("3. Both")
    print("4. Play a custom JSON file")

    try:
        choice = input("Enter choice (1/2/3/4): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "3"  # Default to both if running non-interactively
        print("3")

    if choice == '4':
        # Play a custom JSON file
        try:
            custom_file = input("Enter the path to your JSON file: ").strip()
        except (EOFError, KeyboardInterrupt):
            custom_file = "sample_song.json"
            print("sample_song.json")

        try:
            print(f"\nLoading {custom_file}...")
            custom_notes = piano.parse_sheet_music(custom_file)
            print(f"Loaded {len(custom_notes)} notes")

            print("\nGenerating audio...")
            custom_audio = piano.play_song(custom_notes, send_to_visualizer=False)

            print("\nWhat would you like to do with this audio?")
            print("1. Play through speakers")
            print("2. Save to WAV file")
            print("3. Both")

            try:
                sub_choice = input("Enter choice (1/2/3): ").strip()
            except (EOFError, KeyboardInterrupt):
                sub_choice = "3"
                print("3")

            if sub_choice in ['1', '3']:
                piano.play_audio_realtime(custom_audio)

            if sub_choice in ['2', '3']:
                try:
                    output_name = input("Enter output filename (default: output_song.wav): ").strip()
                except (EOFError, KeyboardInterrupt):
                    output_name = ""
                    print("")

                if not output_name:
                    output_name = "output_song.wav"

                print(f"\nSaving audio file to {output_name}...")
                piano.save_audio(custom_audio, output_name)

        except FileNotFoundError:
            print(f"Error: File '{custom_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: File '{custom_file}' is not valid JSON.")
        except Exception as e:
            print(f"Error: {e}")

    elif choice in ['1', '3']:
        piano.play_audio_realtime(audio_wave)

    if choice in ['2', '3']:
        print("\nSaving audio file...")
        piano.save_audio(audio_wave, "output_song.wav")

    print("\nDone!")
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. No action taken.")


if __name__ == "__main__":
    main()

