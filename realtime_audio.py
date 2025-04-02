"""
Real-time Audio Analysis Tool (Improved Console Version)
- BPM detection
- Pitch tracking
- Onset detection
- Console output
- Fixed FFT size warning
"""

import numpy as np
import sounddevice as sd
import librosa
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Constants
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_LENGTH = 512
BUFFER_DURATION = 3  # seconds
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION
# Make sure n_fft is not larger than BLOCK_SIZE to avoid warnings
N_FFT = BLOCK_SIZE


@dataclass
class AudioFeatures:
    """Container for audio analysis results"""
    tempo: float = 0.0
    pitch_hz: float = 0.0
    pitch_note: str = ""
    onset_detected: bool = False
    onset_strength: float = 0.0
    spectrum: Optional[np.ndarray] = None
    waveform: Optional[np.ndarray] = None
    confidence: float = 0.0


class AudioAnalyzer:
    """Core audio analysis module that can be wrapped as a DLL for Unity"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.last_beat_time = 0
        self.beat_interval = 0.5  # Initial value, will be updated
        self.last_onset_time = 0
        self.min_onset_interval = 0.1  # Minimum time between onsets
        self.last_features = AudioFeatures()
        
        # For onset detection
        self.onset_envelope = np.zeros(buffer_size // HOP_LENGTH, dtype=np.float32)
        self.onset_threshold = 0.5  # Adjustable threshold for onset detection
        self.prev_onset_value = 0
        
        # For tempo estimation
        self.tempo_memory = []
        self.tempo_memory_size = 5  # Remember last N tempo estimates for smoothing
        
    def update_buffer(self, audio_chunk):
        """Add new audio data to the buffer"""
        if len(audio_chunk) > 0:
            # Roll buffer and add new data
            self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
            self.audio_buffer[-len(audio_chunk):] = audio_chunk
    
    def compute_spectrum(self, audio):
        """Compute the frequency spectrum of audio data"""
        if len(audio) == 0:
            return np.zeros(N_FFT // 2)
        
        # Use FFT for faster spectrum analysis
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=N_FFT))
        return spectrum[:len(spectrum) // 2]  # Return only meaningful frequencies
    
    def detect_pitch(self, audio_chunk):
        """Detect pitch using piptrack algorithm from librosa"""
        if len(audio_chunk) < 512 or np.max(np.abs(audio_chunk)) < 0.01:
            return self.last_features.pitch_hz, self.last_features.pitch_note, 0.0
        
        try:
            # Use n_fft parameter to avoid the warning
            pitches, magnitudes = librosa.piptrack(
                y=audio_chunk, 
                sr=self.sample_rate,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Get pitch with highest magnitude
            if pitches.size > 0 and magnitudes.size > 0:
                index = magnitudes[:, 0].argmax()
                pitch_hz = pitches[index, 0]
                confidence = magnitudes[index, 0]
                
                # Only update if we have a decent signal
                if confidence > 0.1:
                    pitch_note = librosa.hz_to_note(pitch_hz)
                    return pitch_hz, pitch_note, confidence
            
            # Return previous values if detection fails
            return self.last_features.pitch_hz, self.last_features.pitch_note, 0.0
            
        except Exception as e:
            print(f"\nError in pitch detection: {e}")
            return self.last_features.pitch_hz, self.last_features.pitch_note, 0.0
    
    def detect_onset(self, now):
        """Detect note onsets in the buffer"""
        try:
            # Calculate onset envelope from audio buffer
            current_onset_env = librosa.onset.onset_strength(
                y=self.audio_buffer[-BLOCK_SIZE*2:], 
                sr=self.sample_rate,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT  # Specify n_fft to avoid warning
            )
            
            # Get the latest onset strength value
            current_value = current_onset_env[-1] if len(current_onset_env) > 0 else 0
            
            # Detect onset using threshold and derivative
            onset_detected = (current_value > self.onset_threshold and 
                              current_value > self.prev_onset_value * 1.5 and
                              now - self.last_onset_time > self.min_onset_interval)
            
            if onset_detected:
                self.last_onset_time = now
            
            self.prev_onset_value = current_value
            return onset_detected, current_value
            
        except Exception as e:
            print(f"\nError in onset detection: {e}")
            return False, 0.0
    
    def estimate_tempo(self):
        """Estimate the tempo (BPM) from the audio buffer"""
        try:
            # Only use a portion of the buffer for faster processing
            analysis_segment = self.audio_buffer[-self.sample_rate*2:]
            
            # Calculate onset envelope
            onset_env = librosa.onset.onset_strength(
                y=analysis_segment, 
                sr=self.sample_rate,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT  # Specify n_fft to avoid warning
            )
            
            # Estimate tempo
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=self.sample_rate,
                hop_length=HOP_LENGTH
            )
            
            # Add to memory for smoothing
            self.tempo_memory.append(tempo)
            if len(self.tempo_memory) > self.tempo_memory_size:
                self.tempo_memory.pop(0)
            
            # Return smoothed tempo
            smoothed_tempo = np.mean(self.tempo_memory)
            return smoothed_tempo
            
        except Exception as e:
            print(f"\nError in tempo estimation: {e}")
            if len(self.tempo_memory) > 0:
                return np.mean(self.tempo_memory)
            return 120.0  # Default fallback
    
    def analyze_audio(self, audio_chunk):
        """Main analysis function that combines all audio processing"""
        # Update buffer with new audio data
        self.update_buffer(audio_chunk)
        
        # Get current time
        now = time.time()
        
        # Create features container
        features = AudioFeatures()
        
        # Compute spectrum for visualization
        features.spectrum = self.compute_spectrum(audio_chunk)
        features.waveform = audio_chunk
        
        # Detect pitch
        features.pitch_hz, features.pitch_note, features.confidence = self.detect_pitch(audio_chunk)
        
        # Detect onset
        features.onset_detected, features.onset_strength = self.detect_onset(now)
        
        # Update tempo less frequently (every 2 seconds) to save CPU
        if now - self.last_beat_time > 2:
            features.tempo = self.estimate_tempo()
            self.beat_interval = 60.0 / features.tempo if features.tempo > 0 else 0.5
            self.last_beat_time = now
        else:
            features.tempo = self.last_features.tempo
        
        # Store features for future reference
        self.last_features = features
        
        return features


class ConsoleDisplay:
    """Simple console display for audio analysis results"""
    
    def __init__(self):
        self.last_update_time = 0
        self.update_interval = 0.1  # Update console every 100ms
        self.onset_count = 0
        self.onset_time = time.time()
        self.signal_peak = 0
        self.peak_decay = 0.05  # How fast the peak level decays
        
    def update_display(self, features):
        """Update the console display with new features"""
        now = time.time()
        
        # Only update display at certain intervals to avoid console spam
        if now - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = now
        
        # Calculate current signal level
        current_level = np.max(np.abs(features.waveform)) if features.waveform is not None else 0
        
        # Update signal peak with decay
        self.signal_peak = max(current_level, self.signal_peak * (1 - self.peak_decay))
        
        # Create status indicators
        tempo_status = f"Tempo: {features.tempo:.1f} BPM"
        
        # Only show pitch if we have a reasonable confidence
        if features.confidence > 0.2:
            pitch_status = f"Pitch: {features.pitch_note} ({features.pitch_hz:.1f} Hz)"
        else:
            pitch_status = "Pitch: --"
        
        # Create onset indicator
        if features.onset_detected:
            self.onset_count += 1
            self.onset_time = now
            onset_status = f"⚡ ONSET! (#{self.onset_count})"
        elif now - self.onset_time < 0.5:  # Keep showing onset for 500ms
            onset_status = f"⚡ ONSET! (#{self.onset_count})"
        else:
            onset_status = "..."
        
        # Create level meter
        normalized_level = current_level / max(0.001, self.signal_peak)
        bars = int(normalized_level * 20)
        level_meter = f"Level: [{'#' * bars}{' ' * (20-bars)}]"
        
        # Clear line and print status
        status = f"{tempo_status} | {pitch_status} | {onset_status} | {level_meter}"
        print(f"\r{status}", end="")


class AudioProcessor:
    """Main class that connects the audio input with analysis and display"""
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        self.display = ConsoleDisplay()
        
        # Queue for thread communication
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Thread for audio processing
        self.process_thread = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice"""
        if status:
            print(f"\nStatus: {status}")
        
        # Extract mono audio
        audio = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        
        # Put audio chunk in the queue
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            pass  # Skip frame if queue is full
    
    def process_audio_thread(self):
        """Thread function for audio processing"""
        while self.running:
            try:
                # Get audio from queue
                audio = self.audio_queue.get(timeout=0.1)
                
                # Analyze audio
                features = self.analyzer.analyze_audio(audio)
                
                # Update display
                self.display.update_display(features)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                pass  # No audio data available
            except Exception as e:
                print(f"\nError in audio processing: {e}")
    
    def start(self):
        """Start audio processing"""
        print("🔊 Real-time Audio Analysis (Console Version)")
        print("--------------------------------------------")
        print("Press Ctrl+C to stop.\n")
        
        self.running = True
        
        # Start audio processing thread
        self.process_thread = threading.Thread(target=self.process_audio_thread)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start audio input stream
        try:
            self.stream = sd.InputStream(
                channels=1,
                callback=self.audio_callback,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            print("Please check your audio device settings.")
            return
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping audio analysis...")
            self.stop()
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        print("\n🛑 Audio processing stopped.")


# Function to be exported for Unity integration later
def analyze_audio_buffer(audio_buffer, sample_rate):
    """Standalone function that can be wrapped in a DLL for Unity"""
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    features = analyzer.analyze_audio(audio_buffer)
    
    # Return a dictionary of features that can be accessed in Unity
    return {
        "tempo": features.tempo,
        "pitch_hz": features.pitch_hz,
        "pitch_note": features.pitch_note,
        "onset_detected": features.onset_detected,
        "onset_strength": features.onset_strength,
        "confidence": features.confidence
    }


# Main execution
if __name__ == "__main__":
    # Create and start audio processor
    processor = AudioProcessor()
    processor.start()