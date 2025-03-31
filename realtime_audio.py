import numpy as np
import sounddevice as sd
import librosa
import time
#import matplotlib.pyplot as plt

#plt.ion()  # Enable interactive mode for live plotting

# Constants
SAMPLERATE = 22050
BLOCKSIZE = 1024
HOP_LENGTH = 512
BUFFER_DURATION = 3  # seconds
BUFFER_SIZE = SAMPLERATE * BUFFER_DURATION
last_beat_time = 0
beat_interval = 0.5  # placeholder, updated by tempo

# Rolling buffer to store audio
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def process_audio(audio):
    global audio_buffer, last_beat_time, beat_interval
    audio_buffer = np.roll(audio_buffer, -len(audio))
    audio_buffer[-len(audio):] = audio
    y_buffer = np.copy(audio_buffer)
    y_chunk = np.copy(audio)

    # Get current time
    now = time.time()

    # Update tempo every 3 seconds
    if now - last_beat_time > 3:
        onset_env = librosa.onset.onset_strength(y=y_buffer, sr=SAMPLERATE)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=SAMPLERATE)
        beat_interval = 60.0 / float(tempo)
        last_beat_time = now  # reset only for tempo, not pitch
        print(f"🕒 Tempo updated: {float(tempo):.1f} BPM")

    # Detect pitch every chunk
    pitch_track = librosa.yin(y_chunk, fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C7'),
                                        sr=SAMPLERATE)
    current_pitch = float(pitch_track[-1])
    pitch_note = librosa.hz_to_note(current_pitch)

    # Trigger pitch update every beat interval
    if now - last_beat_time >= beat_interval:
        print(f"🎵 Pitch: {pitch_note:<4} ({current_pitch:.1f} Hz)")
        last_beat_time = now



def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio = indata[:, 0]  # mono
    process_audio(audio)

# Start stream
with sd.InputStream(channels=1, callback=audio_callback,
                    samplerate=SAMPLERATE, blocksize=BLOCKSIZE):
    print("🔊 Listening... press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 Stopped.")
