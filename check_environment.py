import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa

import sys
print(f"Cursor is using Python at: {sys.executable}")


def check_environment():
    print("\n=== Python Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    print("\n=== Package Versions ===")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Sounddevice version: {sd.__version__}")
    print(f"Librosa version: {librosa.__version__}")
    
    print("\n=== Audio Devices ===")
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    print("\nDefault input device:")
    print(sd.query_devices(kind='input'))
    
    print("\n=== Matplotlib Test ===")
    try:
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.title("Test Plot")
        plt.show(block=False)
        print("✓ Matplotlib visualization working")
    except Exception as e:
        print(f"✗ Matplotlib error: {e}")
    
    print("\n=== Audio Test ===")
    try:
        duration = 0.1  # seconds
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sd.play(audio, sample_rate)
        sd.wait()
        print("✓ Audio playback working")
    except Exception as e:
        print(f"✗ Audio error: {e}")

if __name__ == "__main__":
    check_environment()