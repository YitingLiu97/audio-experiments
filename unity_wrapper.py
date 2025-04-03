"""
Unity DLL Wrapper for Audio Analysis

This script demonstrates how to wrap the audio analysis functionality
into a DLL that can be used in Unity.

Requirements:
- cffi: For creating C-compatible function interfaces
- pythonnet: For .NET/C# integration
- pyinstaller: For creating a standalone executable

Install with:
pip install cffi pythonnet pyinstaller
"""

import os
import numpy as np
import cffi
from audio_analyzer import AudioAnalyzer, AudioFeatures, SAMPLE_RATE

# Create CFFI interface
ffi = cffi.FFI()

# Define C-compatible struct for audio features
ffi.cdef("""
    typedef struct {
        float tempo;
        float pitch_hz;
        char pitch_note[8];
        bool onset_detected;
        float onset_strength;
        float confidence;
    } AudioFeatures;
    
    AudioFeatures analyze_audio(float* buffer, int buffer_size, int sample_rate);
    void init_analyzer();
    void cleanup_analyzer();
"""
)

# Global analyzer instance
g_analyzer = None

# Implementation of C-compatible functions
@ffi.def_extern()
def analyze_audio(buffer_ptr, buffer_size, sample_rate):
    """Analyze audio buffer and return features"""
    global g_analyzer
    
    # Create analyzer if it doesn't exist
    if g_analyzer is None:
        g_analyzer = AudioAnalyzer(sample_rate=sample_rate)
    
    # Convert C buffer to numpy array
    buffer = np.frombuffer(
        ffi.buffer(buffer_ptr, buffer_size * 4),  # 4 bytes per float
        dtype=np.float32
    )
    
    # Analyze audio
    features = g_analyzer.analyze_audio(buffer)
    
    # Create C-compatible result struct
    result = ffi.new("AudioFeatures*")
    result.tempo = features.tempo
    result.pitch_hz = features.pitch_hz
    result.onset_detected = features.onset_detected
    result.onset_strength = features.onset_strength
    result.confidence = features.confidence
    
    # Convert pitch note string (safely)
    pitch_note = features.pitch_note[:7] if features.pitch_note else "Unknown"
    for i, c in enumerate(pitch_note):
        result.pitch_note[i] = ord(c)
    result.pitch_note[len(pitch_note)] = 0  # Null terminator
    
    return result[0]

@ffi.def_extern()
def init_analyzer():
    """Initialize the analyzer"""
    global g_analyzer
    g_analyzer = AudioAnalyzer(sample_rate=SAMPLE_RATE)

@ffi.def_extern()
def cleanup_analyzer():
    """Clean up the analyzer"""
    global g_analyzer
    g_analyzer = None


def build_dll():
    """Build the DLL for Unity integration"""
    # Create the shared library
    library = ffi.verify(
        """
        #include <stdbool.h>
        
        typedef struct {
            float tempo;
            float pitch_hz;
            char pitch_note[8];
            bool onset_detected;
            float onset_strength;
            float confidence;
        } AudioFeatures;
        
        extern AudioFeatures analyze_audio(float* buffer, int buffer_size, int sample_rate);
        extern void init_analyzer();
        extern void cleanup_analyzer();
        """,
        sources=[],  # No C sources needed, using Python implementations
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3'],
        modulename="audio_analysis"
    )
    
    print(f"DLL built successfully: {library._name}")
    return library._name


# Example of how to use the DLL from C#/Unity
CSHARP_EXAMPLE = """
using System;
using System.Runtime.InteropServices;

public class AudioAnalysis
{
    // Import functions from DLL
    [DllImport("audio_analysis")]
    private static extern void init_analyzer();
    
    [DllImport("audio_analysis")]
    private static extern void cleanup_analyzer();
    
    [DllImport("audio_analysis")]
    private static extern AudioFeatures analyze_audio(
        [In] float[] buffer, int buffer_size, int sample_rate);
    
    // Define structure matching the C structure
    [StructLayout(LayoutKind.Sequential)]
    public struct AudioFeatures
    {
        public float tempo;
        public float pitch_hz;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] pitch_note;
        [MarshalAs(UnmanagedType.I1)]
        public bool onset_detected;
        public float onset_strength;
        public float confidence;
        
        public string PitchNote
        {
            get
            {
                int length = 0;
                while (length < pitch_note.Length && pitch_note[length] != 0)
                    length++;
                return System.Text.Encoding.ASCII.GetString(pitch_note, 0, length);
            }
        }
    }
    
    // Unity MonoBehaviour example
    /*
    private void Start()
    {
        // Initialize the analyzer
        init_analyzer();
    }
    
    private void OnAudioFilterRead(float[] data, int channels)
    {
        // Convert stereo to mono if needed
        float[] monoData = new float[data.Length / channels];
        for (int i = 0; i < monoData.Length; i++)
        {
            monoData[i] = 0;
            for (int c = 0; c < channels; c++)
                monoData[i] += data[i * channels + c];
            monoData[i] /= channels;
        }
        
        // Analyze audio
        AudioFeatures features = analyze_audio(monoData, monoData.Length, AudioSettings.outputSampleRate);
        
        // Use the features
        Debug.Log($"Tempo: {features.tempo} BPM, Pitch: {features.PitchNote} ({features.pitch_hz} Hz)");
        if (features.onset_detected)
            Debug.Log("Onset detected!");
    }
    
    private void OnDestroy()
    {
        // Clean up the analyzer
        cleanup_analyzer();
    }
    */
}
"""


if __name__ == "__main__":
    print("Building DLL for Unity integration...")
    dll_path = build_dll()
    
    print("\nC# example for Unity integration:")
    print(CSHARP_EXAMPLE)
    
    print("\nTo create a standalone executable:")
    print("pyinstaller --onefile --hidden-import=cffi audio_wrapper.py")