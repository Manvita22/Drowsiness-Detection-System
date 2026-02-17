"""
Generate alarm sound files for the drowsiness detection system
"""

import numpy as np
import wave
import struct

def create_beep(filename, frequency, duration, sample_rate=44100):
    """Create a simple beep sound at given frequency"""
    num_samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.arange(num_samples)
    waveform = np.sin(2 * np.pi * frequency * t / sample_rate)
    
    # Add fade in and fade out to avoid clicks
    fade_samples = int(sample_rate * 0.05)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    
    # Convert to 16-bit audio
    waveform = (waveform * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())
    
    print(f"Created: {filename} ({frequency} Hz, {duration}s)")

def create_alert_sound(filename):
    """Create alert sound: two short beeps"""
    sample_rate = 44100
    
    # First beep
    t1 = np.arange(int(sample_rate * 0.2))
    beep1 = np.sin(2 * np.pi * 800 * t1 / sample_rate) * 0.5
    
    # Silence
    silence = np.zeros(int(sample_rate * 0.1))
    
    # Second beep
    t2 = np.arange(int(sample_rate * 0.2))
    beep2 = np.sin(2 * np.pi * 800 * t2 / sample_rate) * 0.5
    
    # Combine
    waveform = np.concatenate([beep1, silence, beep2])
    
    # Add fade
    fade_samples = int(sample_rate * 0.05)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    
    # Convert to 16-bit
    waveform = (waveform * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())
    
    print(f"Created: {filename} (alert sound)")

def create_drowsy_sound(filename):
    """Create drowsy sound: continuous alarm"""
    sample_rate = 44100
    duration = 1.0  # 1 second alarm
    
    # Create a repeating pattern of high frequency alert
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples)
    
    # Mix two frequencies for more alarming sound
    freq1 = 1000
    freq2 = 1500
    waveform = 0.4 * np.sin(2 * np.pi * freq1 * t / sample_rate)
    waveform += 0.3 * np.sin(2 * np.pi * freq2 * t / sample_rate)
    
    # Add fade
    fade_samples = int(sample_rate * 0.05)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    waveform[:fade_samples] *= fade_in
    waveform[-fade_samples:] *= fade_out
    
    # Convert to 16-bit
    waveform = (waveform * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())
    
    print(f"Created: {filename} (drowsy alarm sound)")

if __name__ == "__main__":
    print("Generating alarm sounds...")
    print()
    
    # Create alert sound (short warning)
    create_alert_sound("alarm_alert.wav")
    
    # Create drowsy sound (continuous alarm)
    create_drowsy_sound("alarm_drowsy.wav")
    
    print()
    print("✓ Alarm sounds created successfully!")
    print("  - alarm_alert.wav: Warning sound")
    print("  - alarm_drowsy.wav: Drowsy alarm sound")
    print()
    print("Now run: python drowsiness_detection.py")
