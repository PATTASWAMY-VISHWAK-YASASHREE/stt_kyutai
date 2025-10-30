"""Diagnostic script to test audio processing and transcription."""
import numpy as np
import io
import wave
import base64
import json
from src.transcription_engine import FastTranscriptionEngine

# Global transcription engine
transcription_service = None

def generate_test_audio(duration_seconds=3, sample_rate=24000):
    """Generate a test audio file with speech-like characteristics."""
    # Generate time array
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Generate a mix of frequencies that resemble speech (200-4000 Hz)
    signal = (
        0.3 * np.sin(2 * np.pi * 250 * t) +  # Low frequency
        0.2 * np.sin(2 * np.pi * 500 * t) +  # Mid-low
        0.2 * np.sin(2 * np.pi * 1000 * t) + # Mid
        0.15 * np.sin(2 * np.pi * 2000 * t) + # High-mid
        0.15 * np.sin(2 * np.pi * 3000 * t)   # High
    )
    
    # Add amplitude modulation to simulate speech patterns
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    signal = signal * modulation
    
    # Normalize to -1 to 1 range
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Convert to int16
    signal_int16 = (signal * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes = 16 bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int16.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    print(f"✅ Generated {duration_seconds}s test audio: {len(wav_bytes)} bytes")
    return wav_bytes

def test_transcription_with_generated_audio():
    """Test transcription with generated audio."""
    print("\n🔬 Testing with generated audio...")
    
    # Generate test audio
    audio_bytes = generate_test_audio(duration_seconds=3, sample_rate=24000)
    
    # Test transcription
    try:
        result = transcription_service.transcribe(audio_bytes)
        transcription_text = result.text if hasattr(result, 'text') else str(result)
        print(f"✅ Transcription result: '{transcription_text}'")
        if transcription_text and transcription_text.strip() and transcription_text != '...':
            print("✅ Got non-empty transcription!")
        else:
            print("⚠️ Empty or punctuation-only transcription")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

def test_transcription_with_silent_audio():
    """Test transcription with silent audio."""
    print("\n🔇 Testing with silent audio...")
    
    duration = 2
    sample_rate = 24000
    
    # Generate silent audio (all zeros)
    signal = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    # Create WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    print(f"✅ Generated {duration}s silent audio: {len(wav_bytes)} bytes")
    
    # Test transcription
    try:
        result = transcription_service.transcribe(wav_bytes)
        transcription_text = result.text if hasattr(result, 'text') else str(result)
        print(f"📝 Result: '{transcription_text}'")
        if not transcription_text or transcription_text == '...':
            print("✅ Silent audio correctly produces empty result")
        else:
            print(f"⚠️ Silent audio produced: '{transcription_text}'")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")

def test_audio_with_varying_durations():
    """Test with different audio durations."""
    print("\n⏱️ Testing different durations...")
    
    for duration in [0.5, 1.0, 2.0, 3.0, 5.0]:
        audio_bytes = generate_test_audio(duration_seconds=duration, sample_rate=24000)
        try:
            result = transcription_service.transcribe(audio_bytes)
            transcription_text = result.text if hasattr(result, 'text') else str(result)
            print(f"  {duration}s audio -> '{transcription_text}' ({len(transcription_text)} chars)")
        except Exception as e:
            print(f"  {duration}s audio -> ERROR: {e}")

if __name__ == "__main__":
    print("🧪 Audio Transcription Diagnostic")
    print("=" * 60)
    
    # Initialize transcription service
    print("\n🔧 Initializing transcription service...")
    try:
        transcription_service = FastTranscriptionEngine()
        transcription_service.initialize()
        print("✅ Service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 1: Generated speech-like audio
    test_transcription_with_generated_audio()
    
    # Test 2: Silent audio
    test_transcription_with_silent_audio()
    
    # Test 3: Various durations
    test_audio_with_varying_durations()
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic complete!")
