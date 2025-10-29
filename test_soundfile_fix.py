"""Test audio decoding with the fixed code."""
import io
import base64
import numpy as np
import soundfile as sf

# Create a simple test audio
audio_data = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)

# Write to WAV in memory
buf = io.BytesIO()
sf.write(buf, audio_data, 16000, format='WAV')
audio_bytes = buf.getvalue()

print(f"Created test audio: {len(audio_bytes)} bytes")

# Test decoding
try:
    with sf.SoundFile(io.BytesIO(audio_bytes)) as handle:
        print(f"Format: {handle.format}")
        print(f"Sample rate: {handle.samplerate}")
        print(f"Channels: {handle.channels}")
        print(f"Has subtype_info: {hasattr(handle, 'subtype_info')}")
        
        if hasattr(handle, 'subtype_info'):
            subtype = handle.subtype_info
            print(f"  subtype_info type: {type(subtype)}")
            print(f"  subtype_info: {subtype}")
            print(f"  Has 'bits' attr: {hasattr(subtype, 'bits')}")
            if hasattr(subtype, 'bits'):
                print(f"  bits: {subtype.bits}")
        
        # Try the safe approach
        bit_depth = None
        if hasattr(handle, 'subtype_info'):
            subtype = handle.subtype_info
            if hasattr(subtype, 'bits'):
                bit_depth = subtype.bits
        
        print(f"\nSafe bit_depth extraction: {bit_depth}")
        print("✓ Fix will work!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
