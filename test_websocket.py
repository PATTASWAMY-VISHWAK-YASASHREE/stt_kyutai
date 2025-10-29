"""
Test script to send audio to the WebSocket endpoint and verify audio processing.
"""
import asyncio
import websockets
import json
import base64
import numpy as np
import soundfile as sf
import io

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    print("=== WebSocket Audio Test ===")
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("✓ Connected!")
        
        # Test 1: Send stereo WAV audio
        print("\n--- Test 1: Stereo WAV (16kHz) ---")
        left = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * np.arange(16000) / 16000).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        
        buf = io.BytesIO()
        sf.write(buf, stereo, 16000, format='WAV')
        audio_bytes = buf.getvalue()
        
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        message = {"type": "audio", "data": audio_b64}
        
        print(f"Sending stereo WAV ({len(audio_bytes)} bytes)...")
        await websocket.send(json.dumps(message))
        
        response = await websocket.recv()
        result = json.loads(response)
        print(f"✓ Response: {result}")
        
        # Test 2: Send mono WAV audio
        print("\n--- Test 2: Mono WAV (24kHz) ---")
        mono_audio = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
        
        buf2 = io.BytesIO()
        sf.write(buf2, mono_audio, 24000, format='WAV')
        audio_bytes2 = buf2.getvalue()
        
        audio_b64_2 = base64.b64encode(audio_bytes2).decode('utf-8')
        message2 = {"type": "audio", "data": audio_b64_2}
        
        print(f"Sending mono WAV ({len(audio_bytes2)} bytes)...")
        await websocket.send(json.dumps(message2))
        
        response2 = await websocket.recv()
        result2 = json.loads(response2)
        print(f"✓ Response: {result2}")
        
        # Test 3: Send different sample rate
        print("\n--- Test 3: Mono WAV (48kHz) ---")
        high_sr_audio = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
        
        buf3 = io.BytesIO()
        sf.write(buf3, high_sr_audio, 48000, format='WAV')
        audio_bytes3 = buf3.getvalue()
        
        audio_b64_3 = base64.b64encode(audio_bytes3).decode('utf-8')
        message3 = {"type": "audio", "data": audio_b64_3}
        
        print(f"Sending 48kHz WAV ({len(audio_bytes3)} bytes)...")
        await websocket.send(json.dumps(message3))
        
        response3 = await websocket.recv()
        result3 = json.loads(response3)
        print(f"✓ Response: {result3}")
        
        print("\n✅ All WebSocket tests passed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
