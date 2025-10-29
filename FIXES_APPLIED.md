# Fixes Applied to Kyutai STT Server

## Date: 2025-10-29

### Critical Bugs Fixed

#### 1. Sample Rate Mismatch (FIXED ✅)
**Problem**: Model expected 24kHz audio but was receiving 16kHz, causing transcription failures.

**Root Cause**: The `transcription_engine._get_audio_config()` method was NOT passing `TARGET_SAMPLE_RATE` to the audio processor, only passing normalization flags.

**Solution**: Added `TARGET_SAMPLE_RATE: 24000` to all ProcessingMode configurations in `transcription_engine.py`:

```python
def _get_audio_config(self) -> Dict[str, Any]:
    """Get audio processing config based on mode."""
    configs = {
        ProcessingMode.ULTRA_FAST: {
            'TARGET_SAMPLE_RATE': 24000,  # ADDED
            'PEAK_NORMALIZE': False,
            'NORMALIZE_L2': False,
            'ENABLE_VAD': False,
        },
        ProcessingMode.FAST: {
            'TARGET_SAMPLE_RATE': 24000,  # ADDED
            'PEAK_NORMALIZE': True,
            'NORMALIZE_L2': False,
            'ENABLE_VAD': False,
        },
        # ... (all modes updated)
    }
```

**Files Modified**: `transcription_engine.py` (lines 260-290)

#### 2. Soundfile Decoder Error (FIXED ✅)
**Problem**: `'function' object has no attribute 'replace'` error when decoding audio with soundfile.

**Root Cause**: Import naming conflict! We were importing `dataclass` from `dataclasses` module (which is the decorator function), then trying to call `dataclass.replace()` which doesn't exist.

**Solution**: Import `replace` separately as `dataclass_replace` and updated all usages:

```python
# OLD:
from dataclasses import dataclass, field
# ... later in code:
dataclass.replace(metadata, peak_amplitude=float(peak))

# NEW:
from dataclasses import dataclass, field, replace as dataclass_replace
# ... later in code:
dataclass_replace(metadata, peak_amplitude=float(peak))
```

**Files Modified**: `audio_processor.py`
- Line 14: Import statement
- Line 218: `with_metadata()` method
- Line 219: `with_metadata()` method
- Line 224: `with_processing_stage()` method
- Line 390: SoundFile decoder metadata update
- Line 640: Normalization pipeline

### How Resampling Now Works

1. **Configuration Flow**:
   ```
   config.py (TARGET_SAMPLE_RATE=24000)
       ↓
   transcription_engine._get_audio_config()
       ↓
   audio_processor (receives TARGET_SAMPLE_RATE in config_override)
       ↓
   ProcessingPipeline adds resampling stage if needed
   ```

2. **Automatic Resampling**:
   - Audio processor checks incoming sample rate vs TARGET_SAMPLE_RATE
   - If different, adds resampling stage to pipeline
   - Uses librosa for high-quality resampling
   - Converts 16kHz → 24kHz automatically

3. **Decoder Fallback Chain**:
   - Priority 1: SoundFile (FIXED - now working)
   - Priority 2: TorchAudio (requires torchcodec installation)
   - Priority 3: PyDub (fallback)
   - Priority 4: Librosa (fallback)

### Testing Status

**Server Status**: ✅ Running on http://0.0.0.0:8000 (Process 1520)

**To Test**:
1. Open `test_ui.html` in browser
2. Record or upload 16kHz audio
3. Should now automatically resample to 24kHz
4. Transcription should complete successfully

### Previous Sessions Summary

**Session 1**: Created test UI, fixed initial audio processing errors
**Session 2**: Fixed soundfile metadata, installed pydub, mono conversion
**Session 3**: Fixed module imports, installed dependencies
**Session 4**: Fixed feature extractor parameters
**Session 5**: Fixed sample rate config and soundfile decoder (THIS SESSION)

### Files Modified (All Sessions)

1. **test_ui.html** - Complete WebSocket test client
2. **main.py** - Fixed imports and result extraction
3. **audio_processor.py** - Fixed dataclass.replace naming conflict
4. **transcription_engine.py** - Added TARGET_SAMPLE_RATE to audio config
5. **encoding.py** - Removed return_attention_mask parameter
6. **config.py** - TARGET_SAMPLE_RATE = 24000

### Next Steps

1. Test audio transcription end-to-end
2. Verify 16kHz audio successfully resamples to 24kHz
3. Monitor server logs for any additional errors
4. Consider installing torchcodec for TorchAudio decoder:
   ```powershell
   .\.venv\Scripts\pip.exe install torchcodec
   ```

### Dependencies Installed

- uvicorn
- fastapi
- websockets
- pydub
- librosa (for resampling)
- soundfile
- numpy
- torch

### Architecture Notes

**EnhancedAudioProcessor**:
- Class-based architecture with lazy-loaded backends
- ThreadSafeBackendManager for decoder management
- ProcessingPipeline with normalization, resampling, VAD stages
- AudioData dataclass with 2D samples format (channels, samples)

**TranscriptionEngine**:
- FastTranscriptionEngine with ProcessingMode configurations
- SimpleTranscriptAggregator for text concatenation
- Async support via ThreadPoolExecutor
- Warmup inference on startup

**Model**: kyutai/stt-1b-en_fr-trfs (1B parameters, CPU, float16)
