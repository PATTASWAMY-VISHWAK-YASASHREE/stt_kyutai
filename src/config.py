# config.py

"""Centralized runtime configuration for the Kyutai STT stack."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Any


# --- Model Configuration --------------------------------------------------------------------

# Kyutai STT model - optimized for low latency and multilingual support
MODEL_ID: str = "kyutai/stt-1b-en_fr-trfs"

# Device selection: "auto" picks CUDA when available, else CPU
DEVICE_PREFERENCE: Literal["auto", "cpu", "cuda", "mps"] = "auto"

# Quantization options (requires bitsandbytes library)
# ⚠️ Note: Quantization requires CUDA and may reduce quality
LOAD_IN_8BIT: bool = False
LOAD_IN_4BIT: bool = False

# Data type for model weights
# Options: "auto", "float32", "float16", "bfloat16"
# "auto" lets transformers choose based on device
TORCH_DTYPE: Optional[str] = "auto"

# Enable torch.compile for faster inference (PyTorch ≥2.0)
# First inference will be slower due to compilation
ENABLE_TORCH_COMPILE: bool = True

# Reduce RAM usage during model loading (recommended for large models)
LOW_CPU_MEM_USAGE: bool = True


# --- Audio Processing -----------------------------------------------------------------------

# Target sample rate (Kyutai model requires 24000 Hz)
TARGET_SAMPLE_RATE: int = 24000

# Target channels (1 = mono, 2 = stereo)
# Kyutai works best with mono audio
TARGET_CHANNELS: int = 1

# Audio normalization
NORMALIZE_L2: bool = False  # L2 normalization (recommended: False for speech)
PEAK_NORMALIZE: bool = True  # Peak normalization (recommended: True)
REMOVE_DC_OFFSET: bool = True  # Remove DC offset (recommended: True)
SOFT_CLIP: bool = False  # Soft clipping using tanh


# --- Voice Activity Detection (VAD) ---------------------------------------------------------

# Enable energy-based VAD to trim silence
ENABLE_VAD: bool = True

# VAD window settings (in seconds)
VAD_WINDOW_SECONDS: float = 0.03  # Analysis window size
VAD_HOP_SECONDS: float = 0.015  # Hop between windows

# Energy threshold in dB (lower = more sensitive)
# Typical range: -50 to -30 dB
VAD_ENERGY_THRESHOLD_DB: float = -45.0

# Hangover duration to extend speech segments (prevents cutting words)
VAD_HANGOVER_SECONDS: float = 0.2

# Minimum duration for speech and silence segments
VAD_MIN_SPEECH_SECONDS: float = 0.25
VAD_MIN_SILENCE_SECONDS: float = 0.2


# --- Chunking Configuration -----------------------------------------------------------------

# Maximum chunk length for processing long audio
# Kyutai model works well with 12-15 second chunks
CHUNK_LENGTH_SECONDS: float = 12.0

# Overlap between chunks to avoid cutting words
CHUNK_OVERLAP_SECONDS: float = 1.0

# Minimum chunk size (to avoid processing tiny segments)
MIN_CHUNK_SECONDS: float = 0.5


# --- Streaming Configuration ----------------------------------------------------------------

# Chunk size for streaming/incremental transcription
STREAMING_CHUNK_SECONDS: float = 6.0

# Overlap for streaming chunks
STREAMING_OVERLAP_SECONDS: float = 0.75

# Emit empty updates (if False, only emit when text changes)
STREAMING_EMIT_EMPTY_UPDATES: bool = False

# Minimum character change required to emit update
STREAMING_MIN_CHAR_DELTA: int = 6


# --- Generation Settings --------------------------------------------------------------------

# Maximum new tokens to generate
MAX_NEW_TOKENS: int = 448  # Kyutai optimal range: 256-512

# N-gram blocking to reduce repetition
NO_REPEAT_NGRAM_SIZE: int = 3

# Cache implementation (avoids deprecation warnings)
CACHE_IMPLEMENTATION: str = "static"

# Beam search settings
NUM_BEAMS: int = 1  # 1 = greedy decoding (fastest), >1 = beam search
LENGTH_PENALTY: float = 1.0  # >1 encourages longer sequences

# Early stopping
EARLY_STOPPING: bool = False


# --- Server Configuration -------------------------------------------------------------------

SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000

# Maximum concurrent WebSocket connections
MAX_CONNECTIONS: int = 100

# WebSocket timeout in seconds
WS_TIMEOUT: float = 300.0

# Maximum audio file size in MB
MAX_AUDIO_SIZE_MB: int = 10

# Enable CORS (for web interfaces)
ENABLE_CORS: bool = True
ALLOWED_ORIGINS: list = ["*"]  # ["http://localhost:3000"] for specific origins


# --- Performance & Caching ------------------------------------------------------------------

# Enable result caching (speeds up repeated requests)
ENABLE_CACHE: bool = True
CACHE_SIZE: int = 1000  # Maximum cached results

# Thread pool settings
MAX_WORKERS: int = 4  # Number of worker threads

# Batch processing
ENABLE_BATCHING: bool = True
BATCH_SIZE: int = 8


# --- Logging Configuration ------------------------------------------------------------------

LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# --- Advanced Settings ----------------------------------------------------------------------

# Skip model loading at startup (for testing/development)
SKIP_MODEL_LOAD: bool = False

# Processing mode (for transcription_engine.py)
# Options: "ULTRA_FAST", "FAST", "BALANCED", "QUALITY", "MAXIMUM_QUALITY"
PROCESSING_MODE: str = "FAST"


# --- Generation Options Dataclass -----------------------------------------------------------

@dataclass(frozen=True)
class GenerationOptions:
    """
    Fine-grained generation controls.
    These can be overridden per request.
    """
    # Token generation
    max_new_tokens: int = 448
    min_new_tokens: Optional[int] = None
    
    # Decoding strategy
    num_beams: int = 1
    do_sample: bool = False
    
    # Repetition control
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.0
    
    # Length control
    length_penalty: float = 1.0
    early_stopping: bool = False
    
    # Cache (avoids deprecation warning)
    cache_implementation: str = "static"
    
    # Additional controls
    temperature: Optional[float] = None  # Only used if do_sample=True
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, filtering None values.
        
        Returns:
            Dict with non-None values only
        """
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None and not k.startswith('_')
        }
    
    def merge(self, **kwargs) -> 'GenerationOptions':
        """
        Create new GenerationOptions with updated values.
        
        Args:
            **kwargs: Values to update
            
        Returns:
            New GenerationOptions instance
        """
        current = self.to_dict()
        current.update(kwargs)
        return GenerationOptions(**current)


# Default generation options instance
DEFAULT_GENERATION_OPTIONS = GenerationOptions()


# --- Helper Functions -----------------------------------------------------------------------

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dict with device information
    """
    import torch
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def validate_config() -> bool:
    """
    Validate configuration values.
    
    Returns:
        True if config is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate sample rate
    if TARGET_SAMPLE_RATE not in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
        raise ValueError(f"Invalid TARGET_SAMPLE_RATE: {TARGET_SAMPLE_RATE}")
    
    # Validate device preference
    if DEVICE_PREFERENCE not in ["auto", "cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid DEVICE_PREFERENCE: {DEVICE_PREFERENCE}")
    
    # Validate quantization
    if LOAD_IN_8BIT and LOAD_IN_4BIT:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization")
    
    # Validate VAD settings
    if VAD_WINDOW_SECONDS <= 0 or VAD_HOP_SECONDS <= 0:
        raise ValueError("VAD window and hop must be positive")
    
    if VAD_HOP_SECONDS >= VAD_WINDOW_SECONDS:
        raise ValueError("VAD hop must be smaller than window")
    
    # Validate chunk settings
    if CHUNK_LENGTH_SECONDS <= 0:
        raise ValueError("CHUNK_LENGTH_SECONDS must be positive")
    
    if CHUNK_OVERLAP_SECONDS >= CHUNK_LENGTH_SECONDS:
        raise ValueError("CHUNK_OVERLAP_SECONDS must be less than CHUNK_LENGTH_SECONDS")
    
    # Validate generation settings
    if MAX_NEW_TOKENS <= 0:
        raise ValueError("MAX_NEW_TOKENS must be positive")
    
    if NUM_BEAMS <= 0:
        raise ValueError("NUM_BEAMS must be positive")
    
    return True


def get_config_summary() -> str:
    """
    Get human-readable configuration summary.
    
    Returns:
        Formatted configuration string
    """
    import torch
    
    device_info = get_device_info()
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║                  KYUTAI STT CONFIGURATION                      ║
╠════════════════════════════════════════════════════════════════╣
║ Model:                                                          ║
║   ID: {MODEL_ID:<55} ║
║   Device: {DEVICE_PREFERENCE:<51} ║
║   Quantization: {'8-bit' if LOAD_IN_8BIT else '4-bit' if LOAD_IN_4BIT else 'None':<47} ║
║   Compile: {'Enabled' if ENABLE_TORCH_COMPILE else 'Disabled':<50} ║
╠════════════════════════════════════════════════════════════════╣
║ Audio Processing:                                              ║
║   Sample Rate: {TARGET_SAMPLE_RATE} Hz{'':<43} ║
║   VAD: {'Enabled' if ENABLE_VAD else 'Disabled':<54} ║
║   Normalization: Peak={PEAK_NORMALIZE}, L2={NORMALIZE_L2}{'':<30} ║
║   Chunk Length: {CHUNK_LENGTH_SECONDS}s{'':<46} ║
╠════════════════════════════════════════════════════════════════╣
║ Generation:                                                    ║
║   Max Tokens: {MAX_NEW_TOKENS:<47} ║
║   Beam Size: {NUM_BEAMS:<48} ║
║   No Repeat N-gram: {NO_REPEAT_NGRAM_SIZE:<41} ║
╠════════════════════════════════════════════════════════════════╣
║ Server:                                                        ║
║   Address: {SERVER_HOST}:{SERVER_PORT}{'':<45} ║
║   Max Connections: {MAX_CONNECTIONS:<42} ║
║   Cache: {'Enabled' if ENABLE_CACHE else 'Disabled':<50} ║
╠════════════════════════════════════════════════════════════════╣
║ Hardware:                                                      ║
║   CUDA Available: {device_info['cuda_available']:<43} ║
║   MPS Available: {device_info['mps_available']:<44} ║
{'║   GPU: ' + device_info.get('cuda_device_name', 'N/A')[:54] + ' ' * (54 - len(device_info.get('cuda_device_name', 'N/A')[:54])) + '║' if device_info['cuda_available'] else ''}
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


# --- Initialize Configuration ---------------------------------------------------------------

# Validate on import
try:
    validate_config()
except Exception as e:
    import warnings
    warnings.warn(f"Configuration validation failed: {e}")


# Export all config values
__all__ = [
    # Model
    'MODEL_ID', 'DEVICE_PREFERENCE', 'LOAD_IN_8BIT', 'LOAD_IN_4BIT',
    'TORCH_DTYPE', 'ENABLE_TORCH_COMPILE', 'LOW_CPU_MEM_USAGE',
    
    # Audio
    'TARGET_SAMPLE_RATE', 'TARGET_CHANNELS', 'NORMALIZE_L2', 'PEAK_NORMALIZE',
    'REMOVE_DC_OFFSET', 'SOFT_CLIP',
    
    # VAD
    'ENABLE_VAD', 'VAD_WINDOW_SECONDS', 'VAD_HOP_SECONDS',
    'VAD_ENERGY_THRESHOLD_DB', 'VAD_HANGOVER_SECONDS',
    'VAD_MIN_SPEECH_SECONDS', 'VAD_MIN_SILENCE_SECONDS',
    
    # Chunking
    'CHUNK_LENGTH_SECONDS', 'CHUNK_OVERLAP_SECONDS', 'MIN_CHUNK_SECONDS',
    
    # Streaming
    'STREAMING_CHUNK_SECONDS', 'STREAMING_OVERLAP_SECONDS',
    'STREAMING_EMIT_EMPTY_UPDATES', 'STREAMING_MIN_CHAR_DELTA',
    
    # Generation
    'MAX_NEW_TOKENS', 'NO_REPEAT_NGRAM_SIZE', 'CACHE_IMPLEMENTATION',
    'NUM_BEAMS', 'LENGTH_PENALTY', 'EARLY_STOPPING',
    'GenerationOptions', 'DEFAULT_GENERATION_OPTIONS',
    
    # Server
    'SERVER_HOST', 'SERVER_PORT', 'MAX_CONNECTIONS', 'WS_TIMEOUT',
    'MAX_AUDIO_SIZE_MB', 'ENABLE_CORS', 'ALLOWED_ORIGINS',
    
    # Performance
    'ENABLE_CACHE', 'CACHE_SIZE', 'MAX_WORKERS',
    'ENABLE_BATCHING', 'BATCH_SIZE',
    
    # Advanced
    'SKIP_MODEL_LOAD', 'PROCESSING_MODE', 'LOG_LEVEL', 'LOG_FORMAT',
    
    # Helpers
    'get_device_info', 'validate_config', 'get_config_summary',
]