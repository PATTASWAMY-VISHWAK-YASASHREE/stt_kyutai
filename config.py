# config.py

"""Centralized runtime configuration for the Kyutai STT stack."""

from dataclasses import dataclass
from typing import Literal, Optional


# --- Model configuration --------------------------------------------------------------------

# Default checkpoint favors low latency and memory usage while retaining multilingual coverage.
MODEL_ID: str = "kyutai/stt-1b-en_fr-trfs"

# Set to ``"cuda"`` or ``"cpu"`` to force a device; ``"auto"`` picks CUDA when available.
DEVICE_PREFERENCE: Literal["auto", "cpu", "cuda"] = "auto"

# Optional quantization flags (require bitsandbytes when enabled).
LOAD_IN_8BIT: bool = False
LOAD_IN_4BIT: bool = False

# Hint the loader to downcast weights for memory savings on supported hardware.
TORCH_DTYPE: Optional[str] = "auto"  # ``"auto"`` lets Transformers pick, else e.g. "float16".

# Enable torch.compile for slightly faster decoding on repeated inferences (PyTorch ≥2.1).
ENABLE_TORCH_COMPILE: bool = False

# Reduce RAM usage when loading large checkpoints on CPU-only hosts.
LOW_CPU_MEM_USAGE: bool = True

# Control generation behaviour.
MAX_NEW_TOKENS: int = 512
NO_REPEAT_NGRAM_SIZE: int = 3

# --- Audio preprocessing -------------------------------------------------------------------

TARGET_SAMPLE_RATE: int = 24000
TARGET_CHANNELS: int = 1  # Mono

# Energy-based VAD trimming reduces silence workload while keeping latency low.
ENABLE_VAD: bool = True
VAD_WINDOW_SECONDS: float = 0.03
VAD_HOP_SECONDS: float = 0.015
VAD_ENERGY_THRESHOLD_DB: float = -45.0
VAD_HANGOVER_SECONDS: float = 0.2
VAD_MIN_SPEECH_SECONDS: float = 0.25
VAD_MIN_SILENCE_SECONDS: float = 0.2

# Chunks keep memory bounded for long recordings while overlapping to avoid word drops.
CHUNK_LENGTH_SECONDS: float = 12.0
CHUNK_OVERLAP_SECONDS: float = 1.0

# Streaming settings define how often incremental transcripts are emitted.
STREAMING_CHUNK_SECONDS: float = 6.0
STREAMING_OVERLAP_SECONDS: float = 0.75
STREAMING_EMIT_EMPTY_UPDATES: bool = False
STREAMING_MIN_CHAR_DELTA: int = 6

# Normalisation ensures consistent loudness without clipping.
NORMALIZE_L2: bool = True
PEAK_NORMALIZE: bool = True

# --- Server ---------------------------------------------------------------------------------

SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000


@dataclass(frozen=True)
class GenerationOptions:
	"""Fine-grained controls that can be overridden per request."""

	max_new_tokens: int = 256  # Reduced from 512 to avoid warnings on short audio
	no_repeat_ngram_size: int = 3
	# Removed temperature and do_sample as they're not valid for this model


DEFAULT_GENERATION_OPTIONS = GenerationOptions()