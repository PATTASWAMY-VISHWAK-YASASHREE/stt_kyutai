"""Audio decoding, normalization, VAD trimming, and chunking utilities."""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import warnings
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as dataclass_replace
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from . import config

# Configure logging with structured format
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type aliases
AudioArray = NDArray[np.float32]
AudioSegment = Tuple[int, int]
T = TypeVar('T')

# Constants
MAX_CHANNELS = 32
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
EPSILON = 1e-12


class AudioFormat(Enum):
    """Audio format classifications."""
    MONO = auto()
    STEREO = auto()
    SURROUND_5_1 = auto()
    SURROUND_7_1 = auto()
    MULTI_CHANNEL = auto()
    
    @classmethod
    def from_channels(cls, channels: int) -> AudioFormat:
        """Determine format from channel count."""
        mapping = {
            1: cls.MONO,
            2: cls.STEREO,
            6: cls.SURROUND_5_1,
            8: cls.SURROUND_7_1,
        }
        return mapping.get(channels, cls.MULTI_CHANNEL)


class ProcessingStage(Enum):
    """Audio processing pipeline stages."""
    DECODE = auto()
    NORMALIZE = auto()
    RESAMPLE = auto()
    VAD = auto()
    CHUNK = auto()


class AudioException(Exception):
    """Base exception for audio processing errors."""
    pass


class AudioDecodingError(AudioException):
    """Raised when audio cannot be decoded."""
    pass


class AudioProcessingError(AudioException):
    """Raised during audio processing operations."""
    pass


class AudioValidationError(AudioException):
    """Raised when audio data fails validation."""
    pass


@dataclass(frozen=True)
class ProcessingMetrics:
    """Metrics collected during audio processing."""
    stage: ProcessingStage
    duration_ms: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioMetadata:
    """Extended audio metadata."""
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    bit_depth: Optional[int] = None
    file_size: Optional[int] = None
    peak_amplitude: Optional[float] = None
    rms_level: Optional[float] = None
    dc_offset: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class AudioData:
    """Immutable container for audio data with comprehensive metadata."""
    samples: AudioArray  # Shape: (channels, samples)
    sample_rate: int
    channels: int
    duration: float
    metadata: AudioMetadata = field(default_factory=AudioMetadata)
    processing_history: Tuple[ProcessingMetrics, ...] = field(default_factory=tuple)
    
    # Class-level cache for computed properties
    _property_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Validate audio data on initialization."""
        # Basic shape validation
        if self.samples.ndim != 2:
            raise AudioValidationError(
                f"Audio must be 2D (channels, samples), got shape {self.samples.shape}"
            )
        
        # Channel validation
        if self.channels != self.samples.shape[0]:
            raise AudioValidationError(
                f"Channel count mismatch: {self.channels} != {self.samples.shape[0]}"
            )
        
        if not 1 <= self.channels <= MAX_CHANNELS:
            raise AudioValidationError(
                f"Invalid channel count: {self.channels}. Must be between 1 and {MAX_CHANNELS}"
            )
        
        # Sample rate validation
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise AudioValidationError(
                f"Invalid sample rate: {self.sample_rate}. Must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}"
            )
        
        # Duration validation
        expected_duration = self.samples.shape[1] / self.sample_rate
        if abs(self.duration - expected_duration) > 0.001:  # 1ms tolerance
            object.__setattr__(self, 'duration', expected_duration)
        
        # Ensure samples are contiguous and float32
        if not self.samples.flags['C_CONTIGUOUS']:
            object.__setattr__(self, 'samples', np.ascontiguousarray(self.samples, dtype=np.float32))
    
    @property
    def format(self) -> AudioFormat:
        """Get audio format based on channel count."""
        return AudioFormat.from_channels(self.channels)
    
    @property
    def num_samples(self) -> int:
        """Get number of samples per channel."""
        return self.samples.shape[1]
    
    @property
    def byte_size(self) -> int:
        """Estimated memory usage in bytes."""
        return self.samples.nbytes
    
    @lru_cache(maxsize=1)
    def get_channel_statistics(self) -> Dict[int, Dict[str, float]]:
        """Compute statistics for each channel."""
        stats = {}
        for ch in range(self.channels):
            channel_data = self.samples[ch]
            stats[ch] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'rms': float(np.sqrt(np.mean(channel_data**2))),
                'peak': float(np.abs(channel_data).max()),
            }
        return stats
    
    def with_metadata(self, **kwargs) -> AudioData:
        """Create new AudioData with updated metadata."""
        new_metadata = dataclass_replace(self.metadata, **kwargs)
        return dataclass_replace(self, metadata=new_metadata)
    
    def with_processing_stage(self, metrics: ProcessingMetrics) -> AudioData:
        """Add processing stage to history."""
        new_history = self.processing_history + (metrics,)
        return dataclass_replace(self, processing_history=new_history)


class BackendProtocol(Protocol):
    """Protocol for audio backend implementations."""
    
    @property
    def name(self) -> str:
        """Backend name."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if backend is available."""
        ...
    
    def decode(self, audio_bytes: bytes) -> AudioData:
        """Decode audio bytes."""
        ...


class ThreadSafeBackendManager:
    """Thread-safe manager for audio backends with lazy loading."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._backends: Dict[str, Optional[Any]] = {}
        self._import_errors: Dict[str, str] = {}
    
    def _try_import(self, module_name: str, package_name: Optional[str] = None) -> Optional[Any]:
        """Thread-safe module import with caching."""
        with self._lock:
            if module_name in self._backends:
                return self._backends[module_name]
            
            try:
                if package_name:
                    module = __import__(module_name)
                    for attr in package_name.split('.'):
                        module = getattr(module, attr)
                else:
                    module = __import__(module_name)
                
                self._backends[module_name] = module
                logger.debug(f"Successfully imported {module_name}")
                return module
                
            except ImportError as e:
                self._backends[module_name] = None
                self._import_errors[module_name] = str(e)
                logger.debug(f"Failed to import {module_name}: {e}")
                return None
    
    @property
    def soundfile(self) -> Optional[Any]:
        """Get soundfile module if available."""
        return self._try_import('soundfile')
    
    @property
    def torchaudio(self) -> Optional[Any]:
        """Get torchaudio module if available."""
        return self._try_import('torchaudio')
    
    @property
    def pydub(self) -> Optional[Any]:
        """Get pydub AudioSegment if available."""
        module = self._try_import('pydub')
        return getattr(module, 'AudioSegment', None) if module else None
    
    @property
    def librosa(self) -> Optional[Any]:
        """Get librosa module if available."""
        return self._try_import('librosa')
    
    @property
    def available_backends(self) -> List[str]:
        """List available backend names."""
        with self._lock:
            return [name for name, module in self._backends.items() if module is not None]


# Singleton backend manager
_backend_manager = ThreadSafeBackendManager()


class AudioDecoder(ABC):
    """Abstract base class for audio decoders."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Decoder name for logging."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority for decoder selection (lower is higher priority)."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this decoder is available."""
        pass
    
    @abstractmethod
    def decode(self, audio_bytes: bytes) -> AudioData:
        """Decode audio bytes into AudioData."""
        pass
    
    def __lt__(self, other: AudioDecoder) -> bool:
        """Allow sorting by priority."""
        return self.priority < other.priority


class SoundFileDecoder(AudioDecoder):
    """High-quality decoder using soundfile backend."""
    
    name = "soundfile"
    priority = 1
    
    def is_available(self) -> bool:
        return _backend_manager.soundfile is not None
    
    def decode(self, audio_bytes: bytes) -> AudioData:
        sf = _backend_manager.soundfile
        if sf is None:
            raise AudioDecodingError(f"{self.name} backend not available")
        
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as handle:
                audio = handle.read(dtype='float32')
                sample_rate = handle.samplerate
                channels = handle.channels
                
                # Get additional metadata with safe attribute access
                bit_depth = None
                if hasattr(handle, 'subtype_info'):
                    subtype = handle.subtype_info
                    if hasattr(subtype, 'bits'):
                        bit_depth = subtype.bits
                
                # Get codec/format safely
                codec = None
                if hasattr(handle, 'format'):
                    # handle.format might be a property or attribute
                    fmt = handle.format
                    codec = str(fmt) if fmt is not None else None
                
                metadata = AudioMetadata(
                    codec=codec,
                    bit_depth=bit_depth,
                    file_size=len(audio_bytes),
                )
                
                # Ensure 2D format (channels, samples)
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)
                else:
                    # soundfile returns (samples, channels), transpose to (channels, samples)
                    audio = audio.T
                
                # Compute additional metadata
                peak = np.abs(audio).max()
                rms = np.sqrt(np.mean(audio**2))
                
                metadata = dataclass_replace(
                    metadata,
                    peak_amplitude=float(peak),
                    rms_level=float(rms),
                )
                
                duration = audio.shape[1] / sample_rate
                
                return AudioData(
                    samples=np.ascontiguousarray(audio, dtype=np.float32),
                    sample_rate=sample_rate,
                    channels=channels,
                    duration=duration,
                    metadata=metadata,
                )
                
        except Exception as e:
            raise AudioDecodingError(f"{self.name} decoding failed: {e}") from e


class TorchAudioDecoder(AudioDecoder):
    """GPU-accelerated decoder using torchaudio."""
    
    name = "torchaudio"
    priority = 2
    
    def is_available(self) -> bool:
        return _backend_manager.torchaudio is not None
    
    def decode(self, audio_bytes: bytes) -> AudioData:
        torchaudio = _backend_manager.torchaudio
        if torchaudio is None:
            raise AudioDecodingError(f"{self.name} backend not available")
        
        try:
            # Create BytesIO object for torchaudio
            audio_io = io.BytesIO(audio_bytes)
            
            # Load audio
            tensor, sr = torchaudio.load(audio_io)
            audio = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
            
            # Get metadata
            info = torchaudio.info(io.BytesIO(audio_bytes))
            metadata = AudioMetadata(
                codec=info.encoding if hasattr(info, 'encoding') else None,
                bitrate=info.bits_per_sample if hasattr(info, 'bits_per_sample') else None,
                file_size=len(audio_bytes),
            )
            
            # Audio is already in (channels, samples) format
            channels = audio.shape[0]
            duration = audio.shape[1] / sr
            
            return AudioData(
                samples=np.ascontiguousarray(audio, dtype=np.float32),
                sample_rate=int(sr),
                channels=channels,
                duration=duration,
                metadata=metadata,
            )
            
        except Exception as e:
            raise AudioDecodingError(f"{self.name} decoding failed: {e}") from e


class PyDubDecoder(AudioDecoder):
    """Versatile decoder using pydub (requires ffmpeg)."""
    
    name = "pydub"
    priority = 3
    
    def is_available(self) -> bool:
        return _backend_manager.pydub is not None
    
    def decode(self, audio_bytes: bytes) -> AudioData:
        AudioSegment = _backend_manager.pydub
        if AudioSegment is None:
            raise AudioDecodingError(f"{self.name} backend not available")
        
        try:
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Extract samples
            raw = segment.get_array_of_samples()
            arr = np.array(raw, dtype=np.float32) / (2**15 if segment.sample_width == 2 else 2**(8*segment.sample_width - 1))
            
            channels = segment.channels
            # Reshape to (channels, samples)
            if channels > 1:
                arr = arr.reshape(-1, channels).T
            else:
                arr = arr.reshape(1, -1)
            
            # Create metadata
            metadata = AudioMetadata(
                codec=getattr(segment, 'format', None),
                bitrate=segment.frame_rate * segment.sample_width * 8 * channels,
                bit_depth=segment.sample_width * 8,
                file_size=len(audio_bytes),
            )
            
            duration = len(segment) / 1000.0  # Convert ms to seconds
            
            return AudioData(
                samples=np.ascontiguousarray(arr, dtype=np.float32),
                sample_rate=segment.frame_rate,
                channels=channels,
                duration=duration,
                metadata=metadata,
            )
            
        except Exception as e:
            raise AudioDecodingError(f"{self.name} decoding failed: {e}") from e


class LibrosaDecoder(AudioDecoder):
    """Scientific audio decoder using librosa."""
    
    name = "librosa"
    priority = 4
    
    def is_available(self) -> bool:
        return _backend_manager.librosa is not None
    
    def decode(self, audio_bytes: bytes) -> AudioData:
        librosa = _backend_manager.librosa
        if librosa is None:
            raise AudioDecodingError(f"{self.name} backend not available")
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
            
            # Ensure 2D format
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            
            channels = audio.shape[0]
            duration = audio.shape[1] / sr
            
            metadata = AudioMetadata(file_size=len(audio_bytes))
            
            return AudioData(
                samples=np.ascontiguousarray(audio, dtype=np.float32),
                sample_rate=sr,
                channels=channels,
                duration=duration,
                metadata=metadata,
            )
            
        except Exception as e:
            raise AudioDecodingError(f"{self.name} decoding failed: {e}") from e


class ProcessingPipeline:
    """Configurable audio processing pipeline with metrics collection."""
    
    def __init__(
        self,
        stages: Optional[List[Callable[[AudioData], AudioData]]] = None,
        collect_metrics: bool = True,
    ):
        self.stages = stages or []
        self.collect_metrics = collect_metrics
        self._metrics: Deque[ProcessingMetrics] = deque(maxlen=1000)
    
    def add_stage(self, stage: Callable[[AudioData], AudioData]) -> ProcessingPipeline:
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        return self
    
    def process(self, audio_data: AudioData) -> AudioData:
        """Process audio through all stages."""
        result = audio_data
        
        for stage in self.stages:
            if self.collect_metrics:
                import time
                start = time.perf_counter()
                input_shape = result.samples.shape
                
                result = stage(result)
                
                duration_ms = (time.perf_counter() - start) * 1000
                metrics = ProcessingMetrics(
                    stage=ProcessingStage.NORMALIZE,  # This should be dynamic
                    duration_ms=duration_ms,
                    input_shape=input_shape,
                    output_shape=result.samples.shape,
                )
                self._metrics.append(metrics)
                result = result.with_processing_stage(metrics)
            else:
                result = stage(result)
        
        return result
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of processing metrics."""
        if not self._metrics:
            return {}
        
        total_time = sum(m.duration_ms for m in self._metrics)
        return {
            'total_processing_time_ms': total_time,
            'num_stages': len(self.stages),
            'average_time_per_stage_ms': total_time / len(self._metrics) if self._metrics else 0,
            'recent_metrics': list(self._metrics)[-10:],  # Last 10 metrics
        }


class AudioNormalizer:
    """Advanced audio normalization with multiple strategies."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or {}
    
    def normalize(self, audio_data: AudioData) -> AudioData:
        """Apply configured normalization strategies."""
        samples = audio_data.samples.copy()
        
        # DC offset removal
        if self.config.get('REMOVE_DC_OFFSET', False):
            samples = self._remove_dc_offset(samples)
        
        # Peak normalization
        if self.config.get('PEAK_NORMALIZE', False):
            samples = self._peak_normalize(samples)
        
        # RMS normalization
        if self.config.get('RMS_NORMALIZE', False):
            target_rms = self.config.get('TARGET_RMS', 0.1)
            samples = self._rms_normalize(samples, target_rms)
        
        # L2 normalization
        if self.config.get('NORMALIZE_L2', False):
            samples = self._l2_normalize(samples)
        
        # LUFS normalization (if available)
        if self.config.get('LUFS_NORMALIZE', False) and SCIPY_AVAILABLE:
            target_lufs = self.config.get('TARGET_LUFS', -23)
            samples = self._lufs_normalize(samples, audio_data.sample_rate, target_lufs)
        
        # Soft clipping
        if self.config.get('SOFT_CLIP', False):
            samples = self._soft_clip(samples)
        else:
            samples = np.clip(samples, -1.0, 1.0)
        
        return dataclass_replace(audio_data, samples=samples)
    
    @staticmethod
    def _remove_dc_offset(samples: AudioArray) -> AudioArray:
        """Remove DC offset from each channel."""
        for ch in range(samples.shape[0]):
            samples[ch] -= np.mean(samples[ch])
        return samples
    
    @staticmethod
    def _peak_normalize(samples: AudioArray) -> AudioArray:
        """Peak normalize each channel independently."""
        for ch in range(samples.shape[0]):
            peak = np.abs(samples[ch]).max()
            if peak > EPSILON:
                samples[ch] = samples[ch] / peak
        return samples
    
    @staticmethod
    def _rms_normalize(samples: AudioArray, target_rms: float) -> AudioArray:
        """RMS normalize each channel."""
        for ch in range(samples.shape[0]):
            rms = np.sqrt(np.mean(samples[ch]**2))
            if rms > EPSILON:
                samples[ch] = samples[ch] * (target_rms / rms)
        return samples
    
    @staticmethod
    def _l2_normalize(samples: AudioArray) -> AudioArray:
        """L2 normalize each channel."""
        for ch in range(samples.shape[0]):
            l2_norm = np.linalg.norm(samples[ch])
            if l2_norm > EPSILON:
                samples[ch] = samples[ch] / l2_norm
        return samples
    
    @staticmethod
    def _lufs_normalize(samples: AudioArray, sample_rate: int, target_lufs: float) -> AudioArray:
        """LUFS (EBU R128) normalization."""
        # Simplified LUFS calculation - would need proper implementation
        # This is a placeholder for the concept
        return samples
    
    @staticmethod
    def _soft_clip(samples: AudioArray, threshold: float = 0.95) -> AudioArray:
        """Apply soft clipping using tanh."""
        mask = np.abs(samples) > threshold
        samples[mask] = np.sign(samples[mask]) * (
            threshold + np.tanh((np.abs(samples[mask]) - threshold) * 2) * (1 - threshold)
        )
        return samples


class AudioResampler:
    """High-quality audio resampling with multiple backends."""
    
    def __init__(self, method: str = 'auto'):
        self.method = method
    
    def resample(self, audio_data: AudioData, target_rate: int) -> AudioData:
        """Resample audio to target sample rate."""
        if audio_data.sample_rate == target_rate:
            return audio_data
        
        # Try different resampling methods in order of quality
        resamplers = [
            self._resample_scipy,
            self._resample_torchaudio,
            self._resample_numpy,
        ]
        
        for resampler in resamplers:
            try:
                return resampler(audio_data, target_rate)
            except Exception as e:
                logger.debug(f"Resampling with {resampler.__name__} failed: {e}")
        
        raise AudioProcessingError("All resampling methods failed")
    
    def _resample_scipy(self, audio_data: AudioData, target_rate: int) -> AudioData:
        """High-quality resampling using scipy."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy not available")
        
        samples = audio_data.samples
        source_rate = audio_data.sample_rate
        
        # Calculate resampling ratio
        ratio = target_rate / source_rate
        new_length = int(samples.shape[1] * ratio)
        
        # Resample each channel
        resampled = np.zeros((audio_data.channels, new_length), dtype=np.float32)
        for ch in range(audio_data.channels):
            resampled[ch] = signal.resample(samples[ch], new_length)
        
        return AudioData(
            samples=resampled,
            sample_rate=target_rate,
            channels=audio_data.channels,
            duration=new_length / target_rate,
            metadata=audio_data.metadata,
            processing_history=audio_data.processing_history,
        )
    
    def _resample_torchaudio(self, audio_data: AudioData, target_rate: int) -> AudioData:
        """GPU-accelerated resampling using torchaudio."""
        torchaudio = _backend_manager.torchaudio
        if torchaudio is None:
            raise RuntimeError("torchaudio not available")
        
        import torch
        from torchaudio.functional import resample as ta_resample
        
        samples = audio_data.samples
        source_rate = audio_data.sample_rate
        
        # Convert to tensor
        tensor = torch.from_numpy(samples)
        
        # Resample
        resampled = ta_resample(tensor, source_rate, target_rate)
        resampled_np = resampled.cpu().numpy() if resampled.is_cuda else resampled.numpy()
        
        new_duration = resampled_np.shape[1] / target_rate
        
        return AudioData(
            samples=np.ascontiguousarray(resampled_np, dtype=np.float32),
            sample_rate=target_rate,
            channels=audio_data.channels,
            duration=new_duration,
            metadata=audio_data.metadata,
            processing_history=audio_data.processing_history,
        )
    
    def _resample_numpy(self, audio_data: AudioData, target_rate: int) -> AudioData:
        """Basic resampling using NumPy linear interpolation."""
        samples = audio_data.samples
        source_rate = audio_data.sample_rate
        
        # Calculate new length
        duration = samples.shape[1] / source_rate
        target_length = int(round(duration * target_rate))
        
        if target_length <= 0:
            return AudioData(
                samples=np.zeros((audio_data.channels, 0), dtype=np.float32),
                sample_rate=target_rate,
                channels=audio_data.channels,
                duration=0.0,
                metadata=audio_data.metadata,
                processing_history=audio_data.processing_history,
            )
        
        # Resample each channel using linear interpolation
        resampled = np.zeros((audio_data.channels, target_length), dtype=np.float32)
        old_indices = np.arange(samples.shape[1])
        new_indices = np.linspace(0, samples.shape[1] - 1, target_length)
        
        for ch in range(audio_data.channels):
            resampled[ch] = np.interp(new_indices, old_indices, samples[ch])
        
        return AudioData(
            samples=resampled,
            sample_rate=target_rate,
            channels=audio_data.channels,
            duration=target_length / target_rate,
            metadata=audio_data.metadata,
            processing_history=audio_data.processing_history,
        )


class EnhancedAudioProcessor:
    """Production-ready audio processor with advanced features."""
    
    def __init__(
        self,
        config_override: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        max_workers: int = 4,
    ):
        """Initialize processor with configuration."""
        self.config = self._merge_config(config_override)
        self.use_cache = use_cache
        self.max_workers = max_workers
        
        # Initialize components
        self._decoders = sorted([
            SoundFileDecoder(),
            TorchAudioDecoder(),
            PyDubDecoder(),
            LibrosaDecoder(),
        ])
        
        self._normalizer = AudioNormalizer(self.config)
        self._resampler = AudioResampler()
        
        # Cache for processed audio
        self._cache: Dict[str, AudioData] = {}
        self._cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self._validate_config()
    
    def _merge_config(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        base_config = {}
        
        # Extract config attributes
        for key in dir(config):
            if not key.startswith('_'):
                base_config[key] = getattr(config, key)
        
        if override:
            base_config.update(override)
        
        return base_config
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = ['TARGET_SAMPLE_RATE', 'CHUNK_LENGTH_SECONDS']
        for key in required:
            if key not in self.config:
                raise AudioValidationError(f"Missing required config: {key}")
        
        if not MIN_SAMPLE_RATE <= self.config['TARGET_SAMPLE_RATE'] <= MAX_SAMPLE_RATE:
            raise AudioValidationError(
                f"Invalid TARGET_SAMPLE_RATE: {self.config['TARGET_SAMPLE_RATE']}"
            )
    
    def _get_cache_key(self, audio_bytes: bytes) -> str:
        """Generate cache key from audio bytes."""
        return hashlib.sha256(audio_bytes).hexdigest()
    
    def decode_audio(self, audio_bytes: bytes, use_cache: Optional[bool] = None) -> AudioData:
        """Decode audio with caching support."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(audio_bytes)
            with self._cache_lock:
                if cache_key in self._cache:
                    logger.debug("Cache hit for audio decoding")
                    return self._cache[cache_key]
        
        # Try decoders in priority order
        errors = []
        for decoder in self._decoders:
            if not decoder.is_available():
                continue
            
            try:
                audio_data = decoder.decode(audio_bytes)
                
                # Cache result
                if use_cache:
                    with self._cache_lock:
                        self._cache[cache_key] = audio_data
                
                logger.debug(f"Successfully decoded with {decoder.name}")
                return audio_data
                
            except AudioDecodingError as e:
                # Convert exception to string properly
                error_str = str(e)
                errors.append(f"{decoder.name} decoding failed: {error_str}")
                logger.debug(f"Decoder {decoder.name} failed: {error_str}")
        
        # All decoders failed
        error_msg = "Unable to decode audio with any available backend"
        if errors:
            error_msg += f". Errors: {'; '.join(errors)}"
        raise AudioDecodingError(error_msg)
    
    def process_audio(
        self,
        audio_bytes: bytes,
        target_channels: Optional[int] = None,
        preserve_original: bool = False,
    ) -> AudioData:
        """Complete audio processing pipeline."""
        # Decode
        audio_data = self.decode_audio(audio_bytes)
        
        # Channel conversion if requested
        if target_channels is not None and target_channels != audio_data.channels:
            audio_data = self._convert_channels(audio_data, target_channels)
        
        # Create processing pipeline
        pipeline = ProcessingPipeline(collect_metrics=True)
        
        # Add normalization stage
        if any(self.config.get(k, False) for k in [
            'PEAK_NORMALIZE', 'RMS_NORMALIZE', 'NORMALIZE_L2', 'REMOVE_DC_OFFSET'
        ]):
            pipeline.add_stage(self._normalizer.normalize)
        
        # Add resampling stage
        target_rate = self.config.get('TARGET_SAMPLE_RATE', audio_data.sample_rate)
        if target_rate != audio_data.sample_rate:
            pipeline.add_stage(lambda ad: self._resampler.resample(ad, target_rate))
        
        # Add VAD stage if enabled
        if self.config.get('ENABLE_VAD', False):
            pipeline.add_stage(self._apply_vad)
        
        # Process through pipeline
        result = pipeline.process(audio_data)
        
        # Log metrics
        metrics = pipeline.get_metrics_summary()
        if metrics:
            logger.debug(f"Processing metrics: {metrics}")
        
        return result
    
    def _convert_channels(self, audio_data: AudioData, target_channels: int) -> AudioData:
        """Convert audio to target number of channels."""
        if target_channels == audio_data.channels:
            return audio_data
        
        samples = audio_data.samples
        
        if target_channels == 1:
            # Convert to mono by averaging channels
            mono = np.mean(samples, axis=0, keepdims=True)
            new_samples = mono
        
        elif target_channels == 2 and audio_data.channels == 1:
            # Convert mono to stereo by duplicating
            new_samples = np.repeat(samples, 2, axis=0)
        
        else:
            # More complex conversions would go here
            raise AudioProcessingError(
                f"Channel conversion from {audio_data.channels} to {target_channels} not implemented"
            )
        
        return AudioData(
            samples=new_samples,
            sample_rate=audio_data.sample_rate,
            channels=target_channels,
            duration=audio_data.duration,
            metadata=audio_data.metadata,
            processing_history=audio_data.processing_history,
        )
    
    def _apply_vad(self, audio_data: AudioData) -> AudioData:
        """Apply Voice Activity Detection."""
        # Use mono mix for VAD detection
        mono_mix = np.mean(audio_data.samples, axis=0)
        
        segments = voice_activity_segments(
            mono_mix,
            audio_data.sample_rate,
            window_seconds=self.config.get('VAD_WINDOW_SECONDS', 0.02),
            hop_seconds=self.config.get('VAD_HOP_SECONDS', 0.01),
            energy_threshold_db=self.config.get('VAD_ENERGY_THRESHOLD_DB', -40),
            hangover_seconds=self.config.get('VAD_HANGOVER_SECONDS', 0.1),
            min_speech_seconds=self.config.get('VAD_MIN_SPEECH_SECONDS', 0.1),
            min_silence_seconds=self.config.get('VAD_MIN_SILENCE_SECONDS', 0.1),
        )
        
        if not segments:
            logger.warning("VAD detected no speech")
            return audio_data
        
        # Trim all channels based on detected segments
        start, end = segments[0][0], segments[-1][1]
        trimmed_samples = audio_data.samples[:, start:end]
        
        return AudioData(
            samples=trimmed_samples,
            sample_rate=audio_data.sample_rate,
            channels=audio_data.channels,
            duration=trimmed_samples.shape[1] / audio_data.sample_rate,
            metadata=audio_data.metadata,
            processing_history=audio_data.processing_history,
        )
    
    def chunk_audio(
        self,
        audio_data: AudioData,
        chunk_seconds: Optional[float] = None,
        overlap_seconds: Optional[float] = None,
        min_chunk_seconds: Optional[float] = None,
    ) -> Generator[AudioData, None, None]:
        """Generate overlapping chunks with size constraints."""
        chunk_seconds = chunk_seconds or self.config.get('CHUNK_LENGTH_SECONDS', 30.0)
        overlap_seconds = overlap_seconds or self.config.get('CHUNK_OVERLAP_SECONDS', 0.0)
        min_chunk_seconds = min_chunk_seconds or chunk_seconds / 2
        
        if chunk_seconds <= 0:
            yield audio_data
            return
        
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate
        chunk_size = int(chunk_seconds * sample_rate)
        min_chunk_size = int(min_chunk_seconds * sample_rate)
        
        stride = max(1, chunk_size - int(overlap_seconds * sample_rate))
        total_samples = samples.shape[1]
        
        for start in range(0, total_samples, stride):
            end = min(start + chunk_size, total_samples)
            
            # Skip if chunk is too small (except for the last chunk if it's all that's left)
            if end - start < min_chunk_size and end < total_samples:
                continue
            
            chunk_samples = samples[:, start:end]
            
            yield AudioData(
                samples=chunk_samples,
                sample_rate=sample_rate,
                channels=audio_data.channels,
                duration=chunk_samples.shape[1] / sample_rate,
                metadata=audio_data.metadata,
                processing_history=audio_data.processing_history,
            )
            
            if end >= total_samples:
                break
    
    def process_parallel(
        self,
        audio_bytes_list: List[bytes],
        **kwargs
    ) -> List[AudioData]:
        """Process multiple audio files in parallel."""
        futures = [
            self._executor.submit(self.process_audio, audio_bytes, **kwargs)
            for audio_bytes in audio_bytes_list
        ]
        
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                results.append(None)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        with self._cache_lock:
            self._cache.clear()
    
    def __enter__(self) -> EnhancedAudioProcessor:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()


# Keep the original VAD functions for compatibility
def voice_activity_segments(
    audio: AudioArray,
    sample_rate: int,
    *,
    window_seconds: float = 0.02,
    hop_seconds: float = 0.01,
    energy_threshold_db: float = -40,
    hangover_seconds: float = 0.1,
    min_speech_seconds: float = 0.1,
    min_silence_seconds: float = 0.1,
) -> List[AudioSegment]:
    """Detect voice activity segments using energy-based VAD."""
    if audio.size == 0:
        return []
    
    # Ensure 1D for VAD processing
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    
    frame_length = max(1, int(window_seconds * sample_rate))
    hop_length = max(1, int(hop_seconds * sample_rate))
    
    # Calculate short-time energy
    energies = _compute_short_time_energy(audio, frame_length, hop_length)
    if energies.size == 0:
        return []
    
    # Convert to dB and threshold
    log_energy = 10.0 * np.log10(energies + EPSILON)
    voice_mask = log_energy > energy_threshold_db
    
    # Apply hangover
    if hangover_seconds > 0:
        hangover_frames = int(np.ceil(hangover_seconds / hop_seconds))
        kernel = np.ones(2 * hangover_frames + 1)
        voice_mask = np.convolve(voice_mask.astype(float), kernel, mode='same') > 0
    
    # Remove short speech segments
    min_speech_frames = int(np.ceil(min_speech_seconds / hop_seconds))
    voice_mask = _remove_short_segments(voice_mask, min_speech_frames)
    
    # Convert frames to sample indices
    segments = _frames_to_segments(voice_mask, frame_length, hop_length, audio.shape[0])
    
    # Merge close segments
    if min_silence_seconds > 0 and segments:
        min_silence_frames = int(np.ceil(min_silence_seconds / hop_seconds))
        segments = _merge_close_segments(segments, hop_length, min_silence_frames)
    
    return segments


def _compute_short_time_energy(
    audio: AudioArray,
    frame_length: int,
    hop_length: int
) -> AudioArray:
    """Compute short-time energy of audio signal."""
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Pad if necessary
        if audio.shape[0] < frame_length:
            audio = np.pad(audio, (0, frame_length - audio.shape[0]), mode='constant')
        
        windows = sliding_window_view(audio, frame_length)[::hop_length]
        return np.mean(np.square(windows), axis=-1).astype(np.float32)
        
    except ImportError:
        # Fallback for older NumPy versions
        total_frames = 1 + max(0, (audio.shape[0] - frame_length) // hop_length)
        energies = np.zeros(total_frames, dtype=np.float32)
        
        for i in range(total_frames):
            start = i * hop_length
            end = min(start + frame_length, audio.shape[0])
            window = audio[start:end]
            energies[i] = np.mean(np.square(window))
        
        return energies


def _remove_short_segments(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove segments shorter than min_frames."""
    if min_frames <= 1:
        return mask
    
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    
    result = mask.copy()
    for start, stop in zip(starts, stops):
        if stop - start < min_frames:
            result[start:stop] = False
    
    return result


def _frames_to_segments(
    mask: np.ndarray,
    frame_length: int,
    hop_length: int,
    total_samples: int
) -> List[AudioSegment]:
    """Convert frame mask to sample segments."""
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    
    segments = []
    for start_frame, stop_frame in zip(starts, stops):
        start_sample = start_frame * hop_length
        end_sample = min(stop_frame * hop_length + frame_length, total_samples)
        if start_sample < end_sample:
            segments.append((start_sample, end_sample))
    
    return segments


def _merge_close_segments(
    segments: List[AudioSegment],
    hop_length: int,
    min_silence_frames: int
) -> List[AudioSegment]:
    """Merge segments separated by less than min_silence_frames."""
    if not segments:
        return []
    
    merged = [segments[0]]
    min_gap = min_silence_frames * hop_length
    
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


# Global processor instance with thread-safe initialization
_processor: Optional[EnhancedAudioProcessor] = None
_processor_lock = threading.Lock()


def get_processor(config_override: Optional[Dict[str, Any]] = None, **kwargs) -> EnhancedAudioProcessor:
    """Get or create global processor instance (thread-safe).
    
    Note: config_override only applies on first creation. To change config,
    you must create a new processor instance directly.
    """
    global _processor
    
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                all_kwargs = kwargs.copy()
                if config_override:
                    all_kwargs['config_override'] = config_override
                _processor = EnhancedAudioProcessor(**all_kwargs)
    elif config_override:
        # If config_override is provided but processor exists, update its config
        with _processor_lock:
            _processor.config = _processor._merge_config(config_override)
    
    return _processor


# Backward compatibility functions
def decode_audio(audio_bytes: bytes) -> Tuple[AudioArray, int]:
    """Legacy API: Decode audio bytes."""
    processor = get_processor()
    audio_data = processor.decode_audio(audio_bytes)
    return audio_data.samples, audio_data.sample_rate


def prepare_audio(audio_bytes: bytes) -> Tuple[AudioArray, int]:
    """Legacy API: Full processing pipeline returning mono 1D audio."""
    processor = get_processor()
    # Force mono output for backward compatibility
    audio_data = processor.process_audio(audio_bytes, target_channels=1)
    # Convert from 2D (1, samples) to 1D (samples,) for legacy API
    samples = audio_data.samples
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples.squeeze(0)
    return samples, audio_data.sample_rate


def chunk_audio(
    audio: AudioArray,
    sample_rate: int,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
) -> Generator[AudioArray, None, None]:
    """Legacy API: Generate audio chunks."""
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    
    audio_data = AudioData(
        samples=audio,
        sample_rate=sample_rate,
        channels=audio.shape[0],
        duration=audio.shape[1] / sample_rate,
    )
    
    processor = get_processor()
    for chunk in processor.chunk_audio(audio_data, chunk_seconds, overlap_seconds):
        yield chunk.samples