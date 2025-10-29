"""Enhanced utility helpers for preparing inputs and decoding outputs with production features."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F

try:
    from numpy.typing import NDArray
    AudioArray = NDArray[np.float32]
except ImportError:
    # Fallback for older numpy
    AudioArray = np.ndarray

from transformers import (
    BatchFeature,
    PreTrainedTokenizerBase,
)

# Handle imports with proper error checking
try:
    from transformers import KyutaiSpeechToTextProcessor
except ImportError:
    warnings.warn("KyutaiSpeechToTextProcessor not available, using base class")
    KyutaiSpeechToTextProcessor = PreTrainedTokenizerBase

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
TokenSequence = Union[torch.Tensor, List[int], np.ndarray]
AudioInput = Union[np.ndarray, List[np.ndarray]]

# Constants
DEFAULT_MAX_LENGTH = 448
DEFAULT_BATCH_SIZE = 8
MAX_AUDIO_LENGTH_SECONDS = 30.0
MIN_CONFIDENCE_THRESHOLD = 0.5
EPSILON = 1e-12
MAX_CACHE_SIZE = 100
MAX_SIMILARITY_CACHE_SIZE = 1000


class DecodingStrategy(Enum):
    """Decoding strategies for transcription."""
    GREEDY = auto()
    BEAM_SEARCH = auto()
    SAMPLING = auto()
    CONSTRAINED_BEAM_SEARCH = auto()


class AggregationMethod(Enum):
    """Methods for aggregating chunk transcripts."""
    SIMPLE = auto()
    OVERLAP_AWARE = auto()
    SEMANTIC = auto()
    WEIGHTED = auto()


class TranscriptionException(Exception):
    """Base exception for transcription errors."""
    pass


class InputPreparationError(TranscriptionException):
    """Error during input preparation."""
    pass


class DecodingError(TranscriptionException):
    """Error during token decoding."""
    pass


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata for audio input."""
    sample_rate: int
    channels: int
    duration: float
    format: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata."""
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"Invalid channel count: {self.channels}")
        if self.duration < 0:
            raise ValueError(f"Invalid duration: {self.duration}")


@dataclass(frozen=True)
class TranscriptionMetrics:
    """Metrics collected during transcription."""
    preparation_time_ms: float
    inference_time_ms: float
    decoding_time_ms: float
    num_chunks: int
    total_audio_duration_s: float
    tokens_generated: int
    confidence_score: Optional[float] = None
    
    @property
    def total_time_ms(self) -> float:
        """Total processing time."""
        return self.preparation_time_ms + self.inference_time_ms + self.decoding_time_ms
    
    @property
    def real_time_factor(self) -> float:
        """Real-time factor (processing time / audio duration)."""
        if self.total_audio_duration_s > 0:
            return (self.total_time_ms / 1000.0) / self.total_audio_duration_s
        return 0.0


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete transcription result with metadata."""
    text: str
    tokens: Optional[List[int]] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[TranscriptionMetrics] = None
    alternative_texts: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"text": self.text}
        for key, value in [
            ("confidence", self.confidence),
            ("language", self.language),
            ("segments", self.segments),
            ("alternatives", self.alternative_texts),
        ]:
            if value is not None:
                result[key] = value
        
        if self.metrics:
            result["metrics"] = {
                "total_time_ms": self.metrics.total_time_ms,
                "real_time_factor": self.metrics.real_time_factor,
                "tokens_generated": self.metrics.tokens_generated,
            }
        return result


@dataclass
class ProcessorConfig:
    """Configuration for audio processor."""
    max_length: int = DEFAULT_MAX_LENGTH
    padding: Union[bool, str] = True
    truncation: bool = True
    return_attention_mask: bool = True
    return_tensors: str = "pt"
    normalize: bool = True
    device: Optional[torch.device] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 4
    use_cache: bool = True
    
    def __post_init__(self):
        """Set default device if not provided."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SafeInputPreparer:
    """Safe input preparation with validation and error handling."""
    
    def __init__(
        self,
        processor: Any,  # Using Any to avoid type conflicts
        config: Optional[ProcessorConfig] = None,
    ):
        """Initialize with processor and configuration."""
        self.processor = processor
        self.config = config or ProcessorConfig()
        self._validate_processor()
        
        # Cache with size limit
        self._cache: Dict[str, BatchFeature] = {}
        self._cache_order: deque = deque(maxlen=MAX_CACHE_SIZE)
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _validate_processor(self) -> None:
        """Validate processor compatibility."""
        required_attrs = ['__call__']  # Minimum requirement
        for attr in required_attrs:
            if not hasattr(self.processor, attr):
                raise ValueError(f"Processor must have '{attr}' method")
    
    def prepare_single(
        self,
        audio: AudioInput,
        sampling_rate: int,
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare a single audio input for model inference.
        
        Args:
            audio: Audio input (numpy array)
            sampling_rate: Sampling rate in Hz
            **kwargs: Additional processor arguments
            
        Returns:
            Prepared BatchFeature for model input
        """
        # Validate and convert audio
        audio_array, metadata = self._validate_and_convert_audio(audio, sampling_rate)
        
        # Check cache if enabled
        cache_key = None
        if self.config.use_cache:
            cache_key = self._get_cache_key(audio_array, metadata)
            if cache_key in self._cache:
                self._cache_hits += 1
                logger.debug(f"Cache hit rate: {self._get_cache_hit_rate():.2%}")
                return self._move_to_device(self._cache[cache_key])
        
        self._cache_misses += 1
        
        # Prepare features
        try:
            features = self._prepare_features(audio_array, sampling_rate, **kwargs)
        except Exception as e:
            raise InputPreparationError(f"Feature preparation failed: {e}") from e
        
        # Update cache
        if cache_key and self.config.use_cache:
            self._update_cache(cache_key, features)
        
        return self._move_to_device(features)
    
    def prepare_batch(
        self,
        audio_list: Sequence[AudioInput],
        sampling_rates: Union[int, Sequence[int]],
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare batch of audio inputs.
        
        Args:
            audio_list: List of audio inputs
            sampling_rates: Single rate or list of rates
            **kwargs: Additional processor arguments
            
        Returns:
            Batched features for model input
        """
        if not audio_list:
            raise InputPreparationError("Empty audio list provided")
        
        # Handle single sample rate for all
        if isinstance(sampling_rates, int):
            sampling_rates = [sampling_rates] * len(audio_list)
        
        if len(audio_list) != len(sampling_rates):
            raise InputPreparationError(
                f"Length mismatch: {len(audio_list)} audios, {len(sampling_rates)} rates"
            )
        
        # Validate and convert all audio
        processed_audios = []
        target_rate = max(set(sampling_rates), key=sampling_rates.count)
        
        for audio, sr in zip(audio_list, sampling_rates):
            audio_array, _ = self._validate_and_convert_audio(audio, sr)
            
            # Resample if needed
            if sr != target_rate:
                audio_array = self._simple_resample(audio_array, sr, target_rate)
            
            processed_audios.append(audio_array)
        
        # Prepare batch features
        try:
            features = self._prepare_features(processed_audios, target_rate, **kwargs)
        except Exception as e:
            raise InputPreparationError(f"Batch preparation failed: {e}") from e
        
        return self._move_to_device(features)
    
    def _validate_and_convert_audio(
        self,
        audio: AudioInput,
        sampling_rate: int
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """Validate and convert audio to proper format."""
        # Ensure numpy array
        if not isinstance(audio, np.ndarray):
            try:
                audio = np.asarray(audio, dtype=np.float32)
            except Exception as e:
                raise InputPreparationError(f"Cannot convert audio to numpy: {e}")
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Check for NaN or Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.warning("Audio contains NaN or Inf values, replacing with zeros")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle different shapes
        original_shape = audio.shape
        if audio.ndim == 1:
            # Mono audio
            channels = 1
            num_samples = audio.shape[0]
        elif audio.ndim == 2:
            # Could be (channels, samples) or (samples, channels)
            # Assume smaller dimension is channels
            if audio.shape[0] <= 16:  # Max reasonable channel count
                channels = audio.shape[0]
                num_samples = audio.shape[1]
                # Convert to mono by averaging
                audio = np.mean(audio, axis=0)
            else:
                channels = audio.shape[1]
                num_samples = audio.shape[0]
                # Convert to mono
                audio = np.mean(audio, axis=1)
        else:
            raise InputPreparationError(f"Unsupported audio shape: {original_shape}")
        
        # Validate duration
        duration = num_samples / sampling_rate
        if duration > MAX_AUDIO_LENGTH_SECONDS:
            raise InputPreparationError(
                f"Audio too long: {duration:.2f}s > {MAX_AUDIO_LENGTH_SECONDS}s"
            )
        
        # Create metadata
        metadata = AudioMetadata(
            sample_rate=sampling_rate,
            channels=channels,
            duration=duration,
        )
        
        return audio, metadata
    
    def _prepare_features(
        self,
        audio: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: int,
        **kwargs,
    ) -> BatchFeature:
        """Prepare features using the processor."""
        # Merge with config
        processor_kwargs = {
            "sampling_rate": sampling_rate,
            "return_tensors": self.config.return_tensors,
            "padding": self.config.padding,
            "truncation": self.config.truncation,
            "max_length": self.config.max_length,
        }
        
        # Only add return_attention_mask if processor supports it
        if hasattr(self.processor, 'model_input_names'):
            if 'attention_mask' in self.processor.model_input_names:
                processor_kwargs["return_attention_mask"] = self.config.return_attention_mask
        
        processor_kwargs.update(kwargs)
        
        # Call processor
        return self.processor(audio=audio, **processor_kwargs)
    
    def _simple_resample(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if source_rate == target_rate:
            return audio
        
        # Calculate new length
        duration = len(audio) / source_rate
        target_length = int(np.round(duration * target_rate))
        
        if target_length == 0:
            return np.array([], dtype=np.float32)
        
        # Create indices for interpolation
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, target_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio)
        
        return resampled.astype(np.float32)
    
    def _get_cache_key(self, audio: np.ndarray, metadata: AudioMetadata) -> str:
        """Generate cache key for audio."""
        hasher = hashlib.sha256()
        
        # Use audio statistics instead of full array for efficiency
        stats = [
            audio.mean(),
            audio.std(),
            audio.min(),
            audio.max(),
            len(audio),
            metadata.sample_rate,
        ]
        
        for stat in stats:
            hasher.update(str(stat).encode())
        
        return hasher.hexdigest()
    
    def _update_cache(self, key: str, features: BatchFeature) -> None:
        """Update cache with size management."""
        # Remove oldest if at capacity
        if len(self._cache) >= MAX_CACHE_SIZE:
            if self._cache_order:
                oldest = self._cache_order.popleft()
                self._cache.pop(oldest, None)
        
        self._cache[key] = features
        self._cache_order.append(key)
    
    def _move_to_device(self, features: BatchFeature) -> BatchFeature:
        """Move features to configured device."""
        if hasattr(features, 'to'):
            return features.to(self.config.device)
        return features
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class SafeDecoder:
    """Safe token decoder with error handling."""
    
    def __init__(
        self,
        processor: Any,
        config: Optional[ProcessorConfig] = None,
    ):
        """Initialize decoder."""
        self.processor = processor
        self.config = config or ProcessorConfig()
        
        # Get tokenizer
        if hasattr(processor, 'tokenizer'):
            self.tokenizer = processor.tokenizer
        elif hasattr(processor, 'batch_decode'):
            self.tokenizer = processor
        else:
            raise ValueError("Processor must have tokenizer or batch_decode method")
        
        # Compile patterns
        self._whitespace_pattern = re.compile(r'\s+')
        self._punctuation_pattern = re.compile(r'([.!?])\1+')
    
    def decode_single(
        self,
        tokens: TokenSequence,
        skip_special_tokens: bool = True,
        clean_output: bool = True,
    ) -> str:
        """Decode a single token sequence."""
        # Convert to list
        token_list = self._tokens_to_list(tokens)
        
        if not token_list:
            return ""
        
        # Decode
        try:
            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(token_list, skip_special_tokens=skip_special_tokens)
            else:
                # Fallback for batch_decode
                texts = self.tokenizer.batch_decode([token_list], skip_special_tokens=skip_special_tokens)
                text = texts[0] if texts else ""
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return ""
        
        # Clean if requested
        if clean_output:
            text = self._clean_text(text)
        
        return text
    
    def decode_batch(
        self,
        token_batches: Sequence[TokenSequence],
        skip_special_tokens: bool = True,
        clean_output: bool = True,
    ) -> List[str]:
        """Decode multiple token sequences."""
        if not token_batches:
            return []
        
        # Convert all to lists
        token_lists = [self._tokens_to_list(tokens) for tokens in token_batches]
        
        # Decode
        try:
            if hasattr(self.tokenizer, 'batch_decode'):
                texts = self.tokenizer.batch_decode(token_lists, skip_special_tokens=skip_special_tokens)
            else:
                # Fallback to single decode
                texts = [
                    self.decode_single(tokens, skip_special_tokens, False)
                    for tokens in token_lists
                ]
        except Exception as e:
            logger.error(f"Batch decoding failed: {e}")
            return [""] * len(token_batches)
        
        # Clean if requested
        if clean_output:
            texts = [self._clean_text(text) for text in texts]
        
        return texts
    
    def _tokens_to_list(self, tokens: TokenSequence) -> List[int]:
        """Convert various token formats to list."""
        if isinstance(tokens, torch.Tensor):
            # Handle various tensor shapes
            if tokens.dim() == 0:
                return [int(tokens.item())]
            elif tokens.dim() == 1:
                return tokens.cpu().tolist()
            else:
                # Flatten higher dimensions
                return tokens.flatten().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            return tokens.flatten().tolist()
        elif isinstance(tokens, list):
            return tokens
        else:
            try:
                return list(tokens)
            except Exception:
                return []
    
    def _clean_text(self, text: str) -> str:
        """Clean decoded text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = self._whitespace_pattern.sub(' ', text)
        
        # Fix repeated punctuation
        text = self._punctuation_pattern.sub(r'\1', text)
        
        # Strip
        text = text.strip()
        
        # Capitalize first letter if lowercase
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text


class SimpleTranscriptAggregator:
    """Simple and robust transcript aggregation."""
    
    def __init__(self, method: AggregationMethod = AggregationMethod.SIMPLE):
        """Initialize aggregator."""
        self.method = method
        self._cache: Dict[str, str] = {}
    
    def aggregate(
        self,
        chunks: Iterable[str],
        overlaps: Optional[List[float]] = None,
        weights: Optional[List[float]] = None,
    ) -> str:
        """Aggregate transcript chunks."""
        chunk_list = [c for c in chunks if c and c.strip()]
        
        if not chunk_list:
            return ""
        
        if len(chunk_list) == 1:
            return chunk_list[0].strip()
        
        if self.method == AggregationMethod.SIMPLE:
            return self._simple_aggregate(chunk_list)
        elif self.method == AggregationMethod.OVERLAP_AWARE and overlaps:
            return self._overlap_aggregate(chunk_list, overlaps)
        elif self.method == AggregationMethod.WEIGHTED and weights:
            return self._weighted_aggregate(chunk_list, weights)
        else:
            return self._simple_aggregate(chunk_list)
    
    def _simple_aggregate(self, chunks: List[str]) -> str:
        """Simple aggregation with duplicate removal."""
        result = []
        prev = None
        
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and chunk != prev:
                result.append(chunk)
                prev = chunk
        
        return " ".join(result)
    
    def _overlap_aggregate(self, chunks: List[str], overlaps: List[float]) -> str:
        """Overlap-aware aggregation."""
        if not chunks:
            return ""
        
        result = [chunks[0].strip()]
        
        for i, chunk in enumerate(chunks[1:], 0):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            if i < len(overlaps) and overlaps[i] > 0.1:
                # Has overlap, try to merge
                words1 = result[-1].split()
                words2 = chunk.split()
                
                # Find overlap
                overlap_size = min(len(words1), len(words2), int(len(words2) * overlaps[i]))
                
                if overlap_size > 0 and words1[-overlap_size:] == words2[:overlap_size]:
                    # Merge without duplicate
                    result[-1] = " ".join(words1 + words2[overlap_size:])
                else:
                    result.append(chunk)
            else:
                result.append(chunk)
        
        return " ".join(result)
    
    def _weighted_aggregate(self, chunks: List[str], weights: List[float]) -> str:
        """Weighted aggregation."""
        if not weights:
            return self._simple_aggregate(chunks)
        
        # Filter by weight threshold
        threshold = np.mean(weights) * 0.5 if weights else 0.0
        filtered = [
            chunk for chunk, weight in zip(chunks, weights)
            if weight >= threshold
        ]
        
        return self._simple_aggregate(filtered)


# Legacy API functions for backward compatibility
def prepare_inputs(
    processor: Any,
    audio_samples: Union[Sequence[np.ndarray], np.ndarray],
    sampling_rate: int,
    device: torch.device,
    **processor_kwargs: Any,
) -> BatchFeature:
    """Legacy API: Prepare model inputs."""
    config = ProcessorConfig(device=device)
    preparer = SafeInputPreparer(processor, config)
    
    if isinstance(audio_samples, np.ndarray):
        if audio_samples.ndim <= 2:
            # Single audio
            return preparer.prepare_single(audio_samples, sampling_rate, **processor_kwargs)
        else:
            # Batch of audios
            audio_list = [audio_samples[i] for i in range(audio_samples.shape[0])]
            return preparer.prepare_batch(audio_list, sampling_rate, **processor_kwargs)
    else:
        # List of audios
        return preparer.prepare_batch(list(audio_samples), sampling_rate, **processor_kwargs)


def decode_tokens(
    processor: Any,
    token_batches: Sequence[torch.Tensor],
    skip_special_tokens: bool = True,
) -> List[str]:
    """Legacy API: Decode token batches."""
    decoder = SafeDecoder(processor)
    return decoder.decode_batch(token_batches, skip_special_tokens)


def aggregate_transcripts(chunks: Iterable[str]) -> str:
    """Legacy API: Aggregate transcript chunks."""
    aggregator = SimpleTranscriptAggregator()
    return aggregator.aggregate(chunks)


# Export main components
__all__ = [
    'SafeInputPreparer',
    'SafeDecoder',
    'SimpleTranscriptAggregator',
    'TranscriptionResult',
    'TranscriptionMetrics',
    'ProcessorConfig',
    'DecodingStrategy',
    'AggregationMethod',
    'prepare_inputs',
    'decode_tokens',
    'aggregate_transcripts',
]