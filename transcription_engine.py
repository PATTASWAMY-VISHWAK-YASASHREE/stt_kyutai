"""
Ultra-fast unified transcription engine.
Orchestrates all components with aggressive optimization for maximum speed.
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

# Import our modules
from audio_processing import AudioData, AudioProcessor, get_processor as get_audio_processor
from encoding import SafeInputPreparer, SafeDecoder, SimpleAggregator, to_thread
from model_loader import ModelLoader, ModelBundle, get_model_loader

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
AudioArray = NDArray[np.float32]
AudioInput = Union[bytes, np.ndarray, AudioData]


class ProcessingMode(Enum):
    """Processing modes for different speed/quality tradeoffs."""
    ULTRA_FAST = auto()      # Maximum speed, lower quality
    FAST = auto()            # Balanced speed/quality
    BALANCED = auto()        # Standard mode
    QUALITY = auto()         # Higher quality, slower
    MAXIMUM_QUALITY = auto() # Best quality, slowest


class CacheStrategy(Enum):
    """Cache strategies."""
    NONE = auto()
    MEMORY = auto()
    DISK = auto()
    HYBRID = auto()


@dataclass
class PerformanceMetrics:
    """Performance metrics for transcription."""
    audio_decode_ms: float = 0.0
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_ms: float = 0.0
    audio_duration_s: float = 0.0
    real_time_factor: float = 0.0
    cache_hit: bool = False
    
    def __str__(self) -> str:
        return (
            f"Total: {self.total_ms:.1f}ms | "
            f"RTF: {self.real_time_factor:.2f}x | "
            f"Cache: {'HIT' if self.cache_hit else 'MISS'}"
        )


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[PerformanceMetrics] = None
    audio_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"text": self.text}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.segments:
            result["segments"] = self.segments
        if self.metrics:
            result["metrics"] = {
                "total_ms": round(self.metrics.total_ms, 2),
                "real_time_factor": round(self.metrics.real_time_factor, 3),
                "cache_hit": self.metrics.cache_hit,
            }
        if self.audio_duration:
            result["audio_duration_s"] = round(self.audio_duration, 2)
        return result


class ResultCache:
    """Ultra-fast LRU cache with audio fingerprinting."""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _fingerprint(self, audio: AudioInput) -> str:
        """Generate fast audio fingerprint."""
        if isinstance(audio, bytes):
            # Use first/last portions for speed
            if len(audio) > 10000:
                sample = audio[:5000] + audio[-5000:]
            else:
                sample = audio
            return hashlib.blake2b(sample, digest_size=16).hexdigest()
        
        elif isinstance(audio, np.ndarray):
            # Use statistics for numpy arrays
            stats = f"{audio.mean():.6f}_{audio.std():.6f}_{audio.shape}_{audio.min():.6f}_{audio.max():.6f}"
            return hashlib.blake2b(stats.encode(), digest_size=16).hexdigest()
        
        elif isinstance(audio, AudioData):
            return self._fingerprint(audio.samples)
        
        return ""
    
    def get(self, audio: AudioInput) -> Optional[TranscriptionResult]:
        """Get from cache."""
        key = self._fingerprint(audio)
        if not key:
            return None
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                result = self.cache[key]
                # Update metrics to show cache hit
                if result.metrics:
                    result.metrics.cache_hit = True
                return result
            
            self.misses += 1
            return None
    
    def put(self, audio: AudioInput, result: TranscriptionResult) -> None:
        """Put in cache."""
        key = self._fingerprint(audio)
        if not key:
            return
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)
            self.cache[key] = result
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0


class FastTranscriptionEngine:
    """
    Ultra-fast unified transcription engine.
    Combines all components with aggressive optimization.
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.FAST,
        cache_size: int = 1000,
        max_workers: int = 4,
        enable_batching: bool = True,
        batch_size: int = 8,
        enable_streaming: bool = False,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize transcription engine.
        
        Args:
            mode: Processing mode (speed/quality tradeoff)
            cache_size: Maximum cache entries
            max_workers: Maximum worker threads
            enable_batching: Enable batch processing
            batch_size: Batch size for processing
            enable_streaming: Enable streaming mode
            model_id: Model ID override
            device: Device override
        """
        self.mode = mode
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.enable_streaming = enable_streaming
        
        # Initialize components
        logger.info("🚀 Initializing FastTranscriptionEngine...")
        
        # Audio processor
        self.audio_processor = get_audio_processor(
            config_override=self._get_audio_config()
        )
        
        # Model loader
        self.model_loader = get_model_loader(
            model_id=model_id,
            device_preference=device or "auto",
            **self._get_model_config()
        )
        
        # Load model eagerly for speed
        self.bundle: Optional[ModelBundle] = None
        self.input_preparer: Optional[SafeInputPreparer] = None
        self.decoder: Optional[SafeDecoder] = None
        
        # Cache
        self.cache = ResultCache(maxsize=cache_size)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Batch processing queue
        self.batch_queue: deque = deque()
        self.batch_lock = threading.Lock()
        
        # Statistics
        self.total_transcriptions = 0
        self.total_time_ms = 0.0
        self.lock = threading.RLock()
        
        logger.info(f"✅ Engine initialized (mode={mode.name}, cache_size={cache_size})")
    
    def _get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing config based on mode."""
        configs = {
            ProcessingMode.ULTRA_FAST: {
                'PEAK_NORMALIZE': False,
                'NORMALIZE_L2': False,
                'ENABLE_VAD': False,
            },
            ProcessingMode.FAST: {
                'PEAK_NORMALIZE': True,
                'NORMALIZE_L2': False,
                'ENABLE_VAD': False,
            },
            ProcessingMode.BALANCED: {
                'PEAK_NORMALIZE': True,
                'NORMALIZE_L2': False,
                'ENABLE_VAD': True,
            },
            ProcessingMode.QUALITY: {
                'PEAK_NORMALIZE': True,
                'NORMALIZE_L2': True,
                'ENABLE_VAD': True,
            },
            ProcessingMode.MAXIMUM_QUALITY: {
                'PEAK_NORMALIZE': True,
                'NORMALIZE_L2': True,
                'ENABLE_VAD': True,
            },
        }
        return configs.get(self.mode, configs[ProcessingMode.BALANCED])
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model config based on mode."""
        configs = {
            ProcessingMode.ULTRA_FAST: {
                'load_in_8bit': True,
                'enable_compile': True,
            },
            ProcessingMode.FAST: {
                'load_in_8bit': False,
                'load_in_4bit': False,
                'enable_compile': True,
                'torch_dtype': 'float16',
            },
            ProcessingMode.BALANCED: {
                'load_in_8bit': False,
                'load_in_4bit': False,
                'enable_compile': True,
                'torch_dtype': 'auto',
            },
            ProcessingMode.QUALITY: {
                'enable_compile': True,
                'torch_dtype': 'float32',
            },
            ProcessingMode.MAXIMUM_QUALITY: {
                'enable_compile': False,
                'torch_dtype': 'float32',
            },
        }
        return configs.get(self.mode, configs[ProcessingMode.BALANCED])
    
    def warmup(self) -> None:
        """
        Warmup the engine (load model, compile, etc).
        Call this once at startup for faster first transcription.
        """
        logger.info("🔥 Warming up engine...")
        start = time.perf_counter()
        
        # Load model
        self.bundle = self.model_loader.ensure_loaded()
        
        # Initialize input preparer and decoder
        self.input_preparer = SafeInputPreparer(
            self.bundle.processor,
            config=None,
        )
        self.decoder = SafeDecoder(
            self.bundle.processor,
            config=None,
        )
        
        # Warmup inference with dummy audio
        try:
            dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01
            _ = self.transcribe(dummy_audio, sample_rate=16000)
            logger.info(f"✅ Warmup complete ({(time.perf_counter() - start)*1000:.0f}ms)")
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")
    
    def transcribe(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio (synchronous).
        
        Args:
            audio: Audio input (bytes, numpy array, or AudioData)
            sample_rate: Sample rate (required for numpy arrays)
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(audio)
            if cached is not None:
                logger.debug("Cache hit!")
                return cached
        
        # Initialize metrics
        metrics = PerformanceMetrics()
        total_start = time.perf_counter()
        
        try:
            # Ensure model loaded
            if self.bundle is None:
                self.warmup()
            
            # 1. Decode and preprocess audio
            start = time.perf_counter()
            audio_data = self._prepare_audio(audio, sample_rate)
            metrics.audio_decode_ms = (time.perf_counter() - start) * 1000
            metrics.audio_duration_s = audio_data.duration
            
            # 2. Prepare model inputs
            start = time.perf_counter()
            inputs = self._prepare_inputs(audio_data)
            metrics.preprocessing_ms = (time.perf_counter() - start) * 1000
            
            # 3. Run inference
            start = time.perf_counter()
            outputs = self._run_inference(inputs, **kwargs)
            metrics.inference_ms = (time.perf_counter() - start) * 1000
            
            # 4. Decode outputs
            start = time.perf_counter()
            text = self._decode_outputs(outputs)
            metrics.postprocessing_ms = (time.perf_counter() - start) * 1000
            
            # Calculate total metrics
            metrics.total_ms = (time.perf_counter() - total_start) * 1000
            if metrics.audio_duration_s > 0:
                metrics.real_time_factor = (metrics.total_ms / 1000.0) / metrics.audio_duration_s
            
            # Create result
            result = TranscriptionResult(
                text=text,
                metrics=metrics,
                audio_duration=metrics.audio_duration_s,
            )
            
            # Update statistics
            with self.lock:
                self.total_transcriptions += 1
                self.total_time_ms += metrics.total_ms
            
            # Cache result
            if use_cache:
                self.cache.put(audio, result)
            
            logger.info(f"✅ Transcription: '{text[:50]}...' | {metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
    
    async def transcribe_async(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio (asynchronous).
        
        Args:
            audio: Audio input
            sample_rate: Sample rate
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult
        """
        return await to_thread(self.transcribe, audio, sample_rate, **kwargs)
    
    def transcribe_batch(
        self,
        audio_list: List[AudioInput],
        sample_rates: Optional[List[int]] = None,
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch (optimized).
        
        Args:
            audio_list: List of audio inputs
            sample_rates: List of sample rates
            **kwargs: Additional parameters
            
        Returns:
            List of TranscriptionResult
        """
        if not audio_list:
            return []
        
        # Normalize sample rates
        if sample_rates is None:
            sample_rates = [None] * len(audio_list)
        
        if not self.enable_batching or len(audio_list) == 1:
            # Process sequentially
            return [
                self.transcribe(audio, sr, **kwargs)
                for audio, sr in zip(audio_list, sample_rates)
            ]
        
        # Check cache first
        results = []
        uncached_indices = []
        uncached_audios = []
        uncached_rates = []
        
        for i, (audio, sr) in enumerate(zip(audio_list, sample_rates)):
            cached = self.cache.get(audio)
            if cached is not None:
                results.append(cached)
            else:
                uncached_indices.append(i)
                uncached_audios.append(audio)
                uncached_rates.append(sr)
                results.append(None)  # Placeholder
        
        if not uncached_audios:
            return results
        
        # Process uncached in batches
        logger.info(f"Processing {len(uncached_audios)} uncached items in batches")
        
        for i in range(0, len(uncached_audios), self.batch_size):
            batch_audios = uncached_audios[i:i+self.batch_size]
            batch_rates = uncached_rates[i:i+self.batch_size]
            batch_indices = uncached_indices[i:i+self.batch_size]
            
            # Process batch
            batch_results = self._process_batch(batch_audios, batch_rates, **kwargs)
            
            # Fill in results
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result
                # Cache
                self.cache.put(audio_list[idx], result)
        
        return results
    
    def _process_batch(
        self,
        audio_list: List[AudioInput],
        sample_rates: List[Optional[int]],
        **kwargs
    ) -> List[TranscriptionResult]:
        """Process a batch of audio."""
        # Prepare all audio
        audio_data_list = [
            self._prepare_audio(audio, sr)
            for audio, sr in zip(audio_list, sample_rates)
        ]
        
        # Prepare inputs in batch
        all_inputs = [self._prepare_inputs(ad) for ad in audio_data_list]
        
        # Run inference (could potentially batch this too)
        all_outputs = [self._run_inference(inp, **kwargs) for inp in all_inputs]
        
        # Decode all
        texts = [self._decode_outputs(out) for out in all_outputs]
        
        # Create results
        return [
            TranscriptionResult(
                text=text,
                audio_duration=ad.duration,
            )
            for text, ad in zip(texts, audio_data_list)
        ]
    
    def transcribe_file(
        self,
        file_path: str,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult
        """
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        return self.transcribe(audio_bytes, **kwargs)
    
    def transcribe_stream(
        self,
        audio_stream: Generator[AudioInput, None, None],
        chunk_duration: float = 5.0,
        **kwargs
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Transcribe streaming audio.
        
        Args:
            audio_stream: Generator yielding audio chunks
            chunk_duration: Duration of chunks in seconds
            **kwargs: Additional parameters
            
        Yields:
            TranscriptionResult for each chunk
        """
        aggregator = SimpleAggregator()
        chunks = []
        
        for audio_chunk in audio_stream:
            result = self.transcribe(audio_chunk, **kwargs)
            chunks.append(result.text)
            
            # Yield incremental result
            aggregated = aggregator.aggregate(chunks)
            yield TranscriptionResult(
                text=aggregated,
                audio_duration=sum(c.audio_duration or 0 for c in [result]),
            )
    
    def _prepare_audio(
        self,
        audio: AudioInput,
        sample_rate: Optional[int]
    ) -> AudioData:
        """Prepare audio for processing."""
        if isinstance(audio, AudioData):
            return audio
        
        elif isinstance(audio, bytes):
            # Decode audio bytes
            return self.audio_processor.process_audio(audio)
        
        elif isinstance(audio, np.ndarray):
            # Convert numpy array
            if sample_rate is None:
                raise ValueError("sample_rate required for numpy array input")
            
            # Validate and convert
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            elif audio.ndim > 2:
                raise ValueError(f"Invalid audio shape: {audio.shape}")
            
            return AudioData(
                samples=audio.astype(np.float32),
                sample_rate=sample_rate,
                channels=audio.shape[0],
                duration=audio.shape[1] / sample_rate,
            )
        
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
    
    def _prepare_inputs(self, audio_data: AudioData) -> Any:
        """Prepare model inputs."""
        # Convert to mono for model
        if audio_data.channels > 1:
            mono = np.mean(audio_data.samples, axis=0)
        else:
            mono = audio_data.samples[0]
        
        # Use input preparer
        return self.input_preparer.prepare_single(
            mono,
            audio_data.sample_rate,
        )
    
    def _run_inference(self, inputs: Any, **kwargs) -> Any:
        """Run model inference."""
        with torch.no_grad():
            if hasattr(self.bundle.model, 'generate'):
                # Seq2Seq models
                outputs = self.bundle.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 448),
                    num_beams=kwargs.get('num_beams', 1),
                    do_sample=kwargs.get('do_sample', False),
                )
            else:
                # CTC models
                outputs = self.bundle.model(**inputs).logits
                outputs = torch.argmax(outputs, dim=-1)
            
            return outputs
    
    def _decode_outputs(self, outputs: Any) -> str:
        """Decode model outputs."""
        return self.decoder.decode_single(outputs, skip_special_tokens=True, clean=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self.lock:
            avg_time = self.total_time_ms / self.total_transcriptions if self.total_transcriptions > 0 else 0
            
            return {
                'total_transcriptions': self.total_transcriptions,
                'total_time_ms': round(self.total_time_ms, 2),
                'avg_time_ms': round(avg_time, 2),
                'cache_hit_rate': round(self.cache.hit_rate * 100, 2),
                'cache_size': len(self.cache.cache),
                'mode': self.mode.name,
            }
    
    def clear_cache(self) -> None:
        """Clear result cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown engine and cleanup resources."""
        logger.info("Shutting down engine...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Unload model
        if self.model_loader:
            self.model_loader.unload()
        
        # Clear cache
        self.cache.clear()
        
        logger.info("Engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.warmup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global singleton
_engine_lock = threading.RLock()
_global_engine: Optional[FastTranscriptionEngine] = None


def get_engine(**kwargs) -> FastTranscriptionEngine:
    """Get or create global engine instance."""
    global _global_engine
    
    with _engine_lock:
        if _global_engine is None:
            _global_engine = FastTranscriptionEngine(**kwargs)
        return _global_engine


def reset_engine() -> None:
    """Reset global engine."""
    global _global_engine
    
    with _engine_lock:
        if _global_engine is not None:
            _global_engine.shutdown()
            _global_engine = None


# Convenience functions
def transcribe(audio: AudioInput, **kwargs) -> str:
    """Quick transcribe function."""
    engine = get_engine()
    result = engine.transcribe(audio, **kwargs)
    return result.text


async def transcribe_async(audio: AudioInput, **kwargs) -> str:
    """Quick async transcribe."""
    engine = get_engine()
    result = await engine.transcribe_async(audio, **kwargs)
    return result.text


def transcribe_file(file_path: str, **kwargs) -> str:
    """Quick file transcribe."""
    engine = get_engine()
    result = engine.transcribe_file(file_path, **kwargs)
    return result.text


# Export
__all__ = [
    'FastTranscriptionEngine',
    'ProcessingMode',
    'TranscriptionResult',
    'PerformanceMetrics',
    'get_engine',
    'reset_engine',
    'transcribe',
    'transcribe_async',
    'transcribe_file',
]