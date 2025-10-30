"""
Production-ready model handler for Kyutai speech-to-text.
Handles model loading, compilation, and inference with proper error handling.
"""

import logging
import threading
from typing import Optional, Union, Dict, Any
import warnings

import numpy as np
import torch
from numpy.typing import NDArray

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
AudioArray = NDArray[np.float32]

# Constants
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000
DEFAULT_SAMPLE_RATE = 16000


def get_config_value(key: str, default: Any) -> Any:
    """Safely get config value with fallback."""
    try:
        from . import config as _config
        return getattr(_config, key, default)
    except (ImportError, AttributeError):
        logger.warning(f"Config key '{key}' not found, using default: {default}")
        return default


class ModelLoadError(Exception):
    """Raised when model fails to load."""
    pass


class InferenceError(Exception):
    """Raised during inference."""
    pass


class ModelHandler:
    """
    Thread-safe handler for Kyutai speech-to-text model.
    Manages model lifecycle, compilation, and inference.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        """
        Initialize model handler.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda/cpu/mps)
            compile_model: Whether to compile with torch.compile
            compile_mode: Compilation mode (default, reduce-overhead, max-autotune)
        """
        self.model_id = model_id or get_config_value('MODEL_ID', 'kyutai/moshi-speech-to-text')
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        
        # Device setup
        self.device = self._setup_device(device)
        
        # Model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # State tracking
        self._is_loaded = False
        self._is_compiled = False
        self._lock = threading.RLock()
        
        logger.info(f"ModelHandler initialized for {self.model_id} on {self.device}")
    
    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Setup and validate device."""
        if device is None:
            # Auto-detect best device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        
        return device
    
    def load(self) -> None:
        """
        Load model and processor.
        Thread-safe and idempotent.
        """
        with self._lock:
            if self._is_loaded:
                logger.info("Model already loaded")
                return
            
            try:
                self._load_model_components()
                self._configure_model()
                
                if self.compile_model:
                    self._compile_model()
                
                self._is_loaded = True
                logger.info("✅ Model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Model load failed: {e}")
                self._cleanup()
                raise ModelLoadError(f"Failed to load model: {e}") from e
    
    def _load_model_components(self) -> None:
        """Load model, processor, and tokenizer."""
        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )
        except ImportError as e:
            raise ModelLoadError("transformers not installed") from e
        
        logger.info(f"Loading model: {self.model_id}...")
        
        # Try Kyutai-specific processor first
        try:
            from transformers import KyutaiSpeechToTextProcessor
            self.processor = KyutaiSpeechToTextProcessor.from_pretrained(self.model_id)
            logger.info("Loaded KyutaiSpeechToTextProcessor")
        except (ImportError, OSError):
            # Fallback to AutoProcessor
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                logger.info("Loaded AutoProcessor")
            except Exception as e:
                logger.error(f"Failed to load processor: {e}")
                raise
        
        # Load model
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
        except Exception:
            # Fallback to other model classes
            try:
                from transformers import AutoModelForCTC
                self.model = AutoModelForCTC.from_pretrained(self.model_id)
                logger.warning("Loaded as CTC model")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        # Move to device
        self.model.to(self.device)
        
        # Get tokenizer
        if hasattr(self.processor, 'tokenizer'):
            self.tokenizer = self.processor.tokenizer
        elif hasattr(self.processor, 'token_decoder'):
            self.tokenizer = self.processor.token_decoder
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    
    def _configure_model(self) -> None:
        """Configure model for inference."""
        # Set to eval mode
        self.model.eval()
        
        # Configure padding token if needed
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token'):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Update model config
        if hasattr(self.model, 'config'):
            if hasattr(self.tokenizer, 'pad_token_id'):
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Disable dropout for inference
            if hasattr(self.model.config, 'dropout'):
                self.model.config.dropout = 0.0
            if hasattr(self.model.config, 'attention_dropout'):
                self.model.config.attention_dropout = 0.0
    
    def _compile_model(self) -> None:
        """Compile model with torch.compile if available."""
        # Check PyTorch version
        pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if pytorch_version < (2, 0):
            logger.warning(f"torch.compile requires PyTorch 2.0+, got {torch.__version__}")
            return
        
        try:
            logger.info(f"Compiling model with mode='{self.compile_mode}'...")
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self._is_compiled = True
            logger.info("✅ Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, continuing without compilation")
    
    def ensure_loaded(self) -> None:
        """Ensure model is loaded (convenience method)."""
        if not self._is_loaded:
            self.load()
    
    @property
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self._is_loaded and self.model is not None and self.processor is not None
    
    def transcribe(
        self,
        audio: Union[AudioArray, np.ndarray],
        sample_rate: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array (1D or 2D)
            sample_rate: Audio sample rate (Hz)
            **kwargs: Additional generation parameters
            
        Returns:
            Transcribed text or None on error
        """
        if not self.is_ready:
            raise InferenceError("Model not loaded. Call load() first.")
        
        with self._lock:
            try:
                # Validate and prepare input
                audio_array = self._validate_audio(audio, sample_rate)
                
                # Prepare model inputs
                inputs = self._prepare_inputs(audio_array, sample_rate or DEFAULT_SAMPLE_RATE)
                
                # Run inference
                transcription = self._run_inference(inputs, **kwargs)
                
                return transcription
                
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                raise InferenceError(f"Transcription failed: {e}") from e
    
    def _validate_audio(
        self,
        audio: Union[AudioArray, np.ndarray],
        sample_rate: Optional[int]
    ) -> np.ndarray:
        """Validate and convert audio input."""
        # Ensure numpy array
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be numpy array, got {type(audio)}")
        
        # Convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(audio)):
            logger.warning("Audio contains NaN/Inf, replacing with zeros")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to 1D if multi-channel
        if audio.ndim == 2:
            # Average channels
            if audio.shape[0] <= 16:  # Channels first
                audio = np.mean(audio, axis=0)
            else:  # Channels last
                audio = np.mean(audio, axis=1)
        elif audio.ndim > 2:
            raise ValueError(f"Invalid audio shape: {audio.shape}")
        
        # Validate sample rate if provided
        if sample_rate is not None:
            if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        return audio
    
    def _prepare_inputs(self, audio: np.ndarray, sample_rate: int) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model."""
        # Process audio through processor
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _run_inference(self, inputs: Dict[str, torch.Tensor], **kwargs) -> str:
        """Run model inference."""
        with torch.no_grad():
            # Set default generation parameters
            generation_config = {
                'max_new_tokens': kwargs.pop('max_new_tokens', 448),
                'num_beams': kwargs.pop('num_beams', 1),
                'do_sample': kwargs.pop('do_sample', False),
            }
            generation_config.update(kwargs)
            
            # Generate
            if hasattr(self.model, 'generate'):
                # Seq2Seq models
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Decode
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            else:
                # CTC models
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
        
        return transcription.strip()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        self._is_compiled = False
    
    def unload(self) -> None:
        """Unload model and free resources."""
        with self._lock:
            logger.info("Unloading model...")
            self._cleanup()
            logger.info("Model unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._cleanup()
        except Exception:
            pass


# Global instance management (thread-safe)
_global_handler_lock = threading.RLock()
_global_handler: Optional[ModelHandler] = None


def get_model_handler(**kwargs) -> ModelHandler:
    """
    Get or create global model handler instance.
    Thread-safe singleton pattern.
    """
    global _global_handler
    
    with _global_handler_lock:
        if _global_handler is None:
            _global_handler = ModelHandler(**kwargs)
        return _global_handler


def reset_model_handler() -> None:
    """Reset global model handler."""
    global _global_handler
    
    with _global_handler_lock:
        if _global_handler is not None:
            _global_handler.unload()
            _global_handler = None


# Legacy API for backward compatibility
stt_model = None
tokenizer = None


def load_model():
    """Legacy: Load model (backward compatibility)."""
    global stt_model, tokenizer
    
    handler = get_model_handler()
    handler.load()
    
    stt_model = handler.model
    tokenizer = handler.tokenizer


def transcribe(audio_array: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Optional[str]:
    """
    Legacy: Transcribe audio (backward compatibility).
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate in Hz
        
    Returns:
        Transcribed text or None
    """
    try:
        handler = get_model_handler()
        handler.ensure_loaded()
        return handler.transcribe(audio_array, sample_rate)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


# Export main classes and functions
__all__ = [
    'ModelHandler',
    'get_model_handler',
    'reset_model_handler',
    'load_model',
    'transcribe',
    'ModelLoadError',
    'InferenceError',
]