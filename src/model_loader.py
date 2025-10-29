"""Model loading utilities for Kyutai STT checkpoints - Bug-free version."""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

# Configure logging
logger = logging.getLogger(__name__)

# Try importing Kyutai-specific classes
try:
    from transformers import (
        KyutaiSpeechToTextForConditionalGeneration,
        KyutaiSpeechToTextProcessor,
    )
    KYUTAI_AVAILABLE = True
except ImportError:
    logger.warning("Kyutai classes not available, using generic transformers")
    KYUTAI_AVAILABLE = False
    # Fallback imports
    try:
        from transformers import (
            AutoModelForSpeechSeq2Seq as KyutaiSpeechToTextForConditionalGeneration,
            AutoProcessor as KyutaiSpeechToTextProcessor,
        )
    except ImportError as e:
        raise ImportError("transformers library not installed") from e

# Optional quantization support
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    QUANTIZATION_AVAILABLE = False


def get_config_value(key: str, default: Any) -> Any:
    """Safely get config value with fallback."""
    try:
        import config
        value = getattr(config, key, default)
        logger.debug(f"Config {key} = {value}")
        return value
    except (ImportError, AttributeError):
        logger.debug(f"Config {key} not found, using default: {default}")
        return default


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


@dataclass(frozen=True)
class ModelBundle:
    """Container holding the loaded model artifacts."""
    model: Any  # KyutaiSpeechToTextForConditionalGeneration
    processor: Any  # KyutaiSpeechToTextProcessor
    device: torch.device
    dtype: Optional[torch.dtype] = None
    is_quantized: bool = False
    
    def __post_init__(self):
        """Validate bundle on creation."""
        if self.model is None:
            raise ValueError("Model cannot be None")
        if self.processor is None:
            raise ValueError("Processor cannot be None")


class ModelLoader:
    """
    Thread-safe loader for Kyutai STT model/processor pair.
    Handles quantization, device placement, and compilation.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        device_preference: str = "auto",
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        enable_compile: bool = True,
        low_cpu_mem: bool = True,
    ):
        """
        Initialize model loader with configuration.
        
        Args:
            model_id: HuggingFace model ID
            device_preference: Device preference (auto/cuda/cpu/mps)
            torch_dtype: Data type (float32/float16/bfloat16/auto)
            load_in_8bit: Enable 8-bit quantization
            load_in_4bit: Enable 4-bit quantization
            enable_compile: Enable torch.compile
            low_cpu_mem: Use low CPU memory mode
        """
        # Load configuration
        self.model_id = model_id or get_config_value('MODEL_ID', 'kyutai/moshi-speech-to-text')
        self.device_preference = device_preference or get_config_value('DEVICE_PREFERENCE', 'auto')
        self.torch_dtype = torch_dtype or get_config_value('TORCH_DTYPE', 'auto')
        self.load_in_8bit = load_in_8bit or get_config_value('LOAD_IN_8BIT', False)
        self.load_in_4bit = load_in_4bit or get_config_value('LOAD_IN_4BIT', False)
        self.enable_compile = enable_compile and get_config_value('ENABLE_TORCH_COMPILE', True)
        self.low_cpu_mem = low_cpu_mem or get_config_value('LOW_CPU_MEM_USAGE', True)
        
        # State
        self._bundle: Optional[ModelBundle] = None
        self._lock = threading.RLock()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration for consistency."""
        # Check quantization requirements
        if self.load_in_8bit or self.load_in_4bit:
            if not QUANTIZATION_AVAILABLE:
                raise ConfigurationError(
                    "Quantization requested but bitsandbytes not installed. "
                    "Install with: pip install bitsandbytes"
                )
            
            # Quantization requires CUDA
            if not torch.cuda.is_available():
                raise ConfigurationError(
                    "Quantization requires CUDA but CUDA is not available. "
                    "Disable quantization or use a CUDA-enabled environment."
                )
        
        # Validate dtype
        if self.torch_dtype not in (None, 'auto', 'float32', 'float16', 'bfloat16'):
            raise ConfigurationError(f"Invalid torch_dtype: {self.torch_dtype}")
        
        # Check bfloat16 support
        if self.torch_dtype == 'bfloat16':
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                logger.warning("bfloat16 requested but not supported by GPU, using float16")
                self.torch_dtype = 'float16'
    
    def load(self, force_reload: bool = False) -> ModelBundle:
        """
        Load the model if not already loaded.
        Thread-safe and idempotent.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            ModelBundle with model, processor, and device
        """
        with self._lock:
            # Return cached bundle if available
            if self._bundle is not None and not force_reload:
                logger.debug("Returning cached model bundle")
                return self._bundle
            
            # Clean up existing bundle if force reloading
            if force_reload and self._bundle is not None:
                self._cleanup()
            
            try:
                # Resolve device and dtype
                device = self._resolve_device()
                dtype = self._resolve_dtype()
                
                logger.info(
                    f"Loading model '{self.model_id}' on {device} "
                    f"(dtype={dtype}, 8bit={self.load_in_8bit}, 4bit={self.load_in_4bit})"
                )
                
                # Load processor
                processor = self._load_processor()
                
                # Load model
                model, is_quantized = self._load_model(device, dtype)
                
                # Move to device if needed (only for non-quantized models)
                if not is_quantized and model.device != device:
                    logger.debug(f"Moving model from {model.device} to {device}")
                    model = model.to(device)
                
                # Set to eval mode
                model.eval()
                
                # Disable gradients for inference
                for param in model.parameters():
                    param.requires_grad = False
                
                # Compile if requested
                if self.enable_compile:
                    model = self._compile_model(model)
                
                # Create bundle
                self._bundle = ModelBundle(
                    model=model,
                    processor=processor,
                    device=device,
                    dtype=dtype,
                    is_quantized=is_quantized,
                )
                
                logger.info("✅ Model loaded successfully")
                return self._bundle
                
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                self._cleanup()
                raise ModelLoadError(f"Failed to load model: {e}") from e
    
    def _resolve_device(self) -> torch.device:
        """Resolve device from preference."""
        pref = self.device_preference.lower()
        
        if pref == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(pref)
        
        # Validate device is available
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA device requested but not available")
        
        logger.debug(f"Resolved device: {device}")
        return device
    
    def _resolve_dtype(self) -> Optional[torch.dtype]:
        """Resolve dtype from string."""
        if self.torch_dtype in (None, "auto"):
            return None
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        dtype = dtype_map.get(self.torch_dtype.lower())
        if dtype is None:
            raise ConfigurationError(f"Invalid dtype: {self.torch_dtype}")
        
        logger.debug(f"Resolved dtype: {dtype}")
        return dtype
    
    def _load_processor(self) -> Any:
        """Load processor."""
        try:
            processor = KyutaiSpeechToTextProcessor.from_pretrained(self.model_id)
            logger.debug("Processor loaded")
            return processor
        except Exception as e:
            raise ModelLoadError(f"Failed to load processor: {e}") from e
    
    def _load_model(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype]
    ) -> tuple[Any, bool]:
        """
        Load model with proper configuration.
        
        Returns:
            Tuple of (model, is_quantized)
        """
        is_quantized = False
        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": self.low_cpu_mem,
        }
        
        # Handle quantization
        if self.load_in_8bit or self.load_in_4bit:
            if not QUANTIZATION_AVAILABLE:
                raise ModelLoadError("BitsAndBytesConfig not available")
            
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_threshold=6.0,
            )
            
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
            is_quantized = True
            
        else:
            # Non-quantized model
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            
            if device.type == "cuda":
                model_kwargs["device_map"] = "auto"
        
        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        
        try:
            model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            logger.debug("Model loaded")
            return model, is_quantized
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e
    
    def _compile_model(self, model: Any) -> Any:
        """Compile model with torch.compile if available."""
        # Check PyTorch version
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
        
        try:
            logger.info("Compiling model with torch.compile...")
            compiled = torch.compile(model, mode="reduce-overhead")
            logger.info("✅ Model compiled successfully")
            return compiled
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, using uncompiled model")
            return model
    
    def _cleanup(self) -> None:
        """Clean up loaded model."""
        if self._bundle is not None:
            logger.debug("Cleaning up model bundle")
            
            # Delete model
            if self._bundle.model is not None:
                del self._bundle.model
            
            # Delete processor
            if self._bundle.processor is not None:
                del self._bundle.processor
            
            # Clear CUDA cache if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._bundle = None
    
    def unload(self) -> None:
        """Unload model and free memory."""
        with self._lock:
            logger.info("Unloading model...")
            self._cleanup()
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._bundle is not None
    
    def get_bundle(self) -> Optional[ModelBundle]:
        """Get current bundle (may be None)."""
        return self._bundle
    
    def ensure_loaded(self) -> ModelBundle:
        """Ensure model is loaded, load if needed."""
        if not self.is_loaded:
            return self.load()
        return self._bundle
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()


# Global singleton management
_global_loader_lock = threading.RLock()
_global_loader: Optional[ModelLoader] = None


def get_model_loader(**kwargs) -> ModelLoader:
    """
    Get or create global ModelLoader instance.
    Thread-safe singleton pattern.
    """
    global _global_loader
    
    with _global_loader_lock:
        if _global_loader is None:
            _global_loader = ModelLoader(**kwargs)
        return _global_loader


def reset_model_loader() -> None:
    """Reset global model loader."""
    global _global_loader
    
    with _global_loader_lock:
        if _global_loader is not None:
            _global_loader.unload()
            _global_loader = None


# Export main components
__all__ = [
    'ModelBundle',
    'ModelLoader',
    'ModelLoadError',
    'ConfigurationError',
    'get_model_loader',
    'reset_model_loader',
]