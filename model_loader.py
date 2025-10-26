"""Model loading utilities for Kyutai STT checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
)

import config

logger = logging.getLogger(__name__)

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore


@dataclass(frozen=True)
class ModelBundle:
    """Container holding the loaded model artefacts."""

    model: KyutaiSpeechToTextForConditionalGeneration
    processor: KyutaiSpeechToTextProcessor
    device: torch.device


class ModelLoader:
    """Loads and caches the Kyutai STT model/processor pair."""

    def __init__(self) -> None:
        self._bundle: Optional[ModelBundle] = None

    def load(self, force_reload: bool = False) -> ModelBundle:
        """Load the configured model if it is not already resident in memory."""
        if self._bundle is not None and not force_reload:
            return self._bundle

        device = _resolve_device(config.DEVICE_PREFERENCE)
        dtype = _resolve_dtype(config.TORCH_DTYPE)

        quant_config = None
        if (config.LOAD_IN_8BIT or config.LOAD_IN_4BIT) and BitsAndBytesConfig is None:
            raise RuntimeError(
                "Quantized loading requested but bitsandbytes is not installed."
                " Install bitsandbytes==0.43.* and retry, or disable quantization flags."
            )

        if BitsAndBytesConfig is not None and (config.LOAD_IN_8BIT or config.LOAD_IN_4BIT):
            quant_config = BitsAndBytesConfig(
                load_in_8bit=config.LOAD_IN_8BIT,
                load_in_4bit=config.LOAD_IN_4BIT,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_threshold=6.0,
            )

        logger.info(
            "Loading Kyutai STT model '%s' on device=%s (dtype=%s, quantized=%s/%s)",
            config.MODEL_ID,
            device,
            dtype if dtype is not None else "default",
            config.LOAD_IN_8BIT,
            config.LOAD_IN_4BIT,
        )

        processor = KyutaiSpeechToTextProcessor.from_pretrained(config.MODEL_ID)

        model_kwargs = {
            "device_map": "auto" if device.type == "cuda" else None,
            "torch_dtype": dtype,
            "quantization_config": quant_config,
        }

        # Remove None values so Transformers does not complain about unexpected arguments.
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            config.MODEL_ID,
            **model_kwargs,
        )

        if model.device.type != device.type:
            model = model.to(device)

        model.eval()

        if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")  # type: ignore[attr-defined]
                logger.info("Model compiled with torch.compile for faster decoding.")
            except RuntimeError as compile_err:  # pragma: no cover - environment specific
                logger.warning("torch.compile failed: %s", compile_err)

        self._bundle = ModelBundle(model=model, processor=processor, device=device)
        return self._bundle


def _resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _resolve_dtype(dtype_hint: Optional[str]) -> Optional[torch.dtype]:
    if dtype_hint in (None, "auto"):
        return None

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    resolved = mapping.get(str(dtype_hint).lower())
    if resolved is None:
        raise ValueError(f"Unsupported TORCH_DTYPE value: {dtype_hint}")
    return resolved
