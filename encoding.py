"""Utility helpers for preparing inputs and decoding outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from transformers import BatchFeature, KyutaiSpeechToTextProcessor


def prepare_inputs(
    processor: KyutaiSpeechToTextProcessor,
    audio_samples: Sequence[np.ndarray] | np.ndarray,
    sampling_rate: int,
    device: torch.device,
    **processor_kwargs: Any,
) -> BatchFeature:
    """Prepare model inputs on the desired device."""

    features = processor(
        audio=audio_samples,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
        **processor_kwargs,
    )   

def decode_tokens(
    processor: KyutaiSpeechToTextProcessor,
    token_batches: Sequence[torch.Tensor],
    skip_special_tokens: bool = True,
) -> List[str]:
    """Convert generated token ids to text."""

    return processor.batch_decode(token_batches, skip_special_tokens=skip_special_tokens)


def aggregate_transcripts(chunks: Iterable[str]) -> str:
    """Join chunk transcripts with whitespace while avoiding duplicates."""

    cleaned: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if cleaned and chunk == cleaned[-1]:
            continue
        cleaned.append(chunk)
    return " ".join(cleaned)

