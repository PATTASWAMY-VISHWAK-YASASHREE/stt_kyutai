"""Audio decoding, normalisation, VAD trimming, and chunking helpers."""

from __future__ import annotations

import io
import logging
from typing import Any, Generator, Iterable, List, Optional, Tuple
import importlib

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
except Exception:  # pragma: no cover - fallback for older NumPy
    sliding_window_view = None  # type: ignore

import config

logger = logging.getLogger(__name__)

try:  # Optional backend with broad codec support.
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

try:
    import torchaudio  # type: ignore
    from torchaudio.functional import resample as ta_resample  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torchaudio = None  # type: ignore
    ta_resample = None  # type: ignore

try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore


def decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode ``audio_bytes`` into mono float32 samples and return (audio, sample_rate)."""

    backends = (_decode_with_soundfile, _decode_with_torchaudio, _decode_with_pydub)
    errors = []
    for backend in backends:
        if backend is None:
            continue
        try:
            audio, sr = backend(audio_bytes)
            return ensure_mono(audio), sr
        except Exception as exc:  # pragma: no cover - backend specific
            errors.append(f"{backend.__name__}: {exc}")
            logger.debug("Audio decode failed via %s: %s", backend.__name__, exc)

    raise RuntimeError(
        "Unable to decode audio bytes. Tried soundfile, torchaudio, and pydub."
        + (" Errors: " + " | ".join(errors) if errors else "")
    )


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    processed = audio.astype(np.float32, copy=False)

    if config.PEAK_NORMALIZE:
        peak = np.abs(processed).max(initial=1.0)
        if peak > 0.0:
            processed = processed / peak

    if config.NORMALIZE_L2:
        l2 = np.linalg.norm(processed)
        if l2 > 0.0:
            processed = processed / l2

    return np.clip(processed, -1.0, 1.0)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    if ta_resample is not None:
        tensor = torch_from_numpy(audio)
        resampled = ta_resample(tensor, source_rate, target_rate)
        return resampled.cpu().numpy()

    # Fallback to NumPy linear interpolation when torchaudio is unavailable.
    duration = audio.shape[0] / float(source_rate)
    target_length = int(round(duration * target_rate))
    target_indices = np.linspace(0, audio.shape[0] - 1, num=target_length)
    return np.interp(target_indices, np.arange(audio.shape[0]), audio).astype(np.float32)


def prepare_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    audio, sr = decode_audio(audio_bytes)
    audio = normalize_audio(audio)
    audio = resample_audio(audio, sr, config.TARGET_SAMPLE_RATE)

    if config.ENABLE_VAD:
        trimmed = trim_silence(
            audio,
            sample_rate=config.TARGET_SAMPLE_RATE,
            window_seconds=config.VAD_WINDOW_SECONDS,
            hop_seconds=config.VAD_HOP_SECONDS,
            energy_threshold_db=config.VAD_ENERGY_THRESHOLD_DB,
            hangover_seconds=config.VAD_HANGOVER_SECONDS,
            min_speech_seconds=config.VAD_MIN_SPEECH_SECONDS,
            min_silence_seconds=config.VAD_MIN_SILENCE_SECONDS,
        )
        if trimmed is not None:
            audio = trimmed

    return audio.astype(np.float32, copy=False), config.TARGET_SAMPLE_RATE


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = config.CHUNK_LENGTH_SECONDS,
    overlap_seconds: float = config.CHUNK_OVERLAP_SECONDS,
) -> Generator[np.ndarray, None, None]:
    if chunk_seconds <= 0:
        yield audio
        return

    chunk_size = int(chunk_seconds * sample_rate)
    if chunk_size <= 0:
        raise ValueError("chunk_seconds too small relative to sample rate")

    stride = max(1, chunk_size - int(overlap_seconds * sample_rate))
    total = audio.shape[0]

    for start in range(0, total, stride):
        end = min(start + chunk_size, total)
        yield audio[start:end]
        if end == total:
            break


def streaming_chunk_generator(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = config.STREAMING_CHUNK_SECONDS,
    overlap_seconds: float = config.STREAMING_OVERLAP_SECONDS,
) -> Iterable[np.ndarray]:
    """Yield overlapping chunks tuned for incremental streaming emission."""

    yield from chunk_audio(audio, sample_rate, chunk_seconds, overlap_seconds)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float,
    hop_seconds: float,
    energy_threshold_db: float,
    hangover_seconds: float,
    min_speech_seconds: float,
    min_silence_seconds: float,
) -> Optional[np.ndarray]:
    segments = voice_activity_segments(
        audio,
        sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        energy_threshold_db=energy_threshold_db,
        hangover_seconds=hangover_seconds,
        min_speech_seconds=min_speech_seconds,
        min_silence_seconds=min_silence_seconds,
    )

    if not segments:
        return None

    start = segments[0][0]
    end = segments[-1][1]
    return audio[start:end]


def voice_activity_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float,
    hop_seconds: float,
    energy_threshold_db: float,
    hangover_seconds: float,
    min_speech_seconds: float,
    min_silence_seconds: float,
) -> List[Tuple[int, int]]:
    if audio.size == 0:
        return []

    frame_length = max(1, int(window_seconds * sample_rate))
    hop_length = max(1, int(hop_seconds * sample_rate))
    hangover_frames = int(np.ceil(hangover_seconds / hop_seconds)) if hangover_seconds > 0 else 0
    min_speech_frames = int(np.ceil(max(min_speech_seconds, 0.0) / hop_seconds))
    min_silence_frames = int(np.ceil(max(min_silence_seconds, 0.0) / hop_seconds))

    energies = _short_time_energy(audio, frame_length, hop_length)
    if energies.size == 0:
        return []

    log_energy = 10.0 * np.log10(energies + 1e-12)
    mask_frames = log_energy > energy_threshold_db

    if hangover_frames > 0:
        kernel = np.ones(2 * hangover_frames + 1, dtype=int)
        mask_frames = np.convolve(mask_frames.astype(int), kernel, mode="same") > 0

    mask_frames = _remove_short_speech(mask_frames, min_speech_frames)

    segments = _frames_to_segments(mask_frames, frame_length, hop_length, audio.shape[0])
    if not segments:
        return []

    if min_silence_frames > 0:
        segments = _merge_close_segments(segments, hop_length, min_silence_frames)

    return segments


def _short_time_energy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    if audio.shape[0] < frame_length:
        padded = np.pad(audio, (0, frame_length - audio.shape[0]), mode="constant")
    else:
        padded = audio

    if sliding_window_view is not None:
        windows = sliding_window_view(padded, frame_length)[::hop_length]
    else:  # pragma: no cover - legacy NumPy fallback
        total_frames = 1 + max(0, (padded.shape[0] - frame_length) // hop_length)
        windows = np.empty((total_frames, frame_length), dtype=padded.dtype)
        for idx in range(total_frames):
            start = idx * hop_length
            end = start + frame_length
            windows[idx] = padded[start:end]

    if windows.size == 0:
        return np.empty(0, dtype=np.float32)
    energy = np.mean(np.square(windows), axis=-1)
    return energy.astype(np.float32)


def _remove_short_speech(mask_frames: np.ndarray, min_speech_frames: int) -> np.ndarray:
    if min_speech_frames <= 1:
        return mask_frames

    mask = mask_frames.astype(int)
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)

    keep = mask_frames.copy()
    for start, stop in zip(starts, stops):
        if stop - start < min_speech_frames:
            keep[start:stop] = False
    return keep


def _frames_to_segments(
    mask_frames: np.ndarray,
    frame_length: int,
    hop_length: int,
    total_samples: int,
) -> List[Tuple[int, int]]:
    mask = mask_frames.astype(int)
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)

    segments: List[Tuple[int, int]] = []
    for start_frame, stop_frame in zip(starts, stops):
        start_sample = start_frame * hop_length
        end_sample = min(stop_frame * hop_length + frame_length, total_samples)
        if start_sample >= end_sample:
            continue
        segments.append((start_sample, end_sample))
    return segments


def _merge_close_segments(
    segments: List[Tuple[int, int]],
    hop_length: int,
    min_silence_frames: int,
) -> List[Tuple[int, int]]:
    if not segments:
        return []

    merged: List[Tuple[int, int]] = [segments[0]]
    min_gap_samples = min_silence_frames * hop_length

    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap_samples:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _decode_with_soundfile(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile backend unavailable")

    with sf.SoundFile(io.BytesIO(audio_bytes)) as handle:
        audio = handle.read(dtype="float32")
        return audio, handle.samplerate


def _decode_with_torchaudio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio backend unavailable")

    tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))  # type: ignore[arg-type]
    audio = tensor.squeeze(0).numpy()
    return audio, int(sr)


def _decode_with_pydub(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    if AudioSegment is None:
        raise RuntimeError("pydub backend unavailable")

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment = segment.set_channels(config.TARGET_CHANNELS)
    raw = segment.get_array_of_samples()
    audio = np.array(raw).astype(np.float32) / 32768.0
    return audio, segment.frame_rate


def torch_from_numpy(audio: np.ndarray) -> Any:
    torch_module = importlib.import_module("torch")

    tensor = torch_module.from_numpy(audio.astype(np.float32, copy=False))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor