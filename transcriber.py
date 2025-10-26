"""High-level transcription orchestration using the modular components."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch

import audio_processor
import config
import encoding
from model_loader import ModelBundle, ModelLoader


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _State:
	bundle: ModelBundle


class KyutaiTranscriber:
	"""Facade that stitches model loading, audio prep, and decoding together."""

	def __init__(self, loader: Optional[ModelLoader] = None) -> None:
		self._loader = loader or ModelLoader()
		self._state: Optional[_State] = None

	def ensure_ready(self) -> ModelBundle:
		if self._state is None:
			bundle = self._loader.load()
			self._state = _State(bundle=bundle)
		return self._state.bundle

	@property
	def is_ready(self) -> bool:
		return self._state is not None

	async def transcribe_bytes_async(self, audio_bytes: bytes) -> str:
		loop = asyncio.get_running_loop()
		return await loop.run_in_executor(None, self.transcribe_bytes, audio_bytes)

	def transcribe_bytes(self, audio_bytes: bytes) -> str:
		audio, sample_rate = audio_processor.prepare_audio(audio_bytes)
		# If the model hasn't been loaded (skipped at startup to conserve memory),
		# avoid attempting to load it on demand while testing the API/UI. Return
		# a clear placeholder so the client can still validate the request/response
		# flow without requiring the heavy model to be loaded.
		if not self.is_ready:
			logger.info("Model not loaded; returning placeholder transcription for UI testing")
			return "[model not loaded - transcription skipped for lightweight test]"
		return self.transcribe_array(audio, sample_rate)

	def transcribe_file(self, path: str) -> str:
		with open(path, "rb") as handle:
			return self.transcribe_bytes(handle.read())

	def stream_file(
		self,
		path: str,
		chunk_seconds: Optional[float] = None,
		overlap_seconds: Optional[float] = None,
		min_char_delta: Optional[int] = None,
		emit_empty: Optional[bool] = None,
	) -> Iterable[str]:
		with open(path, "rb") as handle:
			yield from self.stream_bytes(
				handle.read(),
				chunk_seconds=chunk_seconds,
				overlap_seconds=overlap_seconds,
				min_char_delta=min_char_delta,
				emit_empty=emit_empty,
			)

	def transcribe_array(
		self,
		audio: np.ndarray,
		sample_rate: int,
		chunk_seconds: Optional[float] = None,
	) -> str:
		bundle = self.ensure_ready()
		chunk_seconds = chunk_seconds if chunk_seconds is not None else config.CHUNK_LENGTH_SECONDS

		if not np.isfinite(audio).all():
			raise ValueError("Audio contains NaN or infinite values after preprocessing.")

		if chunk_seconds and chunk_seconds > 0 and _requires_chunking(audio, sample_rate, chunk_seconds):
			transcripts = []
			for chunk in audio_processor.chunk_audio(audio, sample_rate, chunk_seconds):
				transcripts.append(self._generate_text(bundle, chunk, sample_rate))
			return encoding.aggregate_transcripts(transcripts)

		return self._generate_text(bundle, audio, sample_rate)

	def stream_bytes(
		self,
		audio_bytes: bytes,
		chunk_seconds: Optional[float] = None,
		overlap_seconds: Optional[float] = None,
		min_char_delta: Optional[int] = None,
		emit_empty: Optional[bool] = None,
	) -> Iterable[str]:
		audio, sample_rate = audio_processor.prepare_audio(audio_bytes)
		yield from self.stream_array(
			audio,
			sample_rate,
			chunk_seconds=chunk_seconds,
			overlap_seconds=overlap_seconds,
			min_char_delta=min_char_delta,
			emit_empty=emit_empty,
		)

	def stream_array(
		self,
		audio: np.ndarray,
		sample_rate: int,
		chunk_seconds: Optional[float] = None,
		overlap_seconds: Optional[float] = None,
		min_char_delta: Optional[int] = None,
		emit_empty: Optional[bool] = None,
	) -> Iterable[str]:
		bundle = self.ensure_ready()
		chunk_seconds = chunk_seconds if chunk_seconds is not None else config.STREAMING_CHUNK_SECONDS
		overlap_seconds = overlap_seconds if overlap_seconds is not None else config.STREAMING_OVERLAP_SECONDS
		min_char_delta = (
			min_char_delta if min_char_delta is not None else config.STREAMING_MIN_CHAR_DELTA
		)
		emit_empty = emit_empty if emit_empty is not None else config.STREAMING_EMIT_EMPTY_UPDATES

		if not np.isfinite(audio).all():
			raise ValueError("Audio contains NaN or infinite values after preprocessing.")

		partial: List[str] = []
		last_emitted = ""

		for chunk in audio_processor.streaming_chunk_generator(
			audio,
			sample_rate,
			chunk_seconds=chunk_seconds,
			overlap_seconds=overlap_seconds,
		):
			text = self._generate_text(bundle, chunk, sample_rate)
			if text:
				partial.append(text)
			combined = encoding.aggregate_transcripts(partial)

			if not combined and not emit_empty:
				continue

			if combined == last_emitted:
				continue

			if (
				min_char_delta > 0
				and len(combined) >= len(last_emitted)
				and combined.startswith(last_emitted)
				and (len(combined) - len(last_emitted)) < min_char_delta
			):
				continue

			last_emitted = combined
			yield combined

	def _generate_text(self, bundle: ModelBundle, audio: np.ndarray, sample_rate: int) -> str:
		inputs = encoding.prepare_inputs(
			bundle.processor,
			audio,
			sampling_rate=sample_rate,
			device=bundle.device,
		)

		gen_kwargs = config.DEFAULT_GENERATION_OPTIONS.__dict__.copy()

		with torch.inference_mode():
			output = bundle.model.generate(**inputs, **gen_kwargs)

		decoded = encoding.decode_tokens(bundle.processor, output)
		return decoded[0].strip() if decoded else ""


def _requires_chunking(audio: np.ndarray, sr: int, chunk_seconds: float) -> bool:
	max_samples = int(chunk_seconds * sr)
	return audio.shape[0] > max_samples

