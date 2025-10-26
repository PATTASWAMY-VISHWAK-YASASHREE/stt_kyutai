# Kyutai STT Starter Stack

This folder contains a modular speech-to-text stack built around Kyutai's streaming models on Hugging Face. The code is organised into small, testable modules so you can swap components or integrate them into larger applications.

## Modules

| File | Responsibility |
| --- | --- |
| `config.py` | Central place for model IDs, device preferences, audio preprocessing parameters, and generation defaults. |
| `model_loader.py` | Loads the `KyutaiSpeechToTextForConditionalGeneration` checkpoint together with its processor, handling quantisation hints and `torch.compile` optimisations. |
| `audio_processor.py` | Decodes, normalises, resamples, trims silence with energy-based VAD, and chunks audio using lightweight backends (`soundfile`, `torchaudio`, fallback to `pydub`). |
| `encoding.py` | Wraps processor calls to prepare model inputs and decode generated tokens. |
| `transcriber.py` | High-level façade combining the loader, audio utilities, and decoding logic with optional chunked or streaming transcription. |
| `main.py` | FastAPI application exposing REST + WebSocket endpoints for live transcription. |
| `transcribe_cli.py` | Convenience CLI that can transcribe a local file or download a small sample to validate the pipeline. |

Unit tests live under `tests/` and focus on deterministic helpers such as audio chunking and transcript aggregation.

## Installation

Create a fresh virtual environment and install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r test1/requirements.txt
```

> **Tip:** When running on NVIDIA GPUs, install the CUDA-enabled builds of `torch` and `torchaudio` that match your driver.

Optional dependencies:

- `bitsandbytes`: enable 8-bit or 4-bit loading by toggling `LOAD_IN_8BIT` / `LOAD_IN_4BIT` in `config.py`.
- `pydub`: used only as a last-resort decoder, helpful when ffmpeg is already part of your stack.

## Running the API server

```powershell
python -m uvicorn test1.main:app --host 0.0.0.0 --port 8000
```

The server loads the model on startup and exposes:

- `GET /` – health check with model metadata.
- `GET /docs` – interactive Swagger UI.
- `WS /ws` – stream base64-encoded audio chunks for real-time transcripts.

## CLI usage

Run the helper CLI with an existing audio file:

```powershell
python test1/transcribe_cli.py path\to\audio.wav
```

Or download a tiny sample from Hugging Face and transcribe it:

```powershell
python test1/transcribe_cli.py --download-sample
```

Enable incremental updates while the audio is processed:

```powershell
python test1/transcribe_cli.py path\to\audio.wav --stream
```

## Configuration knobs

Edit `config.py` to adjust:

- `MODEL_ID` – choose between `kyutai/stt-1b-en_fr-trfs` (smaller, multilingual) or `kyutai/stt-2.6b-en-trfs` (highest accuracy).
- `DEVICE_PREFERENCE` – force `"cuda"`, `"cpu"`, or leave `"auto"` for automatic selection.
- `ENABLE_VAD` and related VAD knobs – energy-based voice activity detection trims leading/trailing silence before inference.
- `CHUNK_LENGTH_SECONDS` / `CHUNK_OVERLAP_SECONDS` – control chunking trade-offs for long recordings.
- `STREAMING_*` – tune streaming chunk size, overlap, and emission cadence for incremental transcripts.
- `ENABLE_TORCH_COMPILE` – compile the model on repeated use (requires PyTorch ≥2.1).

## Testing

Lightweight unit tests validate the pure-Python utilities:

```powershell
python -m unittest discover -s test1/tests
```

Tests are skipped automatically if optional dependencies (e.g., `numpy`) are missing.

## Resource considerations

- Prefer the 1B checkpoint for edge devices or CPU-only deployments.
- Quantisation drastically reduces memory but requires `bitsandbytes`; ensure the package is available on your platform before enabling it.
- Audio is always resampled to 24 kHz, the native rate of Kyutai STT models, to balance latency and accuracy.
- Chunk overlap defaults to 1 second based on community heuristics to reduce word truncation at chunk boundaries.
