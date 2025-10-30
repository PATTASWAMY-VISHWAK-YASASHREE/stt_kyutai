"""Command-line helper to run Kyutai STT on an audio file."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from src.transcription_engine import FastTranscriptionEngine

_SAMPLE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
    "resolve/main/test.flac"
)


def _download_sample() -> Path:
    target = Path(tempfile.gettempdir()) / "kyutai_stt_sample.flac"
    urlretrieve(_SAMPLE_URL, target)
    return target


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the audio file to transcribe (wav/flac/mp3).",
    )
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Download a small sample file and transcribe it.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Emit incremental transcripts for each processed chunk.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Chunk size for transcription in seconds.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        help="Overlap between consecutive chunks in seconds.",
    )
    parser.add_argument(
        "--min-char-delta",
        type=int,
        default=None,
        help="Minimum character growth before emitting another streaming update.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.download_sample:
        audio_path = _download_sample()
    elif args.audio:
        audio_path = Path(args.audio).expanduser()
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
    else:
        print("Provide an audio path or use --download-sample", file=sys.stderr)
        return 1

    # Initialize transcription engine
    transcriber = FastTranscriptionEngine()
    # Ensure model is loaded
    transcriber.initialize()
    
    if args.stream:
        # For streaming, we need to adapt the API
        print("⚠️ Streaming mode not yet adapted for FastTranscriptionEngine", file=sys.stderr)
        return 1
    else:
        result = transcriber.transcribe_file(str(audio_path))
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
