"""Audio file validation, FFmpeg-based format conversion, and inspection utilities."""

import os
import shutil
import subprocess

import librosa

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

# Compressed formats that librosa/soundfile cannot reliably decode directly
# and must be converted to WAV via FFmpeg first.
FFMPEG_CONVERTIBLE_FORMATS = {".mp3", ".m4a", ".flac", ".ogg"}

TARGET_SAMPLE_RATE = 16000


class AudioValidationError(Exception):
    """Raised when an audio file fails validation or conversion."""


def validate_audio_file(path: str) -> None:
    """Validate that a file exists, has a supported extension, and is non-empty.

    This does not attempt to decode the file. Some supported formats (m4a,
    mp3, ogg) are not directly decodable by librosa/soundfile and must go
    through convert_to_wav()/prepare_audio_for_transcription() first, which
    perform decodability and duration checks on a guaranteed-readable WAV.
    """
    if not os.path.exists(path):
        raise AudioValidationError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported formats: {supported}"
        )

    if os.path.getsize(path) == 0:
        raise AudioValidationError("Audio file is empty.")


def get_audio_duration(path: str) -> float:
    """Return the duration of an audio file in seconds.

    Raises AudioValidationError if the file cannot be decoded. Only call
    this on formats librosa/soundfile can read directly (WAV, or a WAV
    produced by convert_to_wav).
    """
    try:
        duration = float(librosa.get_duration(path=path))
    except Exception as exc:
        raise AudioValidationError(f"Could not read audio file (corrupted or unreadable): {exc}")

    if duration <= 0:
        raise AudioValidationError("Audio file has zero duration.")

    return duration


def _ffmpeg_path() -> str:
    """Locate the ffmpeg binary, raising a friendly error if it's missing."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioValidationError(
            "FFmpeg is required to process this audio format but was not found "
            "on this system. Install it with: brew install ffmpeg"
        )
    return ffmpeg


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert an audio file to 16kHz mono PCM WAV using FFmpeg.

    Runs ffmpeg via subprocess.run with an explicit argument list (never
    shell=True), so filenames containing spaces or special characters are
    handled safely without any shell-quoting concerns.
    """
    ffmpeg = _ffmpeg_path()

    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise AudioValidationError("Audio conversion timed out.")
    except Exception as exc:
        raise AudioValidationError(f"Failed to run FFmpeg: {exc}")

    if result.returncode != 0:
        stderr_lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
        detail = stderr_lines[-1] if stderr_lines else "unknown FFmpeg error"
        raise AudioValidationError(f"FFmpeg could not convert this audio file: {detail}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise AudioValidationError("FFmpeg produced no output for this audio file.")

    return output_path


def prepare_audio_for_transcription(source_path: str, work_dir: str) -> str:
    """Return the path to a WAV file ready for duration checks and transcription.

    .wav files pass through unchanged (existing WAV handling). Compressed
    formats (.mp3, .m4a, .flac, .ogg) are converted via FFmpeg to 16kHz
    mono PCM WAV in work_dir first.
    """
    ext = os.path.splitext(source_path)[1].lower()

    if ext == ".wav":
        return source_path

    if ext in FFMPEG_CONVERTIBLE_FORMATS:
        output_path = os.path.join(work_dir, "converted.wav")
        return convert_to_wav(source_path, output_path)

    supported = ", ".join(sorted(SUPPORTED_FORMATS))
    raise AudioValidationError(
        f"Unsupported audio format '{ext}'. Supported formats: {supported}"
    )
