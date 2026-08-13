"""ASR transcription service built on faster-whisper."""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from faster_whisper import WhisperModel

from src.audio import get_audio_duration

# Maps UI-facing language names to Whisper language codes.
# Add new Indian languages here (e.g. "Tamil": "ta") to extend support.
LANGUAGE_CODES = {
    "Hindi": "hi",
    "English": "en",
}

VALID_MODEL_SIZES = {"tiny", "base", "small"}


class ASRError(Exception):
    """Raised when model loading or transcription fails."""


@dataclass
class TranscriptionResult:
    transcription: str
    language: str
    processing_time_seconds: float
    audio_duration_seconds: float
    rtf: float


def _resolve_device() -> Tuple[str, str]:
    """Pick device and compute_type. Uses CUDA if available, else CPU int8."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class ASRService:
    """Wraps a faster-whisper model. Loads the model lazily on first use.

    Cache instances of this class (e.g. via st.cache_resource in Streamlit)
    so the model is loaded once per model_size/device, not per request.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        if model_size not in VALID_MODEL_SIZES:
            raise ASRError(
                f"Unsupported model size '{model_size}'. "
                f"Choose from: {', '.join(sorted(VALID_MODEL_SIZES))}"
            )
        self.model_size = model_size

        if device is None or device == "auto":
            resolved_device, resolved_compute_type = _resolve_device()
            device = resolved_device
            compute_type = compute_type or resolved_compute_type
        else:
            compute_type = compute_type or ("float16" if device == "cuda" else "int8")

        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
            except Exception as exc:
                raise ASRError(f"Failed to load Whisper model '{self.model_size}': {exc}")
        return self._model

    def transcribe(self, audio_path: str, language: str) -> TranscriptionResult:
        """Transcribe an audio file in the given language.

        Args:
            audio_path: path to a validated audio file.
            language: UI-facing language name, e.g. "Hindi" or "English".
        """
        lang_code = LANGUAGE_CODES.get(language)
        if lang_code is None:
            raise ASRError(
                f"Unsupported language '{language}'. Supported: {', '.join(LANGUAGE_CODES)}"
            )

        audio_duration = get_audio_duration(audio_path)
        model = self._load_model()

        start = time.perf_counter()
        try:
            segments, _info = model.transcribe(audio_path, language=lang_code)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise ASRError(f"Transcription failed: {exc}")
        processing_time = time.perf_counter() - start

        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0

        return TranscriptionResult(
            transcription=text,
            language=language,
            processing_time_seconds=processing_time,
            audio_duration_seconds=audio_duration,
            rtf=rtf,
        )
