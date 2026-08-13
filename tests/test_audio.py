import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    AudioValidationError,
    convert_to_wav,
    prepare_audio_for_transcription,
    validate_audio_file,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def test_rejects_unsupported_extension():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not audio")
        tmp_path = tmp.name

    try:
        with pytest.raises(AudioValidationError, match="Unsupported audio format"):
            validate_audio_file(tmp_path)
    finally:
        os.remove(tmp_path)


def test_rejects_missing_file():
    with pytest.raises(AudioValidationError, match="File not found"):
        validate_audio_file("/tmp/does_not_exist_indicspeechlab_test.wav")


def test_prepare_audio_passes_wav_through_unchanged(tmp_path):
    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), np.zeros(1600, dtype=np.float32), 16000)

    result = prepare_audio_for_transcription(str(wav_path), str(tmp_path))

    assert result == str(wav_path)


def test_prepare_audio_rejects_unsupported_extension(tmp_path):
    bogus_path = tmp_path / "notes.txt"
    bogus_path.write_text("hello")

    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        prepare_audio_for_transcription(str(bogus_path), str(tmp_path))


def test_convert_to_wav_missing_ffmpeg_raises_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    input_path = tmp_path / "clip.m4a"
    input_path.write_bytes(b"fake audio bytes")

    with pytest.raises(AudioValidationError, match="FFmpeg"):
        convert_to_wav(str(input_path), str(tmp_path / "out.wav"))


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed on this system")
def test_convert_to_wav_produces_16k_mono_pcm(tmp_path):
    """End-to-end: encode a real M4A with ffmpeg, then convert it back via our pipeline."""
    source_wav = tmp_path / "source.wav"
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(source_wav), tone, sr)

    m4a_path = tmp_path / "clip.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source_wav), str(m4a_path)],
        capture_output=True,
        check=True,
    )

    output_path = convert_to_wav(str(m4a_path), str(tmp_path / "converted.wav"))

    info = sf.info(output_path)
    assert info.samplerate == 16000
    assert info.channels == 1


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed on this system")
def test_prepare_audio_converts_m4a_via_ffmpeg(tmp_path):
    """The full routing path: prepare_audio_for_transcription sends .m4a through FFmpeg."""
    source_wav = tmp_path / "source.wav"
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, sr))).astype(np.float32)
    sf.write(str(source_wav), tone, sr)

    m4a_path = tmp_path / "clip with spaces.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source_wav), str(m4a_path)],
        capture_output=True,
        check=True,
    )

    result_path = prepare_audio_for_transcription(str(m4a_path), str(tmp_path))

    assert result_path != str(m4a_path)
    info = sf.info(result_path)
    assert info.samplerate == 16000
    assert info.channels == 1
