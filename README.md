# IndicSpeechLab

IndicSpeechLab is a lightweight ASR evaluation platform for measuring speech recognition quality and inference performance on Indian-language audio.

## Problem

Evaluating ASR (Automatic Speech Recognition) quality for Indian languages usually requires stitching together separate scripts for transcription, error-rate calculation, and result tracking. IndicSpeechLab bundles this into a single local tool: upload audio, transcribe it with Whisper, compare against a reference transcript, and get standard ASR metrics — all from one dashboard.

## Features

- Upload an audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`)
- Transcribe with an open-source Whisper model (via `faster-whisper`), running locally
- Select language: Hindi or English
- Optionally supply a ground-truth reference transcript
- Automatic calculation of **Word Error Rate (WER)** and **Character Error Rate (CER)**
- Inference timing and **Real-Time Factor (RTF)**
- Results persisted to CSV, viewable and downloadable from the dashboard
- Hindi-safe text normalization (no transliteration, Devanagari preserved)

## Architecture

```
Audio
 ↓
Whisper ASR
 ↓
Transcription
 ↓
Text Normalization
 ↓
WER / CER
 ↓
Latency / RTF
 ↓
CSV Results
 ↓
Streamlit Dashboard
```

## Tech Stack

- **Python 3.10+**
- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper inference (CPU int8 by default, CUDA if available)
- **Streamlit** — dashboard UI
- **jiwer** — WER/CER calculation
- **pandas** — CSV result storage and display
- **librosa** / **soundfile** — audio duration and validation
- **numpy**
- **FFmpeg** — converts compressed uploads (M4A, MP3, FLAC, OGG) to 16kHz mono PCM WAV before transcription

## Installation

Install FFmpeg first (required to decode M4A, MP3, FLAC, and OGG uploads):

```bash
brew install ffmpeg
```

Then set up the Python environment:

```bash
cd indic-speech-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to change the default model:

```
WHISPER_MODEL=small   # tiny | base | small
DEVICE=auto           # auto | cpu | cuda
```

## Usage

```bash
streamlit run app.py
```

1. Pick a Whisper model and language in the sidebar.
2. Upload an audio file.
3. (Optional) Paste a reference transcript.
4. Click **Transcribe & Evaluate**.
5. Review the transcription, WER/CER, processing time, and RTF.
6. Browse or download past evaluations from the **Previous Evaluations** table.

Run tests:

```bash
pytest
```

## Metrics

- **WER (Word Error Rate)** — the fraction of words in the reference that were substituted, deleted, or inserted incorrectly in the hypothesis. Lower is better.
- **CER (Character Error Rate)** — the same idea at the character level. More forgiving of minor spelling/segmentation differences than WER, useful for morphologically rich languages.
- **RTF (Real-Time Factor)** — `processing_time / audio_duration`. An RTF below 1.0 means the system transcribed the audio faster than real time (e.g. RTF = 0.3 means a 10-second clip took 3 seconds to transcribe).

If no reference transcript is provided, WER and CER are reported as `N/A` rather than a misleading zero.

## Project Structure

```
indic-speech-lab/
│
├── app.py                   # Streamlit dashboard
│
├── src/
│   ├── asr.py                # ASRService: faster-whisper wrapper, lazy model loading
│   ├── metrics.py             # WER/CER calculation, text normalization
│   ├── audio.py                # Audio validation, FFmpeg conversion, duration extraction
│   └── storage.py              # CSV-backed result storage
│
├── data/results/               # evaluations.csv (gitignored)
├── samples/                     # sample audio files (not committed)
├── tests/                       # pytest test suite
│
├── requirements.txt
├── .env.example
└── .gitignore
```

```bash
source .venv/bin/activate
streamlit run app.py
```

## Future Improvements

- Additional Indian languages: Tamil, Telugu, Bengali, Marathi, Hinglish
- Multiple ASR model backends (Hugging Face Transformers, other Whisper variants)
- Batch evaluation and multi-file comparison
- Noise robustness experiments and SNR-based evaluation
- Voice activity detection (VAD) and silence removal preprocessing
- Charts and trend visualization across evaluation history
- GPU-accelerated inference tuning
- PostgreSQL-backed storage (storage.py is already isolated for this swap)
- FastAPI backend for programmatic access
- Docker packaging and cloud deployment
