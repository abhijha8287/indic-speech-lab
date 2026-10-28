# IndicSpeechLab

A Streamlit tool that transcribes Hindi or English audio with Whisper and scores the result with WER, CER and real-time factor.

## Why this exists
Evaluating speech recognition for Indian languages usually means stitching together separate scripts for transcription, text cleanup and scoring. This puts the whole loop in one place, running locally.

## What it does
- Accepts `.wav`, `.mp3`, `.m4a`, `.flac` or `.ogg` uploads
- Transcribes locally with `faster-whisper` (tiny, base or small model) in Hindi or English
- Normalises text without transliterating, so Devanagari stays intact
- Computes WER and CER against an optional reference transcript (shown as N/A without one)
- Measures processing time and real-time factor (processing time / audio length)
- Saves every evaluation to CSV, viewable and downloadable in the app

## Project structure
```
indic-speech-lab/
├── src/
│   ├── asr.py        Whisper loading and transcription
│   ├── audio.py      Audio decoding and duration
│   ├── metrics.py    Normalisation, WER, CER, RTF
│   └── storage.py    CSV results storage
├── tests/            Audio and metric tests
├── data/results/     Saved evaluation results
├── samples/          Place test audio here
├── app.py            Streamlit app
├── packages.txt      System packages (ffmpeg)
├── .env.example      Model and device settings
└── requirements.txt  Python dependencies
```

## Getting started
Requirements: Python 3.10+, FFmpeg.
```bash
git clone https://github.com/abhijha8287/indic-speech-lab.git
cd indic-speech-lab
brew install ffmpeg            # or: sudo apt install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
Pick a model and language in the sidebar, upload audio, optionally paste a reference transcript, then click **Transcribe & Evaluate**.

## Configuration
| Variable | Purpose |
|:--|:--|
| `WHISPER_MODEL` | `tiny`, `base` or `small` |
| `DEVICE` | Where to run Whisper (default `auto`) |

## Tests
```bash
pytest
```

## Tech stack
Python, faster-whisper, jiwer, librosa, soundfile, FFmpeg, pandas, Streamlit, pytest

## Notes
Planned: more Indian languages (Tamil, Telugu, Bengali, Marathi, Hinglish), more ASR backends, batch evaluation, noise and SNR tests, VAD preprocessing, charts over history, PostgreSQL storage, a FastAPI backend and Docker packaging.
