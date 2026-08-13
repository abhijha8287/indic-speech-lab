"""CSV-backed storage for ASR evaluation results.

This module is the only place that knows results live in a CSV file.
Swapping to a database later (e.g. PostgreSQL) means changing the
implementation of save_evaluation/load_evaluations, not their callers.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

RESULTS_DIR = os.path.join("data", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "evaluations.csv")

COLUMNS = [
    "timestamp",
    "filename",
    "language",
    "model",
    "audio_duration_seconds",
    "processing_time_seconds",
    "rtf",
    "reference",
    "transcription",
    "wer",
    "cer",
]


def _ensure_csv_exists(csv_path: str = RESULTS_CSV) -> None:
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)


def save_evaluation(
    filename: str,
    language: str,
    model: str,
    audio_duration_seconds: float,
    processing_time_seconds: float,
    rtf: float,
    reference: str,
    transcription: str,
    wer: Optional[float],
    cer: Optional[float],
    csv_path: str = RESULTS_CSV,
) -> None:
    """Append one evaluation result row to the CSV store."""
    _ensure_csv_exists(csv_path)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "language": language,
        "model": model,
        "audio_duration_seconds": audio_duration_seconds,
        "processing_time_seconds": processing_time_seconds,
        "rtf": rtf,
        "reference": reference or "",
        "transcription": transcription,
        "wer": wer if wer is not None else "",
        "cer": cer if cer is not None else "",
    }

    df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(csv_path, mode="a", header=False, index=False)


def load_evaluations(csv_path: str = RESULTS_CSV) -> pd.DataFrame:
    """Load all saved evaluation results, most recent first."""
    _ensure_csv_exists(csv_path)
    df = pd.read_csv(csv_path)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df
