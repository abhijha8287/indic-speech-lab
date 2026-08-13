"""Text normalization and ASR quality metrics (WER, CER)."""

import re
import unicodedata
from typing import Optional

import jiwer


def normalize_text(text: str) -> str:
    """Normalize text for fair WER/CER comparison.

    Lowercases Latin script (a no-op on Devanagari, which is uncased),
    strips punctuation/symbols while keeping letters, numbers, and
    combining marks (e.g. Devanagari matras) intact, and collapses
    whitespace. Does not transliterate between scripts.
    """
    if not text:
        return ""

    text = text.strip().lower()

    kept_chars = []
    for ch in text:
        if ch.isspace():
            kept_chars.append(ch)
            continue
        category = unicodedata.category(ch)
        # L* = letters, N* = numbers, M* = combining marks (e.g. matras)
        if category[0] in ("L", "N", "M"):
            kept_chars.append(ch)

    normalized = "".join(kept_chars)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_wer(reference: str, hypothesis: str) -> Optional[float]:
    """Calculate Word Error Rate. Returns None if reference is empty/missing."""
    if not reference or not reference.strip():
        return None

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if not ref_norm:
        return None

    return jiwer.wer(ref_norm, hyp_norm)


def calculate_cer(reference: str, hypothesis: str) -> Optional[float]:
    """Calculate Character Error Rate. Returns None if reference is empty/missing."""
    if not reference or not reference.strip():
        return None

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if not ref_norm:
        return None

    return jiwer.cer(ref_norm, hyp_norm)
