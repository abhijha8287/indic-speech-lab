from src.metrics import calculate_cer, calculate_wer, normalize_text


def test_identical_reference_hypothesis_gives_zero_wer():
    text = "the quick brown fox jumps over the lazy dog"
    assert calculate_wer(text, text) == 0.0


def test_identical_reference_hypothesis_gives_zero_cer():
    text = "the quick brown fox"
    assert calculate_cer(text, text) == 0.0


def test_different_sentence_gives_nonzero_wer():
    reference = "the quick brown fox"
    hypothesis = "a slow green turtle"
    assert calculate_wer(reference, hypothesis) > 0.0


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!!!") == "hello world"


def test_normalize_text_preserves_hindi_unicode():
    hindi = "नमस्ते दुनिया"
    normalized = normalize_text(hindi)
    assert normalized
    assert "न" in normalized
    assert normalized == hindi.strip()


def test_missing_reference_returns_none():
    assert calculate_wer("", "some hypothesis") is None
    assert calculate_cer(None, "some hypothesis") is None
