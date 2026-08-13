"""IndicSpeechLab — Indian Language ASR Evaluation & Benchmarking dashboard."""

import os
import tempfile

import pandas as pd
import streamlit as st

from src.asr import ASRError, ASRService, LANGUAGE_CODES, VALID_MODEL_SIZES
from src.audio import (
    AudioValidationError,
    SUPPORTED_FORMATS,
    prepare_audio_for_transcription,
    validate_audio_file,
)
from src.metrics import calculate_cer, calculate_wer
from src.storage import load_evaluations, save_evaluation

st.set_page_config(page_title="IndicSpeechLab", layout="wide")

MODEL_ORDER = ["tiny", "base", "small"]


@st.cache_resource(show_spinner=False)
def get_asr_service(model_size: str) -> ASRService:
    """Load and cache the Whisper model so it's only loaded once per model size."""
    return ASRService(model_size=model_size)


def format_metric(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def main():
    st.title("IndicSpeechLab")
    st.caption("Indian Language ASR Evaluation & Benchmarking")

    default_model = os.environ.get("WHISPER_MODEL", "small")
    if default_model not in VALID_MODEL_SIZES:
        default_model = "small"

    with st.sidebar:
        st.header("Configuration")
        model_options = [m for m in MODEL_ORDER if m in VALID_MODEL_SIZES]
        model_size = st.selectbox(
            "Whisper model",
            options=model_options,
            index=model_options.index(default_model),
        )
        language = st.selectbox("Language", options=list(LANGUAGE_CODES.keys()))

    st.subheader("1. Upload Audio")
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_FORMATS)],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        st.subheader("2. Reference Transcript (optional)")
        reference_text = st.text_area(
            "Paste the ground-truth transcript here to calculate WER/CER",
            height=120,
        )

        st.subheader("3. Run Evaluation")
        run_clicked = st.button("Transcribe & Evaluate", type="primary")

        if run_clicked:
            try:
                with tempfile.TemporaryDirectory() as work_dir:
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    original_path = os.path.join(work_dir, f"original{suffix}")
                    with open(original_path, "wb") as original_file:
                        original_file.write(uploaded_file.getvalue())

                    with st.status("Running evaluation...", expanded=True) as status:
                        status.write("Validating audio file...")
                        validate_audio_file(original_path)

                        status.write("Preparing audio (converting if needed)...")
                        wav_path = prepare_audio_for_transcription(original_path, work_dir)

                        status.write(f"Loading Whisper model ({model_size})...")
                        asr_service = get_asr_service(model_size)

                        status.write("Transcribing audio...")
                        result = asr_service.transcribe(wav_path, language)

                        status.write("Calculating metrics...")
                        wer = calculate_wer(reference_text, result.transcription)
                        cer = calculate_cer(reference_text, result.transcription)

                        status.write("Saving results...")
                        save_evaluation(
                            filename=uploaded_file.name,
                            language=result.language,
                            model=model_size,
                            audio_duration_seconds=result.audio_duration_seconds,
                            processing_time_seconds=result.processing_time_seconds,
                            rtf=result.rtf,
                            reference=reference_text,
                            transcription=result.transcription,
                            wer=wer,
                            cer=cer,
                        )
                        status.update(label="Evaluation complete", state="complete")
                    # work_dir (original upload + any converted WAV) is removed
                    # automatically once this `with` block exits.

                st.markdown("---")
                st.subheader("Transcription")
                st.markdown(f"**Detected/Selected Language:** {result.language}")
                if result.transcription:
                    st.text_area(
                        "Transcription", value=result.transcription, height=150, disabled=True
                    )
                else:
                    st.warning(
                        "The model returned an empty transcription. "
                        "The audio may be silent, too short, or unclear."
                    )

                st.markdown("---")
                st.subheader("Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Audio Duration", f"{result.audio_duration_seconds:.2f} sec")
                col2.metric("Processing Time", f"{result.processing_time_seconds:.2f} sec")
                col3.metric("RTF", f"{result.rtf:.3f}")
                st.caption("RTF < 1.0 means the system processed audio faster than real time.")

                col4, col5 = st.columns(2)
                col4.metric("WER", format_metric(wer))
                col5.metric("CER", format_metric(cer))
                if wer is None:
                    st.info("No reference transcript provided — WER/CER shown as N/A.")

            except AudioValidationError as exc:
                st.error(f"Audio problem: {exc}")
            except ASRError as exc:
                st.error(f"Transcription problem: {exc}")
            except Exception as exc:
                st.error(
                    "Something went wrong while processing this file. "
                    "Try a different file, language, or model."
                )
                st.caption(f"Details: {exc}")

    st.markdown("---")
    st.subheader("Previous Evaluations")
    try:
        history_df = load_evaluations()
    except Exception:
        history_df = pd.DataFrame()

    if history_df.empty:
        st.caption("No evaluations yet. Run one above to see results here.")
    else:
        st.dataframe(history_df, use_container_width=True)
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="evaluations.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
