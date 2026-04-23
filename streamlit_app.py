"""
streamlit_app.py
----------------
Streamlit UI for the Ancient Inscription Translator.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import preprocessing
import ocr_engine
import dataset_matcher
import genai_translator
import utils
from main import InscriptionTranslator

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ancient Inscription Translator",
    page_icon=None,
    layout="wide",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Cached resources (reload only when the selection changes)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_translator(backend: str, tess_langs: str, easy_langs_key: str) -> InscriptionTranslator:
    """Build the pipeline once per (backend, lang) combo and cache it."""
    easy_langs = list(easy_langs_key) if easy_langs_key else None
    return InscriptionTranslator(
        llm_backend=backend,
        tesseract_langs=tess_langs,
        easyocr_langs=easy_langs,
    )


def _pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    if bgr is None:
        return None
    if len(bgr.shape) == 2:
        return Image.fromarray(bgr)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _bgr_from_upload(uploaded) -> np.ndarray:
    """Decode any Streamlit UploadedFile into an OpenCV BGR array."""
    data = uploaded.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image. Is it a valid PNG/JPEG?")
    return img


# ---------------------------------------------------------------------------
# Sidebar: pipeline configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline settings")

    backend = st.selectbox(
        "LLM backend",
        options=["auto", "gemini", "groq", "local", "stub"],
        index=0,
        help="'auto' uses Gemini first and fails over to Groq.",
    )

    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    st.caption(
        f"Gemini key: {'detected' if has_gemini else 'missing'}  •  "
        f"Groq key: {'detected' if has_groq else 'missing'}"
    )

    script_hint = st.selectbox(
        "Script hint (optional)",
        options=["(auto)", "brahmi", "tamil-brahmi", "sanskrit", "prakrit", "pali", "devanagari", "tamil"],
        index=0,
    )
    if script_hint == "(auto)":
        script_hint = None

    apply_perspective = st.toggle(
        "Try perspective correction",
        value=False,
        help="Detect the largest quadrilateral in the photo and warp it to a frontal view.",
    )

    st.divider()
    st.subheader("OCR languages")
    tess_langs = st.text_input(
        "Tesseract langs",
        value="eng+san+tam+hin",
        help="'+' separated list of installed Tesseract traineddata languages.",
    )
    default_easy = ["en", "hi", "ta"]
    easy_langs = st.multiselect(
        "EasyOCR langs",
        options=["en", "hi", "ta", "sa", "mr", "ne", "bn", "gu", "kn", "te", "pa"],
        default=default_easy,
        help="Tamil ('ta') is auto-split into its own compatible group.",
    )

    st.divider()
    show_intermediates = st.toggle("Show preprocessing stages", value=True)
    show_bboxes = st.toggle("Highlight detected text regions", value=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Ancient Inscription Translator")
st.write(
    "Upload a photograph of a temple inscription (Brahmi, Tamil-Brahmi, "
    "Sanskrit, Prakrit, Pali, or medieval Devanagari/Tamil) and get a "
    "readable modern English & Hindi translation."
)

uploaded = st.file_uploader(
    "Upload an inscription image",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
    accept_multiple_files=False,
)

col_run, col_status = st.columns([1, 3])
with col_run:
    run = st.button("Translate", type="primary", disabled=uploaded is None, use_container_width=True)
with col_status:
    if uploaded is None:
        st.info("Pick an image to begin.")

if run and uploaded is not None:
    try:
        img_bgr = _bgr_from_upload(uploaded)
    except Exception as e:
        st.error(f"Image decode failed: {e}")
        st.stop()

    # ----- Build pipeline -----
    try:
        pipe = _get_translator(backend, tess_langs, tuple(easy_langs))
    except Exception as e:
        st.error(f"Could not initialise the pipeline: {e}")
        st.stop()

    # ----- Stage 1: preprocessing -----
    with st.spinner("Preprocessing image …"):
        try:
            stages = preprocessing.preprocess_inscription(
                img_bgr,
                apply_perspective=apply_perspective,
                return_intermediates=True,
            )
            cleaned = stages["cleaned"]
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    # ----- Stage 2: OCR -----
    with st.spinner("Running OCR ensemble …"):
        try:
            ocr_result = pipe.ocr(cleaned)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

    # ----- Stage 3: dictionary mapping -----
    with st.spinner("Mapping tokens against the script dataset …"):
        mapped_text, _details = pipe.dictionary_map(ocr_result.raw_text)

    # ----- Stage 4: GenAI translation -----
    with st.spinner("Calling the LLM for clean-up + translation …"):
        try:
            llm_out = pipe.translate(ocr_result.raw_text, mapped_text, script_hint=script_hint)
        except Exception as e:
            st.error(f"LLM translation failed: {e}")
            llm_out = genai_translator.TranslationOutput(
                cleaned_text=mapped_text,
                english="",
                hindi="",
                notes=f"LLM error: {e}",
                confidence=0.0,
                backend="error",
            )

    # ----- Assemble report -----
    report = utils.PipelineReport(
        source_image=uploaded.name,
        extracted_text=ocr_result.raw_text,
        mapped_text=mapped_text,
        cleaned_text=llm_out.cleaned_text,
        english=llm_out.english,
        hindi=llm_out.hindi,
        alternatives=llm_out.alternatives,
        notes=llm_out.notes,
        ocr_confidence=ocr_result.mean_confidence,
        llm_confidence=llm_out.confidence,
    )

    # -----------------------------------------------------------------
    # Results — image stages
    # -----------------------------------------------------------------
    st.subheader("Pipeline stages")
    img_cols = st.columns(3)
    with img_cols[0]:
        st.caption("Original")
        st.image(_pil_from_bgr(img_bgr), use_column_width=True)
    with img_cols[1]:
        st.caption("Preprocessed (binary)")
        st.image(_pil_from_bgr(cleaned), use_column_width=True, clamp=True)
    with img_cols[2]:
        if show_bboxes:
            vis = ocr_engine.draw_boxes(img_bgr, ocr_result)
            st.caption(f"Detected regions ({len(ocr_result.boxes)})")
            st.image(_pil_from_bgr(vis), use_column_width=True)
        else:
            st.caption("Edges")
            st.image(_pil_from_bgr(stages.get("edges")), use_column_width=True, clamp=True)

    if show_intermediates:
        with st.expander("All preprocessing stages", expanded=False):
            stage_cols = st.columns(4)
            for i, (name, arr) in enumerate(stages.items()):
                with stage_cols[i % 4]:
                    st.caption(name)
                    st.image(_pil_from_bgr(arr), use_column_width=True, clamp=True)

    # -----------------------------------------------------------------
    # Confidence + backend summary
    # -----------------------------------------------------------------
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OCR engine", ocr_result.engine or "—")
    m2.metric("OCR mean conf", f"{ocr_result.mean_confidence:.1f}")
    m3.metric("LLM backend", llm_out.backend or "—")
    m4.metric("Overall conf", f"{report.overall_confidence:.2f}")

    # -----------------------------------------------------------------
    # Translation output
    # -----------------------------------------------------------------
    st.subheader("Translation")
    tab_eng, tab_hin, tab_clean, tab_raw = st.tabs(
        ["Modern English", "Modern Hindi", "Cleaned transliteration", "Raw OCR + mapping"]
    )

    with tab_eng:
        if llm_out.english:
            st.markdown(f"> {llm_out.english}")
        else:
            st.info("No English output produced.")
        if llm_out.alternatives:
            with st.expander("Alternative readings"):
                for alt in llm_out.alternatives:
                    st.markdown(f"- {alt}")
        if llm_out.notes:
            with st.expander("Epigraphist notes"):
                st.write(llm_out.notes)

    with tab_hin:
        if llm_out.hindi:
            st.markdown(f"> {llm_out.hindi}")
        else:
            st.info("No Hindi output produced.")

    with tab_clean:
        st.caption("LLM-reconstructed reading (with ⟨…⟩ marking damaged portions).")
        st.code(llm_out.cleaned_text or "(empty)", language="text")

    with tab_raw:
        st.caption("Direct OCR output before LLM post-processing.")
        st.code(ocr_result.raw_text or "(empty)", language="text")
        st.caption("Dictionary-mapped form.")
        st.code(mapped_text or "(empty)", language="text")
        if ocr_result.boxes:
            with st.expander(f"Per-token boxes ({len(ocr_result.boxes)})"):
                rows = [
                    {
                        "text": b.text,
                        "confidence": round(b.confidence, 1),
                        "x": b.bbox[0], "y": b.bbox[1],
                        "w": b.bbox[2], "h": b.bbox[3],
                    }
                    for b in ocr_result.boxes
                ]
                st.dataframe(rows, use_container_width=True)

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------
    st.download_button(
        "Download full report as JSON",
        data=json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
        file_name=f"{os.path.splitext(uploaded.name)[0]}_translation.json",
        mime="application/json",
        use_container_width=False,
    )

else:
    with st.expander("About this pipeline", expanded=False):
        st.markdown(
            """
            This UI wraps the `InscriptionTranslator` pipeline:

            1. **Preprocessing** — OpenCV: grayscale → denoise → CLAHE → adaptive
               threshold → morphology. Perspective correction is optional.
            2. **OCR ensemble** — Tesseract + EasyOCR at multiple scales; best
               result wins.
            3. **Dictionary mapping** — Brahmi / Tamil-Brahmi / Sanskrit tokens
               are looked up in `data/script_mappings.json`. Unknown tokens fall
               back to Levenshtein fuzzy matching.
            4. **GenAI clean-up + translation** — Gemini (primary) with Groq
               fallback. Produces modern English, modern Hindi, alternative
               readings, and a confidence score.

            Set `GEMINI_API_KEY` and/or `GROQ_API_KEY` in your environment
            before starting Streamlit for the LLM stage to work.
            """
        )
