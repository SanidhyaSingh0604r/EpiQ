"""
ocr_engine.py
-------------
Multi-engine OCR wrapper for ancient inscriptions.

Strategy
--------
No single OCR engine reliably handles ancient scripts (Brahmi, Tamil-Brahmi,
Prakrit, etc.), so we try several in priority order and keep whichever
extracts the most high-confidence text:

    1. Tesseract (with `san`, `tam`, `hin`, `eng` language packs)
    2. EasyOCR (fallback — handles Devanagari and Tamil well)
    3. (Optional) TrOCR / a fine-tuned transformer model — plug-in point

Handling eroded inscriptions
----------------------------
- We run OCR at multiple scales and keep the best result.
- We also detect line-level bounding boxes so the downstream GenAI step
  knows which fragments sit on which line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Optional imports — we don't hard-fail if a backend is missing.
try:
    import pytesseract
    from pytesseract import Output
    _HAS_TESS = True
except ImportError:
    _HAS_TESS = False

try:
    import easyocr
    _HAS_EASY = True
except ImportError:
    _HAS_EASY = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class OCRResult:
    raw_text: str = ""
    boxes: List[OCRBox] = field(default_factory=list)
    engine: str = ""
    mean_confidence: float = 0.0
    language_hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

def _tesseract_ocr(img: np.ndarray, langs: str = "eng+san+tam+hin") -> OCRResult:
    """
    Run Tesseract. `langs` is a '+' separated list of installed language packs.
    Install Sanskrit/Tamil/Hindi traineddata into your tessdata directory.
    """
    if not _HAS_TESS:
        return OCRResult(engine="tesseract-missing")

    # PSM 6 = treat as a single uniform block of text (good for inscriptions).
    config = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(img, lang=langs, config=config, output_type=Output.DICT)
    except pytesseract.TesseractError as e:
        log.warning("Tesseract failed for langs=%s: %s", langs, e)
        return OCRResult(engine="tesseract-error")

    boxes: List[OCRBox] = []
    confs: List[float] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            c = float(data["conf"][i])
        except (ValueError, TypeError):
            c = -1
        if c < 0:
            continue
        boxes.append(OCRBox(
            text=txt,
            confidence=c,
            bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
        ))
        confs.append(c)

    raw = " ".join(b.text for b in boxes)
    return OCRResult(
        raw_text=raw,
        boxes=boxes,
        engine="tesseract",
        mean_confidence=float(np.mean(confs)) if confs else 0.0,
        language_hint=langs,
    )


# ---------------------------------------------------------------------------
# EasyOCR backend
# ---------------------------------------------------------------------------

_EASY_READER_CACHE = {}

# EasyOCR compatibility rules.
# Tamil ('ta') can ONLY be combined with English ('en').
# Devanagari-family languages (hi, mr, ne, sa, ...) can be grouped with en.
# To cover the typical Indic inscription case we run multiple *compatible*
# groups sequentially and merge the results.
_EASYOCR_INCOMPATIBLE = {"ta"}  # languages that must be run in isolation (with 'en')


def _split_into_compatible_groups(langs: List[str]) -> List[List[str]]:
    """
    Split a requested language list into EasyOCR-compatible groups.
    Each returned group is safe to pass to easyocr.Reader(...).
    """
    langs = list(dict.fromkeys(langs))  # dedupe, keep order
    isolated = [l for l in langs if l in _EASYOCR_INCOMPATIBLE]
    shared = [l for l in langs if l not in _EASYOCR_INCOMPATIBLE]

    groups: List[List[str]] = []
    if shared:
        groups.append(shared if "en" in shared else shared + ["en"])
    for l in isolated:
        groups.append([l, "en"])  # EasyOCR requires English as the companion
    if not groups:
        groups.append(["en"])
    return groups


def _get_easy_reader(langs: List[str]):
    """Cache EasyOCR readers since model loading is expensive."""
    key = tuple(sorted(langs))
    if key not in _EASY_READER_CACHE:
        log.info("Loading EasyOCR reader for languages: %s", langs)
        _EASY_READER_CACHE[key] = easyocr.Reader(langs, gpu=False)
    return _EASY_READER_CACHE[key]


def _easyocr_ocr(img: np.ndarray, langs: Optional[List[str]] = None) -> OCRResult:
    """
    EasyOCR fallback. Supports Devanagari ('hi'), Tamil ('ta'), English ('en').
    Note: EasyOCR has no native Brahmi model — we use Devanagari / Tamil as
    the closest available approximation.

    Because EasyOCR refuses some language combinations (e.g. Tamil cannot be
    combined with Hindi), we split the requested list into compatible groups
    and run one reader per group, then merge the outputs.
    """
    if not _HAS_EASY:
        return OCRResult(engine="easyocr-missing")

    requested = langs or ["en", "hi", "ta"]
    groups = _split_into_compatible_groups(requested)

    all_boxes: List[OCRBox] = []
    all_confs: List[float] = []
    used_groups: List[str] = []

    for group in groups:
        try:
            reader = _get_easy_reader(group)
        except ValueError as e:
            # EasyOCR rejected the combination — skip this group.
            log.warning("EasyOCR rejected %s: %s", group, e)
            continue
        except Exception as e:
            log.warning("EasyOCR reader init failed for %s: %s", group, e)
            continue

        try:
            result = reader.readtext(img)
        except Exception as e:
            log.warning("EasyOCR readtext failed for %s: %s", group, e)
            continue

        for bbox_pts, text, conf in result:
            if not text.strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            all_boxes.append(OCRBox(
                text=text.strip(),
                confidence=float(conf) * 100,
                bbox=(x, y, w, h),
            ))
            all_confs.append(float(conf) * 100)
        used_groups.append("+".join(group))

    if not all_boxes:
        return OCRResult(engine="easyocr", language_hint="|".join(used_groups))

    raw = " ".join(b.text for b in all_boxes)
    return OCRResult(
        raw_text=raw,
        boxes=all_boxes,
        engine="easyocr",
        mean_confidence=float(np.mean(all_confs)),
        language_hint="|".join(used_groups),
    )


# ---------------------------------------------------------------------------
# Multi-scale + ensemble wrapper
# ---------------------------------------------------------------------------

def _rescale(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def extract_text(
    img: np.ndarray,
    tesseract_langs: str = "eng+san+tam+hin",
    easyocr_langs: Optional[List[str]] = None,
    scales: Tuple[float, ...] = (1.0, 1.5, 2.0),
) -> OCRResult:
    """
    Run the OCR ensemble. We try multiple scales because inscription glyphs
    often need upscaling for Tesseract to latch onto strokes, but aggressive
    upscaling amplifies noise — so we keep whichever scale wins.
    """
    best: OCRResult = OCRResult(engine="none")

    for s in scales:
        scaled = _rescale(img, s)

        if _HAS_TESS:
            try:
                r = _tesseract_ocr(scaled, langs=tesseract_langs)
                if r.mean_confidence > best.mean_confidence and r.raw_text.strip():
                    best = r
            except Exception as e:
                log.warning("Tesseract crashed at scale=%.2f: %s", s, e)

        if _HAS_EASY:
            try:
                r = _easyocr_ocr(scaled, langs=easyocr_langs)
                if r.mean_confidence > best.mean_confidence and r.raw_text.strip():
                    best = r
            except Exception as e:
                log.warning("EasyOCR crashed at scale=%.2f: %s", s, e)

    if not best.raw_text:
        log.warning("No OCR engine produced text. Is Tesseract or EasyOCR installed?")
    return best


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_boxes(img: np.ndarray, result: OCRResult,
               color: Tuple[int, int, int] = (0, 200, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of `img` with detected text regions drawn on."""
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for b in result.boxes:
        x, y, w, h = b.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label = f"{b.text} ({b.confidence:.0f})"
        cv2.putText(vis, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis
