"""
utils.py
--------
Shared helpers: image loading, visualisation, and confidence aggregation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read an image with OpenCV, raising a helpful error on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
# Final report container
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    source_image: str
    extracted_text: str = ""
    mapped_text: str = ""
    cleaned_text: str = ""
    english: str = ""
    hindi: str = ""
    alternatives: List[str] = field(default_factory=list)
    notes: str = ""
    ocr_confidence: float = 0.0
    llm_confidence: float = 0.0

    @property
    def overall_confidence(self) -> float:
        # Weighted blend — OCR floor * LLM judgement. Normalised to [0, 1].
        ocr = max(0.0, min(1.0, self.ocr_confidence / 100.0))
        llm = max(0.0, min(1.0, self.llm_confidence))
        if ocr == 0 and llm == 0:
            return 0.0
        return round(0.4 * ocr + 0.6 * llm, 3)

    def pretty(self) -> str:
        return (
            f"=== Ancient Inscription Translation Report ===\n"
            f"Source        : {self.source_image}\n"
            f"Extracted     : {self.extracted_text}\n"
            f"Mapped        : {self.mapped_text}\n"
            f"Cleaned       : {self.cleaned_text}\n"
            f"English       : {self.english}\n"
            f"Hindi         : {self.hindi}\n"
            f"Alternatives  : {self.alternatives}\n"
            f"Notes         : {self.notes}\n"
            f"OCR conf      : {self.ocr_confidence:.1f}\n"
            f"LLM conf      : {self.llm_confidence:.2f}\n"
            f"Overall conf  : {self.overall_confidence:.2f}\n"
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["overall_confidence"] = self.overall_confidence
        return d
