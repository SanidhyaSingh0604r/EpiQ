"""
preprocessing.py
----------------
Image preprocessing pipeline for ancient temple inscriptions.

Stages:
    1. Grayscale conversion
    2. Noise reduction (Gaussian + Median)
    3. Illumination normalization (CLAHE)
    4. Adaptive thresholding (binarization)
    5. Morphological cleanup
    6. Edge enhancement / inscription region isolation
    7. Optional perspective correction (warp to a frontal view)

All functions accept and return numpy arrays (OpenCV-style BGR/GRAY images).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core preprocessing steps
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to single-channel grayscale. If already gray, returns a copy."""
    if img is None:
        raise ValueError("Input image is None.")
    if len(img.shape) == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, gaussian_ksize: int = 3, median_ksize: int = 3) -> np.ndarray:
    """
    Two-stage denoising: Gaussian blur smooths sensor noise,
    median blur removes salt-and-pepper noise common on weathered stone.
    """
    blurred = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)
    blurred = cv2.medianBlur(blurred, median_ksize)
    return blurred


def normalize_illumination(gray: np.ndarray, clip_limit: float = 2.5,
                           tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Essential for inscriptions photographed in uneven lighting
    (e.g., shadows cast by pillars or sun glare on stone).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def adaptive_binarize(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    """
    Adaptive Gaussian thresholding. Much better than global thresholding
    for stone surfaces where lighting varies across the inscription.
    `block_size` must be odd.
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C,
    )


def morphological_cleanup(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """
    Close small gaps inside characters, then open to strip specks.
    Helpful for eroded Brahmi/Tamil-Brahmi glyphs where strokes are broken.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k, iterations=1)
    return opened


def enhance_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map — useful for isolating inscription panels from surrounding stone."""
    return cv2.Canny(gray, low, high)


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def perspective_correct(img: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Attempts to detect the inscription panel as the largest quadrilateral
    and warp it to a frontal rectangle. Falls back to the original image
    if no suitable quad is found.
    """
    gray = to_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * img.shape[0] * img.shape[1]:
            rect = _order_corners(approx.reshape(4, 2))
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0, 0], [maxW - 1, 0],
                            [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxW, maxH))
            if debug:
                print(f"[preprocessing] Perspective corrected to {maxW}x{maxH}")
            return warped

    if debug:
        print("[preprocessing] No quadrilateral panel found — skipping warp.")
    return img


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_inscription(
    img: np.ndarray,
    apply_perspective: bool = False,
    return_intermediates: bool = False,
) -> np.ndarray | dict:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image from `cv2.imread`.
    apply_perspective : bool
        If True, try to rectify the inscription panel before OCR.
    return_intermediates : bool
        If True, return a dict of every stage (useful for debugging / UI).

    Returns
    -------
    Either the final binary image (default) or a dict of all stages.
    """
    if img is None:
        raise ValueError("preprocess_inscription received None.")

    stages = {"original": img}

    if apply_perspective:
        img = perspective_correct(img)
        stages["perspective"] = img

    gray = to_grayscale(img)
    stages["grayscale"] = gray

    denoised = denoise(gray)
    stages["denoised"] = denoised

    normalized = normalize_illumination(denoised)
    stages["normalized"] = normalized

    binary = adaptive_binarize(normalized)
    stages["binary"] = binary

    cleaned = morphological_cleanup(binary)
    stages["cleaned"] = cleaned

    edges = enhance_edges(normalized)
    stages["edges"] = edges

    if return_intermediates:
        return stages
    return cleaned
