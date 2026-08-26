"""Low-DPI flatbed scan preprocessor for EasyOCR.

Problem: bank and credit-card statements are often flatbed-scanned at 150-200 DPI.
At that resolution text characters are only 8-12 pixels tall, below EasyOCR's
comfort zone.  Heavy denoising makes the problem worse by smearing the few ink
pixels that exist.

Algorithm:
1. Convert the incoming page to grayscale.
2. Estimate page skew with a projection-profile sweep (coarse + fine).  We score
   each candidate angle by the variance of the horizontal projection of an
   inverted binary version; a horizontal text baseline produces sharp peaks.
3. Rotate the original grayscale page to correct the skew.
4. Assume / clamp the source DPI to the 150-200 flatbed range and compute the
   scale factor needed to reach 400 DPI equivalent.
5. Upscale with Lanczos4 interpolation BEFORE any denoising.
6. Apply a mild unsharp mask to restore edge definition.
7. Apply CLAHE to correct uneven flatbed lighting.
8. Return the enhanced grayscale image as RGB for EasyOCR, plus an adaptive
   Gaussian binarization for downstream word segmentation.

No bilateral filtering, no heavy Gaussian blur, and no morphological denoising
is applied to the recognition image; those operations are postponed to the
binary segmentation copy only when useful.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

# Flatbed scans in this project are assumed to fall in this range.
MIN_SOURCE_DPI = 150.0
MAX_SOURCE_DPI = 200.0
DEFAULT_SOURCE_DPI = 175.0
TARGET_DPI = 400.0

# Skew search range in degrees.
MAX_SKEW_DEGREES = 15.0

# Fast projection-profile scoring uses a thumbnail no wider than this.
SKEW_ESTIMATE_MAX_DIM = 512


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel copy of a BGR, RGB, or grayscale image."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _binarize_for_projection(gray: np.ndarray) -> np.ndarray:
    """Fast inverted binary image for projection-profile scoring."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _projection_score(binary: np.ndarray) -> float:
    """Score a binary image by horizontal projection variance.

    Horizontal text baselines produce tall, narrow peaks; the angle that
    maximizes projection variance is the best deskew estimate.
    """
    proj = np.sum(binary > 0, axis=1, dtype=np.float64)
    if proj.size <= 1:
        return 0.0
    return float(np.var(proj))


def estimate_skew_projection_profile(
    gray: np.ndarray,
    max_angle: float = MAX_SKEW_DEGREES,
) -> float:
    """Estimate page skew using a coarse-to-fine projection-profile search.

    Returns the estimated rotation angle in degrees; sign follows OpenCV's
    rotation convention (positive = counter-clockwise).
    """
    h, w = gray.shape[:2]
    scale = min(1.0, SKEW_ESTIMATE_MAX_DIM / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = _binarize_for_projection(small)

    center = (small.shape[1] / 2.0, small.shape[0] / 2.0)

    def _score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary,
            matrix,
            (small.shape[1], small.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return _projection_score(rotated)

    # Coarse sweep: 1-degree steps.
    best_angle = 0.0
    best_score = -1.0
    for angle in np.linspace(-max_angle, max_angle, 31):
        score = _score(angle)
        if score > best_score:
            best_score = score
            best_angle = angle

    # Fine sweep: +/- 1 degree around coarse best, 0.1-degree steps.
    for angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 21):
        score = _score(angle)
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the given angle, filling new borders with white."""
    angle = max(-MAX_SKEW_DEGREES, min(MAX_SKEW_DEGREES, angle))
    if abs(angle) < 0.3:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    border: int | tuple
    if image.ndim == 3:
        border = (255, 255, 255)
    else:
        border = 255

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _clamp_source_dpi(dpi: float | None) -> float:
    """All raster pages are treated as low-DPI flatbed scans."""
    if dpi is None or not (MIN_SOURCE_DPI <= float(dpi) <= MAX_SOURCE_DPI):
        return DEFAULT_SOURCE_DPI
    return float(dpi)


def _upscale(gray: np.ndarray, source_dpi: float, target_dpi: float) -> tuple[np.ndarray, float]:
    """Lanczos upscale from source to target DPI."""
    scale = target_dpi / source_dpi
    if scale <= 1.05:
        return gray, 1.0
    upscaled = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LANCZOS4,
    )
    return upscaled, scale


def _unsharp_mask(gray: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """Mild sharpening via unsharp mask."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, amount, blurred, amount - 1.0, 0)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Adaptive contrast for uneven flatbed lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding for clean word segmentation."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )


def preprocess_scan(
    image: np.ndarray,
    source_dpi: float | None = None,
    target_dpi: float = TARGET_DPI,
) -> Dict[str, np.ndarray | float]:
    """Prepare a scanned page for OCR and segmentation.

    Args:
        image: BGR or grayscale numpy array (e.g. from PyMuPDF rendering).
        source_dpi: Estimated native DPI of the embedded raster.  If None or
            outside 150-200, a default flatbed value (175) is assumed.
        target_dpi: Desired DPI equivalent for recognition (default 400).

    Returns:
        Dictionary with keys:
          - rgb: enhanced RGB image for EasyOCR recognition.
          - gray: enhanced grayscale image.
          - binary: inverted binary image for word segmentation.
          - scale: total upscale factor applied.
          - skew_angle: deskew angle in degrees.
          - source_dpi: assumed native DPI.
          - effective_dpi: pixel DPI of the returned images.
    """
    gray = _to_grayscale(image)

    # 1. Deskew using projection profile on the original grayscale.
    skew_angle = estimate_skew_projection_profile(gray)
    if abs(skew_angle) >= 0.3:
        gray = deskew(gray, skew_angle)

    # 2. Enforce low-DPI assumption and compute upscale factor.
    source_dpi = _clamp_source_dpi(source_dpi)

    # 3. Upscale to target DPI with Lanczos (before denoising).
    upscaled, scale = _upscale(gray, source_dpi, target_dpi)

    # 4. Mild sharpening to restore edges softened by interpolation.
    sharpened = _unsharp_mask(upscaled)

    # 5. CLAHE to compensate for scanner glass shadows / uneven lighting.
    contrasted = _apply_clahe(sharpened)

    # 6. Binarization copy for downstream segmentation; recognition uses gray/RGB.
    binary = _binarize(contrasted)

    # 7. RGB output for EasyOCR.
    rgb = cv2.cvtColor(contrasted, cv2.COLOR_GRAY2RGB)

    return {
        "rgb": rgb,
        "gray": contrasted,
        "binary": binary,
        "scale": scale,
        "skew_angle": skew_angle,
        "source_dpi": source_dpi,
        "effective_dpi": source_dpi * scale,
    }
