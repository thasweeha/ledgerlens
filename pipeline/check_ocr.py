"""Check detection and ICR for scanned bank statement pages.

Algorithm:
1. Focus on the bottom 25-40% of a preprocessed RGB page, where physical checks
   are normally scanned.
2. Look for large rectangular boundary contours; if none are found, fall back to
   a full-width band that contains a MICR-style numeric line at its bottom.
3. Inside each candidate check region:
   - Extract the amount from the right-hand numeric box (allowlist: digits,
     decimal point, comma, dollar sign).
   - Extract the payee name from the "pay to the order of" band (allowlist:
     upper/lower letters, spaces, and common punctuation).
   - Extract the check number from the MICR line at the bottom of the check
     (allowlist: digits).
4. Return structured check records with page_number, check_number, amount,
   payee_name, and a confidence score.

This module uses EasyOCR in recognition-only mode on small cropped regions, so
it stays CPU-friendly and does not load any PyTorch models.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.money import parse_amount_decimal

# Region of the page where checks are expected.
CHECK_BOTTOM_START = 0.60
CHECK_BOTTOM_END = 1.00

# Minimum relative size of a check rectangle vs the page width/height.
MIN_CHECK_WIDTH_RATIO = 0.40
MIN_CHECK_HEIGHT_PX = 120

# MICR line sits at the very bottom of a check.
MICR_BAND_TOP = 0.78
MICR_BAND_BOTTOM = 0.96
MICR_MIN_WIDTH_RATIO = 0.35

# Amount box is the upper-right quadrant of the check.
AMOUNT_ROI = (0.55, 0.12, 0.98, 0.42)  # x0, y0, x1, y1 as ratios

# Payee line sits just below the "pay to the order of" prompt.
PAYEE_ROI = (0.12, 0.22, 0.65, 0.42)


_ALLOWLIST_DIGITS = "0123456789"
_ALLOWLIST_AMOUNT = _ALLOWLIST_DIGITS + ".,$"
_ALLOWLIST_PAYEE = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-&'"
)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel copy of an RGB/gray numpy array."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Inverted adaptive binary image for contour detection."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15,
    )


def _find_rectangular_check_regions(page_rgb: np.ndarray) -> List[Dict[str, np.ndarray | Tuple[int, ...]]]:
    """Find rectangular check-shaped regions in the bottom of the page."""
    h, w = page_rgb.shape[:2]
    y0 = int(h * CHECK_BOTTOM_START)
    y1 = int(h * CHECK_BOTTOM_END)
    bottom_band = page_rgb[y0:y1, :]

    gray = _to_grayscale(bottom_band)
    binary = _binarize(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict[str, np.ndarray | Tuple[int, ...]]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < w * MIN_CHECK_WIDTH_RATIO or ch < MIN_CHECK_HEIGHT_PX:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # Accept quadrilaterals or large near-rectangular blobs.
        is_rect = len(approx) == 4
        area = cv2.contourArea(contour)
        bbox_area = cw * ch
        if not (is_rect or (area > bbox_area * 0.7 and bbox_area > w * 0.25 * MIN_CHECK_HEIGHT_PX)):
            continue

        abs_bbox = (x, y0 + y, x + cw, y0 + y + ch)
        regions.append(
            {
                "bbox": abs_bbox,
                "roi": page_rgb[abs_bbox[1] : abs_bbox[3], abs_bbox[0] : abs_bbox[2]],
            }
        )

    return regions


def _find_micr_line(check_roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate the MICR numeric line at the bottom of a check ROI."""
    h, w = check_roi.shape[:2]
    y0 = int(h * MICR_BAND_TOP)
    y1 = int(h * MICR_BAND_BOTTOM)
    band = check_roi[y0:y1, :]
    gray = _to_grayscale(band)
    binary = _binarize(gray)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[Tuple[int, int, int, int]] = None
    best_width = 0
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < w * MICR_MIN_WIDTH_RATIO or ch < 12 or ch > 60:
            continue
        if cw > best_width:
            best_width = cw
            best = (x, y0 + y, x + cw, y0 + y + ch)
    return best


def _ocr_region(
    roi: np.ndarray,
    recognizer,
    allowlist: str | None = None,
) -> Tuple[str, float]:
    """Run EasyOCR recognition on a small RGB/gray ROI."""
    if not roi.size:
        return "", 0.0
    return recognizer.recognize(roi, allowlist=allowlist)


def _extract_amount(check_roi: np.ndarray, recognizer) -> Tuple[Optional[Decimal], float]:
    """Extract the numeric amount from the upper-right box of the check."""
    h, w = check_roi.shape[:2]
    x0, y0, x1, y1 = AMOUNT_ROI
    region = check_roi[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
    text, conf = _ocr_region(region, recognizer, allowlist=_ALLOWLIST_AMOUNT)
    amount = parse_amount_decimal(text)
    return amount, conf


def _extract_payee(check_roi: np.ndarray, recognizer) -> Tuple[str, float]:
    """Extract the payee name from the center band of the check."""
    h, w = check_roi.shape[:2]
    x0, y0, x1, y1 = PAYEE_ROI
    region = check_roi[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
    text, conf = _ocr_region(region, recognizer, allowlist=_ALLOWLIST_PAYEE)
    return text.strip(), conf


def _extract_check_number(check_roi: np.ndarray, recognizer) -> Tuple[Optional[str], float]:
    """Extract the check number from the MICR line at the bottom."""
    micr = _find_micr_line(check_roi)
    if micr is None:
        return None, 0.0

    mx0, my0, mx1, my1 = micr
    region = check_roi[my0:my1, mx0:mx1]
    text, conf = _ocr_region(region, recognizer, allowlist=_ALLOWLIST_DIGITS + " ")
    # MICR lines are groups of digits; the check number is usually the first
    # or second numeric group.  We return the first plausible group.
    digits = re.findall(r"\d{2,}", text)
    if digits:
        return digits[0], conf
    return None, conf


def detect_checks(
    page_rgb: np.ndarray,
    page_index: int,
    recognizer,
) -> List[Dict[str, object]]:
    """Detect and ICR physical checks in the bottom region of a page.

    Args:
        page_rgb: Preprocessed RGB page image (from pipeline.preprocessor).
        page_index: 0-based page index.
        recognizer: EasyOCR recognizer instance.

    Returns:
        List of check dicts with keys: page_number, check_number, amount,
        payee_name, confidence, bbox.
    """
    regions = _find_rectangular_check_regions(page_rgb)

    # If no rectangular check boundary is found, still inspect the bottom band
    # as a single fallback region in case the check edge was cropped.
    if not regions:
        h, w = page_rgb.shape[:2]
        y0 = int(h * CHECK_BOTTOM_START)
        y1 = int(h * CHECK_BOTTOM_END)
        bbox = (0, y0, w, y1)
        regions.append(
            {
                "bbox": bbox,
                "roi": page_rgb[y0:y1, :],
            }
        )

    checks: List[Dict[str, object]] = []
    for region in regions:
        roi = region["roi"]
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        check_number, cn_conf = _extract_check_number(roi, recognizer)
        amount, amount_conf = _extract_amount(roi, recognizer)
        payee_name, payee_conf = _extract_payee(roi, recognizer)

        confidences = [c for c in (cn_conf, amount_conf, payee_conf) if c > 0]
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        checks.append(
            {
                "page_number": page_index,
                "check_number": check_number,
                "amount": amount,
                "payee_name": payee_name,
                "confidence": confidence,
                "bbox": region["bbox"],
            }
        )

    return checks
