"""Hybrid ingestion router (spec §1).

Evaluates every PDF page at ingestion and marks it MODE_DIGITAL_VECTOR or
MODE_FLATBED_RASTER so mixed-mode documents route each page down the right
pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pymupdf as fitz

from pipeline.tokens import PageMode

MIN_TEXT_WORDS = 12
FRAGMENT_MAX_RATIO = 0.25
IMAGE_COVER_THRESHOLD = 0.85


def _is_fragmented(word_text: str) -> bool:
    """Detect broken font encodings: cid dumps, replacement glyphs, mojibake."""
    text = word_text.strip()
    if not text:
        return True
    if "\ufffd" in text or "cid:" in text.lower():
        return True
    alnum = sum(ch.isalnum() for ch in text)
    return alnum / len(text) < 0.4


def image_coverage(page) -> float:
    """Fraction of the page rectangle covered by embedded raster images."""
    page_rect = page.rect
    page_area = abs(page_rect.width * page_rect.height)
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        ix0, iy0 = max(x0, page_rect.x0), max(y0, page_rect.y0)
        ix1, iy1 = min(x1, page_rect.x1), min(y1, page_rect.y1)
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return min(1.0, covered / page_area)


def classify_page(page) -> PageMode:
    """Decide the execution mode for a single PDF page."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    if not words:
        return PageMode.FLATBED_RASTER

    coverage = image_coverage(page)
    if len(words) < MIN_TEXT_WORDS:
        # A thin text layer stamped over a full-page scan is still a raster page.
        return PageMode.FLATBED_RASTER if coverage >= IMAGE_COVER_THRESHOLD else PageMode.DIGITAL_VECTOR

    fragmented = sum(1 for w in words if _is_fragmented(w[4]))
    if fragmented / len(words) > FRAGMENT_MAX_RATIO:
        return PageMode.FLATBED_RASTER
    return PageMode.DIGITAL_VECTOR


def route_document(path: str | Path) -> Dict:
    """Classify every page of a PDF; returns the routing plan for ingestion."""
    path = Path(path)
    plan: Dict = {
        "path": str(path),
        "exists": path.exists(),
        "page_count": 0,
        "modes": [],
        "digital_pages": [],
        "raster_pages": [],
        "error": None,
    }
    if not path.exists():
        plan["error"] = f"file not found: {path}"
        return plan
    try:
        with fitz.open(path) as doc:
            plan["page_count"] = len(doc)
            for index, page in enumerate(doc):
                mode = classify_page(page)
                plan["modes"].append(mode.value)
                (plan["digital_pages"] if mode is PageMode.DIGITAL_VECTOR else plan["raster_pages"]).append(index)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure to caller
        plan["error"] = str(exc)
    return plan
