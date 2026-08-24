"""Unified token model shared by the digital-vector and raster OCR pipelines.

All bounding boxes are expressed in PDF point space (72 DPI origin) so tokens
produced by PyMuPDF and tokens produced by the 300 DPI neural OCR pipeline can
be projected into the same column lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


class PageMode(str, Enum):
    DIGITAL_VECTOR = "DIGITAL_VECTOR"
    FLATBED_RASTER = "FLATBED_RASTER"

    @property
    def short(self) -> str:
        return "DIGITAL" if self is PageMode.DIGITAL_VECTOR else "RASTER"


@dataclass(frozen=True)
class Token:
    """One word with geometry, provenance, and OCR confidence."""

    text: str
    page: int
    bbox: BBox
    mode: PageMode = PageMode.DIGITAL_VECTOR
    confidence: float = 1.0
    font_size: Optional[float] = None

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    def scaled(self, factor: float) -> "Token":
        """Return a copy with bbox scaled by `factor` (e.g. pixels -> points)."""
        x0, y0, x1, y1 = self.bbox
        return replace(self, bbox=(x0 * factor, y0 * factor, x1 * factor, y1 * factor))

    def provenance(self, file_name: str) -> str:
        return (
            f"Source: {file_name} | Page: {self.page + 1} | Mode: {self.mode.short} "
            f"| BBox: [{self.x0:.0f}, {self.y0:.0f}, {self.x1:.0f}, {self.y1:.0f}]"
        )


def overlap_fraction(token: Token, lane_start: float, lane_end: float) -> float:
    """Horizontal overlap of a token with a lane interval, relative to token width."""
    if token.width <= 0:
        return 1.0 if lane_start <= token.x_center <= lane_end else 0.0
    overlap = min(token.x1, lane_end) - max(token.x0, lane_start)
    if overlap <= 0:
        return 0.0
    return overlap / token.width


def cluster_baselines(tokens: List[Token], tolerance: float) -> List[List[Token]]:
    """Group tokens into visual baselines (rows) using y-center proximity.

    Tokens must share one page; rows come back top-to-bottom, tokens inside a
    row left-to-right.
    """
    rows: List[List[Token]] = []
    for token in sorted(tokens, key=lambda t: (t.y_center, t.x0)):
        placed = False
        for row in reversed(rows):
            reference = sum(t.y_center for t in row) / len(row)
            if abs(token.y_center - reference) <= tolerance:
                row.append(token)
                placed = True
                break
            if token.y_center - reference > tolerance:
                break
        if not placed:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda t: t.x0)
    return rows


def median(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    n = len(items)
    mid = n // 2
    if n % 2:
        return items[mid]
    return (items[mid - 1] + items[mid]) / 2.0


def page_heights(tokens: List[Token]) -> dict:
    """Per-page (max_y, median_token_height) used for table-region trimming."""
    info: dict = {}
    for token in tokens:
        entry = info.setdefault(token.page, {"max_y": 0.0, "heights": []})
        entry["max_y"] = max(entry["max_y"], token.y1)
        if token.height > 0:
            entry["heights"].append(token.height)
    return {
        page: (entry["max_y"], median(entry["heights"]) or 10.0)
        for page, entry in info.items()
    }
