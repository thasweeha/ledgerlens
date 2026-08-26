"""Path B: flatbed scanned pipeline for low-DPI bank statements.

Workflow per raster page:
1. Render the PDF page at its detected native DPI (typically 150-200).
2. Run pipeline.preprocessor.preprocess_scan to deskew, upscale to 400 DPI
   equivalent with Lanczos, sharpen, apply CLAHE, and binarize.
3. Segment words from the binary image and cluster them into visual text lines.
4. Extract each line/chunk from the enhanced RGB image and recognize it with
   EasyOCR (detector disabled).
5. Project recognized text back onto the segmented word boxes and emit Token
   objects scaled to PDF point space.

The confidence threshold is intentionally low (0.25) so marginal but correct
OCR results on noisy scans still survive to the column parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline import preprocessor
from pipeline.tokens import PageMode, Token

MAX_SKEW_DEGREES = 15.0
MIN_WORD_HEIGHT_PX = 12
MIN_WORD_AREA_PX = 80
OCR_BATCH_SIZE = 16
LINE_TARGET_HEIGHT_PX = 64
LINE_MAX_ASPECT = 8.0
OCR_CONFIDENCE_THRESHOLD = 0.25


def _chunk_line_boxes(
    line: List[Tuple[int, int, int, int]],
    max_aspect: float = LINE_MAX_ASPECT,
) -> List[List[Tuple[int, int, int, int]]]:
    """Split a baseline into horizontal chunks no wider than max_aspect x
    line-height, cutting only at word-box boundaries so glyphs survive."""
    chunks: List[List[Tuple[int, int, int, int]]] = []
    current: List[Tuple[int, int, int, int]] = []
    for box in line:
        if current:
            x0 = min(b[0] for b in current)
            height = max(b[3] - b[1] for b in current + [box])
            if (box[2] - x0) / max(1, height) > max_aspect:
                chunks.append(current)
                current = [box]
                continue
        current.append(box)
    if current:
        chunks.append(current)
    return chunks


def render_page(path: str | Path, page_index: int = 0, dpi: int = 150) -> np.ndarray:
    """Render one PDF page into a high-resolution BGR image."""
    import pymupdf as fitz

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with fitz.open(path) as doc:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page index {page_index} out of range (0..{len(doc) - 1})")
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def detect_native_dpi(
    path: str | Path, page_index: int, floor: float = 96.0, cap: float = 400.0
) -> float:
    """Effective resolution of the dominant embedded raster image.

    Scanned PDFs often embed low-DPI JPEGs; rendering above the native
    resolution only interpolates soft strokes and degrades both thresholding
    and neural recognition.  We render at native DPI and let the preprocessor
    perform a controlled Lanczos upscale.
    """
    import pymupdf as fitz

    with fitz.open(path) as doc:
        page = doc.load_page(page_index)
        page_rect = page.rect
        page_area = max(1.0, abs(page_rect.width * page_rect.height))
        best_dpi: Optional[float] = None
        best_coverage = 0.0
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # noqa: BLE001 - malformed xrefs must not kill routing
                continue
            for rect in rects:
                coverage = abs(rect.width * rect.height) / page_area
                if coverage <= best_coverage or rect.width < 1:
                    continue
                best_coverage = coverage
                best_dpi = info[2] / rect.width * 72.0
    if best_dpi is None:
        return min(cap, max(floor, preprocessor.DEFAULT_SOURCE_DPI))
    return min(cap, max(floor, best_dpi))


def _merge_boxes(
    boxes: List[Tuple[int, int, int, int]],
    gap_ratio: float = 0.10,
) -> List[Tuple[int, int, int, int]]:
    """Reunite glyph fragments of the same word (tiny gaps only); real
    inter-word gaps must survive so lane assignment stays meaningful."""

    def vertical_overlap(a, b) -> float:
        top = max(a[1], b[1])
        bottom = min(a[3], b[3])
        return max(0, bottom - top)

    merged = True
    while merged:
        merged = False
        out: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            absorbed = False
            for i, other in enumerate(out):
                gap = max(box[0], other[0]) - min(box[2], other[2])
                reference_height = min(box[3] - box[1], other[3] - other[1])
                if (
                    gap <= reference_height * gap_ratio
                    and vertical_overlap(box, other) > 0.5 * reference_height
                ):
                    out[i] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    absorbed = True
                    merged = True
                    break
            if not absorbed:
                out.append(box)
        boxes = out
    return boxes


def segment_word_bboxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fuse glyphs into words with a horizontal morphological close, then
    return word bounding boxes sorted top-to-bottom, left-to-right."""
    width = binary.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 400), 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < MIN_WORD_HEIGHT_PX or w < 4 or w * h < MIN_WORD_AREA_PX:
            continue
        boxes.append((x, y, x + w, y + h))
    boxes = _merge_boxes(boxes)
    boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    return boxes


def group_lines(
    boxes: List[Tuple[int, int, int, int]],
    height_factor: float = 0.6,
) -> List[List[Tuple[int, int, int, int]]]:
    """Cluster word boxes into visual text lines by baseline proximity."""
    ordered = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    heights = sorted(b[3] - b[1] for b in ordered)
    median_h = heights[len(heights) // 2] if heights else 10
    tolerance = max(6.0, median_h * height_factor)
    lines: List[List[Tuple[int, int, int, int]]] = []
    current: List[Tuple[int, int, int, int]] = []
    center: float = 0.0
    for box in ordered:
        yc = (box[1] + box[3]) / 2
        if not current or abs(yc - center) <= tolerance:
            current.append(box)
            center = sum((b[1] + b[3]) / 2 for b in current) / len(current)
        else:
            lines.append(current)
            current = [box]
            center = yc
    if current:
        lines.append(current)
    return [sorted(line, key=lambda b: b[0]) for line in lines]


def _distribute_line_text(
    text: str,
    line_boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[str, Tuple[int, int, int, int], bool]]:
    """Project recognized line text back onto detected word boxes.

    Words are placed at their proportional x-offset inside the line strip and
    assigned to the nearest detected box; several recognized words may land on
    one box (joined with spaces), empty boxes yield no token. The boolean marks
    whether the box received any text.
    """
    if not text.strip() or not line_boxes:
        return []
    x0 = min(b[0] for b in line_boxes)
    x1 = max(b[2] for b in line_boxes)
    centers = [(b[0] + b[2]) / 2 for b in line_boxes]
    span = max(1.0, x1 - x0)

    pieces = text.split()
    total_chars = len(text.strip())
    out: List[Tuple[str, Tuple[int, int, int, int], bool]] = []
    offset = 0
    for piece in pieces:
        start = text.find(piece, offset)
        if start < 0:
            continue
        offset = start + len(piece)
        proj = x0 + span * ((start + len(piece) / 2) / total_chars)
        nearest = min(range(len(line_boxes)), key=lambda i: abs(centers[i] - proj))
        box = line_boxes[nearest]
        for idx, (existing, ebox, _) in enumerate(out):
            if ebox == box:
                out[idx] = (f"{existing} {piece}", ebox, True)
                break
        else:
            out.append((piece, box, True))
    return out


def extract_page_tokens(
    path: str | Path,
    page_index: int,
    recognizer=None,
    dpi: Optional[int] = None,
    extract_checks: bool = False,
) -> dict:
    """Full raster path for one page.

    Renders at the embedded image's native DPI, preprocesses for low-DPI OCR,
    segments words from the binary image, groups them into lines, recognizes
    each line strip with EasyOCR, and projects text back onto word boxes.
    Recognition runs on the enhanced RGB image (never binarized).  Bounding
    boxes are returned in PDF point space.

    Args:
        path: PDF file path.
        page_index: 0-based page index.
        recognizer: EasyOCR recognizer instance (created lazily if None).
        dpi: Optional render DPI override.  None means native DPI.
        extract_checks: If True, run check detection on the preprocessed page
            and return detected checks in the result dict.

    Returns:
        Dict with keys: tokens, skew_angle, words_detected, lines_recognized,
        low_confidence_words, source_dpi, effective_dpi, and optionally checks.
    """
    if recognizer is None:
        from pipeline.ocr_easyocr import EasyOCRRecognizer

        recognizer = EasyOCRRecognizer()

    source_dpi = detect_native_dpi(path, page_index)
    # All raster pages in this project are assumed to be low-DPI flatbed scans.
    source_dpi = float(np.clip(source_dpi, preprocessor.MIN_SOURCE_DPI, preprocessor.MAX_SOURCE_DPI))

    if dpi is None:
        dpi = int(round(source_dpi))

    image = render_page(path, page_index, dpi)
    processed = preprocessor.preprocess_scan(image, source_dpi=source_dpi)

    rgb = processed["rgb"]
    binary = processed["binary"]
    effective_dpi = float(processed["effective_dpi"])
    scale = 72.0 / effective_dpi

    boxes = segment_word_bboxes(binary)
    lines = group_lines(boxes)

    pad = 4
    crops: List[np.ndarray] = []
    chunk_map: List[List[Tuple[int, int, int, int]]] = []
    for line in lines:
        for chunk in _chunk_line_boxes(line):
            x0 = min(b[0] for b in chunk)
            y0 = max(0, min(b[1] for b in chunk) - pad)
            x1 = min(rgb.shape[1], max(b[2] for b in chunk))
            y1 = min(rgb.shape[0], max(b[3] for b in chunk) + pad)
            strip = rgb[y0:y1, x0:x1]
            if not strip.size:
                strip = np.zeros((8, 8, 3), np.uint8)
            if strip.shape[0] < LINE_TARGET_HEIGHT_PX:
                factor = LINE_TARGET_HEIGHT_PX / strip.shape[0]
                strip = cv2.resize(
                    strip,
                    None,
                    fx=factor,
                    fy=factor,
                    interpolation=cv2.INTER_LANCZOS4,
                )
            crops.append(strip)
            chunk_map.append(chunk)

    texts: List[str] = []
    confidences: List[float] = []
    for start in range(0, len(crops), OCR_BATCH_SIZE):
        chunk_crops = crops[start : start + OCR_BATCH_SIZE]
        results = recognizer.recognize_batch(chunk_crops)
        texts.extend(t for t, _ in results)
        confidences.extend(c for _, c in results)

    tokens: List[Token] = []
    low_confidence = 0
    for chunk, text, confidence in zip(chunk_map, texts, confidences):
        for piece, box, _used in _distribute_line_text(text, chunk):
            if confidence < OCR_CONFIDENCE_THRESHOLD:
                low_confidence += 1
            tokens.append(
                Token(
                    text=piece,
                    page=page_index,
                    bbox=(
                        box[0] * scale,
                        box[1] * scale,
                        box[2] * scale,
                        box[3] * scale,
                    ),
                    mode=PageMode.FLATBED_RASTER,
                    confidence=round(float(confidence), 4),
                )
            )

    # Sort top-to-bottom, left-to-right so the column parser sees logical rows.
    tokens.sort(key=lambda t: (t.y_center, t.x0))

    result: dict = {
        "tokens": tokens,
        "skew_angle": round(float(processed["skew_angle"]), 2),
        "words_detected": len(boxes),
        "lines_recognized": len(lines),
        "low_confidence_words": low_confidence,
        "source_dpi": round(source_dpi, 1),
        "effective_dpi": round(effective_dpi, 1),
    }

    if extract_checks:
        from pipeline import check_ocr

        result["checks"] = check_ocr.detect_checks(rgb, page_index, recognizer)

    return result
