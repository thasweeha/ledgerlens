"""Path A: digital vector extraction (spec §2).

Pulls word tokens, font sizes, and exact bounding boxes straight from the PDF
vector stream via PyMuPDF - no rasterization, no image distortion.
"""

from __future__ import annotations

import pymupdf as fitz

from pipeline.tokens import PageMode, Token


def _split_span_words(span: dict) -> list:
    """Split a styled span into words with proportional sub-bboxes."""
    text = span["text"]
    x0, y0, x1, y1 = span["bbox"]
    words = []
    run_start = None
    for pos, char in enumerate(text + " "):
        if char.isspace():
            if run_start is not None:
                width = x1 - x0
                left = x0 + width * (run_start / len(text))
                right = x0 + width * (pos / len(text))
                words.append((left, y0, right, y1, text[run_start:pos]))
                run_start = None
        elif run_start is None:
            run_start = pos
    return words


def extract_page_tokens(page, page_index: int) -> list:
    """Extract one Token per word with native geometry and font size."""
    tokens = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span["text"].strip():
                    continue
                size = float(span.get("size", 0.0)) or None
                for wx0, wy0, wx1, wy1, word in _split_span_words(span):
                    if not word.strip():
                        continue
                    tokens.append(
                        Token(
                            text=word,
                            page=page_index,
                            bbox=(wx0, wy0, wx1, wy1),
                            mode=PageMode.DIGITAL_VECTOR,
                            confidence=1.0,
                            font_size=size,
                        )
                    )
    tokens.sort(key=lambda t: (t.y_center, t.x0))
    return tokens
