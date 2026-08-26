"""EasyOCR-backed text recognition for raster pages.

Public interface:
    recognizer = EasyOCRRecognizer()
    results = recognizer.recognize_batch(list_of_crops)  # [(text, conf)]
    text, conf = recognizer.recognize(crop, allowlist='0123456789')

Crops are grayscale or color numpy arrays.  Confidence is returned as a float
in [0, 1].  An optional ``allowlist`` restricts the recognition alphabet for
field-specific ICR (e.g. checks).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Require offline mode -- models must come from the local bundled directory.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image


def _resolve_model_dir() -> str:
    """Locate the bundled EasyOCR models directory.

    Dev run:      <repo root>/models/easyocr-models
    PyInstaller:  <_MEIPASS>/models/easyocr-models
    Override:     LEDGERLENS_EASYOCR_MODEL_DIR environment variable
    """
    override = os.environ.get("LEDGERLENS_EASYOCR_MODEL_DIR")
    if override:
        return override
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # noqa: SLF001
    else:
        base = Path(__file__).resolve().parent.parent
    return str(base / "models" / "easyocr-models")


DEFAULT_MODEL_DIR = _resolve_model_dir()


class EasyOCRRecognizer:
    """Recognition-only EasyOCR wrapper.

    Uses the lightweight english_g2 recognition model (14 MB).  The detector
    is disabled (``detector=False``) because pipeline/raster.py already segments
    lines before handing crops to the recognizer.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        gpu: bool = False,
    ):
        import easyocr

        self._reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            detector=False,          # raster.py already segments lines
            recognizer=True,
            download_enabled=False,  # must already be on disk
            model_storage_directory=model_dir or DEFAULT_MODEL_DIR,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recognize_batch(
        self,
        crops: Sequence[np.ndarray],
        allowlist: str | None = None,
    ) -> List[Tuple[str, float]]:
        """Recognize a batch of image crops; order is preserved."""
        if not crops:
            return []

        results: List[Tuple[str, float]] = []
        for crop in crops:
            text, conf = self._recognize_one(crop, allowlist=allowlist)
            results.append((text, conf))
        return results

    def recognize(
        self,
        crop: np.ndarray,
        allowlist: str | None = None,
    ) -> Tuple[str, float]:
        """Single-crop convenience wrapper."""
        return self._recognize_one(crop, allowlist=allowlist)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recognize_one(
        self,
        crop: np.ndarray,
        allowlist: str | None = None,
    ) -> Tuple[str, float]:
        """Run EasyOCR recognition on a single grayscale/color numpy array."""
        pil = self._to_pil(crop)
        img_array = np.array(pil.convert("L"))  # grayscale numpy

        kwargs: dict = {"detail": 1}
        if allowlist:
            kwargs["allowlist"] = allowlist

        raw = self._reader.recognize(img_array, **kwargs)

        if not raw:
            return ("", 0.0)

        _bbox, text, conf = raw[0]
        return (str(text).strip(), float(conf))

    @staticmethod
    def _to_pil(crop: np.ndarray) -> Image.Image:
        """Accept grayscale or BGR arrays, return an RGB PIL Image."""
        if crop.ndim == 2:
            return Image.fromarray(crop).convert("RGB")
        if crop.ndim == 3 and crop.shape[2] == 3:
            import cv2

            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        raise ValueError(f"unsupported crop shape: {crop.shape}")
