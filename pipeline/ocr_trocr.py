"""TrOCR-backed text recognition for raster pages (spec §2, Path B).

Wraps microsoft/trocr-{base,small}-printed behind a tiny interface so the
raster pipeline stays recognizer-agnostic:

    recognizer = TrOCRRecognizer()
    results = recognizer.recognize_batch(list_of_gray_crops)  # [(text, conf)]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Force HuggingFace offline mode BEFORE any transformers/huggingface_hub import:
# any accidental Hub attempt must fail loudly instead of silently reaching the
# network. All model loading goes through local bundled weights (DEFAULT_MODEL).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image


def _resolve_default_model_path() -> str:
    """Locate the bundled trocr-small-printed weights on local disk.

    Dev run:      <repo root>/models/trocr-small-printed
    PyInstaller:  <_MEIPASS>/models/trocr-small-printed  (spec data entry)
    Override:     LEDGERLENS_TROCR_MODEL environment variable
    """
    override = os.environ.get("LEDGERLENS_TROCR_MODEL")
    if override:
        return override
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # noqa: SLF001 - PyInstaller standard
    else:
        base = Path(__file__).resolve().parent.parent
    return str(base / "models" / "trocr-small-printed")


DEFAULT_MODEL = _resolve_default_model_path()


class TrOCRRecognizer:
    """Printed-text recognizer returning per-crop text plus mean token
    confidence in [0, 1]."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_new_tokens: int = 32,
    ):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
        except ValueError:
            import json
            import logging

            import transformers
            from transformers import ViTImageProcessor

            logging.getLogger(__name__).warning(
                "TrOCRProcessor.from_pretrained failed (%s); transformers>=5 cannot "
                "auto-convert TrOCR's sentencepiece tokenizer, building processor "
                "from explicit components instead.",
                model_name,
            )
            model_dir = Path(model_name)
            with open(model_dir / "tokenizer_config.json", encoding="utf-8") as fh:
                tokenizer_class = json.load(fh).get("tokenizer_class", "XLMRobertaTokenizer")
            self.processor = TrOCRProcessor(
                image_processor=ViTImageProcessor.from_pretrained(model_name),
                tokenizer=getattr(transformers, tokenizer_class).from_pretrained(model_name),
            )
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _to_pil(self, crop: np.ndarray) -> Image.Image:
        """Accept grayscale or BGR arrays; TrOCR expects natural-looking
        images, so never feed binarized bitmaps."""
        if crop.ndim == 2:
            return Image.fromarray(crop).convert("RGB")
        if crop.ndim == 3 and crop.shape[2] == 3:
            import cv2

            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        raise ValueError(f"unsupported crop shape: {crop.shape}")

    @torch.inference_mode()
    def recognize_batch(self, crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize a batch of image crops; order is preserved."""
        if not crops:
            return []
        pixel_values = self.processor(
            images=[self._to_pil(crop) for crop in crops],
            return_tensors="pt",
        ).pixel_values.to(self.device)

        generated = self.model.generate(
            pixel_values,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        texts = self.processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )

        # Geometric mean of per-step max probabilities over each sequence's
        # real tokens (scores[t] produced sequences[:, t + 1]).
        step_probs = [
            torch.softmax(step_scores, dim=-1).max(dim=-1).values
            for step_scores in generated.scores
        ]
        eos_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id

        confidences: List[float] = []
        for row, sequence in enumerate(generated.sequences):
            log_total = 0.0
            steps_used = 0
            for t, token in enumerate(sequence.tolist()[1:]):
                if t >= len(step_probs) or token in (eos_id, pad_id):
                    break
                log_total += float(torch.log(step_probs[t][row]))
                steps_used += 1
            confidence = (
                float(torch.exp(torch.tensor(log_total / steps_used)))
                if steps_used
                else 0.0
            )
            confidences.append(confidence)

        results = []
        for text, confidence in zip(texts, confidences):
            results.append((text.strip(), confidence))
        return results

    def recognize(self, crop: np.ndarray) -> Tuple[str, float]:
        return self.recognize_batch([crop])[0]
