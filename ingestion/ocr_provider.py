"""
OCR provider abstraction layer.

Defines the ``OCRProvider`` interface and ships a single concrete
implementation backed by **PaddleOCR** (CPU-only).

Design notes
------------
* PaddleOCR and PaddlePaddle are imported lazily — they are heavy
  dependencies (~1 GB) that should not penalize startup when OCR is
  never invoked.
* The PaddleOCR model is loaded once (singleton) on first use.
* The interface is deliberately simple so a future swap to Google
  Vision, Azure Document Intelligence, or AWS Textract only requires
  a new ``OCRProvider`` subclass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OCRResult:
    """Immutable container returned by every ``OCRProvider``."""

    text: str
    """Extracted text (may be empty if nothing was detected)."""

    confidence: float
    """Average confidence score in the range [0.0, 1.0]."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class OCRProvider(ABC):
    """Contract every OCR backend must satisfy."""

    @abstractmethod
    def extract_text(self, image: Image.Image) -> OCRResult:
        """Run OCR on a single PIL image and return the result.

        Parameters
        ----------
        image:
            An RGB PIL ``Image`` instance.

        Returns
        -------
        OCRResult
            Extracted text and an average confidence score.
        """

    def extract_text_from_images(self, images: list[Image.Image]) -> OCRResult:
        """OCR multiple images and merge into one result.

        Subclasses may override this for batched / optimized paths.
        The default implementation simply iterates.

        Parameters
        ----------
        images:
            A list of RGB PIL ``Image`` instances (e.g. one per PDF page).

        Returns
        -------
        OCRResult
            Merged text separated by page breaks, with overall avg
            confidence.
        """
        if not images:
            return OCRResult(text="", confidence=0.0)

        texts: list[str] = []
        confidences: list[float] = []

        for idx, img in enumerate(images, start=1):
            logger.debug("OCR processing image %d / %d", idx, len(images))
            result = self.extract_text(img)
            if result.text:
                texts.append(result.text)
                confidences.append(result.confidence)

        merged_text = "\n\n".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(text=merged_text, confidence=avg_confidence)


# ---------------------------------------------------------------------------
# PaddleOCR implementation
# ---------------------------------------------------------------------------
class PaddleOCRProvider(OCRProvider):
    """PaddleOCR-backed provider (CPU-only, lazily initialized)."""

    _instance: PaddleOCRProvider | None = None
    _engine = None  # Will hold the PaddleOCR object

    def __new__(cls) -> PaddleOCRProvider:
        """Singleton — reuse the same PaddleOCR engine across calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_engine(self) -> None:
        """Lazily import and initialize the PaddleOCR model."""
        if self.__class__._engine is not None:
            return

        logger.info("Lazy-loading PaddleOCR engine (CPU) …")
        try:
            from paddleocr import PaddleOCR  # noqa: WPS433 — lazy import

            try:
                # Try PaddleOCR 3.x+ arguments (device='cpu', use_textline_orientation=True)
                self.__class__._engine = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                    device="cpu",
                    enable_mkldnn=False,
                )
            except (ValueError, TypeError) as err:
                logger.debug("PaddleOCR 3.x arguments failed (%s), trying 2.x fallback...", err)
                # Fallback to PaddleOCR 2.x arguments (use_gpu=False, use_angle_cls=True, show_log=False)
                self.__class__._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                    use_gpu=False,
                    enable_mkldnn=False,
                )
            logger.info("PaddleOCR engine initialized successfully.")
        except ImportError as exc:
            logger.error(
                "PaddleOCR is not installed. "
                "Install with: pip install paddleocr paddlepaddle"
            )
            raise RuntimeError(
                "PaddleOCR is required for OCR processing but is not installed."
            ) from exc

    # -- public interface ----------------------------------------------------

    def extract_text(self, image: Image.Image) -> OCRResult:
        """Run PaddleOCR on a single PIL image."""
        self._ensure_engine()

        img_array = np.array(image)

        try:
            raw_results = self._engine.ocr(img_array, cls=True)
        except TypeError:
            raw_results = self._engine.ocr(img_array)

        if not raw_results or not raw_results[0]:
            return OCRResult(text="", confidence=0.0)

        lines: list[str] = []
        confidences: list[float] = []

        for line_info in raw_results[0]:
            # Each line_info is [box_coords, (text, confidence)]
            text_conf = line_info[1]
            text = text_conf[0]
            conf = float(text_conf[1])

            if text.strip():
                lines.append(text.strip())
                confidences.append(conf)

        joined_text = "\n".join(lines)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(text=joined_text, confidence=avg_confidence)
