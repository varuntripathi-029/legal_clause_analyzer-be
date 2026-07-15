"""
OCR processing orchestrator.

Sits between the file readers (``pdf_reader``, ``image_reader``) and the
rest of the pipeline.  Responsibilities:

1. Accept one or more PIL images.
2. Delegate to the configured ``OCRProvider``.
3. Run post-OCR text cleaning.
4. Return a clean ``OCRProcessingResult``.

The module does **not** know where images come from — callers convert
PDFs / image files into PIL images before handing them here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ingestion.ocr_provider import OCRProvider, PaddleOCRProvider, OCRResult
from ingestion.text_cleaner import clean_ocr_text

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OCRProcessingResult:
    """Final output of the OCR + cleaning pipeline."""

    raw_text: str
    """Text straight out of the OCR engine (before cleaning)."""

    cleaned_text: str
    """Text after the full cleaning pipeline."""

    confidence: float
    """Average OCR confidence in [0.0, 1.0]."""

    page_count: int
    """Number of images / pages processed."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _get_provider() -> OCRProvider:
    """Return the default (singleton) PaddleOCR provider."""
    return PaddleOCRProvider()


def ocr_single_image(
    image: Image.Image,
    provider: OCRProvider | None = None,
) -> OCRProcessingResult:
    """Run OCR on one image and return cleaned text.

    Parameters
    ----------
    image:
        An RGB PIL image.
    provider:
        Optional custom ``OCRProvider``.  Defaults to ``PaddleOCRProvider``.
    """
    prov = provider or _get_provider()
    result: OCRResult = prov.extract_text(image)

    cleaned = clean_ocr_text(result.text)

    return OCRProcessingResult(
        raw_text=result.text,
        cleaned_text=cleaned,
        confidence=result.confidence,
        page_count=1,
    )


def ocr_multiple_images(
    images: list[Image.Image],
    provider: OCRProvider | None = None,
) -> OCRProcessingResult:
    """Run OCR on multiple images (e.g. scanned PDF pages) and merge.

    Parameters
    ----------
    images:
        List of RGB PIL images — typically one per PDF page.
    provider:
        Optional custom ``OCRProvider``.  Defaults to ``PaddleOCRProvider``.
    """
    if not images:
        return OCRProcessingResult(
            raw_text="",
            cleaned_text="",
            confidence=0.0,
            page_count=0,
        )

    prov = provider or _get_provider()

    logger.info("Running OCR on %d page(s) …", len(images))
    result: OCRResult = prov.extract_text_from_images(images)

    cleaned = clean_ocr_text(result.text)

    return OCRProcessingResult(
        raw_text=result.text,
        cleaned_text=cleaned,
        confidence=result.confidence,
        page_count=len(images),
    )
