"""
Image file reader — runs OCR on uploaded image files.

Supported formats: JPG, JPEG, PNG.

The module opens the image with Pillow, converts to RGB if needed,
and delegates to the OCR processor.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image

from ingestion.ocr_processor import ocr_single_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImageReadResult:
    """Result of reading and OCR-ing an image file."""

    text: str
    """Cleaned OCR text."""

    ocr_confidence: float
    """Average OCR confidence in [0.0, 1.0]."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_image(image_bytes: bytes) -> ImageReadResult:
    """Open an image from raw bytes and run OCR.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the uploaded image file (JPG / JPEG / PNG).

    Returns
    -------
    ImageReadResult

    Raises
    ------
    ValueError
        If the file cannot be opened as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.error("Failed to open image: %s", exc)
        raise ValueError(
            "The uploaded file could not be opened as an image. "
            "Please ensure it is a valid JPG, JPEG, or PNG file."
        ) from exc

    # Ensure RGB — PaddleOCR expects 3-channel images
    if img.mode != "RGB":
        img = img.convert("RGB")

    logger.info("Running OCR on uploaded image (%dx%d) …", img.width, img.height)

    result = ocr_single_image(img)

    return ImageReadResult(
        text=result.cleaned_text,
        ocr_confidence=result.confidence,
    )
