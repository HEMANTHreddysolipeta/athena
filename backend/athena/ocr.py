"""Local OCR for scanned PDF pages (RapidOCR: PaddleOCR models on ONNX Runtime).

Runs fully offline on the CPU; the models ship inside the pip package.
Loaded on first use only, because most documents never need it.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    logger.info("Loading OCR engine")
    return RapidOCR()


def ocr_image_bytes(png: bytes) -> str:
    """Text of an image, top-to-bottom, one detected line per output line."""
    result, _ = _engine()(png)
    if not result:
        return ""
    # result rows: [box, text, confidence]; box[0] is the top-left corner.
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    return "\n".join(text for _, text, confidence in rows if float(confidence) >= 0.5)
