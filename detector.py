"""PDF complexity classification to predict extraction success."""

import logging
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF

from models import PDFType

logger = logging.getLogger(__name__)


class PDFClassifier:
    """Analyzes PDF documents to predict extraction complexity."""

    def __init__(
        self,
        text_threshold: float = 0.3,
        image_threshold: float = 0.5,
    ):
        """Initialize PDF classifier with thresholds.

        Args:
            text_threshold: Minimum text ratio to classify as text-based.
            image_threshold: Minimum image ratio to classify as image-based.
        """
        self.text_threshold = text_threshold
        self.image_threshold = image_threshold

    def classify(self, pdf_path: str) -> PDFType:
        """Classify PDF document complexity.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            PDFType enum indicating document characteristics.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
                return PDFType.COMPLEX_LAYOUT

            doc = fitz.open(pdf_path)
            page_count = doc.page_count

            if page_count == 0:
                return PDFType.COMPLEX_LAYOUT

            text_ratio, image_ratio = self._analyze_pages(doc, page_count)
            doc.close()

            return self._determine_type(text_ratio, image_ratio)

        except Exception as e:
            logger.error(f"Error classifying PDF {pdf_path}: {e}")
            return PDFType.COMPLEX_LAYOUT

    def _analyze_pages(self, doc: fitz.Document, sample_size: int = None) -> Tuple[float, float]:
        """Analyze sample pages to determine text and image ratios.

        Args:
            doc: PyMuPDF document.
            sample_size: Number of pages to sample (None = all pages).

        Returns:
            Tuple of (text_ratio, image_ratio).
        """
        if sample_size is None:
            sample_size = min(doc.page_count, 10)

        step = max(1, doc.page_count // sample_size)
        text_count = 0
        image_count = 0

        for page_num in range(0, doc.page_count, step):
            page = doc[page_num]
            text_blocks = page.get_text("blocks")
            images = page.get_images()

            text_count += len([b for b in text_blocks if b[4].strip()])
            image_count += len(images)

        total_elements = text_count + image_count
        if total_elements == 0:
            return 0.0, 0.0

        text_ratio = text_count / total_elements
        image_ratio = image_count / total_elements

        return text_ratio, image_ratio

    def _determine_type(self, text_ratio: float, image_ratio: float) -> PDFType:
        """Determine PDF type based on content ratios.

        Args:
            text_ratio: Ratio of text content.
            image_ratio: Ratio of image content.

        Returns:
            PDFType classification.
        """
        if text_ratio >= self.text_threshold and image_ratio < 0.1:
            return PDFType.TEXT_EMBEDDED
        elif image_ratio >= self.image_threshold and text_ratio < 0.1:
            return PDFType.IMAGE_BASED
        elif text_ratio > 0.1 and image_ratio > 0.1:
            return PDFType.MIXED
        else:
            return PDFType.COMPLEX_LAYOUT


def classify_pdf(pdf_path: str) -> PDFType:
    """Convenience function to classify PDF complexity with default settings.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        PDFType enum indicating document characteristics.
    """
    classifier = PDFClassifier()
    return classifier.classify(pdf_path)
