"""Core extraction engine with three-library cascade for PDF text extraction."""

import io
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader

from models import ExtractionMethod, ExtractionResult
from quality import QualityScorer

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """Cascade-based PDF text extraction with fallback strategy."""

    def __init__(
        self,
        pypdf2_quality_threshold: float = 0.5,
        pymupdf_quality_threshold: float = 0.5,
        min_words: int = 50,
        tesseract_config: str = "--psm 6",
    ):
        """Initialize extraction engine with quality thresholds.

        Args:
            pypdf2_quality_threshold: Minimum quality score for PyPDF2 result.
            pymupdf_quality_threshold: Minimum quality score for PyMuPDF result.
            min_words: Minimum words required for acceptable extraction.
            tesseract_config: Tesseract command-line configuration.
        """
        self.pypdf2_threshold = pypdf2_quality_threshold
        self.pymupdf_threshold = pymupdf_quality_threshold
        self.quality_scorer = QualityScorer(min_words=min_words)
        self.tesseract_config = tesseract_config

    def extract(self, pdf_path: str) -> ExtractionResult:
        """Extract text from PDF using cascade strategy.

        Attempts extraction in order: PyPDF2 -> PyMuPDF -> Tesseract OCR.
        Each stage is only attempted if previous stage returns insufficient quality.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            ExtractionResult with text, method, confidence, and timing.
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return ExtractionResult(
                text="",
                method_used=ExtractionMethod.PYPDF2,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error=f"File not found: {pdf_path}",
            )

        try:
            page_count = self._get_page_count(str(pdf_path))

            # Stage 1: PyPDF2
            text, quality_score = self._extract_pypdf2(str(pdf_path))
            if self.quality_scorer.is_acceptable(quality_score) and quality_score.overall >= self.pypdf2_threshold:
                logger.info(
                    f"Stage 1 (PyPDF2) success: {quality_score.word_count} words, "
                    f"score {quality_score.overall:.2%}"
                )
                return ExtractionResult(
                    text=text,
                    method_used=ExtractionMethod.PYPDF2,
                    confidence_score=quality_score.overall,
                    processing_time=time.time() - start_time,
                    quality_score=quality_score,
                    page_count=page_count,
                    has_images=False,
                )

            logger.debug(f"PyPDF2 quality insufficient ({quality_score.overall:.2%}), advancing to PyMuPDF")

            # Stage 2: PyMuPDF
            text, quality_score, has_images = self._extract_pymupdf(str(pdf_path))
            if self.quality_scorer.is_acceptable(quality_score) and quality_score.overall >= self.pymupdf_threshold:
                logger.info(
                    f"Stage 2 (PyMuPDF) success: {quality_score.word_count} words, "
                    f"score {quality_score.overall:.2%}"
                )
                return ExtractionResult(
                    text=text,
                    method_used=ExtractionMethod.PYMUPDF,
                    confidence_score=quality_score.overall,
                    processing_time=time.time() - start_time,
                    quality_score=quality_score,
                    page_count=page_count,
                    has_images=has_images,
                )

            logger.debug(f"PyMuPDF quality insufficient ({quality_score.overall:.2%}), advancing to Tesseract")

            # Stage 3: Tesseract OCR
            text, quality_score = self._extract_tesseract(str(pdf_path))
            logger.info(
                f"Stage 3 (Tesseract) completion: {quality_score.word_count} words, "
                f"score {quality_score.overall:.2%}"
            )

            return ExtractionResult(
                text=text,
                method_used=ExtractionMethod.TESSERACT,
                confidence_score=quality_score.overall,
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                page_count=page_count,
                has_images=True,
            )

        except Exception as e:
            logger.error(f"Extraction pipeline failed for {pdf_path}: {e}")
            return ExtractionResult(
                text="",
                method_used=ExtractionMethod.PYPDF2,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error=str(e),
            )

    def _extract_pypdf2(self, pdf_path: str) -> Tuple[str, "QualityScore"]:
        """Extract text using PyPDF2.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (extracted_text, quality_score).
        """
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

            quality_score = self.quality_scorer.score(text)
            return text, quality_score

        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return "", self.quality_scorer.score("")

    def _extract_pymupdf(self, pdf_path: str) -> Tuple[str, "QualityScore", bool]:
        """Extract text using PyMuPDF (fitz).

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (extracted_text, quality_score, has_images).
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            total_images = 0

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
                total_images += len(page.get_images())

            doc.close()
            quality_score = self.quality_scorer.score(text)
            has_images = total_images > 0

            return text, quality_score, has_images

        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return "", self.quality_scorer.score(""), False

    def _extract_tesseract(self, pdf_path: str) -> Tuple[str, "QualityScore"]:
        """Extract text using Tesseract OCR.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (extracted_text, quality_score).
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("ppm")

                image = Image.open(io.BytesIO(image_bytes))
                page_text = pytesseract.image_to_string(image, config=self.tesseract_config)
                full_text += page_text + "\n"

            doc.close()
            quality_score = self.quality_scorer.score(full_text)
            return full_text, quality_score

        except Exception as e:
            logger.warning(f"Tesseract extraction failed: {e}")
            return "", self.quality_scorer.score("")

    @staticmethod
    def _get_page_count(pdf_path: str) -> Optional[int]:
        """Get page count from PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Number of pages or None if unable to determine.
        """
        try:
            doc = fitz.open(pdf_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception:
            return None


def extract_text(pdf_path: str) -> ExtractionResult:
    """Convenience function to extract text from PDF with default settings.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        ExtractionResult with extracted text and metadata.
    """
    engine = ExtractionEngine()
    return engine.extract(pdf_path)
