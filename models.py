"""Data models for document extraction system."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExtractionMethod(Enum):
    """Extraction method used by the cascade."""
    PYPDF2 = "pypdf2"
    PYMUPDF = "pymupdf"
    TESSERACT = "tesseract"


class PDFType(Enum):
    """Classification of PDF document type based on content characteristics."""
    TEXT_EMBEDDED = "text_embedded"
    IMAGE_BASED = "image_based"
    MIXED = "mixed"
    COMPLEX_LAYOUT = "complex_layout"


@dataclass
class QualityScore:
    """Quality metrics for extracted text."""
    overall: float
    readability: float
    completeness: float
    character_validity: float
    garbage_ratio: float
    encoding_health: bool
    word_count: int


@dataclass
class ExtractionResult:
    """Result of PDF text extraction with metadata."""
    text: str
    method_used: ExtractionMethod
    confidence_score: float
    processing_time: float
    quality_score: Optional[QualityScore] = None
    page_count: Optional[int] = None
    has_images: Optional[bool] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether extraction produced usable text."""
        return bool(self.text and self.confidence_score > 0)
