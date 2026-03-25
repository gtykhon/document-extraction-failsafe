"""Text quality scoring for extracted PDF content."""

import re
from typing import Tuple

from models import QualityScore


class QualityScorer:
    """Evaluates quality of extracted text using multiple metrics."""

    def __init__(
        self,
        min_words: int = 50,
        max_garbage_ratio: float = 0.05,
        min_readability: float = 0.6,
    ):
        """Initialize quality scorer with thresholds.

        Args:
            min_words: Minimum word count to consider text valid.
            max_garbage_ratio: Maximum acceptable garbage character ratio.
            min_readability: Minimum readability score (0-1) for acceptable text.
        """
        self.min_words = min_words
        self.max_garbage_ratio = max_garbage_ratio
        self.min_readability = min_readability

    def score(self, text: str) -> QualityScore:
        """Score extracted text quality.

        Args:
            text: Extracted text to evaluate.

        Returns:
            QualityScore with detailed metrics.
        """
        if not text:
            return QualityScore(
                overall=0.0,
                readability=0.0,
                completeness=0.0,
                character_validity=0.0,
                garbage_ratio=1.0,
                encoding_health=False,
                word_count=0,
            )

        word_count = self._count_words(text)
        character_validity = self._check_character_validity(text)
        garbage_ratio = self._calculate_garbage_ratio(text)
        readability = self._calculate_readability(text)
        encoding_health = self._check_encoding_health(text)
        completeness = self._calculate_completeness(word_count)

        overall = self._calculate_overall_score(
            character_validity, readability, completeness, garbage_ratio
        )

        return QualityScore(
            overall=overall,
            readability=readability,
            completeness=completeness,
            character_validity=character_validity,
            garbage_ratio=garbage_ratio,
            encoding_health=encoding_health,
            word_count=word_count,
        )

    def is_acceptable(self, quality_score: QualityScore) -> bool:
        """Determine if quality score meets acceptance threshold.

        Args:
            quality_score: Quality metrics to evaluate.

        Returns:
            True if text meets minimum quality standards.
        """
        return (
            quality_score.word_count >= self.min_words
            and quality_score.garbage_ratio <= self.max_garbage_ratio
            and quality_score.readability >= self.min_readability
            and quality_score.encoding_health
        )

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text.

        Args:
            text: Text to count.

        Returns:
            Word count.
        """
        return len(text.split())

    @staticmethod
    def _check_character_validity(text: str) -> float:
        """Check validity of characters in text.

        Calculates the ratio of printable/ASCII characters to total characters.

        Args:
            text: Text to check.

        Returns:
            Ratio of valid characters (0-1).
        """
        if not text:
            return 0.0

        valid_chars = sum(
            1 for char in text
            if char.isprintable() or char.isspace() or ord(char) < 128
        )
        return valid_chars / len(text)

    @staticmethod
    def _calculate_garbage_ratio(text: str) -> float:
        """Calculate ratio of garbage/noise characters.

        Detects sequences of non-ASCII, non-letter characters that indicate
        encoding errors or OCR artifacts.

        Args:
            text: Text to analyze.

        Returns:
            Ratio of garbage characters (0-1).
        """
        if not text:
            return 0.0

        garbage_pattern = r"[^\w\s\-.,;:!?\'\"()\[\]\{\}/@#$%&*+=]"
        garbage_matches = len(re.findall(garbage_pattern, text))

        return min(1.0, garbage_matches / max(len(text), 1))

    @staticmethod
    def _calculate_readability(text: str) -> float:
        """Calculate readability score based on sentence structure.

        Uses simple heuristics: presence of common words, sentence length,
        and punctuation patterns.

        Args:
            text: Text to analyze.

        Returns:
            Readability score (0-1).
        """
        if not text:
            return 0.0

        common_words = [
            "the", "and", "to", "of", "a", "in", "is", "for", "that", "with"
        ]
        text_lower = text.lower()
        common_word_count = sum(1 for word in common_words if word in text_lower)

        sentences = len(re.split(r"[.!?]+", text)) - 1
        avg_sentence_length = len(text.split()) / max(sentences, 1)

        punctuation_score = (
            len(re.findall(r"[.!?,;:]", text)) / max(len(text.split()), 1)
        )

        common_words_score = min(common_word_count / len(common_words), 1.0)
        length_score = 1.0 if 10 < avg_sentence_length < 25 else 0.7
        punct_score = 1.0 if 0.05 < punctuation_score < 0.2 else 0.7

        return (common_words_score * 0.4 + length_score * 0.3 + punct_score * 0.3)

    @staticmethod
    def _check_encoding_health(text: str) -> bool:
        """Check for encoding problems.

        Args:
            text: Text to check.

        Returns:
            True if text appears properly encoded.
        """
        if not text:
            return False

        try:
            text.encode("utf-8").decode("utf-8")
            return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False

    def _calculate_completeness(self, word_count: int) -> float:
        """Calculate completeness based on word count.

        Args:
            word_count: Number of words extracted.

        Returns:
            Completeness score (0-1).
        """
        if word_count >= self.min_words * 5:
            return 1.0
        elif word_count >= self.min_words:
            return 0.5 + (word_count / (self.min_words * 2)) * 0.5
        else:
            return word_count / self.min_words * 0.5

    def _calculate_overall_score(
        self,
        character_validity: float,
        readability: float,
        completeness: float,
        garbage_ratio: float,
    ) -> float:
        """Calculate overall quality score.

        Args:
            character_validity: Character validity ratio.
            readability: Readability score.
            completeness: Completeness score.
            garbage_ratio: Garbage character ratio.

        Returns:
            Overall quality score (0-1).
        """
        garbage_penalty = 1.0 - garbage_ratio
        combined = (
            character_validity * 0.25
            + readability * 0.35
            + completeness * 0.25
            + garbage_penalty * 0.15
        )
        return max(0.0, min(1.0, combined))


def score_text_quality(text: str) -> QualityScore:
    """Convenience function to score text quality with default thresholds.

    Args:
        text: Text to score.

    Returns:
        QualityScore with detailed metrics.
    """
    scorer = QualityScorer()
    return scorer.score(text)
