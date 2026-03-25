"""Unit tests for text quality scoring."""

import pytest

from quality import QualityScorer


class TestQualityScorer:
    """Tests for QualityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a QualityScorer instance."""
        return QualityScorer(min_words=50, max_garbage_ratio=0.05)

    def test_empty_text(self, scorer):
        """Empty text should return zero scores."""
        quality = scorer.score("")
        assert quality.overall == 0.0
        assert quality.readability == 0.0
        assert quality.word_count == 0

    def test_valid_text(self, scorer):
        """Valid English text should score well."""
        text = "The quick brown fox jumps over the lazy dog. " * 5
        quality = scorer.score(text)
        assert quality.overall > 0.6
        assert quality.word_count > 50
        assert quality.character_validity > 0.9
        assert quality.garbage_ratio < 0.05
        assert quality.encoding_health is True

    def test_text_with_garbage(self, scorer):
        """Text with garbage characters should score lower."""
        text = "The quick brown fox \x00\x01\x02 jumps over the lazy dog. " * 5
        quality = scorer.score(text)
        assert quality.garbage_ratio > 0.05
        assert quality.overall < 0.7

    def test_short_text(self, scorer):
        """Text below word count threshold should have low completeness."""
        text = "The quick brown fox jumps."
        quality = scorer.score(text)
        assert quality.word_count < 50
        assert quality.completeness < 0.5

    def test_text_acceptability_threshold(self, scorer):
        """is_acceptable should respect minimum thresholds."""
        short_text = "Word."
        short_quality = scorer.score(short_text)
        assert not scorer.is_acceptable(short_quality)

        valid_text = "The quick brown fox jumps over the lazy dog. " * 10
        valid_quality = scorer.score(valid_text)
        assert scorer.is_acceptable(valid_quality)

    def test_readability_calculation(self, scorer):
        """Readability should reflect sentence structure."""
        # Well-structured text
        good_text = "This is a sentence. This is another sentence. " * 5
        good_quality = scorer.score(good_text)

        # Fragmented text
        bad_text = "word word word word word. " * 10
        bad_quality = scorer.score(bad_text)

        assert good_quality.readability > bad_quality.readability

    def test_encoding_health(self, scorer):
        """Properly encoded text should pass encoding checks."""
        text = "Valid UTF-8 text with special characters: éèêë ñ"
        quality = scorer.score(text)
        assert quality.encoding_health is True

    def test_character_validity(self, scorer):
        """Should correctly identify valid vs invalid characters."""
        valid_text = "Normal English text with numbers 123 and punctuation."
        valid_quality = scorer.score(valid_text)
        assert valid_quality.character_validity > 0.9

    def test_word_count_accuracy(self, scorer):
        """Word count should be accurate."""
        text = "one two three four five"
        quality = scorer.score(text)
        assert quality.word_count == 5

        text_with_punctuation = "one, two. three; four: five!"
        quality = scorer.score(text_with_punctuation)
        assert quality.word_count == 5

    def test_overall_score_composition(self, scorer):
        """Overall score should be bounded between 0 and 1."""
        texts = [
            "",
            "short",
            "The quick brown fox " * 5,
            "The quick brown fox jumps over the lazy dog. " * 20,
        ]
        for text in texts:
            quality = scorer.score(text)
            assert 0.0 <= quality.overall <= 1.0
            assert 0.0 <= quality.readability <= 1.0
            assert 0.0 <= quality.completeness <= 1.0

    def test_quality_scorer_thresholds(self):
        """QualityScorer should respect custom thresholds."""
        scorer_strict = QualityScorer(min_words=100, max_garbage_ratio=0.02)
        scorer_lenient = QualityScorer(min_words=10, max_garbage_ratio=0.2)

        text = "The quick brown fox jumps over the lazy dog. " * 3

        strict_quality = scorer_strict.score(text)
        lenient_quality = scorer_lenient.score(text)

        assert not scorer_strict.is_acceptable(strict_quality)
        assert scorer_lenient.is_acceptable(lenient_quality)


class TestQualityModule:
    """Tests for module-level quality functions."""

    def test_score_text_quality_function(self):
        """Module function should work with default thresholds."""
        from quality import score_text_quality

        text = "The quick brown fox jumps over the lazy dog. " * 5
        quality = score_text_quality(text)

        assert quality.overall > 0
        assert quality.word_count > 0
        assert isinstance(quality.overall, float)
