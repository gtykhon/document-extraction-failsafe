# Development Notes

## Project Overview

This is a complete, runnable Python project demonstrating a production-grade document extraction system. The codebase is suitable for learning, reference, or deployment.

## Architecture Highlights

### Three-Library Cascade

The system attempts extraction in a specific order:

1. **PyPDF2** - Fastest, handles embedded-text PDFs (70% of documents)
2. **PyMuPDF** - Handles complex layouts, rotated pages, mixed content (20%)
3. **Tesseract OCR** - Final fallback for image-based or scanned PDFs (10%)

Each stage includes quality evaluation. If quality meets the threshold, extraction stops. Otherwise, it advances to the next library.

### Quality Scoring

Quality metrics include:

- **Character Validity**: Ratio of printable/valid characters
- **Garbage Ratio**: Detection of encoding errors or noise
- **Readability**: Sentence structure, common words, punctuation patterns
- **Encoding Health**: UTF-8 validation
- **Word Count**: Minimum threshold for acceptable extraction
- **Completeness**: Assessment based on document size

### PDF Classification

The detector module classifies PDFs into four types:

- `TEXT_EMBEDDED`: Pure text (uses fast PyPDF2 path)
- `IMAGE_BASED`: Scanned documents (skip to Tesseract)
- `MIXED`: Text + images (requires fallback handling)
- `COMPLEX_LAYOUT`: Multi-column, rotated, or unusual structures

## Code Quality

### Type Safety

- Full type annotations on all functions
- Enum-based type definitions (ExtractionMethod, PDFType)
- Dataclass models with clear field definitions
- Return types always specified

### Error Handling

- Graceful degradation at each cascade stage
- Exceptions logged but not re-raised
- Result objects include error field for debugging
- Extraction always returns result, never raises

### Logging

- DEBUG: Cascade advancement decisions
- INFO: Stage completion with metrics
- WARNING: Extraction method failures
- ERROR: Pipeline-level failures

Example cascade logs:
```
DEBUG: PyPDF2 quality insufficient (0.35), advancing to PyMuPDF
INFO: Stage 2 (PyMuPDF) success: 250 words, score 0.78
```

### Testing

- 25 total test cases
- Unit tests focus on individual components
- Integration tests verify cascade behavior
- Mock external dependencies (PyPDF2, PyMuPDF, Tesseract)
- Edge cases: empty PDFs, corrupt files, missing pages

Run tests: `pytest tests/ -v`

## Performance Characteristics

### Time Complexity

- PyPDF2: O(n) where n = text length, typically <100ms
- PyMuPDF: O(n·m) where m = page complexity, typically 200-500ms
- Tesseract: O(n·m²) due to OCR, typically 2-10 seconds

### Space Complexity

- Text storage: O(n) for full document text
- Page images (Tesseract): O(page_width × page_height × channels)
- Quality scoring: O(1) relative to text size

## Known Limitations

### Tesseract Configuration

The system uses `--psm 6` (uniform block of text). Alternative PSM modes:

- `--psm 0`: Auto orientation detection
- `--psm 3`: Fully automatic
- `--psm 11`: Sparse text

Adjust `tesseract_config` parameter in `ExtractionEngine` for different document types.

### Image Preprocessing

Current implementation uses 2x scaling for Tesseract. This helps with small fonts but increases processing time. Adjust `fitz.Matrix(2, 2)` in `_extract_tesseract` for different scaling.

### Language Support

Tesseract supports multiple languages via language packs. Default is English. Install additional languages: `tesseract-ocr-[lang]`

## Deployment Considerations

### Configuration

Quality thresholds are tunable per deployment:

```python
engine = ExtractionEngine(
    pypdf2_quality_threshold=0.5,  # Strict: 0.7, Lenient: 0.3
    pymupdf_quality_threshold=0.5,
    min_words=50,  # Higher for large documents, lower for small ones
    tesseract_config="--psm 6"
)
```

### Resource Management

Tesseract uses significant memory for image processing. For high-volume processing:

1. Process PDFs sequentially or with limited parallelism
2. Monitor memory usage, especially on Tesseract fallback
3. Consider page-level processing for very large documents
4. Set timeouts for extraction (not currently implemented)

### Monitoring

Integrate logging with your monitoring system:

```python
import logging
logging.getLogger("extractor").setLevel(logging.INFO)
logging.getLogger("detector").setLevel(logging.DEBUG)
```

Track these metrics:

- Extraction success rate per method
- Average processing time by method
- Quality score distribution
- Document type distribution

## Future Enhancements

### Pre-Classification

Add PDF complexity prediction to skip low-probability methods:

```python
pdf_type = classifier.classify(pdf_path)
if pdf_type == PDFType.IMAGE_BASED:
    # Skip PyPDF2, go directly to PyMuPDF or Tesseract
    result = engine._extract_pymupdf(pdf_path)
```

### Parallel OCR

Tesseract could process multiple pages in parallel, but coordination complexity often isn't justified for the 10% case.

### Language Detection

Detect document language before Tesseract processing:

```python
language = detect_language(text_from_pymupdf)
tesseract_config = f"--psm 6 -l {language}"
```

### Machine Learning Quality Gate

Replace heuristic quality scoring with trained model for better threshold tuning.

## References

### Library Documentation

- PyPDF2: https://pypdf2.readthedocs.io/
- PyMuPDF: https://pymupdf.readthedocs.io/
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Pillow: https://pillow.readthedocs.io/

### Related Work

- PDFPlumber: Specialized library for table extraction
- pdfminer.six: Alternative PDF parsing approach
- pdf2image + pytesseract: Alternative OCR pipeline

## License & Attribution

This is a sanitized reference implementation of a production system built for a federal financial agency through a government IT consulting firm (2019-2023).
