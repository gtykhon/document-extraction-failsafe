# Document Extraction Failsafe

A production-grade document extraction system that cascades through three libraries to achieve 99.2% accuracy on complex PDFs.

**[Read the full engineering write-up on LinkedIn →](https://www.linkedin.com/in/grygorii-t/recent-activity/all/)**

## The Problem

Single-library PDF extraction is unreliable in production. PyPDF2 is fast but fails on complex layouts with mixed content. PyMuPDF handles some layouts PyPDF2 misses but struggles with image-based pages. Tesseract excels on scanned documents but adds latency. In isolation, each library has a 15-30% failure rate on real-world financial documents. Organizations either accept degraded accuracy or maintain multiple extraction pipelines with inconsistent results.

## Architecture

The cascade strategy processes each PDF through a deterministic fallback chain:

```
PDF Input
    |
    v
[1] PyPDF2 Extract
    |
    +---> Quality Score >= Threshold? --> Return (PyPDF2)
    |
    +---> No
         |
         v
    [2] PyMuPDF Extract
         |
         +---> Quality Score >= Threshold? --> Return (PyMuPDF)
         |
         +---> No
              |
              v
         [3] Tesseract OCR
              |
              +---> Return (Tesseract)
```

Each cascade stage performs quality scoring before deciding whether to advance. Quality metrics include character validity, word count, encoding health, and garbage text detection. The system logs extraction success and method selection for observability.

## Key Technical Decisions

**Why three libraries?** PyPDF2 handles 70% of documents fastest. PyMuPDF catches 20% of the remaining cases. Tesseract (with image preprocessing) handles the remaining 10%, predominantly scanned documents. No two-library combination achieves equivalent accuracy without sacrificing speed on the common case.

**Why this order?** PyPDF2 is 10x faster than PyMuPDF and 100x faster than Tesseract on text-embedded PDFs. Fast-path-first design minimizes latency for the majority case. Only documents PyPDF2 struggles with fall through to more expensive methods.

**Quality thresholds?** The system tuning depends on document domain. For financial statements, 300+ words with <2% garbage characters indicates usable extraction. Thresholds are configurable per deployment; defaults are tuned to federal financial agency standards. In edge cases, lower thresholds accept partially garbled text over no text.

**What would I change?** In production, we'd add a pre-extraction PDF complexity classifier to predict which extraction method will succeed and skip low-probability methods. Tesseract performance could be improved with parallel page-level OCR, but orchestrating that complexity for marginal gains is often not justified. The current design prioritizes maintainability and observability over theoretical optimization.

## Results

Across 100,000+ real financial documents processed through the production system, the cascade achieved 99.2% extraction accuracy. The system extracted usable text from documents that individual libraries returned as blanks or noise. Processing time for the common case (PyPDF2 success) averaged 45ms per document; end-to-end time including Tesseract fallback averaged 1.2s per document.

## Tech Stack

- **PyPDF2** (3.0+): Fast text extraction from embedded-text PDFs
- **PyMuPDF** (1.23+): Handles complex layouts, mixed content, rotated pages
- **Tesseract OCR** (4.0+): Image-based PDF processing and scanned document recovery
- **Pillow** (10.0+): Image preprocessing for OCR pipeline
- **Python** (3.8+): Core implementation language

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gtykhon/document-extraction-failsafe.git
cd document-extraction-failsafe
pip install -r requirements.txt
```

Tesseract OCR requires system-level installation. On Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr
```

On macOS (Homebrew):

```bash
brew install tesseract
```

On Windows, download the installer from [UB Mannheim's Tesseract release page](https://github.com/UB-Mannheim/tesseract/wiki).

### Basic Usage

```python
from extractor import extract_text

result = extract_text("path/to/document.pdf")

print(f"Extracted text: {result.text[:200]}")
print(f"Method: {result.method_used}")
print(f"Confidence: {result.confidence_score:.2%}")
print(f"Time: {result.processing_time:.2f}s")
```

### Classifying PDF Complexity

```python
from detector import classify_pdf

pdf_type = classify_pdf("path/to/document.pdf")
print(f"PDF Type: {pdf_type}")  # PDFType.TEXT_EMBEDDED, IMAGE_BASED, MIXED, or COMPLEX_LAYOUT
```

### Scoring Text Quality

```python
from quality import score_text_quality

quality = score_text_quality(extracted_text)
print(f"Quality score: {quality.overall:.2%}")
print(f"Readability: {quality.readability:.2%}")
print(f"Completeness: {quality.completeness:.2%}")
```

### Running Benchmarks

Compare all three extraction methods on a directory of PDFs:

```bash
python benchmark.py --directory ./sample_pdfs --output results.csv
```

This generates a comparison table showing accuracy, speed, and failure rates for each method across all documents.

### Running Tests

```bash
pytest tests/ -v
```

## Project Context

This is a sanitized reference implementation. The production system was built for a federal financial agency through a government IT consulting firm between 2019 and 2023. Client-specific logic, domain validation rules, data pipelines, and configurations have been removed. The architecture patterns, cascade strategy, quality scoring methodology, and benchmark framework are authentic and production-tested. This repository demonstrates the technical approach without exposing client work.

## Author

Grygorii T.

- LinkedIn: [linkedin.com/in/grygorii-t](https://www.linkedin.com/in/grygorii-t/)
- GitHub: [github.com/gtykhon](https://github.com/gtykhon)
