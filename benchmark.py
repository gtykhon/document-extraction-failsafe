"""Benchmarking tool to compare extraction methods across PDF documents."""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import fitz
import pytesseract
from PyPDF2 import PdfReader

from extractor import ExtractionEngine
from models import ExtractionMethod
from quality import QualityScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExtractionBenchmark:
    """Benchmark extraction methods across a directory of PDFs."""

    def __init__(self, output_file: str = "benchmark_results.csv"):
        """Initialize benchmark tool.

        Args:
            output_file: CSV file to write results to.
        """
        self.output_file = output_file
        self.results: List[Dict] = []
        self.quality_scorer = QualityScorer()

    def benchmark_directory(self, directory: str, limit: int = None) -> None:
        """Run extraction benchmarks on all PDFs in directory.

        Args:
            directory: Path to directory containing PDF files.
            limit: Maximum number of PDFs to process (None = all).
        """
        pdf_dir = Path(directory)
        if not pdf_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return

        pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
        if limit:
            pdf_files = pdf_files[:limit]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        engine = ExtractionEngine()

        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Processing {pdf_file.name}")

            result = self._benchmark_single_pdf(pdf_file, engine)
            self.results.append(result)

        self._write_results()
        self._print_summary()

    def _benchmark_single_pdf(self, pdf_file: Path, engine: ExtractionEngine) -> Dict:
        """Benchmark all extraction methods on a single PDF.

        Args:
            pdf_file: Path to PDF file.
            engine: ExtractionEngine instance.

        Returns:
            Dictionary with benchmark results for this PDF.
        """
        result_dict = {
            "filename": pdf_file.name,
            "file_size_kb": pdf_file.stat().st_size / 1024,
        }

        # Get page count
        try:
            doc = fitz.open(str(pdf_file))
            result_dict["page_count"] = doc.page_count
            doc.close()
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            result_dict["page_count"] = None

        # Run cascade extraction
        start = time.time()
        cascade_result = engine.extract(str(pdf_file))
        cascade_time = time.time() - start

        result_dict["cascade_method"] = cascade_result.method_used.value
        result_dict["cascade_time_s"] = round(cascade_time, 3)
        result_dict["cascade_words"] = len(cascade_result.text.split()) if cascade_result.text else 0
        result_dict["cascade_quality"] = round(cascade_result.confidence_score, 3) if cascade_result.quality_score else 0
        result_dict["cascade_success"] = cascade_result.success

        # Benchmark individual methods
        for method_name, method_fn in [
            ("pypdf2", self._extract_pypdf2),
            ("pymupdf", self._extract_pymupdf),
            ("tesseract", self._extract_tesseract),
        ]:
            text, elapsed = method_fn(str(pdf_file))
            words = len(text.split()) if text else 0
            quality = self.quality_scorer.score(text)

            result_dict[f"{method_name}_time_s"] = round(elapsed, 3)
            result_dict[f"{method_name}_words"] = words
            result_dict[f"{method_name}_quality"] = round(quality.overall, 3)
            result_dict[f"{method_name}_success"] = quality.overall > 0.5

        return result_dict

    @staticmethod
    def _extract_pypdf2(pdf_path: str) -> tuple:
        """Extract text using PyPDF2 and measure time.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (text, elapsed_time_in_seconds).
        """
        try:
            start = time.time()
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text, time.time() - start
        except Exception as e:
            logger.warning(f"PyPDF2 failed on {pdf_path}: {e}")
            return "", 0.0

    @staticmethod
    def _extract_pymupdf(pdf_path: str) -> tuple:
        """Extract text using PyMuPDF and measure time.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (text, elapsed_time_in_seconds).
        """
        try:
            start = time.time()
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                text += doc[page_num].get_text()
            doc.close()
            return text, time.time() - start
        except Exception as e:
            logger.warning(f"PyMuPDF failed on {pdf_path}: {e}")
            return "", 0.0

    @staticmethod
    def _extract_tesseract(pdf_path: str) -> tuple:
        """Extract text using Tesseract OCR and measure time.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (text, elapsed_time_in_seconds).
        """
        try:
            import io
            from PIL import Image

            start = time.time()
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(image_bytes))
                page_text = pytesseract.image_to_string(image, config="--psm 6")
                full_text += page_text + "\n"

            doc.close()
            return full_text, time.time() - start

        except Exception as e:
            logger.warning(f"Tesseract failed on {pdf_path}: {e}")
            return "", 0.0

    def _write_results(self) -> None:
        """Write benchmark results to CSV file."""
        if not self.results:
            logger.warning("No results to write")
            return

        fieldnames = list(self.results[0].keys())
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        logger.info(f"Results written to {self.output_file}")

    def _print_summary(self) -> None:
        """Print summary statistics of benchmark results."""
        if not self.results:
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total PDFs processed: {len(self.results)}")

        # Cascade performance
        cascade_successes = sum(1 for r in self.results if r["cascade_success"])
        print(f"\nCascade Extraction Success Rate: {cascade_successes}/{len(self.results)} ({100*cascade_successes/len(self.results):.1f}%)")

        methods = ["pypdf2", "pymupdf", "tesseract"]
        print("\nMethod Comparison:")
        print(f"{'Method':<15} {'Success Rate':<15} {'Avg Time (s)':<15} {'Avg Words':<15} {'Avg Quality':<15}")
        print("-" * 75)

        for method in methods:
            success_key = f"{method}_success"
            time_key = f"{method}_time_s"
            words_key = f"{method}_words"
            quality_key = f"{method}_quality"

            successes = sum(1 for r in self.results if r.get(success_key, False))
            success_rate = 100 * successes / len(self.results) if self.results else 0
            avg_time = sum(r.get(time_key, 0) for r in self.results) / len(self.results) if self.results else 0
            avg_words = sum(r.get(words_key, 0) for r in self.results) / len(self.results) if self.results else 0
            avg_quality = sum(r.get(quality_key, 0) for r in self.results) / len(self.results) if self.results else 0

            print(f"{method:<15} {success_rate:>6.1f}%{'':<7} {avg_time:>6.3f}s{'':<7} {avg_words:>8.0f}{'':<5} {avg_quality:>6.3f}")

        print("=" * 80)


def main():
    """CLI entry point for benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark PDF extraction methods across a directory"
    )
    parser.add_argument(
        "--directory",
        "-d",
        required=True,
        help="Directory containing PDF files to benchmark",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_results.csv",
        help="Output CSV file for results (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of PDFs to process (default: all)",
    )

    args = parser.parse_args()

    benchmark = ExtractionBenchmark(output_file=args.output)
    benchmark.benchmark_directory(args.directory, limit=args.limit)


if __name__ == "__main__":
    main()
