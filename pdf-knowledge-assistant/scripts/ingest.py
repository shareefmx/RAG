"""CLI Document Ingestion Utility.

Usage:
    python scripts/ingest.py path/to/document1.pdf path/to/document2.pdf
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import get_settings, setup_logging
from app.state import get_app_state


def main():
    parser = argparse.ArgumentParser(description="Ingest and index PDF documents into the vector store.")
    parser.add_argument("pdf_files", nargs="+", type=Path, help="Paths to PDF documents to index.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("ingest_cli")

    state = get_app_state()

    logger.info("Starting ingestion for %d documents...", len(args.pdf_files))
    success_count = 0

    for file_path in args.pdf_files:
        if not file_path.exists():
            logger.error("File does not exist: %s", file_path)
            continue
        try:
            res = state.index_pdf_file(file_path)
            logger.info(
                "Successfully indexed '%s': %d pages, %d chunks. Total store chunks: %d",
                res["filename"],
                res["pages_extracted"],
                res["chunks_created"],
                res["total_indexed_chunks"],
            )
            success_count += 1
        except Exception as e:
            logger.error("Failed to index '%s': %s", file_path.name, e)

    logger.info("Ingestion finished. %d/%d documents successfully indexed.", success_count, len(args.pdf_files))


if __name__ == "__main__":
    main()
