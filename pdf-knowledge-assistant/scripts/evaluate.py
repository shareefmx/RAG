"""RAG Evaluation Benchmark Runner Script.

Runs precision, recall, hit-rate, MRR, and faithfulness evaluations over
benchmark queries using an isolated in-memory index to prevent polluting
the main application vector store.

Usage:
    python scripts/evaluate.py
"""

import json
import logging
from pathlib import Path
import sys
import tempfile

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pymupdf as fitz

from app.config import get_settings, setup_logging
from app.embeddings.embedding_service import get_embedding_service
from app.evaluation.dataset import BENCHMARK_DATASET
from app.evaluation.evaluator import RAGEvaluator
from app.generation.llm import LLMServiceFactory
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.pdf_loader import PDFLoader
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import DocumentRetriever
from app.vectorstore.faiss_store import FAISSVectorStore


def create_benchmark_pdf(target_path: Path) -> Path:
    """Creates the standard benchmark PDF corresponding to BENCHMARK_DATASET."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Introduction to Retrieval-Augmented Generation\n\n"
        "RAG combines parametric knowledge in neural models with non-parametric external retrieval.\n"
        "The primary benefit of external retrieval in RAG is that it prevents hallucinations "
        "and provides direct citations for human verification.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "Methodology and Experimental Setup\n\n"
        "The research evaluated FAISS vector indices with sentence-transformer embeddings.\n"
        "Evaluation demonstrated a 94% hit rate on domain queries when using recursive character chunking.",
    )

    doc.save(str(target_path))
    doc.close()
    return target_path


def main():
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("eval_runner")

    logger.info("Initializing isolated RAG evaluation benchmark...")

    # Initialize isolated components to protect the active user's vector store
    embedder = get_embedding_service(settings.embedding_model)
    eval_store = FAISSVectorStore(dimension=embedder.dimension)
    loader = PDFLoader()
    chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    with tempfile.TemporaryDirectory() as temp_dir:
        bench_pdf_path = Path(temp_dir) / "sample_research.pdf"
        create_benchmark_pdf(bench_pdf_path)

        # Ingest benchmark PDF into the isolated evaluation store
        pages = loader.load(bench_pdf_path)
        chunks = chunker.chunk_documents(pages)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        eval_store.add_chunks(chunks, embeddings)

        eval_retriever = DocumentRetriever(
            vector_store=eval_store,
            embedding_service=embedder,
            similarity_threshold=settings.similarity_threshold,
            default_top_k=settings.top_k,
        )

        llm_service = LLMServiceFactory.create(
            provider=settings.llm_provider,
            gemini_api_key=settings.gemini_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            model_name=settings.llm_model,
        )

        eval_pipeline = RAGPipeline(
            retriever=eval_retriever,
            llm_service=llm_service,
        )

        # Run evaluator
        evaluator = RAGEvaluator(retriever=eval_retriever, pipeline=eval_pipeline)
        report = evaluator.evaluate(BENCHMARK_DATASET, top_k=5)

    # Print markdown report
    print("\n" + "=" * 60)
    print("           📊 RAG BENCHMARK EVALUATION RESULTS")
    print("=" * 60 + "\n")
    print(report.summary_table())
    print("\nQuery Breakdown:")
    for d in report.query_details:
        status = "✅ HIT" if d["hit"] else "❌ MISS"
        rank_str = f"Rank: #{d['first_rank']}" if d["first_rank"] else "Not in Top-K"
        print(f"- [{status}] '{d['question']}' ({rank_str}, Prec: {d['precision']}, Faith: {d['faithfulness']})")

    # Save results to data/processed/eval_results.json
    output_file = settings.get_processed_path() / "eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_queries": report.total_queries,
                "hit_rate": report.hit_rate,
                "mrr": report.mrr,
                "avg_precision": report.avg_precision,
                "avg_faithfulness": report.avg_faithfulness,
                "details": report.query_details,
            },
            f,
            indent=2,
        )
    logger.info("Saved evaluation report to '%s'", output_file)


if __name__ == "__main__":
    main()
