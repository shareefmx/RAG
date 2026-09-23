"""RAG Evaluation Metrics Module.

Computes quantitative retrieval metrics (HitRate@K, MRR@K, Precision@K)
and generation quality metrics (Keyword Recall, Faithfulness).
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from app.evaluation.dataset import EvalSample
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import DocumentRetriever


logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Contains aggregate metrics for retrieval and generation."""

    total_queries: int
    hit_rate: float
    mrr: float
    avg_precision: float
    avg_faithfulness: float
    query_details: List[Dict[str, Any]]

    def summary_table(self) -> str:
        """Renders metrics as a formatted markdown table."""
        return (
            f"| Metric | Score |\n"
            f"| :--- | :--- |\n"
            f"| Total Evaluated Queries | {self.total_queries} |\n"
            f"| **Hit Rate@K** | **{self.hit_rate * 100:.1f}%** |\n"
            f"| **Mean Reciprocal Rank (MRR)** | **{self.mrr:.3f}** |\n"
            f"| **Average Precision@K** | **{self.avg_precision * 100:.1f}%** |\n"
            f"| **Keyword Faithfulness** | **{self.avg_faithfulness * 100:.1f}%** |\n"
        )


class RAGEvaluator:
    """Evaluates retrieval accuracy and answer quality against benchmark datasets."""

    def __init__(self, retriever: DocumentRetriever, pipeline: Optional[RAGPipeline] = None):
        self.retriever = retriever
        self.pipeline = pipeline

    def evaluate(self, samples: List[EvalSample], top_k: int = 5) -> EvaluationReport:
        """Runs evaluation over the provided samples."""
        hits = 0
        reciprocal_ranks = []
        precisions = []
        faithfulness_scores = []
        details = []

        for sample in samples:
            # 1. Evaluate Retrieval (use threshold 0.0 to inspect full top-K candidate ranking)
            retrieval_res = self.retriever.retrieve(sample.question, top_k=top_k, similarity_threshold=0.0)
            retrieved_chunks = retrieval_res.chunks

            # Find if target page was retrieved
            found_hit = False
            first_rank = 0

            for rank, search_res in enumerate(retrieved_chunks, start=1):
                c = search_res.chunk
                if c.page == sample.expected_page and c.filename == sample.expected_filename:
                    if not found_hit:
                        found_hit = True
                        first_rank = rank
                        break

            if found_hit:
                hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Precision@K: count relevant chunks / top_k
            rel_count = sum(
                1 for r in retrieved_chunks
                if r.chunk.page == sample.expected_page and r.chunk.filename == sample.expected_filename
            )
            precisions.append(rel_count / max(1, len(retrieved_chunks)))

            # 2. Evaluate Generation if pipeline is available
            faith_score = 0.0
            answer_text = ""
            if self.pipeline:
                try:
                    rag_res = self.pipeline.answer_question(
                        sample.question,
                        top_k=top_k,
                        similarity_threshold=0.10,
                    )
                    answer_text = rag_res.answer
                    # Compute key terms presence
                    matched_terms = sum(
                        1 for term in sample.key_terms
                        if term.lower() in answer_text.lower()
                    )
                    faith_score = matched_terms / max(1, len(sample.key_terms))
                except Exception as e:
                    logger.warning("Generation eval failed for '%s': %s", sample.question, e)

            faithfulness_scores.append(faith_score)

            details.append({
                "question": sample.question,
                "hit": found_hit,
                "first_rank": first_rank if found_hit else None,
                "precision": round(precisions[-1], 3),
                "faithfulness": round(faith_score, 3),
                "answer": answer_text[:100] + "..." if len(answer_text) > 100 else answer_text,
            })

        n = len(samples)
        return EvaluationReport(
            total_queries=n,
            hit_rate=hits / n if n else 0.0,
            mrr=sum(reciprocal_ranks) / n if n else 0.0,
            avg_precision=sum(precisions) / n if n else 0.0,
            avg_faithfulness=sum(faithfulness_scores) / n if n else 0.0,
            query_details=details,
        )
