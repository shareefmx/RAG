"""RAG Evaluation Benchmark Dataset."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalSample:
    """A ground-truth question and answer pair for RAG evaluation."""

    question: str
    expected_answer: str
    expected_page: int
    expected_filename: str
    key_terms: List[str]


# Standard benchmark test cases against the sample research paper
BENCHMARK_DATASET: List[EvalSample] = [
    EvalSample(
        question="What methodology was used in the research?",
        expected_answer="The research evaluated FAISS vector indices with sentence-transformer embeddings.",
        expected_page=2,
        expected_filename="sample_research.pdf",
        key_terms=["FAISS", "sentence-transformer", "methodology", "embeddings"],
    ),
    EvalSample(
        question="What was the hit rate achieved in the evaluation?",
        expected_answer="Evaluation demonstrated a 94% hit rate on domain queries.",
        expected_page=2,
        expected_filename="sample_research.pdf",
        key_terms=["94%", "hit rate", "domain queries"],
    ),
    EvalSample(
        question="What does RAG combine according to the paper?",
        expected_answer="RAG combines parametric knowledge in neural models with non-parametric external retrieval.",
        expected_page=1,
        expected_filename="sample_research.pdf",
        key_terms=["parametric", "non-parametric", "external retrieval"],
    ),
    EvalSample(
        question="What is the benefit of external retrieval in RAG?",
        expected_answer="It prevents hallucinations and provides direct citations.",
        expected_page=1,
        expected_filename="sample_research.pdf",
        key_terms=["hallucinations", "citations"],
    ),
]

