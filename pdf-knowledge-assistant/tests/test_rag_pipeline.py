"""Unit tests for the RAGPipeline."""

import pytest

from app.generation.llm import MockLLMService
from app.ingestion.chunker import DocumentChunk
from app.rag.pipeline import RAGPipeline, SourceCitation
from app.retrieval.retriever import DocumentRetriever
from app.vectorstore.base import SearchResult
from app.vectorstore.faiss_store import FAISSVectorStore


class MockRetriever:
    """Mock retriever returning canned responses for pipeline unit testing."""

    def __init__(self, should_find: bool = True):
        self.should_find = should_find
        self.default_top_k = 5

    def retrieve(self, query: str, top_k=None, similarity_threshold=None, filter_dict=None):
        from app.retrieval.retriever import RetrievalResult

        if not self.should_find:
            return RetrievalResult(query=query, has_sufficient_context=False)

        chunk = DocumentChunk(
            chunk_id="chunk_test_001",
            document_id="doc_alpha",
            filename="climate_study.pdf",
            page=7,
            section="Methodology",
            chunk_index=1,
            text="The research measured atmospheric carbon dioxide levels using LIDAR sensors.",
        )
        sr = SearchResult(chunk=chunk, score=0.885)
        return RetrievalResult(
            query=query,
            chunks=[sr],
            has_sufficient_context=True,
            best_score=0.885,
            all_candidates=[sr],
        )


def test_rag_pipeline_success():
    retriever = MockRetriever(should_find=True)
    llm = MockLLMService(canned_response="LIDAR sensors were utilized to record atmospheric carbon dioxide.")
    pipeline = RAGPipeline(retriever=retriever, llm_service=llm)

    response = pipeline.answer_question("What equipment was used to measure CO2?")

    assert response.has_sufficient_context is True
    assert "LIDAR sensors" in response.answer
    assert len(response.sources) == 1

    citation = response.sources[0]
    assert citation.filename == "climate_study.pdf"
    assert citation.page == 7
    assert citation.chunk_id == "chunk_test_001"
    assert citation.score == 0.885
    assert "LIDAR sensors" in citation.snippet


def test_rag_pipeline_insufficient_context():
    retriever = MockRetriever(should_find=False)
    llm = MockLLMService()
    pipeline = RAGPipeline(retriever=retriever, llm_service=llm)

    response = pipeline.answer_question("What is the capital of Mars?")

    assert response.has_sufficient_context is False
    assert response.answer == RAGPipeline.INSUFFICIENT_CONTEXT_MESSAGE
    assert len(response.sources) == 0

