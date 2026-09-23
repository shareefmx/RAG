"""Unit tests for DocumentRetriever."""

import numpy as np
import pytest

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import DocumentChunk
from app.retrieval.retriever import DocumentRetriever
from app.vectorstore.faiss_store import FAISSVectorStore


@pytest.fixture(scope="module")
def shared_embedder():
    return EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def populated_retriever(shared_embedder):
    store = FAISSVectorStore(dimension=shared_embedder.dimension)

    chunks = [
        DocumentChunk(
            chunk_id="chunk_rag_01",
            document_id="doc_rag",
            filename="rag_paper.pdf",
            page=1,
            section="Introduction",
            chunk_index=1,
            text="Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations by supplying factual document chunks.",
        ),
        DocumentChunk(
            chunk_id="chunk_rag_02",
            document_id="doc_rag",
            filename="rag_paper.pdf",
            page=4,
            section="Evaluation",
            chunk_index=2,
            text="The experimental results showed a 92% answer faithfulness score on the test set.",
        ),
    ]

    embeddings = shared_embedder.embed_documents([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)

    return DocumentRetriever(
        vector_store=store,
        embedding_service=shared_embedder,
        similarity_threshold=0.35,
        default_top_k=2,
    )


def test_retriever_relevant_query(populated_retriever):
    result = populated_retriever.retrieve("How does RAG reduce hallucinations?")

    assert result.has_sufficient_context is True
    assert len(result.chunks) >= 1
    assert result.chunks[0].chunk.chunk_id == "chunk_rag_01"
    assert result.best_score > 0.40


def test_retriever_threshold_filtering(populated_retriever):
    # Unrelated query about making pizza
    result = populated_retriever.retrieve(
        "What is the best dough temperature for Neapolitan pizza?",
        similarity_threshold=0.45,
    )

    # Should not qualify any chunks under strict threshold
    assert result.has_sufficient_context is False
    assert len(result.chunks) == 0


def test_retriever_empty_query(populated_retriever):
    result = populated_retriever.retrieve("   ")
    assert result.has_sufficient_context is False
    assert len(result.chunks) == 0


def test_bm25_lexical_search():
    from app.retrieval.bm25 import BM25Index

    index = BM25Index()
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            filename="doc1.pdf",
            page=1,
            section="Sec1",
            chunk_index=1,
            text="Muhammed Shareef is an AI/ML Engineer with expertise in React and Python.",
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="doc1",
            filename="doc1.pdf",
            page=2,
            section="Sec2",
            chunk_index=2,
            text="He completed his education at New Horizon College of Engineering.",
        ),
    ]
    index.index_chunks(chunks)

    # Keyword match for education / college
    matches = index.search("New Horizon College education", k=2)
    assert len(matches) >= 1
    assert matches[0].chunk.chunk_id == "c2"

    # Stopwords should not cause false positive matches
    unrelated = index.search("what is the dough pizza", k=2)
    assert len(unrelated) == 0


