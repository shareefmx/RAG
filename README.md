# 🧠 RAG Learning Lab

A structured collection of Retrieval-Augmented Generation (RAG) projects, experiments, implementations, and learning notes.

This repository documents the journey from understanding the foundational mechanics of RAG (parsing, chunking, embeddings, vector indexing) to building production-ready, agentic, and multi-modal knowledge systems.

---

## 🎯 Core Goal

The goal of this repository is to master and implement the complete RAG ecosystem:

```
Documents
   ↓
Document Loading (PyMuPDF, Parsers)
   ↓
Text Extraction & Structure Detection
   ↓
Cleaning & Normalization
   ↓
Recursive Chunking & Metadata Enrichment
   ↓
Embeddings (Sentence Transformers)
   ↓
Vector Database (FAISS, Chroma, Managed Stores)
   ↓
Similarity Search & Retrieval Filtering
   ↓
Cross-Encoder Reranking
   ↓
Context Construction
   ↓
Prompt Engineering & Guardrails
   ↓
LLM Generation (Gemini, OpenRouter)
   ↓
Answer + Source Citations
   ↓
Retrieval & Generation Evaluation (Precision@K, Hit Rate, Faithfulness)
   ↓
Production Deployment (FastAPI, Gradio, Hugging Face Spaces, Docker)
```

---

## 📚 Projects Roadmap

| # | Project | Description | Status |
|---|---------|-------------|--------|
| **01** | [**PDF Knowledge Assistant**](./pdf-knowledge-assistant) | End-to-end PDF RAG with PyMuPDF, sentence-transformers, FAISS/Chroma, reranking, source citations, FastAPI, and Gradio UI. | **Active / Production-Ready** |
| **02** | **Resume + Job Description RAG** | Parsing resumes & JDs, skill-gap semantic matching, candidate scoring & interview question synthesis. | Planned |
| **03** | **Multi-Document RAG** | Cross-document reasoning, multi-PDF synthesis, document-level metadata filtering & conflict resolution. | Planned |
| **04** | **Hybrid Search RAG** | Combining semantic dense retrieval with sparse lexical BM25 search using Reciprocal Rank Fusion (RRF). | Planned |
| **05** | **Reranking RAG** | In-depth benchmarks of Cross-Encoders, FlashRank, and Cohere Rerank on diverse domain corpora. | Planned |
| **06** | **MongoDB RAG** | Building scalable retrieval with MongoDB Atlas Vector Search and metadata-rich document storage. | Planned |
| **07** | **Agentic RAG** | Autonomous query routing, multi-step tool-use, iterative retrieval, self-correction, and planning. | Planned |
| **08** | **Multimodal RAG** | Extracting and searching across text, charts, diagrams, tables, and images using Vision-Language Models. | Planned |
| **09** | **Graph RAG** | Extracting entities and relations into knowledge graphs, community summarization, and hybrid graph traversal. | Planned |
| **10** | **Production RAG** | High-throughput caching, semantic cache, observability (OpenTelemetry), guardrails, rate limiting, and Kubernetes/Docker scaling. | Planned |

---

## 🛠️ Technology Stack

- **Languages & Frameworks:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn
- **UI:** Gradio, Streamlit
- **Document Processing:** PyMuPDF (`fitz`), pdfplumber
- **Vector Search & Storage:** FAISS (`faiss-cpu`), ChromaDB
- **Embeddings & Reranking:** Hugging Face `sentence-transformers`, Cross-Encoders
- **LLM Providers:** Google Gemini API (`google-genai`), OpenRouter (OpenAI-compatible)
- **Evaluation & Testing:** Pytest, Custom Precision@K / Recall@K / Hit Rate evaluators
- **Deployment:** Docker, Hugging Face Spaces

---

## 📖 Key RAG Concepts Covered

- **Document Ingestion:** Converting raw multi-format files into searchable, structured text units.
- **Chunking Strategy:** Balancing semantic coherence and context size using recursive character boundaries and sliding overlap.
- **Embeddings:** Vector embeddings map semantic meaning into dense high-dimensional geometry where cosine similarity corresponds to conceptual closeness.
- **Vector Indexing:** Exact and approximate nearest neighbor search (L2 / Cosine Inner Product).
- **Reranking:** Two-stage retrieval where high-recall vector search provides candidates and a cross-encoder computes full attention between query and candidate text.
- **Grounded Generation:** Enforcing strict prompt constraints where the model must synthesize answers solely from retrieved evidence, citing exact source pages.
- **Hallucination Prevention:** Similarity score thresholds and explicit fallback responses when evidence is insufficient.

---

## 🚀 Getting Started

To explore Project 1 (PDF Knowledge Assistant):

```bash
cd pdf-knowledge-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY or OPENROUTER_API_KEY in .env
python app.py
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
