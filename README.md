# 🧠 RAG Learning Lab

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/shareefmx/pdf-knowledge-assistant)
[![Live Web App](https://img.shields.io/badge/Web%20App-Direct%20Interface-success?style=for-the-badge)](https://shareefmx-pdf-knowledge-assistant.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/shareefmx/RAG)

A structured collection of Retrieval-Augmented Generation (RAG) projects, experiments, implementations, and learning notes.

This repository documents the journey from understanding the foundational mechanics of RAG (parsing, chunking, embeddings, hybrid vector/lexical indexing) to building production-ready, agentic, and multi-modal knowledge systems.

---

## 🌐 Live Hugging Face Deployment

The **PDF Knowledge Assistant** is deployed and running live on Hugging Face Spaces with **ZeroGPU (Nvidia A10G)** hardware acceleration:

- **Hugging Face Space:** [https://huggingface.co/spaces/shareefmx/pdf-knowledge-assistant](https://huggingface.co/spaces/shareefmx/pdf-knowledge-assistant)
- **Direct Fullscreen Interface:** [https://shareefmx-pdf-knowledge-assistant.hf.space/](https://shareefmx-pdf-knowledge-assistant.hf.space/)

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
Cleaning & Normalization (Token Boundary Fixes)
   ↓
Recursive Chunking & Metadata Enrichment
   ↓
Embeddings (Sentence Transformers) + BM25 Lexical Index
   ↓
Vector Database (FAISS, Chroma, Managed Stores)
   ↓
Hybrid Search & Reciprocal Rank Fusion (RRF)
   ↓
Similarity Search & Out-of-Domain Filtering
   ↓
Cross-Encoder Reranking
   ↓
Context Construction
   ↓
Prompt Engineering & Anti-Hallucination Guardrails
   ↓
LLM Generation (Gemini Multi-Model Fallback, OpenRouter)
   ↓
Answer + Verified Source Citations (Document & Page)
   ↓
Production Deployment (FastAPI, Gradio, Hugging Face Spaces, Docker)
```

---

## 📚 Projects

| Project | Description | Live Demo | Status |
|---------|-------------|-----------|--------|
| [**PDF Knowledge Assistant**](./pdf-knowledge-assistant) | End-to-end PDF RAG with PyMuPDF, sentence-transformers, Okapi BM25 hybrid search, FAISS/Chroma, Gemini multi-model fallback, source citations down to exact page numbers, FastAPI, and Gradio UI. | [**Hugging Face Space**](https://huggingface.co/spaces/shareefmx/pdf-knowledge-assistant) | **🟢 Active / Live on Hugging Face Spaces** |

---

## 🛠️ Technology Stack

- **Languages & Frameworks:** Python 3.11+, FastAPI, Pydantic v2, Uvicorn
- **UI:** Gradio 6.x
- **Document Processing:** PyMuPDF (`fitz`), layout preservation, regex token-boundary cleaning
- **Vector Search & Storage:** FAISS (`faiss-cpu`), ChromaDB
- **Retrieval Engine:** Okapi BM25 + Dense Sentence Transformers (`all-MiniLM-L6-v2`) with Reciprocal Rank Fusion (RRF)
- **Reranking:** Cross-Encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM Providers:** Google Gemini API (`google-genai` with `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-2.5-flash` fallback chain), OpenRouter (OpenAI-compatible)
- **Evaluation & Testing:** Pytest, Precision@K / Recall@K / Hit Rate evaluators
- **Deployment:** Hugging Face Spaces (ZeroGPU `zero-a10g`), GitHub

---

## 📖 Key RAG Concepts Covered

- **Document Ingestion:** Converting raw multi-format files into searchable, structured text units.
- **Chunking Strategy:** Balancing semantic coherence and context size using recursive character boundaries and sliding overlap.
- **Embeddings:** Vector embeddings map semantic meaning into dense high-dimensional geometry where cosine similarity corresponds to conceptual closeness.
- **Hybrid Retrieval & RRF:** Combining dense vector semantics with sparse BM25 lexical keyword matching via Reciprocal Rank Fusion to maximize both recall and precision.
- **Vector Indexing:** Exact and approximate nearest neighbor search (L2 / Cosine Inner Product).
- **Reranking:** Two-stage retrieval where high-recall vector search provides candidates and a cross-encoder computes full attention between query and candidate text.
- **Grounded Generation:** Enforcing strict prompt constraints where the model must synthesize answers solely from retrieved evidence, citing exact source pages.
- **Hallucination Prevention:** Similarity score thresholds and explicit fallback responses when evidence is insufficient or out of domain.

---

## 🚀 Getting Started

### 1. Try Online (Zero Setup)
Launch the application directly in your browser:
👉 **[Open on Hugging Face Spaces](https://huggingface.co/spaces/shareefmx/pdf-knowledge-assistant)**

### 2. Run Locally

```bash
cd pdf-knowledge-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set your GEMINI_API_KEY in .env
python app.py
```

Open your browser to `http://localhost:7860` to access the Gradio interface.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
