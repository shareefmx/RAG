"""Prompt Engineering and System Templates for Grounded RAG.

Enforces strict anti-hallucination guardrails, factuality constraints,
and source attribution standards.
"""

from typing import List
from app.vectorstore.base import SearchResult


SYSTEM_RAG_PROMPT = """You are a precise, document-grounded question answering assistant.

Your task is to answer the user's question using ONLY the retrieved document excerpts provided in the CONTEXT section below.

CRITICAL RULES:
1. STRICT GROUNDING: Rely exclusively on the facts directly stated in the CONTEXT. Do NOT assume, extrapolate, or bring in outside knowledge.
2. NO HALLUCINATION: If the context does not contain sufficient facts to answer the question completely, clearly state: "I couldn't find enough relevant information in the uploaded documents to answer this question."
3. SOURCE CITATIONS: Whenever stating a fact, explicitly reference the source document and page number in your response using the format: [Document: <filename>, Page: <page_number>].
4. CONCISE & OBJECTIVE: Maintain a professional, objective tone. Provide clear, structured explanations without unnecessary fluff.
5. NEVER SPECULATE: Never guess or invent author names, numerical figures, dates, or technical claims not present in the CONTEXT.
"""


def build_context_string(chunks: List[SearchResult]) -> str:
    """Formats retrieved SearchResult chunks into a structured, numbered context block.

    Args:
        chunks: List of SearchResult objects passing similarity threshold.

    Returns:
        Structured context string with clear source and page demarcations.
    """
    if not chunks:
        return "NO RELEVANT CONTEXT FOUND."

    context_parts: List[str] = []
    for i, res in enumerate(chunks, start=1):
        c = res.chunk
        header = (
            f"[SOURCE {i}]\n"
            f"Document: {c.filename}\n"
            f"Page: {c.page}\n"
            f"Section: {c.section}\n"
            f"Chunk ID: {c.chunk_id}\n"
            f"Similarity Score: {res.score:.4f}\n"
            f"Content:\n{c.text}\n"
        )
        context_parts.append(header)

    return "\n" + "=" * 40 + "\n\n".join(context_parts) + "\n" + "=" * 40


def build_rag_prompt(question: str, context: str) -> str:
    """Constructs the complete generation prompt incorporating retrieved context and the user query."""
    return f"""CONTEXT INFORMATION:
{context}

USER QUESTION:
{question}

Please answer the question based strictly on the CONTEXT above:"""


QUERY_REWRITE_SYSTEM_PROMPT = """You are an expert query reformulator.
Given a conversation history between a User and an Assistant, and a follow-up User question,
rewrite the follow-up question into a standalone search query that can be understood without the conversation history.

RULES:
1. Replace all ambiguous pronouns (it, they, that, this, the model, the paper) with the specific entities mentioned previously.
2. Do NOT answer the question.
3. Output ONLY the rewritten standalone question and nothing else.
4. If the question is already fully self-contained, return it unchanged.
"""


def build_query_rewrite_prompt(history_str: str, current_question: str) -> str:
    """Constructs the prompt for conversational query reformulation."""
    return f"""CONVERSATION HISTORY:
{history_str}

LATEST QUESTION:
{current_question}

STANDALONE SEARCH QUERY:"""

