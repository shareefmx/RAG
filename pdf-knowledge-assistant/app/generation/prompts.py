"""Prompt Engineering and System Templates for Grounded RAG.

Enforces strict anti-hallucination guardrails, factuality constraints,
and source attribution standards.
"""

from typing import List
from app.vectorstore.base import SearchResult


SYSTEM_RAG_PROMPT = """You are an expert, precise, document-grounded knowledge assistant.

Your task is to provide clear, helpful, well-structured, and accurate answers to the user's questions based EXCLUSIVELY on the retrieved document excerpts in the CONTEXT section.

CRITICAL RULES:
1. STRICT DOCUMENT GROUNDING: Answer factually using only the facts directly mentioned in the CONTEXT. Do NOT invent, assume, extrapolate, or introduce outside information.
2. SOURCE CITATIONS: Back up facts by referencing the source document and page number in your response using the format: [Document: <filename>, Page: <page_number>].
3. STRUCTURED & DETAILED REPLIES: Present answers in an organized, easy-to-read manner. Use bullet points, bold key terms, and clear sections to highlight important details (e.g. degrees, dates, tools, responsibilities, statistics).
4. PARTIAL CONTEXT HANDLING: If the context contains partial information relevant to the question, present everything that is found in the documents accurately. If certain specific sub-details asked by the user are not mentioned, clearly and concisely point out what was not found without refusing to answer what IS found.
5. OUT-OF-DOMAIN REFUSAL: ONLY if the context contains NO relevant information whatsoever to address the question, respond with:
"I couldn't find enough relevant information in the uploaded documents to answer this question."
6. NO SPECULATION: Never guess names, numbers, dates, or technical claims not present in the CONTEXT.
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

