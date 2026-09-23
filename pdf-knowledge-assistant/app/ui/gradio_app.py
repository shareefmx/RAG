"""Gradio Web Interface for PDF Knowledge Assistant.

Provides an interactive user interface for:
- Uploading and indexing single or multiple PDF documents
- Real-time status reporting (document counts, page counts, chunk counts)
- Context-aware question answering with conversational query rewriting
- Verified source citations with expandable text snippets and similarity scores
- Document deletion and management
- Configurable retrieval parameters (Top-K, similarity threshold, reranking)
"""

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from app.state import get_app_state

try:
    import spaces
except Exception:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if callable(fn):
                return fn
            def wrapper(f):
                return f
            return wrapper
    spaces = _MockSpaces()


logger = logging.getLogger(__name__)


@spaces.GPU(duration=10)
def _gpu_worker_init() -> bool:
    """Registered @spaces.GPU function to satisfy Hugging Face ZeroGPU startup validator."""
    return True


def format_citations_html(citations) -> str:
    """Renders source citations as formatted HTML cards in a sleek, high-contrast dark black-and-white theme."""
    if not citations:
        return """
        <div style="background-color: #0d0d0e; border: 1px solid #27272a; border-radius: 8px; padding: 14px; color: #a1a1aa; font-size: 13px;">
            No sources cited for this query.
        </div>
        """

    html_blocks = []
    for i, c in enumerate(citations, 1):
        score_percent = int(c.score * 100) if 0.0 <= c.score <= 1.0 else f"{c.score:.2f}"
        snippet_escaped = html.escape(c.snippet)

        block = f"""
        <div style="border: 1px solid #27272a; border-radius: 8px; padding: 14px; margin-bottom: 12px; background-color: #0d0d0e; color: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #27272a; padding-bottom: 8px;">
                <span style="font-weight: 700; font-size: 14px; color: #ffffff;">📄 {html.escape(c.filename)} &mdash; Page {c.page}</span>
                <span style="background-color: #000000; border: 1px solid #ffffff; color: #ffffff; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; font-family: monospace;">
                    Match: {score_percent}%
                </span>
            </div>
            <div style="font-size: 12px; color: #a1a1aa; margin-bottom: 8px;">
                <strong style="color: #d4d4d8;">Section:</strong> <span style="color: #ffffff;">{html.escape(c.section)}</span> &nbsp;|&nbsp; <strong style="color: #d4d4d8;">Chunk ID:</strong> <code style="background-color: #1f1f23; border: 1px solid #3f3f46; color: #ffffff; padding: 1px 6px; border-radius: 4px; font-size: 11px;">{html.escape(c.chunk_id)}</code>
            </div>
            <details style="margin-top: 8px;">
                <summary style="cursor: pointer; color: #ffffff; font-size: 12px; font-weight: 600; text-decoration: underline; outline: none;">View Matching Excerpt</summary>
                <div style="background-color: #000000; border-left: 3px solid #ffffff; border: 1px solid #27272a; padding: 10px; margin-top: 8px; font-size: 12px; color: #f4f4f5; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; line-height: 1.5; border-radius: 4px;">{snippet_escaped}</div>
            </details>
        </div>
        """
        html_blocks.append(block)

    return "".join(html_blocks)


def get_indexed_docs_markdown() -> str:
    """Generates a summary markdown list of all currently indexed documents."""
    state = get_app_state()
    state.reload_vector_store_if_needed()
    docs = state.vector_store.list_documents()
    if not docs:
        return "ℹ️ *No documents indexed yet. Upload one or more PDFs above to get started.*"

    total_chunks = state.vector_store.total_chunks()
    lines = [f"**Currently Indexed Documents** ({len(docs)} documents, {total_chunks} total chunks):\n"]
    for d in docs:
        sections_str = ", ".join(d['sections'][:3]) + ("..." if len(d['sections']) > 3 else "")
        lines.append(
            f"- **{d['filename']}** (`{d['document_id']}`): {d['page_count']} pages, {d['chunk_count']} chunks. *Sections: {sections_str or 'General'}*"
        )
    return "\n".join(lines)


def get_doc_dropdown_choices() -> List[Tuple[str, str]]:
    """Returns dropdown choices formatted as (Display Name, document_id)."""
    state = get_app_state()
    state.reload_vector_store_if_needed()
    docs = state.vector_store.list_documents()
    choices = [("All Documents (Global Search)", "ALL")]
    for d in docs:
        choices.append((f"{d['filename']} ({d['document_id']})", d["document_id"]))
    return choices


def index_files(files) -> Tuple[str, str, Any]:
    """Handles multi-file PDF upload and ingestion."""
    if not files:
        return "⚠️ Please select at least one PDF file.", get_indexed_docs_markdown(), gr.update()

    state = get_app_state()
    results = []

    for file_obj in files:
        file_path = Path(file_obj.name)
        try:
            stat = state.index_pdf_file(file_path)
            results.append(f"✅ **{stat['filename']}**: {stat['pages_extracted']} pages, {stat['chunks_created']} chunks indexed.")
        except Exception as e:
            results.append(f"❌ **{file_path.name}**: Indexing error ({e})")

    status_msg = "\n".join(results)
    doc_markdown = get_indexed_docs_markdown()
    new_choices = get_doc_dropdown_choices()

    return status_msg, doc_markdown, gr.update(choices=new_choices, value="ALL")


def delete_selected_document(doc_id: str) -> Tuple[str, str, Any]:
    """Deletes a selected document from the vector store."""
    if not doc_id or doc_id == "ALL":
        return "⚠️ Please select a specific document to delete.", get_indexed_docs_markdown(), gr.update()

    state = get_app_state()
    deleted_count = state.vector_store.delete_document(doc_id)
    state.vector_store.save(state.settings.get_vectorstore_path())

    status_msg = f"🗑️ Deleted document `{doc_id}` ({deleted_count} chunks removed)."
    doc_markdown = get_indexed_docs_markdown()
    new_choices = get_doc_dropdown_choices()

    return status_msg, doc_markdown, gr.update(choices=new_choices, value="ALL")


def answer_query(
    question: str,
    doc_filter: str,
    top_k: int,
    threshold: float,
    use_reranker: bool,
    chat_history: Optional[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """Processes user query through RAG pipeline, updates chat and sources cards."""
    history = list(chat_history) if chat_history else []
    if not question.strip():
        return history, "", "", ""

    state = get_app_state()
    state.reload_vector_store_if_needed()
    selected_doc_id = None if (not doc_filter or doc_filter == "ALL") else doc_filter

    response = state.pipeline.answer_question(
        question=question,
        conversation_history=history,
        document_id_filter=selected_doc_id,
        top_k=int(top_k),
        similarity_threshold=float(threshold),
    )

    # Append new turn in Gradio 6 messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    new_history = list(history)
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": response.answer})

    sources_html = format_citations_html(response.sources)
    rewritten_info = f"🔄 *Standalone Query:* {response.rewritten_question}" if response.rewritten_question else ""

    return new_history, "", sources_html, rewritten_info


def create_ui() -> gr.Blocks:
    """Builds and returns the full Gradio Blocks application."""
    state = get_app_state()

    with gr.Blocks(title="📚 PDF Knowledge Assistant") as demo:
        gr.Markdown(
            """
            # 📚 PDF Knowledge Assistant — Production RAG
            Ask factual questions over your PDF documents. Answers are strictly grounded in retrieved evidence with exact page citations.
            """
        )

        with gr.Row():
            # Left Column: Upload, Document Management, Settings
            with gr.Column(scale=4):
                gr.Markdown("### 1. Document Management")
                file_input = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                index_btn = gr.Button("📥 Index Documents", variant="primary")
                index_status = gr.Markdown(value="*Upload PDF files to start indexing.*")

                docs_summary = gr.Markdown(value=get_indexed_docs_markdown())

                with gr.Accordion("⚙️ Document Filter & Deletion", open=False):
                    doc_dropdown = gr.Dropdown(
                        label="Filter Query to Document",
                        choices=get_doc_dropdown_choices(),
                        value="ALL",
                    )
                    delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")
                    delete_status = gr.Markdown()

                with gr.Accordion("🛠️ Advanced Retrieval Parameters", open=False):
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=state.settings.top_k,
                        step=1,
                        label="Top-K Retrieved Chunks",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=state.settings.similarity_threshold,
                        step=0.01,
                        label="Relevance Sensitivity Threshold (Out-of-Domain Filter)",
                        info="Default 0.05 prevents false refusals. Higher values strictly reject low-confidence chunks.",
                    )
                    reranker_checkbox = gr.Checkbox(
                        label="Enable Cross-Encoder Reranking",
                        value=state.settings.use_reranker,
                    )

            # Right Column: Chat QA, Verified Citations, Sample Questions
            with gr.Column(scale=6):
                gr.Markdown("### 2. Conversational Q&A")
                chatbot = gr.Chatbot(
                    label="Knowledge Assistant",
                    height=420,
                )

                query_info = gr.Markdown(value="")

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a question about your documents",
                        placeholder="e.g., What methodology was used in the research?",
                        scale=9,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

                clear_btn = gr.Button("🧹 Clear Conversation", size="sm")

                gr.Markdown("### 3. Source Citations & Evidence")
                sources_output = gr.HTML(
                    value="""
                    <div style="background-color: #0d0d0e; border: 1px solid #27272a; border-radius: 8px; padding: 16px; color: #a1a1aa; font-size: 13px;">
                        📄 Verified source citations and matching excerpts will appear here in high-contrast dark view after asking a question.
                    </div>
                    """
                )

        # Event Handlers
        index_btn.click(
            fn=index_files,
            inputs=[file_input],
            outputs=[index_status, docs_summary, doc_dropdown],
        )

        delete_btn.click(
            fn=delete_selected_document,
            inputs=[doc_dropdown],
            outputs=[delete_status, docs_summary, doc_dropdown],
        )

        ask_btn.click(
            fn=answer_query,
            inputs=[query_input, doc_dropdown, top_k_slider, threshold_slider, reranker_checkbox, chatbot],
            outputs=[chatbot, query_input, sources_output, query_info],
        )

        query_input.submit(
            fn=answer_query,
            inputs=[query_input, doc_dropdown, top_k_slider, threshold_slider, reranker_checkbox, chatbot],
            outputs=[chatbot, query_input, sources_output, query_info],
        )

        clear_btn.click(lambda: ([], "", "", ""), outputs=[chatbot, query_input, sources_output, query_info])

    return demo
