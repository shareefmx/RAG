"""Query Rewriting Module for Conversational Context Resolution.

Reformulates follow-up queries using conversation history into self-contained,
standalone search queries suitable for dense vector retrieval.
"""

import logging
from typing import Dict, List, Optional

from app.generation.llm import BaseLLMService
from app.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_query_rewrite_prompt,
)


logger = logging.getLogger(__name__)


class QueryRewriter:
    """Reformulates conversational follow-up questions into standalone search queries."""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

    def rewrite(self, current_question: str, history: List[Dict[str, str]]) -> str:
        """Takes chat history and the current user question and produces a standalone query.

        Args:
            current_question: The latest user question.
            history: List of chat messages, e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}].

        Returns:
            Rewritten standalone query string.
        """
        if not history:
            return current_question.strip()

        # Format past turns into a readable transcript
        transcript_lines = []
        # Use up to last 4 turns to avoid context overflow
        for turn in history[-4:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            transcript_lines.append(f"{role}: {content}")

        history_str = "\n".join(transcript_lines)
        prompt = build_query_rewrite_prompt(history_str, current_question)

        try:
            rewritten = self.llm_service.generate(
                prompt=prompt,
                system_instruction=QUERY_REWRITE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=128,
            )
            clean_rewritten = rewritten.strip().strip('"').strip("'")
            if clean_rewritten:
                logger.info("Original question: '%s' -> Standalone query: '%s'", current_question, clean_rewritten)
                return clean_rewritten
        except Exception as e:
            logger.warning("Query rewrite error: %s. Using original question.", e)

        return current_question.strip()

