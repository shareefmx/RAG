"""LLM Service Abstraction Module.

Decouples the RAG pipeline from specific LLM providers.
Supports Google Gemini API, OpenRouter (OpenAI-compatible), and a deterministic Mock provider.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM generation failures."""
    pass


class BaseLLMService(ABC):
    """Abstract interface defining standard LLM text generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generates a text completion for the given prompt.

        Args:
            prompt: Formatted user prompt containing retrieved context and question.
            system_instruction: System prompt enforcing ground rules.
            temperature: Sampling temperature (0.0 for factual grounded generation).
            max_tokens: Maximum output tokens to generate.

        Returns:
            Generated response string.
        """
        pass


class GeminiLLMService(BaseLLMService):
    """Google Gemini API implementation with multi-model quota resilience."""

    FALLBACK_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-2.5-flash",
    ]

    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash-lite"):
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file or environment."
            )
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Google GenAI Client: %s", e)
                raise LLMError(f"Could not initialize Google GenAI client: {e}") from e
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction if system_instruction else None,
        )

        # Build prioritized model candidate list with user model first
        models_to_try = [self.model_name]
        for fb in self.FALLBACK_MODELS:
            if fb not in models_to_try:
                models_to_try.append(fb)

        last_error = None
        for current_model in models_to_try:
            try:
                logger.info("Attempting generation with Gemini model '%s'", current_model)
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )

                if response.text and response.text.strip():
                    return response.text.strip()
                logger.warning("Gemini model '%s' returned empty response; trying fallback", current_model)

            except Exception as e:
                err_str = str(e)
                logger.warning("Gemini generation on '%s' failed: %s", current_model, err_str)
                last_error = e
                # If rate-limited (429), unavailable (503), or not found (404), try next model
                if "429" in err_str or "503" in err_str or "404" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    continue
                # For fatal API errors, break or raise
                break

        logger.error("All Gemini models in fallback chain failed: %s", last_error)
        raise LLMError(f"Language model service temporarily unavailable: {last_error}") from last_error


class OpenRouterLLMService(BaseLLMService):
    """OpenRouter API implementation using OpenAI-compatible client."""

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3-8b-instruct:free"):
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Please set OPENROUTER_API_KEY in your .env file or environment."
            )
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key,
                )
            except Exception as e:
                logger.error("Failed to initialize OpenAI/OpenRouter client: %s", e)
                raise LLMError(f"Could not initialize OpenRouter client: {e}") from e
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        logger.info("Sending generation request to OpenRouter model '%s'", self.model_name)
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            return content.strip()

        except Exception as e:
            logger.error("OpenRouter generation failed: %s", e)
            raise LLMError(f"OpenRouter service temporarily unavailable: {e}") from e


class MockLLMService(BaseLLMService):
    """Deterministic mock LLM for testing, evaluation benchmarks, and offline environments."""

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.canned_response:
            return self.canned_response
        return "Based on the provided context, the methodology used was empirical benchmarking with FAISS."


class LLMServiceFactory:
    """Factory creating the appropriate LLMService according to application settings."""

    @staticmethod
    def create(
        provider: str = "gemini",
        gemini_api_key: str = "",
        openrouter_api_key: str = "",
        model_name: Optional[str] = None,
    ) -> BaseLLMService:
        provider = provider.lower().strip()

        if provider == "gemini":
            key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
            if not key:
                logger.warning("No GEMINI_API_KEY found. Falling back to MockLLMService.")
                return MockLLMService()
            return GeminiLLMService(api_key=key, model_name=model_name or "gemini-2.5-flash")

        elif provider == "openrouter":
            key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
            if not key:
                logger.warning("No OPENROUTER_API_KEY found. Falling back to MockLLMService.")
                return MockLLMService()
            return OpenRouterLLMService(
                api_key=key,
                model_name=model_name or "meta-llama/llama-3-8b-instruct:free"
            )

        elif provider == "mock":
            return MockLLMService()

        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported: 'gemini', 'openrouter', 'mock'.")

