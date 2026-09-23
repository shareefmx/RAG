import pytest
from app.generation.llm import (
    BaseLLMService,
    LLMError,
    LLMServiceFactory,
    MockLLMService,
    MultiProviderResilientLLMService,
)


class FailingLLMService(BaseLLMService):
    def generate(self, prompt: str, system_instruction=None, temperature=0.0, max_tokens=1024) -> str:
        raise LLMError("Simulated outage / 403 Forbidden")


class WorkingLLMService(BaseLLMService):
    def __init__(self, response: str = "Fallback Success"):
        self.response = response

    def generate(self, prompt: str, system_instruction=None, temperature=0.0, max_tokens=1024) -> str:
        return self.response


def test_mock_llm_service():
    mock = MockLLMService(canned_response="Test canned response")
    assert mock.generate("Any prompt") == "Test canned response"


def test_multiprovider_failover_success():
    failing = FailingLLMService()
    working = WorkingLLMService("Recovered via failover")

    resilient = MultiProviderResilientLLMService(
        providers=[
            ("Provider A", failing),
            ("Provider B", working),
        ]
    )

    result = resilient.generate("Test query")
    assert result == "Recovered via failover"
    assert resilient.active_provider_name == "Provider B"


def test_multiprovider_all_failed():
    failing1 = FailingLLMService()
    failing2 = FailingLLMService()

    resilient = MultiProviderResilientLLMService(
        providers=[
            ("Provider 1", failing1),
            ("Provider 2", failing2),
        ]
    )

    with pytest.raises(LLMError, match="All LLM providers in resilient failover chain failed"):
        resilient.generate("Test query")


def test_factory_fallback_to_mock_when_no_keys():
    service = LLMServiceFactory.create(provider="auto", gemini_api_key="", nvidia_api_key="", openrouter_api_key="")
    assert isinstance(service, MockLLMService)
