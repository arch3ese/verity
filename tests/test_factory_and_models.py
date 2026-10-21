"""Tests for Model Factory configuration, temperature forwarding, and 404 handling."""

import os
import pytest
from verity.models.factory import get_chat_model, ResilientChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel


class FailingChatModel(BaseChatModel):
    error_to_raise: Exception

    @property
    def _llm_type(self) -> str:
        return "failing-mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.error_to_raise


class SucceedingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "succeeding-mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=HumanMessage(content="success"))])


def test_temperature_forwarded_to_google_chat_model():
    """Verify temperature parameter reaches ChatGoogleGenerativeAI constructor (MOD-02)."""
    model = get_chat_model("gemini", "gemini-3.5-flash-lite", temperature=0.7, api_key="dummy_key")
    # Verify primary model receives configured temperature
    assert hasattr(model, "primary_model")
    assert model.primary_model.temperature == 0.7


def test_model_id_configurable_via_env(monkeypatch):
    """Verify default model ID is configurable via ALIGN_DEFAULT_MODEL env var (MOD-01)."""
    monkeypatch.setenv("ALIGN_DEFAULT_MODEL", "gemini-3.8-flash")
    model = get_chat_model("gemini", None, api_key="dummy_key")
    assert model.primary_model.model == "gemini-3.8-flash"


def test_resilient_model_does_not_swallow_404():
    """Verify 404 / not_found errors are NOT swallowed by fallback chain (MOD-01)."""
    primary = FailingChatModel(error_to_raise=RuntimeError("404 NotFound: Model 'gemini-nonexistent' does not exist."))
    fallback = SucceedingChatModel()
    resilient = ResilientChatModel(primary_model=primary, fallback_models=[fallback])

    with pytest.raises(Exception) as exc_info:
        resilient._generate([HumanMessage(content="hello")])

    err = str(exc_info.value).lower()
    assert "404" in err or "not found" in err
