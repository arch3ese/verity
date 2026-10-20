"""
Unified Model Factory supporting Hybrid architectures:
Cloud APIs (Google Gemini, OpenAI) + Local open-weight models (Ollama) + Mock models for testing.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()


class MockChatModel(BaseChatModel):
    """
    Deterministic mock chat model for unit tests, offline benchmarking,
    and fast CI/CD execution without API keys or external services.
    """
    mock_response: str = "This is a deterministic mock model response for testing."
    custom_responses: Dict[str, Any] = {}
    response_handler: Optional[Any] = None

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self.mock_response
        last_msg = messages[-1].content if messages else ""
        if isinstance(last_msg, str):
            if callable(self.response_handler):
                content = self.response_handler(last_msg)
            else:
                for pattern, resp in self.custom_responses.items():
                    if pattern.lower() in last_msg.lower():
                        if callable(resp):
                            content = resp(last_msg)
                        else:
                            content = resp
                        break

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


import logging

logger = logging.getLogger("aligncraft.models.factory")


class ResilientChatModel(BaseChatModel):
    """
    Wrapper around LangChain ChatModels that catches rate limits (429),
    server high demand (503), or quota exhaustion, immediately failing over
    to specified fallback models without hanging or crashing.
    """
    primary_model: BaseChatModel
    fallback_models: List[BaseChatModel] = []

    @property
    def _llm_type(self) -> str:
        return "resilient-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        candidates = [self.primary_model] + list(self.fallback_models)
        last_error = None

        for idx, model in enumerate(candidates):
            try:
                return model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                # 404 / not_found is a persistent configuration error, NOT a transient quota issue.
                if any(token in err_str for token in ["404", "not_found", "not found"]):
                    logger.error(f"Model ID not found (404): {e}")
                    raise ValueError(
                        f"Requested model was not found (404). Please verify model ID configuration "
                        f"or set ALIGN_DEFAULT_MODEL: {e}"
                    ) from e

                is_recoverable = any(
                    token in err_str
                    for token in ["429", "resource_exhausted", "503", "unavailable", "rate limit", "quota", "deadline"]
                )
                if is_recoverable and idx < len(candidates) - 1:
                    logger.warning(
                        f"Model generation failed with recoverable error: {e}. "
                        f"Attempting failover to next candidate model..."
                    )
                    last_error = e
                    continue
                raise e

        if last_error:
            raise last_error
        raise RuntimeError("No chat models available in ResilientChatModel pool.")


def get_chat_model(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Creates and returns a LangChain chat model instance.

    Supported providers:
    - 'gemini' / 'google': ChatGoogleGenerativeAI (default: gemini-3.5-flash-lite with fallback)
    - 'openai': ChatOpenAI (e.g. gpt-4o, gpt-4o-mini)
    - 'ollama': ChatOllama (e.g. llama3.2, qwen2.5)
    - 'mock': MockChatModel (for tests / CI)
    """
    provider_clean = provider.strip().lower()

    if provider_clean in ["gemini", "google", "google_genai"]:
        api_key = kwargs.pop("api_key", None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file or environment."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Target model: Configurable via ALIGN_DEFAULT_MODEL or ALIGN_GEMINI_MODEL, default gemini-3.5-flash-lite
        default_model = os.getenv("ALIGN_DEFAULT_MODEL") or os.getenv("ALIGN_GEMINI_MODEL") or "gemini-3.5-flash-lite"
        target_model = model_name if model_name and model_name.strip() else default_model

        # Temperature handling (MOD-02): forward explicitly, with optional ALIGN_TEMPERATURE env override
        effective_temp = temperature
        if os.getenv("ALIGN_TEMPERATURE") is not None:
            try:
                effective_temp = float(os.getenv("ALIGN_TEMPERATURE"))
            except ValueError:
                pass

        max_retries = kwargs.pop("max_retries", 1)
        timeout = kwargs.pop("timeout", 15.0)

        # Build primary model (forward temperature to ChatGoogleGenerativeAI)
        primary = ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=api_key,
            temperature=effective_temp,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )

        # Build fallback model pool: configurable via ALIGN_FALLBACK_MODELS
        env_fallbacks = os.getenv("ALIGN_FALLBACK_MODELS")
        if env_fallbacks is not None:
            raw_names = [f.strip() for f in env_fallbacks.split(",") if f.strip() and f.strip().lower() not in ["none", "false", "0"]]
        else:
            raw_names = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]
        fallback_names = [m for m in raw_names if m != target_model]

        fallbacks = [
            ChatGoogleGenerativeAI(
                model=m,
                google_api_key=api_key,
                temperature=effective_temp,
                max_retries=max_retries,
                timeout=timeout,
                **kwargs,
            )
            for m in fallback_names
        ]

        return ResilientChatModel(primary_model=primary, fallback_models=fallbacks)

    elif provider_clean == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set it in your .env file or environment."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )

    elif provider_clean == "ollama":
        base_url = kwargs.pop("base_url", None) or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or "llama3.2",
            base_url=base_url,
            temperature=temperature,
            **kwargs,
        )

    elif provider_clean in ["mock", "test", "offline"]:
        mock_resp = kwargs.get("mock_response", "Deterministic evaluation output.")
        custom_resps = kwargs.get("custom_responses", {})
        resp_handler = kwargs.get("response_handler")
        return MockChatModel(
            mock_response=mock_resp,
            custom_responses=custom_resps,
            response_handler=resp_handler,
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported options: 'gemini', 'openai', 'ollama', 'mock'."
        )
