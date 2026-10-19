"""
Unified Model Factory supporting Hybrid architectures:
Cloud APIs (Google Gemini, OpenAI) + Local open-weight models (Ollama) + Mock models for testing.
"""

import os
from typing import Any, Dict, Optional
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


def get_chat_model(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Creates and returns a LangChain chat model instance.

    Supported providers:
    - 'gemini' / 'google': ChatGoogleGenerativeAI (e.g. gemini-2.5-flash)
    - 'openai': ChatOpenAI (e.g. gpt-4o, gpt-4o-mini)
    - 'ollama': ChatOllama (e.g. llama3.2, qwen2.5)
    - 'mock': MockChatModel (for tests / CI)
    """
    provider_clean = provider.strip().lower()

    if provider_clean in ["gemini", "google", "google_genai"]:
        api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file or environment."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        actual_model = model_name if model_name and "gemini-3" in model_name else "gemini-3.8-flash"
        return ChatGoogleGenerativeAI(
            model=actual_model,
            google_api_key=api_key,
            temperature=temperature,
            **kwargs,
        )

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
        base_url = kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model_name,
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
