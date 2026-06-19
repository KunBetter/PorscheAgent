from abc import ABC, abstractmethod
from openai import OpenAI


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send messages to the LLM and return the assistant message dict."""
        ...

    @property
    @abstractmethod
    def embedding_client(self):
        """Return the underlying OpenAI client for embedding API calls."""
        ...


class DeepSeekProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    @property
    def embedding_client(self):
        return self.client

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.model_dump(exclude_none=True)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature

    @property
    def embedding_client(self):
        return self.client

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.model_dump(exclude_none=True)


def create_provider(config) -> LLMProvider:
    from porsche_agent.config import Config

    if config.llm_provider == "deepseek":
        return DeepSeekProvider(
            api_key=config.api_key,
            model=config.model or "deepseek-v4-pro",
            base_url=config.base_url or "https://api.deepseek.com",
            temperature=config.temperature,
        )
    elif config.llm_provider == "openai":
        return OpenAIProvider(
            api_key=config.api_key,
            model=config.model or "gpt-4o",
            base_url=config.base_url,
            temperature=config.temperature,
        )
    raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
