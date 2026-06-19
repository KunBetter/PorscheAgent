from dataclasses import dataclass
import os


@dataclass
class Config:
    llm_provider: str = "deepseek"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    max_iterations: int = 10
    system_prompt: str | None = None
    temperature: float = 0.0
    context_window: int = 20
    context_summary_trigger: int = 10
    short_term_memory: bool = True
    long_term_memory: bool = False
    vector_store_url: str = "http://127.0.0.1:9876"
    embedding_model: str = "deepseek-embedding"

    @classmethod
    def from_env(cls, provider: str | None = None) -> "Config":
        provider = provider or os.getenv("PORSCHE_LLM_PROVIDER", "deepseek")
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        if not api_key:
            env_var = f"{provider.upper()}_API_KEY"
            raise ValueError(
                f"API key not found. Set the {env_var} environment variable."
            )

        return cls(
            llm_provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_iterations=int(os.getenv("PORSCHE_MAX_ITERATIONS", "10")),
            system_prompt=os.getenv("PORSCHE_SYSTEM_PROMPT"),
            temperature=float(os.getenv("PORSCHE_TEMPERATURE", "0.0")),
            context_window=int(os.getenv("PORSCHE_CONTEXT_WINDOW", "20")),
            context_summary_trigger=int(
                os.getenv("PORSCHE_CONTEXT_SUMMARY_TRIGGER", "10")
            ),
            short_term_memory=os.getenv("PORSCHE_SHORT_TERM_MEMORY", "true").lower()
            != "false",
            long_term_memory=os.getenv("PORSCHE_LONG_TERM_MEMORY", "false").lower()
            == "true",
            vector_store_url=os.getenv(
                "PORSCHE_VECTOR_STORE_URL", "http://127.0.0.1:9876"
            ),
            embedding_model=os.getenv("PORSCHE_EMBEDDING_MODEL", "deepseek-embedding"),
        )
