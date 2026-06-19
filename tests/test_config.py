import os
import pytest
from porsche_agent.config import Config


class TestConfigFromEnv:
    def test_deepseek_defaults(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        monkeypatch.delenv("PORSCHE_MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("PORSCHE_SYSTEM_PROMPT", raising=False)
        monkeypatch.delenv("PORSCHE_TEMPERATURE", raising=False)

        config = Config.from_env()

        assert config.llm_provider == "deepseek"
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-v4-pro"
        assert config.max_iterations == 10
        assert config.system_prompt is None
        assert config.temperature == 0.0

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            Config.from_env()

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

        config = Config.from_env(provider="openai")

        assert config.llm_provider == "openai"
        assert config.api_key == "sk-openai-test"
        assert config.model == "gpt-4o-mini"

    def test_custom_iterations(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("PORSCHE_MAX_ITERATIONS", "20")

        config = Config.from_env()
        assert config.max_iterations == 20

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            Config.from_env(provider="anthropic")
