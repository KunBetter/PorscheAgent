# PorscheAgent

Minimal LLM agent framework. DeepSeek first, OpenAI-compatible.

## Quickstart

```bash
export DEEPSEEK_API_KEY=sk-your-api-key
pip install -e .
porsche chat
```

## Usage

### CLI

```bash
# Interactive chat with built-in tools
porsche chat

# Run a custom script
porsche run examples/basic_agent.py
```

### Library

```python
from porsche_agent import Agent, DeepSeekProvider, tool

@tool(description="Get weather for a city")
def get_weather(city: str) -> str:
    return "Sunny, 25°C"

llm = DeepSeekProvider(api_key="sk-...")
agent = Agent(llm=llm, tools=[get_weather])
print(agent.run("What's the weather in Beijing?"))
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API base URL |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Model name |
| `PORSCHE_MAX_ITERATIONS` | `10` | Max tool-use iterations |
| `PORSCHE_SYSTEM_PROMPT` | — | Default system prompt for CLI |
