# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v

# Go vector store (for long-term memory)
cd vectorstore && go build -o vectorstore . && ./vectorstore
```

## Architecture

PorscheAgent is a minimal LLM agent framework. Single Python dependency: `openai>=1.0.0`.

### Core Loop (`agent.py`)

ReAct pattern: send messages + tools to LLM → execute tool calls → append tool results → repeat until direct text response.

Agent accepts optional memory modules: `context_manager`, `short_term_memory`, `long_term_memory`. When absent, falls back to raw `self.history` list.

### Provider Layer (`llm.py`)

`LLMProvider` ABC with `chat()` (single HTTP call) and `embedding_client` (returns OpenAI client for embeddings). `DeepSeekProvider` / `OpenAIProvider` use OpenAI SDK.

### Memory Subsystem (`memory/`)

Three independent layers, all optional:

| Layer | Scope | Backend | Key Feature |
|-------|-------|---------|-------------|
| **ContextManager** | Session | Python | Sliding window + LLM summarization of older messages |
| **ShortTermMemory** | Session | Python `dict` | Key-value with TTL, lazy expiration, `remember`/`recall` tools |
| **LongTermMemory** | Cross-session | Go vector store | Embedding-based semantic retrieval, `remember_forever`/`search_memory` tools |

Memory tools auto-register in `Agent.__init__` when the corresponding module is provided.

Context flow in `Agent.run()`:
1. LTM search → enrich user message with relevant long-term facts
2. STM `to_prompt_fragment()` → inject session facts
3. `ContextManager.build_context()` → summary + windowed messages
4. ReAct loop

### Tool System (`tools.py`)

`@tool` decorator converts Python functions → `Tool` dataclasses. Type annotations (str/int/float/bool/Optional) inferred to JSON Schema.

### Go Vector Store (`vectorstore/`)

Standalone Go binary, zero dependencies. Brute-force cosine similarity search with JSON persistence. HTTP API on `:9876`: `/health` `/add` `/search` `/delete` `/save` `/load`.

### CLI (`cli.py`)

- `porsche chat` — interactive, wires all memory modules per config
- `porsche run SCRIPT` — executes user script with full library access

### Config (`config.py`)

`Config.from_env()` reads env vars. Key vars:
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`
- `PORSCHE_CONTEXT_WINDOW` (20), `PORSCHE_CONTEXT_SUMMARY_TRIGGER` (10)
- `PORSCHE_SHORT_TERM_MEMORY` (true), `PORSCHE_LONG_TERM_MEMORY` (false)
- `PORSCHE_VECTOR_STORE_URL`, `PORSCHE_EMBEDDING_MODEL`
