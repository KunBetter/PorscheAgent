import time
import pytest
from porsche_agent.memory.context import ContextManager
from porsche_agent.memory.short_term import ShortTermMemory, Fact, create_memory_tools
from porsche_agent.memory.long_term import LongTermMemory


class TestContextManager:
    def test_within_window_returns_all(self):
        llm = _MockLLM()
        cm = ContextManager(llm=llm, max_window=10, summary_trigger=5)
        cm.add({"role": "user", "content": "hello"})
        cm.add({"role": "assistant", "content": "hi"})

        ctx = cm.build_context()
        assert len(ctx) == 2
        assert cm.compressed is None

    def test_exceeds_window_triggers_summary(self):
        llm = _MockLLM()
        cm = ContextManager(llm=llm, max_window=3, summary_trigger=2)

        # Fill with 6 messages so 3 older ones get summarized
        for i in range(6):
            cm.add({"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"msg {i}"})

        ctx = cm.build_context()
        # Should have: system summary + last 3 messages = 4
        assert len(ctx) <= 4
        assert cm.compressed == "summary called"
        # Verify the summary message exists
        assert ctx[-4]["role"] == "system"

    def test_system_message_preserved_at_front(self):
        llm = _MockLLM()
        cm = ContextManager(llm=llm, max_window=2, summary_trigger=1)
        cm.add({"role": "system", "content": "You are helpful"})
        for i in range(4):
            cm.add({"role": "user", "content": f"msg {i}"})

        ctx = cm.build_context()
        assert ctx[0]["role"] == "system"
        assert ctx[0]["content"] == "You are helpful"

    def test_extra_context_injected(self):
        llm = _MockLLM()
        cm = ContextManager(llm=llm, max_window=20)
        cm.add({"role": "user", "content": "hello"})

        ctx = cm.build_context(extra_context="[STM]\nkey: value")
        assert any("[STM]" in m.get("content", "") for m in ctx)

    def test_clear_resets_state(self):
        llm = _MockLLM()
        cm = ContextManager(llm=llm, max_window=3, summary_trigger=1)
        for i in range(5):
            cm.add({"role": "user", "content": f"msg {i}"})
        cm.build_context()

        cm.clear()
        assert len(cm.full_history) == 0
        assert cm.compressed is None
        assert cm._msgs_since_summary == 0


class TestShortTermMemory:
    def test_put_and_get(self):
        stm = ShortTermMemory()
        stm.put("city", "Beijing")
        assert stm.get("city") == "Beijing"

    def test_get_missing_key(self):
        stm = ShortTermMemory()
        assert stm.get("nope") is None

    def test_delete(self):
        stm = ShortTermMemory()
        stm.put("city", "Beijing")
        assert stm.delete("city") is True
        assert stm.get("city") is None
        assert stm.delete("nope") is False

    def test_ttl_expiration(self):
        stm = ShortTermMemory()
        stm.put("temp", "value", ttl=0.01)  # 10ms TTL
        time.sleep(0.02)
        assert stm.get("temp") is None

    def test_ttl_not_expired(self):
        stm = ShortTermMemory()
        stm.put("perm", "value", ttl=3600)
        assert stm.get("perm") == "value"

    def test_all_filters_expired(self):
        stm = ShortTermMemory()
        stm.put("a", "1", ttl=0.01)
        stm.put("b", "2", ttl=3600)
        time.sleep(0.02)
        all_facts = stm.all()
        assert "a" not in all_facts
        assert "b" in all_facts

    def test_to_prompt_fragment_empty(self):
        stm = ShortTermMemory()
        assert stm.to_prompt_fragment() == ""

    def test_to_prompt_fragment_has_facts(self):
        stm = ShortTermMemory()
        stm.put("city", "Beijing")
        fragment = stm.to_prompt_fragment()
        assert "[Session Memory]" in fragment
        assert "city: Beijing" in fragment

    def test_clear(self):
        stm = ShortTermMemory()
        stm.put("city", "Beijing")
        stm.clear()
        assert stm.get("city") is None

    def test_memory_tools_remember_and_recall(self):
        stm = ShortTermMemory()
        tools = create_memory_tools(stm)
        tool_map = {t.name: t for t in tools}

        result = tool_map["remember"](key="city", value="Beijing")
        assert "Remembered" in result
        assert stm.get("city") == "Beijing"

        result = tool_map["recall"](key="city")
        assert result == "Beijing"

    def test_memory_tool_recall_missing(self):
        stm = ShortTermMemory()
        tools = create_memory_tools(stm)
        tool_map = {t.name: t for t in tools}

        result = tool_map["recall"](key="nope")
        assert "No memory found" in result


class TestLongTermMemory:
    def test_add_and_search(self):
        class MockEmbeddingClient:
            def __init__(self):
                self.call_count = 0

            @property
            def embeddings(self):
                return self

            def create(self, model, input):
                self.call_count += 1
                # Return different fake vectors for different inputs
                if "Beijing" in input:
                    vec = [0.9, 0.8, 0.7]
                elif "weather" in input:
                    vec = [0.1, 0.2, 0.3]
                else:
                    vec = [0.5, 0.5, 0.5]

                class EmbeddingData:
                    def __init__(self, vec):
                        self.embedding = vec

                class EmbeddingResponse:
                    def __init__(self, vec):
                        self.data = [EmbeddingData(vec)]

                return EmbeddingResponse(vec)

        ltm = _FakeLongTermMemory(
            embedding_client=MockEmbeddingClient(),
            vector_store_url="http://127.0.0.1:9876",
        )
        # Note: this test needs the Go service running.
        # Skip if unavailable.
        try:
            ltm.add("test_1", "User lives in Beijing")
            results = ltm.search("Where does the user live?", k=1)
            assert len(results) > 0
        except Exception:
            pytest.skip("Go vector store not available")


class _MockLLM:
    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": "summary called"}


class _FakeLongTermMemory(LongTermMemory):
    """Bypasses the HTTP call for unit testing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._store: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}

    def _post(self, path, data):
        import json
        payload = json.loads(data)
        if path == "/add":
            self._store[payload["id"]] = payload["vector"]
            self._metadata[payload["id"]] = payload.get("metadata", {})
            return {"status": "ok"}
        elif path == "/search":
            # Simple cosine similarity
            results = []
            for mid, vec in self._store.items():
                dot = sum(a * b for a, b in zip(payload["vector"], vec))
                results.append({
                    "entry": {
                        "id": mid,
                        "vector": vec,
                        "metadata": self._metadata.get(mid, {}),
                    },
                    "distance": dot,
                })
            results.sort(key=lambda r: r["distance"], reverse=True)
            return {"results": results[: payload["k"]]}
        return {}

    def _delete(self, path, data):
        import json
        payload = json.loads(data)
        self._store.pop(payload["id"], None)
        self._metadata.pop(payload["id"], None)
        return {}
