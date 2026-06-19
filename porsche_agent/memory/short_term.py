import time
from dataclasses import dataclass, field
from porsche_agent.tools import Tool, tool


@dataclass
class Fact:
    key: str
    value: str
    created_at: float = field(default_factory=time.time)
    ttl: float | None = None


class ShortTermMemory:
    def __init__(self):
        self._store: dict[str, Fact] = {}

    def put(self, key: str, value: str, ttl: float | None = None) -> None:
        self._store[key] = Fact(key=key, value=value, ttl=ttl)

    def get(self, key: str) -> str | None:
        fact = self._store.get(key)
        if fact is None:
            return None
        if fact.ttl is not None and (time.time() - fact.created_at) > fact.ttl:
            del self._store[key]
            return None
        return fact.value

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def all(self) -> dict[str, str]:
        now = time.time()
        expired = [
            k
            for k, f in self._store.items()
            if f.ttl is not None and (now - f.created_at) > f.ttl
        ]
        for k in expired:
            del self._store[k]
        return {k: f.value for k, f in self._store.items()}

    def clear(self) -> None:
        self._store.clear()

    def to_prompt_fragment(self) -> str:
        facts = self.all()
        if not facts:
            return ""
        lines = [f"  {k}: {v}" for k, v in facts.items()]
        return "[Session Memory]\n" + "\n".join(lines)


def create_memory_tools(stm: ShortTermMemory) -> list[Tool]:
    @tool(description="Store a fact in short-term memory for later recall during this session")
    def remember(key: str, value: str) -> str:
        stm.put(key, value)
        return f"Remembered: {key} = {value}"

    @tool(description="Recall a fact from short-term memory by its key")
    def recall(key: str) -> str:
        result = stm.get(key)
        if result is None:
            return f"No memory found for key: {key}"
        return result

    return [remember, recall]
