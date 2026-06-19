import json
import urllib.request
import urllib.error
from porsche_agent.tools import Tool, tool


class LongTermMemory:
    def __init__(
        self,
        embedding_client,
        embedding_model: str = "deepseek-embedding",
        vector_store_url: str = "http://127.0.0.1:9876",
    ):
        self._client = embedding_client
        self._model = embedding_model
        self._url = vector_store_url.rstrip("/")

    def add(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        vector = self._embed(text)
        payload = json.dumps({
            "id": memory_id,
            "vector": vector,
            "metadata": metadata or {"text": text},
        })
        self._post("/add", payload)

    def search(self, query: str, k: int = 5) -> list[dict]:
        vector = self._embed(query)
        payload = json.dumps({"vector": vector, "k": k})
        resp = self._post("/search", payload)
        return resp.get("results", [])

    def delete(self, memory_id: str) -> bool:
        payload = json.dumps({"id": memory_id})
        try:
            self._delete("/delete", payload)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def save(self, path: str = "porsche_memory.json") -> None:
        self._post("/save", json.dumps({"path": path}))

    def load(self, path: str = "porsche_memory.json") -> None:
        self._post("/load", json.dumps({"path": path}))

    def _embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    def _post(self, path: str, data: str) -> dict:
        url = self._url + path
        req = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _delete(self, path: str, data: str) -> dict:
        url = self._url + path
        req = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))


def create_long_term_memory_tools(ltm: LongTermMemory) -> list[Tool]:
    @tool(description="Store information in long-term semantic memory for future retrieval across sessions")
    def remember_forever(key: str, content: str) -> str:
        ltm.add(key, content)
        return f"Stored in long-term memory: {key}"

    @tool(description="Search long-term memory for information relevant to the query")
    def search_memory(query: str) -> str:
        try:
            results = ltm.search(query, k=5)
        except Exception as e:
            return f"Search failed: {e}"
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            meta = r.get("entry", r).get("metadata", {})
            text = meta.get("text", r.get("entry", {}).get("id", ""))
            dist = r.get("distance", 0)
            lines.append(f"- [{dist:.2f}] {text}")
        return "\n".join(lines)

    return [remember_forever, search_memory]
