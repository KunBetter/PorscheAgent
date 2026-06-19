class ContextManager:
    def __init__(
        self,
        llm,
        max_window: int = 20,
        summary_trigger: int = 10,
        summary_prompt: str | None = None,
    ):
        self.llm = llm
        self.max_window = max_window
        self.summary_trigger = summary_trigger
        self.summary_prompt = summary_prompt or (
            "Summarize this conversation in a few sentences. "
            "Preserve key facts, decisions, user preferences, and important context."
        )
        self.full_history: list[dict] = []
        self.compressed: str | None = None
        self._msgs_since_summary = 0

    def add(self, message: dict) -> None:
        self.full_history.append(message)
        self._msgs_since_summary += 1

    def build_context(self, extra_context: str | None = None) -> list[dict]:
        if len(self.full_history) <= self.max_window:
            messages = list(self.full_history)
        else:
            if self._msgs_since_summary >= self.summary_trigger:
                self._summarize()

            messages = []
            if self.full_history and self.full_history[0]["role"] == "system":
                messages.append(self.full_history[0])

            if self.compressed:
                from porsche_agent.agent import system_msg

                messages.append(
                    system_msg(f"[Conversation Summary]\n{self.compressed}")
                )

            messages.extend(self.full_history[-self.max_window :])

        if extra_context:
            from porsche_agent.agent import system_msg

            messages.append(system_msg(extra_context))

        return messages

    def clear(self) -> None:
        self.full_history.clear()
        self.compressed = None
        self._msgs_since_summary = 0

    def _summarize(self) -> None:
        older = self.full_history[: -self.max_window]
        if not older:
            return

        from porsche_agent.agent import user_msg

        lines = []
        for msg in older:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                content += f" [tool_calls: {names}]"
            if content.strip():
                lines.append(f"[{role}]: {content}")

        prev = f"Previous summary: {self.compressed}\n\n" if self.compressed else ""
        prompt = prev + self.summary_prompt + "\n\n" + "\n".join(lines)

        response = self.llm.chat([user_msg(prompt)])
        self.compressed = response.get("content", "") or ""
        self._msgs_since_summary = 0
