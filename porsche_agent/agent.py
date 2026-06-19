import json
from porsche_agent.tools import Tool


def system_msg(content: str) -> dict:
    return {"role": "system", "content": content}


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def tool_msg(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class Agent:
    def __init__(
        self,
        llm,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        context_manager=None,
        short_term_memory=None,
        long_term_memory=None,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.context = context_manager
        self.stm = short_term_memory
        self.ltm = long_term_memory
        self.history: list[dict] = []

        if self.stm:
            from porsche_agent.memory.short_term import create_memory_tools

            for t in create_memory_tools(self.stm):
                self.tools[t.name] = t
        if self.ltm:
            from porsche_agent.memory.long_term import create_long_term_memory_tools

            for t in create_long_term_memory_tools(self.ltm):
                self.tools[t.name] = t

    def run(self, user_message: str) -> str:
        extra_context = ""

        if self.ltm:
            try:
                results = self.ltm.search(user_message, k=3)
                if results:
                    lines = []
                    for r in results:
                        meta = r.get("metadata", {})
                        text = meta.get("text", r.get("id", ""))
                        lines.append(f"- {text}")
                    extra_context = "[Long-term memory]\n" + "\n".join(lines)
            except Exception:
                pass

        if self.stm:
            stm_fragment = self.stm.to_prompt_fragment()
            if stm_fragment:
                sep = "\n\n" if extra_context else ""
                extra_context += sep + stm_fragment

        if self.context:
            if self.system_prompt and not self.context.full_history:
                self.context.add(system_msg(self.system_prompt))
            if extra_context:
                user_message = (
                    extra_context + "\n\n---\n[User Message]\n" + user_message
                )
            self.context.add(user_msg(user_message))
        else:
            if self.system_prompt and not self.history:
                self.history.append(system_msg(self.system_prompt))
            if extra_context:
                user_message = (
                    extra_context + "\n\n---\n[User Message]\n" + user_message
                )
            self.history.append(user_msg(user_message))

        tool_schemas = (
            [t.to_openai_schema() for t in self.tools.values()]
            if self.tools
            else None
        )

        for _ in range(self.max_iterations):
            if self.context:
                messages = self.context.build_context()
            else:
                messages = list(self.history)

            response = self.llm.chat(
                messages=messages,
                tools=tool_schemas,
            )

            if self.context:
                self.context.add(response)
            else:
                self.history.append(response)

            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                return response.get("content") or ""

            for tc in tool_calls:
                tc_id = tc["id"]
                tc_name = tc["function"]["name"]
                tool = self.tools.get(tc_name)

                if tool is None:
                    result = f"Error: unknown tool '{tc_name}'"
                else:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        result = f"Error parsing tool arguments: {e}"
                    else:
                        try:
                            result = tool(**args)
                        except Exception as e:
                            result = f"Tool error: {e}"

                if self.context:
                    self.context.add(tool_msg(tc_id, result))
                else:
                    self.history.append(tool_msg(tc_id, result))

        raise RuntimeError(
            f"Agent exceeded max iterations ({self.max_iterations}) "
            "without a final response."
        )
