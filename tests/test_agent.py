import pytest
from porsche_agent.agent import Agent, system_msg, user_msg, tool_msg


class TestMessageFactories:
    def test_system_msg(self):
        assert system_msg("hello") == {"role": "system", "content": "hello"}

    def test_user_msg(self):
        assert user_msg("hi") == {"role": "user", "content": "hi"}

    def test_tool_msg(self):
        assert tool_msg("call_1", "result") == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result",
        }


class TestAgent:
    def test_run_without_tools(self):
        class MockLLM:
            def chat(self, messages, tools=None):
                return {"role": "assistant", "content": "Hello, world!"}

        agent = Agent(llm=MockLLM())
        result = agent.run("Hi!")

        assert result == "Hello, world!"
        # Should have: system (optional), user, assistant
        assert len(agent.history) >= 2
        assert agent.history[-1] == {"role": "assistant", "content": "Hello, world!"}

    def test_run_with_system_prompt(self):
        class MockLLM:
            def chat(self, messages, tools=None):
                return {"role": "assistant", "content": "OK"}

        agent = Agent(llm=MockLLM(), system_prompt="Be helpful")
        agent.run("test")

        assert agent.history[0] == {"role": "system", "content": "Be helpful"}

    def test_run_with_tool_call(self):
        call_count = [0]

        class MockLLM:
            def chat(self, messages, tools=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "greet",
                                    "arguments": '{"name": "World"}',
                                },
                            }
                        ],
                    }
                return {"role": "assistant", "content": "Greeting sent!"}

        from porsche_agent.tools import tool

        @tool(description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        agent = Agent(llm=MockLLM(), tools=[greet])
        result = agent.run("Say hello")

        assert result == "Greeting sent!"
        assert call_count[0] == 2

    def test_max_iterations_exceeded(self):
        class MockLLM:
            def chat(self, messages, tools=None):
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                }

        from porsche_agent.tools import tool

        @tool(description="No-op")
        def noop() -> str:
            return "done"

        agent = Agent(llm=MockLLM(), tools=[noop], max_iterations=2)
        with pytest.raises(RuntimeError, match="max iterations"):
            agent.run("test")

    def test_history_persists_between_runs(self):
        class MockLLM:
            def chat(self, messages, tools=None):
                return {"role": "assistant", "content": "Response"}

        agent = Agent(llm=MockLLM())
        agent.run("First")
        agent.run("Second")

        # Messages should accumulate across runs
        assert len(agent.history) == 4  # user, assistant, user, assistant
