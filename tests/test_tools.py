import pytest
from porsche_agent.tools import tool, Tool, _python_type_to_json_type, _build_json_schema


class TestPythonTypeToJsonType:
    def test_str(self):
        assert _python_type_to_json_type(str) == "string"

    def test_int(self):
        assert _python_type_to_json_type(int) == "integer"

    def test_float(self):
        assert _python_type_to_json_type(float) == "number"

    def test_bool(self):
        assert _python_type_to_json_type(bool) == "boolean"

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            _python_type_to_json_type(list)


class TestBuildJsonSchema:
    def test_simple_params(self):
        def foo(name: str, count: int):
            pass

        schema = _build_json_schema(foo)
        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["count"] == {"type": "integer"}
        assert "name" in schema["required"]
        assert "count" in schema["required"]

    def test_optional_param(self):
        def foo(name: str, count: int | None):
            pass

        schema = _build_json_schema(foo)
        assert schema["properties"]["count"] == {"type": "integer", "nullable": True}
        assert "count" not in schema["required"]

    def test_default_param(self):
        def foo(name: str, count: int = 5):
            pass

        schema = _build_json_schema(foo)
        assert "count" not in schema["required"]


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        result = add(a=1, b=2)
        assert result == "3"

    def test_tool_custom_name(self):
        @tool(name="my_tool", description="Test")
        def some_func(x: str) -> str:
            return x

        assert some_func.name == "my_tool"

    def test_to_openai_schema(self):
        @tool(description="Test tool")
        def test_tool(query: str) -> str:
            return query

        schema = test_tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_docstring_as_description(self):
        @tool()
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}"

        assert greet.description == "Say hello to someone."
