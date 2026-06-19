from dataclasses import dataclass
from collections.abc import Callable
import inspect
import typing


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable

    def __call__(self, **kwargs) -> str:
        result = self.func(**kwargs)
        return str(result)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _python_type_to_json_type(tp: type) -> str:
    origin = typing.get_origin(tp)
    if origin is not None:
        tp = origin

    if tp is str:
        return "string"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is bool:
        return "boolean"
    raise TypeError(f"Unsupported type for tool parameter: {tp}")


def _build_json_schema(func: Callable) -> dict:
    hints = typing.get_type_hints(func)
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue
        annotation = hints.get(name, str)
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        is_optional = False
        inner = annotation
        if origin is not None and origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                is_optional = True
                inner = non_none[0]

        json_type = _python_type_to_json_type(inner)
        prop: dict = {"type": json_type}
        if is_optional:
            prop["nullable"] = True
        properties[name] = prop

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    schema: dict = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], Tool]:
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description
        if tool_desc is None:
            doc = inspect.getdoc(func)
            tool_desc = doc.split("\n")[0].strip() if doc else ""
        parameters = _build_json_schema(func)
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            func=func,
        )

    return decorator
