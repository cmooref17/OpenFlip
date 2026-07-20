"""Tool registry, @tool decorator, and ToolResult container.

Every callable tool the framework exposes lives in this package. Each is decorated with
@tool, which introspects the function's signature to produce the Ollama-compatible JSON
schema, and registers it in TOOL_REGISTRY.

Tools never touch Discord directly. They return a ToolResult; the tool_executor handles
all I/O (file uploads, captions, errors).
"""
from __future__ import annotations
import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints, get_args, get_origin, Union, Literal


@dataclass
class ToolResult:
    text: Optional[str] = None              # what posts to Discord (subject to silent_to_discord + media_only)
    attachments: list[Path] = field(default_factory=list)  # files for Discord; always posted
    error: Optional[str] = None
    model_feedback: Optional[str] = None    # what the model sees as the tool result. If None, framework
                                            # generates a reasonable string from text/attachments/error.
    posted_urls: list[str] = field(default_factory=list)   # Discord CDN URLs filled in by tool_executor
                                                           # AFTER channel.send so build_model_feedback can
                                                           # include them — lets the model say "again" and
                                                           # call edit_image with the prior image's URL.
    # Outcome of the user-facing post, filled in by tool_executor after
    # _post_tool_result. Tri-state:
    #   True  — the send call(s) succeeded (URL presence is NOT the signal:
    #           non-Discord transports can succeed without returning a URL).
    #   False — posting was attempted and FAILED; post_fail_reason says why.
    #           build_model_feedback then tells the model the user has NOT
    #           seen the output, so it can't claim "sent it!" over a failure.
    #   None  — posting wasn't attempted (silent turn / tool error / nothing
    #           to post).
    posted_ok: Optional[bool] = None
    post_fail_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def fail(cls, msg: str) -> "ToolResult":
        return cls(error=msg)


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters_schema: dict
    is_async: bool
    # If True, this tool's text output is fed to the model but NOT posted to Discord.
    # Set on a per-tool basis after registration (e.g., web_search). Attachments always
    # post regardless — this only affects the text payload.
    silent_to_discord: bool = False


TOOL_REGISTRY: dict[str, Tool] = {}


def _python_type_to_schema(tp) -> dict:
    """Translate a Python type hint to a JSON-schema fragment."""
    origin = get_origin(tp)
    args = get_args(tp)

    # Handle Optional[T] / Union[T, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_schema(non_none[0])
        return {"type": "string"}

    # Handle Literal[...]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Handle list[T]
    if origin in (list, tuple):
        item_schema = _python_type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Plain types
    if tp is str:   return {"type": "string"}
    if tp is int:   return {"type": "integer"}
    if tp is float: return {"type": "number"}
    if tp is bool:  return {"type": "boolean"}
    if tp is dict:  return {"type": "object"}
    if tp is list:  return {"type": "array"}
    return {"type": "string"}


def _build_parameters_schema(func: Callable, param_descriptions: dict[str, str]) -> dict:
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties = {}
    required = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        annotation = hints.get(pname, str)
        prop = _python_type_to_schema(annotation)
        if pname in param_descriptions:
            prop["description"] = param_descriptions[pname]
        # Propagate the function's actual default into the schema so the
        # tool's JSON schema reflects what the Python function actually
        # accepts. Without this, optional params would show up as
        # default=None and dispatch None into functions written as
        # `def foo(file: str = "")`, crashing on `.strip()`.
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = param.default
        properties[pname] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _parse_param_descriptions(docstring: str, param_names: set[str] | None = None) -> tuple[str, dict[str, str]]:
    """Pull "name: description" lines out of an Args: block.

    A line inside the Args block starts a new parameter only when the text
    before the colon is one of the function's actual parameter names —
    otherwise it's a continuation of the previous parameter's description
    (multi-line arg docs routinely contain colons, e.g. 'Example: "..."').
    """
    if not docstring:
        return "", {}
    lines = [l.rstrip() for l in docstring.strip().split("\n")]
    desc_lines = []
    in_args = False
    current: Optional[str] = None
    param_descriptions: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            current = None
            continue
        if in_args:
            if not stripped:
                in_args = False
                current = None
                continue
            pname, sep, pdesc = stripped.partition(":")
            pname = pname.strip()
            starts_param = bool(sep) and pname.isidentifier() and (
                param_names is None or pname in param_names
            )
            if starts_param:
                current = pname
                param_descriptions[current] = pdesc.strip()
            elif current:
                param_descriptions[current] = (param_descriptions[current] + " " + stripped).strip()
        else:
            desc_lines.append(stripped)
    return " ".join(l for l in desc_lines if l).strip(), param_descriptions


def tool(func: Callable) -> Callable:
    """Decorator: registers a tool. Function must have a docstring whose first line/paragraph
    is the description shown to the model. An Args: block can document parameters."""
    sig_param_names = {
        p for p in inspect.signature(func).parameters if p not in ("self", "cls")
    }
    description, param_descriptions = _parse_param_descriptions(func.__doc__ or "", sig_param_names)
    if not description:
        description = func.__name__.replace("_", " ")
    schema = _build_parameters_schema(func, param_descriptions)
    is_async = asyncio.iscoroutinefunction(func)
    TOOL_REGISTRY[func.__name__] = Tool(
        name=func.__name__,
        description=description,
        func=func,
        parameters_schema=schema,
        is_async=is_async,
    )
    # Providers (anthropic/openai conversation builders) read this attribute to
    # send real parameter schemas to the API. Without it they fall back to an
    # empty additionalProperties schema and the model has to guess arg names.
    func.tool_spec = {
        "name": func.__name__,
        "description": description,
        "input_schema": schema,
    }
    return func


def get_tool(name: str) -> Optional[Tool]:
    return TOOL_REGISTRY.get(name)


def tool_callable_for_ollama(name: str) -> Optional[Callable]:
    """Ollama's chat() takes a list of callables; it introspects them via the same mechanism
    we use here. Returning the raw function lets ollama do its own thing — but we override
    its schema by also passing the schema explicitly elsewhere if needed."""
    t = TOOL_REGISTRY.get(name)
    return t.func if t else None
