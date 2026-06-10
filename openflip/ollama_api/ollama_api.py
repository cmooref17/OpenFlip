import asyncio
import os
import json
import inspect
import re
from typing import Callable, Sequence, Any, Union, cast
from pathlib import Path
from .utils import *
import ollama
from pydantic import BaseModel

config: dict = {}
_ollama_client: ollama.AsyncClient | None = None


def load_config(path: str = None) -> dict:
    """Load config from a JSON file. If path is omitted, looks for config.json next to this module."""
    global config, _ollama_client
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(path, "r") as file:
            config = json.loads(file.read())
    except FileNotFoundError:
        print_ts(f"Missing config file at: {path}")
        config = {}
    except Exception as e:
        print_ts(f"Unknown error while loading config: \"{e}\"")
        config = {}
    _ollama_client = ollama.AsyncClient(config.get('host'))
    return config


def set_host(host: str):
    """Override the Ollama host without reloading the whole config."""
    global _ollama_client
    config['host'] = host
    _ollama_client = ollama.AsyncClient(host)


load_config()

class Options(dict):
    # Type hints for IDE support
    num_ctx: int
    temperature: float
    repeat_penalty: float
    repeat_last_n: int
    num_predict: int
    top_k: int
    top_p: float
    min_p: float
    mirostat: int
    mirostat_eta: float
    mirostat_tau: float

    def __init__(self, options: dict = None, num_ctx: int = None, temperature: float = None, repeat_penalty: float = None, repeat_last_n: int = None, num_predict: int = None, top_k: int = None, top_p: float = None, min_p: float = None, mirostat: int = None, mirostat_eta: float = None, mirostat_tau: float = None):
        _options:dict = config.get('default_options', {}).copy()
        _options.update(options or {})
        explicit = {'num_ctx': num_ctx, 'temperature': temperature, 'repeat_penalty': repeat_penalty, 'repeat_last_n': repeat_last_n, 'num_predict': num_predict, 'top_k': top_k, 'top_p': top_p, 'min_p': min_p, 'mirostat': mirostat, 'mirostat_eta': mirostat_eta, 'mirostat_tau': mirostat_tau}
        _options.update({k: v for k, v in explicit.items() if v is not None})
        super().__init__(_options)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self) -> 'Options':
        return Options(super().copy())


class Model(dict):
    _cache:dict[str, 'Model'] = {}

    # Type hints for ollama.show() fields
    modified_at: str
    template: str
    modelfile: str
    license: str
    details: dict
    model_info: dict
    capabilities: list[str]
    parameters: str

    @staticmethod
    def _extract_name(model:dict) -> str|None:
        """Extract model name from modelfile '# FROM name' comment."""
        modelfile = model.get('modelfile', '')
        for line in modelfile.split('\n'):
            if line.startswith('# FROM '):
                return line[7:].strip()
        return None

    def __new__(cls, model:dict) -> 'Model':
        name = cls._extract_name(model)
        if name and name in cls._cache:
            return cls._cache[name]
        instance = cast('Model', super().__new__(cls))
        if name:
            cls._cache[name] = instance
        return instance

    def __init__(self, model:dict):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        super().__init__()
        self._name:str = self._extract_name(model)
        self.update(model)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self[name] = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters_parsed(self) -> dict:
        """Parameters parsed into a dict."""
        raw = self.get('parameters')
        if not raw:
            return {}
        params = {}
        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, value = parts
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value
        return params

    @property
    def family(self) -> str|None:
        return self.get('details', {}).get('family')

    @property
    def families(self) -> list[str]:
        return self.get('details', {}).get('families', [])

    @property
    def parameter_size(self) -> str|None:
        return self.get('details', {}).get('parameter_size')

    @property
    def quantization_level(self) -> str|None:
        return self.get('details', {}).get('quantization_level')

    @property
    def context_length(self) -> int|None:
        return self.get('model_info', {}).get(f'{self.family}.context_length')

    @property
    def supports_tools(self) -> bool:
        return 'tools' in self.get('capabilities', [])

    @property
    def supports_vision(self) -> bool:
        return 'vision' in self.get('capabilities', [])

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()


class _SyntheticToolCall:
    """Stand-in for ollama.Message.ToolCall when we recover a tool call from
    chat content (the JSON-as-text leak some chat-tuned models exhibit)."""
    class _Function:
        def __init__(self, name: str, arguments: dict):
            self.name = name
            self.arguments = arguments
    def __init__(self, name: str, arguments: dict):
        self.function = _SyntheticToolCall._Function(name, arguments)


def _flatten_tool_call_args(raw_params: Any) -> dict:
    """Tool-call leak content commonly nests each argument under a schema-shaped
    dict ({"type": ..., "description": ..., "value": X}). Lift the actual value
    out so the args dict matches what the tool function expects."""
    if not isinstance(raw_params, dict):
        return {}
    flat: dict = {}
    for k, v in raw_params.items():
        if isinstance(v, dict) and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    return flat


def _recover_tool_call_from_content(content: str, tools_map: dict[str, Callable]) -> tuple[str, dict] | None:
    """Detect and recover a tool call leaked into chat content.
    Returns (function_name, args_dict) on success, None otherwise.
    Conservative: only recovers content that parses as a single JSON object
    with the canonical {"name": ..., "parameters": ...} shape AND names a tool
    actually present in tools_map."""
    if not content or not isinstance(content, str):
        return None
    s = content.strip()
    if not (s.startswith("{") and (s.endswith("}") or s.endswith("]"))):
        return None
    # Try strict parse first.
    try:
        parsed = json.loads(s)
    except Exception:
        # Some leaks omit the trailing brace. Try adding one.
        try:
            parsed = json.loads(s + "}")
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    params = parsed.get("parameters") if isinstance(parsed.get("parameters"), dict) else parsed.get("arguments")
    if not isinstance(name, str) or name not in tools_map:
        return None
    return name, _flatten_tool_call_args(params or {})


class ToolCall:
    def __init__(self, tool_call:'ollama.Message.ToolCall | _SyntheticToolCall', tools_map:dict[str, Callable] = None):
        self._function_name:str = tool_call.function.name
        self._args:dict[str, Any] = dict(tool_call.function.arguments)
        self.function:Callable = tools_map.get(self._function_name) if tools_map else None

    @property
    def function_name(self) -> str:
        return self._function_name

    @property
    def args(self) -> dict[str, Any]:
        return self._args

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.function) if callable(self.function) else False

    async def invoke(self) -> Any:
        if not self.function:
            raise ValueError(f"No function found for tool call: {self._function_name}")
        result = self.function(**self.args)
        if inspect.isawaitable(result):
            return await result
        return result

    def __str__(self) -> str:
        return f"ToolCall - {self.function_name}"

    def __repr__(self) -> str:
        return self.__str__()


from ollama import Image


class ChatMessage(dict):
    # Type hints for IDE support
    role:str
    content:str
    images:Sequence[Image]

    def __init__(self, role:str = None, content:str = None, images:list[str] = None, message_dict:dict = None):
        if message_dict:
            message_dict['images'] = message_dict.get('images', None)
            super().__init__(message_dict)
        else:
            if not role or not isinstance(content, str):
                raise ValueError("Role and content are required if not providing a message_dict object")
            self['role'] = role
            self['content'] = content
            self['images'] = images
        #for key, value in kwargs.items():
            #self['key'] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self:
            self[name] = value
        else:
            object.__setattr__(self, name, value)

    @property
    def role_user(self) -> bool:
        return self.role == 'user'

    @property
    def role_assistant(self) -> bool:
        return self.role == 'assistant'

    @property
    def role_system(self) -> bool:
        return self.role == 'system'

    def copy(self) -> 'ChatMessage':
        return ChatMessage(message_dict=super().copy())


class AIChatMessage(ChatMessage):
    def __init__(self, ollama_response:ollama.ChatResponse, chat_format:BaseModel = None, tools_map:dict[str, Callable] = None):
        super().__init__(message_dict=dict(ollama_response.message))
        tool_calls = ollama_response.message.tool_calls
        self.pop('tool_calls', None)

        self.raw_response:ollama.ChatResponse = ollama_response
        self.chat_format:ChatFormat|None = ChatFormat.model_validate_json(ollama_response.message.content) if chat_format else None
        if self.chat_format:
            self.content = str(self.chat_format)
        self.model:str = ollama_response.model
        self.created_at:str = ollama_response.created_at
        self.done:bool = ollama_response.done
        self.done_reason:str = ollama_response.done_reason  # stop, length, user
        self.load_duration:float = ollama_response.load_duration / 1e9 if ollama_response.load_duration and isinstance(ollama_response.load_duration, (int, float)) else 0  # duration in seconds to load model, if unloaded
        self.prompt_eval_duration:float = ollama_response.prompt_eval_duration / 1e9 if ollama_response.prompt_eval_duration and isinstance(ollama_response.prompt_eval_duration, (int, float)) else 0  # duration in seconds to process prompt
        self.total_duration:float = ollama_response.total_duration / 1e9 if ollama_response.total_duration and isinstance(ollama_response.total_duration, (int, float)) else 0  # duration in seconds for full response
        self.prompt_eval_count:int = ollama_response.prompt_eval_count if ollama_response.prompt_eval_count and isinstance(ollama_response.prompt_eval_count, int) else 0 # num tokens processed from prompt
        self.eval_count:int = ollama_response.eval_count if ollama_response.eval_count and isinstance(ollama_response.eval_count, int) else 0  # num tokens generated in response
        self.images:list[Union[str, bytes, Path]] = [i for i in ollama_response.message.images] if ollama_response.message.images else []
        self.tool_calls:list[ToolCall] = [ToolCall(t, tools_map) for t in tool_calls] if tool_calls else []

        # JSON-as-text tool-call recovery: some chat-tuned models leak tool calls
        # back as raw JSON in the content field instead of using the protocol.
        # If we got no real tool_calls AND the content parses as a single
        # {name, parameters} object naming a known tool, lift it into tool_calls
        # and clear the content so the assistant turn is treated as a tool call.
        if not self.tool_calls and tools_map:
            recovered = _recover_tool_call_from_content(self.content, tools_map)
            if recovered is not None:
                fn_name, fn_args = recovered
                self.tool_calls = [ToolCall(_SyntheticToolCall(fn_name, fn_args), tools_map)]
                self.content = ""

        # Check for thinking in message.thinking field
        self.thinking: str | None = None
        if hasattr(ollama_response.message, 'thinking') and ollama_response.message.thinking:
            self.thinking = ollama_response.message.thinking
        # Fallback: Separate <think></think> or <response></response> tags from content
        elif "</think>" in self.content_text or "</response>" in self.content_text:
            # Find opening tag position (-1 if not found)
            think_start = self.content_text.find('<think>')
            response_start = self.content_text.find('<response>')

            # Pick the earliest valid opening tag
            if think_start >= 0 and response_start >= 0:
                index0 = min(think_start, response_start)
            elif think_start >= 0:
                index0 = think_start
            elif response_start >= 0:
                index0 = response_start
            else:
                index0 = 0

            # Find closing tag and its length
            think_end = self.content_text.rfind('</think>')
            response_end = self.content_text.rfind('</response>')

            if think_end >= 0 and response_end >= 0:
                if think_end > response_end:
                    index1 = think_end
                    closing_tag_len = len('</think>')
                else:
                    index1 = response_end
                    closing_tag_len = len('</response>')
            elif think_end >= 0:
                index1 = think_end
                closing_tag_len = len('</think>')
            elif response_end >= 0:
                index1 = response_end
                closing_tag_len = len('</response>')
            else:
                index1 = -1
                closing_tag_len = 0

            if index1 >= 0:
                self.thinking = self.content_text[index0:index1 + closing_tag_len]
                self.content_text = self.content_text[index1 + closing_tag_len:]
            else:
                print_ts(f'Failed to find closing </think> or </response> tag.\n"""\n{self.content_text}\n"""', error=True)

        self.content_text = self.content_text.strip() # Remove leading/trailing characters
        self.content_text = re.sub(r' {2,}', ' ', self.content_text) # Remove double spaces

        while self.content_text.startswith('"') and self.content_text.endswith('"'):
            self.content_text = self.content_text[1:-1]

    @property
    def content_text(self):
        return self.chat_format.verbal_response_to_user_message if self.chat_format else self.content

    @content_text.setter
    def content_text(self, content):
        if self.chat_format:
            self.chat_format.verbal_response_to_user_message = content
        else:
            self.content = content

    @property
    def user_prefixed_content(self):
        print_ts(yellow("Avoid calling user_prefixed_content on a AIChatMessage object."))
        return self.content

    @property
    def tokens(self) -> int:
        return self.eval_count

    @property
    def input_tokens(self) -> int:
        return self.prompt_eval_count

    @property
    def tokens_per_second(self) -> float:
        if self.response_time <= 0:
            return 0.0
        return round((self.total_duration - self.prompt_eval_duration - self.load_duration) / self.response_time, 2)

    @property
    def response_time(self) -> float:
        return round(self.total_duration - self.load_duration, 2)

    async def invoke_tools(self):
        if self.tool_calls:
            for t in self.tool_calls:
                await t.invoke()

    def print_request_stats(self):
        print_ts(f"{COLOR_GREEN}Generated response in {COLOR_END}{self.total_duration - self.load_duration:.2f}{COLOR_GREEN} seconds{COLOR_END}", log=True, include_timestamp=False)
        if self.load_duration > 0.1:
            print(yellow(f"Loaded model\t{self.load_duration:.2f} seconds\t{self.model}"))
        print(yellow(f"Prompt      \t{self.prompt_eval_count} tokens\t\t{self.prompt_eval_duration:.4f} seconds\t({self.prompt_eval_count / max(self.prompt_eval_duration, 0.0001):.4f} t/s)"))
        print(yellow(f"Response    \t{self.eval_count} tokens\t\t{self.total_duration - self.prompt_eval_duration - self.load_duration:.4f} seconds\t({self.eval_count / max(self.total_duration - self.prompt_eval_duration - self.load_duration, 0.0001):.4f} t/s)"))
        print()


class Conversation:
    def __init__(self, conversation_id: str, model: str = None, system_message: str = None, options: 'Options | dict' = None, **kwargs):
        super().__init__()
        self.conversation_id: str = str(conversation_id)
        self.messages: list[ChatMessage] = []

        self.model: str = model or config.get('default_model')
        self.options: Options = Options(options) if options is not None else Options(config.get('default_options', {}))
        self.system_message: str = system_message or ""
        self.debug: bool = False

        if self.system_message:
            self.append_system_message(self.system_message)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def append_message(self, role:str, content:str):
        self.messages.append(ChatMessage(role, content))

    def append_system_message(self, content:str):
        return self.append_message(role='system', content=content)

    def append_user_message(self, content:str):
        return self.append_message(role='user', content=content)

    def append_assistant_message(self, content:str):
        return self.append_message(role='assistant', content=content)

    def reset_conversation(self):
        self.messages = []
        self.append_system_message(self.system_message)

    def undo_message(self, count:int = 1) -> list[ChatMessage] | None:
        if not count or not self.messages or len(self.messages) <= 1 or self.messages[-1].role == 'user':
            return None
        undo_index = -1
        for i in range(len(self.messages)):
            index = len(self.messages) - i - 1
            message = self.messages[index]
            if message.role == 'user':
                undo_index = index
                count -= 1
                if count <= 0:
                    break
        if undo_index != -1:
            removed_messages = self.messages[undo_index:]
            self.messages = self.messages[:undo_index]
            return removed_messages
        return None # failed to undo any messages

    async def redo_message(self) -> AIChatMessage | None:
        if not self.messages or len(self.messages) <= 1 or self.messages[-1].role == 'user':
            return None
        redo_index = -1
        for i in range(len(self.messages)):
            index = len(self.messages) - i - 1
            message = self.messages[index]
            if message.role == 'user':
                redo_index = index
                break
        if redo_index != -1:
            self.messages = self.messages[:redo_index + 1]
            response = await chat(messages=self.messages, model=self.model, options=self.options)
            return response
        return None # failed to redo

    def update_options(self, options:Options | dict = None, num_ctx:int = None, temperature:float = None, repeat_penalty:float = None, repeat_last_n:int = None, num_predict:int = None, top_k:int = None, top_p:float = None, min_p:float = None, mirostat:int = None, mirostat_eta:float = None, mirostat_tau:float = None) -> Options:
        """Update multiple options at once. Accepts a dict and/or individual parameters. Only non-None values are applied. Returns current options."""
        if options:
            for key, value in options.items():
                if value is not None:
                    self.options[key] = value
        local_opts = {
            'num_ctx': num_ctx,
            'temperature': temperature,
            'repeat_penalty': repeat_penalty,
            'repeat_last_n': repeat_last_n,
            'num_predict': num_predict,
            'top_k': top_k,
            'top_p': top_p,
            'min_p': min_p,
            'mirostat': mirostat,
            'mirostat_eta': mirostat_eta,
            'mirostat_tau': mirostat_tau
        }
        for key, value in local_opts.items():
            if value is not None:
                self.options[key] = value
        return self.options

    async def chat(self, user_prompt:str | dict | ChatMessage = None, model:str = None, options:Options | dict = None, think:bool = None, tools:list[Callable] = None, tool_model:str = None, tool_router_system_prompt:str = None, tool_router_user_prompt:str = None, chat_format = None, keep_alive:float | str = '15m') -> AIChatMessage | None:
        """Send a message and get a response. Does not modify conversation history."""
        messages = self.messages.copy()

        if user_prompt:
            if isinstance(user_prompt, str):
                user_prompt = ChatMessage(role='user', content=user_prompt)
            elif isinstance(user_prompt, dict):
                user_prompt = ChatMessage(message_dict=user_prompt)
            messages.append(user_prompt)

        return await chat(messages=messages, model=model or self.model, options=options or self.options, think=think, tools=tools, tool_model=tool_model, tool_router_system_prompt=tool_router_system_prompt, tool_router_user_prompt=tool_router_user_prompt, chat_format=chat_format, keep_alive=keep_alive)




DEFAULT_TOOL_ROUTER_SYSTEM_PROMPT = """You are a tool-routing assistant. Given a prior user message and an assistant response, choose the single most appropriate tool to call based on the assistant's response.

Only call a tool if the assistant is clearly performing that action right now in their response. If no tool is appropriate, call `do_nothing`.

Do not invent actions the assistant did not state. Be literal."""

DEFAULT_TOOL_ROUTER_USER_PROMPT = "Based on the messages above, choose the single most relevant tool to call. If none apply, call `do_nothing`."


def do_nothing(reason:str, **kwargs) -> bool:
    """
    No action is being performed right now.

    Call this when the AI is not currently performing any action - just talking, asking, refusing, or chatting.

    Args:
        reason: Why no action applies
    """
    return True

async def chat(messages:list[ChatMessage|dict], model:str = None, options:Options | dict = None, think:bool = None, tools:list[Callable] = None, tool_model:str = None, tool_router_system_prompt:str = None, tool_router_user_prompt:str = None, chat_format = None, keep_alive:float | str = None) -> AIChatMessage | None:
    model = model or config.get('default_model')

    # When using tool_model, automatically include do_nothing if not present
    if tools and tool_model:
        tool_names = [f.__name__ for f in tools]
        if 'do_nothing' not in tool_names:
            tools = [do_nothing] + list(tools)

    tools_map = {f.__name__: f for f in tools} if tools else None

    if tools and tool_model:
        # Two-step flow: chat model for response, tool_model for tool selection
        response = await _ollama_client.chat(messages=messages, model=model, options=options, think=think, format=chat_format, keep_alive=keep_alive)
        if not response:
            return None

        ai_message = AIChatMessage(response)

        last_user_content = ""
        for m in reversed(messages):
            role = m.get('role') if isinstance(m, dict) else getattr(m, 'role', None)
            if role == 'user':
                last_user_content = m.get('content') if isinstance(m, dict) else getattr(m, 'content', '')
                break

        tools_payload = [
            ChatMessage('system', tool_router_system_prompt or DEFAULT_TOOL_ROUTER_SYSTEM_PROMPT),
            ChatMessage('user', f'User: {last_user_content}'),
            ChatMessage('assistant', f'Assistant: {ai_message.content}'),
            ChatMessage('user', tool_router_user_prompt or DEFAULT_TOOL_ROUTER_USER_PROMPT),
        ]

        tool_options = Options(options.copy() if options else {})
        tool_options.temperature = 0

        tool_response = await _ollama_client.chat(messages=tools_payload, model=tool_model, options=tool_options, tools=tools, think=False, keep_alive=keep_alive)

        if tool_response:
            ai_tool_message = AIChatMessage(tool_response, tools_map=tools_map)
            if ai_tool_message.tool_calls:
                ai_message.tool_calls = ai_tool_message.tool_calls

        return ai_message
    else:
        response = await _ollama_client.chat(messages=messages, model=model, options=options, think=think, tools=tools, format=chat_format, keep_alive=keep_alive)
        if not response:
            return None
        return AIChatMessage(response, tools_map=tools_map)


async def ollama_pull(model:str, insecure: bool = False):
    return await _ollama_client.pull(model=model, insecure=insecure)


async def ollama_list() -> list[dict]:
    raw_models = dict(await _ollama_client.list()).get('models')
    models = [{"model":m["model"], "modified_at":m["modified_at"], "size":m["size"], "details":dict(m["details"])} for m in raw_models]
    return models


async def ollama_show(model:str|Model) -> dict:
    raw_model = await _ollama_client.show(model)
    model_json = raw_model.model_dump_json(indent=4)
    model = json.loads(model_json)
    return model


async def ollama_ps() -> list[dict]:
    raw_models = dict(await _ollama_client.ps()).get('models')
    models = [{"model":m["model"], "name":m["name"], "expires_at":m["expires_at"], "size":m["size"], "size_vram":m["size_vram"], "details":dict(m["details"])} for m in raw_models]
    return models


async def ollama_unload(model:str|Model|None = None):
    if model and not isinstance(model, str):
        return
    # Unload all models, or only specified model
    loaded_models = [m['model'] for m in (await ollama_ps() or [])]
    for m in loaded_models:
        if not model or m == model:
            process = await asyncio.create_subprocess_exec("ollama", "stop", m)
            await process.wait()
# Alias
async def ollama_stop(model:str|Model|None = None):
    return await ollama_unload(model=model)


if __name__ == '__main__':
    pass