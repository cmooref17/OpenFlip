"""claude_code provider: drive the unmodified Claude Code binary through the
Claude Agent SDK.

Why this exists: the `anthropic` provider imitates Claude Code over raw HTTP
(its OAuth token, user agent, beta flags and billing block). This provider
runs the real binary instead — whatever `claude` login the box has is used by
Claude Code itself; OpenFlip never reads, stores or refreshes a credential.

Shape: mirrors AnthropicConversation's public surface (ctor, load/save,
messages, system_message, chat/chat_stream, clear_history, reapply_agent,
SessionOverridesMixin) so runtime.py, commands.py and text_commands.py keep
working. The differences are structural:

  * Claude Code owns the transcript and the tool loop. One ClaudeSDKClient per
    OpenFlip conversation, resumed by session id (persisted in the .meta.json
    sidecar). `self.messages` is a read model mirrored into the usual JSONL
    for the dashboard, /status, /undo and provider migration — not the thing
    sent to the API.
  * Tools are served to Claude Code as an in-process MCP server
    (claude_code_tools) and executed through tool_executor exactly as on the
    other providers. chat() therefore returns with tool_calls=[] and the
    final text; runtime's tool loop runs once per turn.
  * Messages runtime appends to `self.messages` after the last turn (the
    framed user text, [FRAMEWORK] notes, nudges, inject_context plants) are
    delivered as the next prompt. `_cc_synced` tracks how much of the mirror
    Claude Code has already seen.
  * Options that change the Claude Code process (system prompt, model, effort,
    thinking, tool set) reconnect the client with `resume`, so hot-reloaded
    SOUL/AGENT/TOOLS files and /model, /effort take effect on the next turn.

Config (config.json `claude_code` block, all optional):
  cli_path             path to a specific `claude` binary (default: the SDK's bundled one)
  idle_disconnect_s    seconds a conversation's client may sit idle (default 600)
  max_buffer_size      bytes for one CLI message (default 50 MB; big tool results)
  migrate_history_chars  chars of mirrored history replayed into a NEW Claude
                       Code session when a conversation already has JSONL
                       history but no session (provider migration; default 60000)
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import time
from typing import Any, AsyncIterator, Callable, Optional

from .agent import Agent
from .session_overrides import SessionOverridesMixin
from .utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END, load_json, save_json
from . import _conversation_io as _cio
from .anthropic_conversation import ChatMessage, AnthropicAIChatMessage, AnthropicToolCall
from ._anthropic_stream import (
    MessageStartEvent, ContentBlockStartEvent, ContentBlockDeltaEvent,
    ContentBlockStopEvent, MessageDeltaEvent, MessageStopEvent, FrameworkErrorEvent,
    _dispatch_event,
)
from .claude_code_tools import TurnContext, build_mcp_server, full_tool_name, bare_tool_name
from .claude_code_hooks import build_hooks

_DEFAULT_IDLE_DISCONNECT_S = 600
_DEFAULT_MAX_BUFFER = 50 * 1024 * 1024
_DEFAULT_MIGRATE_CHARS = 60_000
_SYSTEM_REMINDER_NOTE = (
    "The application adds system reminders and [FRAMEWORK] notes to this conversation. "
    "Treat them as context from the application, not as messages from the user."
)
_CC_BUILTIN_CONTEXT_OFF = {
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}
_CC_SETTINGS = '{"includeGitInstructions": false, "attribution": {"commit": "", "pr": ""}}'


def _cc_config() -> dict:
    from .config_global import get_config
    return (get_config() or {}).get("claude_code") or {}


class ClaudeCodeConversation(SessionOverridesMixin):
    """Conversation backed by a Claude Code session (Agent SDK)."""

    _provider_name = "claude_code"

    def __init__(self, conversation_id: str, agent: Agent):
        self.conversation_id = conversation_id
        self.agent = agent
        self._init_overrides()
        self.model = self._normalize_model(agent.model)
        self.system_message = agent.system_message
        self.messages: list[ChatMessage] = []
        self._persisted_count = 0
        self.last_usage: dict | None = None
        # Set by the PreCompact hook during chat(); runtime reads it after.
        self.compacted_this_turn: bool = False
        # /compact sets both (text_commands._do_compact / commands.compact_cmd);
        # the next chat() sends Claude Code's own /compact first.
        self.force_compact_next: bool = False
        self.force_compact_trigger_override: bool = False
        self.retry_notice_cb: Optional[Callable] = None
        self._pending_image_attachments: list[dict] = []
        self._persist_guard = None
        # --- Claude Code session state ---
        self._cc_session_id: str | None = None
        self._cc_synced: int = 0          # messages[:n] already seen by Claude Code
        self._client = None               # ClaudeSDKClient
        self._client_sig: tuple | None = None
        self._client_tools: list[str] = []
        self._idle_task: asyncio.Task | None = None
        self._cc_turn: TurnContext | None = None
        self._prompt_path: str | None = None

    # ------------------------------------------------------------ naming
    @staticmethod
    def _normalize_model(model_str: str) -> str:
        m = model_str or ""
        if "/" in m:
            m = m.split("/", 1)[1]
        if m.endswith("-1m"):
            m = m[:-3]
        return m

    def _wants_1m_context(self) -> bool:
        return (self._effective_raw_model() or "").endswith("-1m")

    def _agent_dir(self) -> str:
        return os.path.dirname(self.agent.path)

    def _conversation_path(self) -> str:
        return _cio.conversation_path(self._agent_dir(), self.conversation_id)

    def _meta_path(self) -> str:
        return os.path.join(
            self._agent_dir(), "conversations", f"{_cio.fs_encode(self.conversation_id)}.meta.json"
        )

    def _content_extractor(self, m) -> str:
        return getattr(m, "content_text", None) or m.get("content", "") or ""

    # ------------------------------------------------------------ persistence
    def _save_meta(self):
        payload: dict = {}
        if self.last_usage is not None:
            payload["last_usage"] = self.last_usage
        if self._cc_session_id:
            payload["claude_code"] = {"session_id": self._cc_session_id, "synced": self._cc_synced}
        _ov = self.overrides_meta_payload()
        if _ov:
            payload["overrides"] = _ov
        if not payload:
            try:
                if os.path.isfile(self._meta_path()):
                    save_json(self._meta_path(), payload)
            except OSError:
                pass
            return
        save_json(self._meta_path(), payload)

    def load(self):
        _cio.migrate_legacy_to_jsonl(self._agent_dir(), self.conversation_id, log_agent_id=self.agent.id)
        msgs = _cio.read_all_messages(self._conversation_path())
        meta = load_json(self._meta_path(), default={}) or {}
        stored_usage = meta.get("last_usage")
        if isinstance(stored_usage, dict):
            self.last_usage = stored_usage
        self.load_overrides_from_meta(meta)
        cc = meta.get("claude_code") or {}
        if isinstance(cc, dict) and cc.get("session_id"):
            self._cc_session_id = str(cc["session_id"])
            try:
                self._cc_synced = max(0, min(int(cc.get("synced", 0) or 0), len(msgs)))
            except (TypeError, ValueError):
                self._cc_synced = 0
        if not msgs:
            return
        for entry in msgs:
            self.messages.append(ChatMessage(entry["role"], entry.get("content", "")))
        self._persisted_count = len(msgs)
        print_ts(f"Loaded {len(msgs)} messages from disk (claude_code)", agent=self.agent.id)

    def save(self):
        non_system = [m for m in self.messages if m.get("role") != "system"]
        new_count = len(non_system) - self._persisted_count
        if new_count > 0:
            _cio.append_messages(
                self._conversation_path(), non_system[-new_count:],
                content_extractor=self._content_extractor,
            )
        self._persisted_count = len(non_system)
        try:
            self._save_meta()
        except Exception:
            pass

    def clear_history(self):
        _cio.delete_conversation_files(
            self._agent_dir(), self.conversation_id,
            extra_paths=[self._meta_path()], backup_tag="pre_reset",
        )
        self._persisted_count = 0
        self.last_usage = None
        self._cc_session_id = None
        self._cc_synced = 0
        self._drop_client()

    def reapply_agent(self):
        self.model = self._normalize_model(self._effective_raw_model())
        self.system_message = self.agent.system_message

    def on_history_cut(self):
        """runtime.undo_last_turn cut the mirror. Claude Code's transcript no
        longer matches, so the next turn starts a fresh session seeded with
        the surviving history (see _migration_preamble)."""
        print_ts("claude_code: history cut (/undo) — next turn reseeds a new session from the mirror",
                 agent=self.agent.id)
        self._cc_session_id = None
        self._cc_synced = 0
        self._drop_client()
        try:
            self._save_meta()
        except Exception:
            pass

    async def aclose(self):
        await self._disconnect()

    # ------------------------------------------------------------ compat shims
    def effective_max_tokens(self) -> int:
        from .config_global import get_max_tokens
        return get_max_tokens(self._effective_raw_model(), "anthropic")

    def effective_compaction_trigger(self) -> int:
        from .config_global import get_compaction_trigger
        return get_compaction_trigger(self._effective_raw_model(), "anthropic")

    def will_compact_this_turn(self) -> bool:
        return bool(self.force_compact_next)

    # ------------------------------------------------------------ client
    def _system_prompt_text(self) -> str:
        # runtime._run_turn already folds the memory index block and any
        # per-turn extension into self.system_message before chat().
        parts = [self.system_message or "", _SYSTEM_REMINDER_NOTE]
        return "\n\n".join(p for p in parts if p)

    def _thinking_config(self, think: bool | None) -> dict | None:
        if think is None:
            think = self.agent.think
        if think is None:
            return None
        return {"type": "adaptive"} if think else {"type": "disabled"}

    def _build_options(self, prompt_text: str, tool_names: list[str], think: bool | None, *, resume: str | None):
        from claude_agent_sdk import ClaudeAgentOptions
        from .config_global import get_max_tool_turns
        cfg = _cc_config()
        env = dict(_CC_BUILTIN_CONTEXT_OFF)
        opts: dict[str, Any] = dict(
            model=self.model,
            system_prompt={"type": "custom", "prompt": prompt_text, "snapshot": False},
            tools=[],
            mcp_servers={"openflip": build_mcp_server(self, tool_names)},
            strict_mcp_config=True,
            allowed_tools=[full_tool_name(n) for n in tool_names],
            permission_mode="bypassPermissions",
            setting_sources=[],
            env=env,
            settings=_CC_SETTINGS,
            cwd=self._agent_dir(),
            include_partial_messages=True,
            max_turns=get_max_tool_turns(),
            max_buffer_size=int(cfg.get("max_buffer_size") or _DEFAULT_MAX_BUFFER),
            hooks=build_hooks(self),
            stderr=self._on_stderr,
        )
        effort = self.effective_effort()
        if effort:
            opts["effort"] = effort
        thinking = self._thinking_config(think)
        if thinking:
            opts["thinking"] = thinking
        if self._wants_1m_context():
            opts["betas"] = ["context-1m-2025-08-07"]
        cli_path = str(cfg.get("cli_path") or "").strip()
        if cli_path:
            opts["cli_path"] = cli_path
        if resume:
            opts["resume"] = resume
        return ClaudeAgentOptions(**opts)

    def _on_stderr(self, line: str) -> None:
        line = (line or "").rstrip()
        if line:
            print_ts(f"claude_code stderr: {line[:400]}", agent=self.agent.id)

    def _client_signature(self, prompt_text: str, tool_names: list[str], think: bool | None) -> tuple:
        # The session id is deliberately NOT part of the signature: the client
        # that opened a new session already owns it, and resume only matters
        # when something else forces a reconnect.
        return (
            prompt_text, self.model, self.effective_effort() or "", json.dumps(self._thinking_config(think) or {}, sort_keys=True),
            self._wants_1m_context(), tuple(tool_names),
        )

    async def _ensure_client(self, tool_names: list[str], think: bool | None):
        from claude_agent_sdk import ClaudeSDKClient
        prompt_text = self._system_prompt_text()
        sig = self._client_signature(prompt_text, tool_names, think)
        if self._client is not None and sig == self._client_sig:
            return self._client
        await self._disconnect()
        opts = self._build_options(prompt_text, tool_names, think, resume=self._cc_session_id)
        client = ClaudeSDKClient(opts)
        t0 = time.monotonic()
        await client.connect()
        self._client = client
        self._client_sig = sig
        self._client_tools = list(tool_names)
        print_ts(
            f"claude_code: client connected in {time.monotonic() - t0:.2f}s "
            f"({'resume ' + self._cc_session_id[:8] if self._cc_session_id else 'new session'}, "
            f"{len(tool_names)} tools, model {self.model})",
            agent=self.agent.id,
        )
        return client

    async def _disconnect(self):
        self._cancel_idle()
        client, self._client, self._client_sig = self._client, None, None
        if client is not None:
            try:
                await asyncio.wait_for(client.disconnect(), timeout=15)
            except Exception as e:
                print_ts(f"claude_code: disconnect error (ignored): {e}", agent=self.agent.id)

    def _drop_client(self):
        """Sync variant for callers that can't await (clear_history)."""
        client, self._client, self._client_sig = self._client, None, None
        self._cancel_idle()
        if client is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._safe_disconnect(client))

    async def _safe_disconnect(self, client):
        try:
            await asyncio.wait_for(client.disconnect(), timeout=15)
        except Exception:
            pass

    def _cancel_idle(self):
        t, self._idle_task = self._idle_task, None
        if t is not None and not t.done():
            t.cancel()

    def _schedule_idle_disconnect(self):
        self._cancel_idle()
        secs = _cc_config().get("idle_disconnect_s", _DEFAULT_IDLE_DISCONNECT_S)
        try:
            secs = float(secs)
        except (TypeError, ValueError):
            secs = _DEFAULT_IDLE_DISCONNECT_S
        if secs <= 0:
            return

        async def _idle():
            try:
                await asyncio.sleep(secs)
            except asyncio.CancelledError:
                return
            if self._cc_turn is None:
                print_ts(f"claude_code: idle {int(secs)}s — disconnecting client", agent=self.agent.id)
                await self._disconnect()

        self._idle_task = asyncio.get_running_loop().create_task(_idle())

    # ------------------------------------------------------------ prompt assembly
    def _pending_messages(self) -> list[ChatMessage]:
        """Mirror entries Claude Code hasn't seen: everything appended since the
        last turn (user text, [FRAMEWORK] notes, nudges, inject_context)."""
        if len(self.messages) < self._cc_synced:
            # A nudge runtime appended and popped again, or a cut it already
            # told us about via on_history_cut(). Just realign.
            self._cc_synced = len(self.messages)
        return [m for m in self.messages[self._cc_synced:]
                if m.get("role") not in ("system", "assistant")]

    def _migration_preamble(self) -> str:
        """When a conversation has JSONL history but no Claude Code session yet
        (agent switched provider, or /undo reseed), replay the tail as text so
        the agent keeps continuity."""
        cap = int(_cc_config().get("migrate_history_chars") or _DEFAULT_MIGRATE_CHARS)
        if cap <= 0:
            return ""
        lines: list[str] = []
        total = 0
        for m in reversed([m for m in self.messages[:self._cc_synced] if m.get("role") != "system"]):
            role = m.get("role", "")
            text = (self._content_extractor(m) or "").strip()
            if not text:
                continue
            line = f"[{role}]: {text}"
            if total + len(line) > cap:
                break
            lines.append(line)
            total += len(line) + 1
        if not lines:
            return ""
        lines.reverse()
        return ("[FRAMEWORK]: Earlier history of this conversation (migrated from a previous session; "
                "oldest first):\n" + "\n".join(lines) + "\n[FRAMEWORK]: end of migrated history.")

    def _compose_prompt(self) -> tuple[str, list[dict]]:
        pending = self._pending_messages()
        parts: list[str] = []
        if not self._cc_session_id and self._cc_synced == 0 and len(self.messages) > len(pending):
            # First turn on Claude Code for a conversation with prior history.
            self._cc_synced = len(self.messages) - len(pending)
            pre = self._migration_preamble()
            if pre:
                parts.append(pre)
        for m in pending:
            role = m.get("role", "user")
            text = self._content_extractor(m) or ""
            if not text.strip():
                continue
            if role == "tool":
                parts.append(f"[Previous tool result: {text}]")
            else:
                parts.append(text)
        images = list(self._pending_image_attachments or [])
        self._pending_image_attachments = []
        return "\n\n".join(parts), images

    @staticmethod
    def _image_blocks(images: list[dict]) -> list[dict]:
        blocks = []
        for entry in images:
            path = entry.get("path")
            if not path:
                continue
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
            except OSError as e:
                print_ts(f"{COLOR_YELLOW}claude_code: image read failed ({path}): {e}{COLOR_END}")
                continue
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": entry.get("content_type") or "image/png", "data": data,
            }})
        return blocks

    # ------------------------------------------------------------ chat
    async def _run_query(self, tools: list[Callable] | None, think: bool | None):
        """Send the pending prompt and yield every SDK message of the response.
        Handles /compact, session bookkeeping, idle timer and interrupt."""
        tool_names = [f.__name__ for f in (tools or [])]
        client = await self._ensure_client(tool_names, think)
        prompt_text, images = self._compose_prompt()
        if not prompt_text.strip() and not images:
            prompt_text = "(continue)"
        self.compacted_this_turn = False
        try:
            if self.force_compact_next:
                self.force_compact_next = False
                self.force_compact_trigger_override = False
                await client.query("/compact")
                async for _m in client.receive_response():
                    pass
                self.compacted_this_turn = True
            if images:
                blocks = [{"type": "text", "text": prompt_text}] + self._image_blocks(images)

                async def _one():
                    yield {"type": "user", "parent_tool_use_id": None, "session_id": "default",
                           "message": {"role": "user", "content": blocks}}
                await client.query(_one())
            else:
                await client.query(prompt_text)
            self._cc_synced = len(self.messages)
            async for m in client.receive_response():
                yield m
        except asyncio.CancelledError:
            # /stop or a runtime timeout cancelled the turn: stop Claude Code
            # too, then drain so the next query starts clean.
            asyncio.get_running_loop().create_task(self._interrupt_and_drain(client))
            raise
        finally:
            self._schedule_idle_disconnect()

    async def _interrupt_and_drain(self, client):
        try:
            await asyncio.wait_for(client.interrupt(), timeout=10)
            async for _m in client.receive_response():
                pass
        except Exception:
            # Can't recover the stream: drop the client; next turn reconnects
            # with resume.
            if self._client is client:
                await self._disconnect()

    async def chat_stream(self, tools: list[Callable] = None, think: bool = None,
                          tool_choice: dict | None = None, _retry_attempt: int = 0):
        """Yield _anthropic_stream events mapped from the SDK's raw API stream
        events. Consumers wanting the final assembled message use chat()."""
        from claude_agent_sdk import StreamEvent, ResultMessage
        content_blocks: dict[int, dict] = {}
        tool_input_buffers: dict[int, str] = {}
        usage_ref: dict = {}
        try:
            async for m in self._run_query(tools, think):
                if isinstance(m, StreamEvent):
                    ev = m.event or {}
                    et = ev.get("type")
                    if et == "message_start":
                        content_blocks.clear(); tool_input_buffers.clear()
                        msg = ev.get("message") or {}
                        yield MessageStartEvent(model=msg.get("model", self.model), usage=msg.get("usage") or {})
                    elif et in ("content_block_start", "content_block_delta"):
                        _dispatch_event(et, ev, content_blocks=content_blocks,
                                        tool_input_buffers=tool_input_buffers, usage_ref=usage_ref)
                        if et == "content_block_start":
                            idx = ev.get("index", 0)
                            blk = content_blocks.get(idx) or {}
                            yield ContentBlockStartEvent(index=idx, block_type=blk.get("type", ""), partial_block=dict(blk))
                        else:
                            yield ContentBlockDeltaEvent(index=ev.get("index", 0), delta=ev.get("delta") or {})
                    elif et == "content_block_stop":
                        idx = ev.get("index", 0)
                        blk = dict(content_blocks.get(idx) or {})
                        if blk.get("type") == "tool_use":
                            try:
                                blk["input"] = json.loads(tool_input_buffers.get(idx) or "{}")
                            except json.JSONDecodeError:
                                blk["input"] = {}
                        yield ContentBlockStopEvent(index=idx, completed_block=blk)
                    elif et == "message_delta":
                        delta = ev.get("delta") or {}
                        yield MessageDeltaEvent(stop_reason=delta.get("stop_reason"),
                                                usage=ev.get("usage") or {}, stop_details=delta.get("stop_details"))
                elif isinstance(m, ResultMessage):
                    err = self._result_error(m)
                    if err:
                        yield FrameworkErrorEvent(message=err[1], kind=err[0])
                        return
            yield MessageStopEvent()
        except FrameworkErrorEvent as fe:  # pragma: no cover - defensive
            yield fe
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield FrameworkErrorEvent(message=f"claude_code: {type(e).__name__}: {e}", kind=self._error_kind(e))

    @staticmethod
    def _error_kind(e: Exception) -> str:
        name = type(e).__name__
        if name in ("CLINotFoundError", "CLIConnectionError", "ProcessError", "CLIJSONDecodeError"):
            return "transport"
        return "transport"

    @staticmethod
    def _result_error(result) -> tuple[str, str] | None:
        """(kind, message) when a ResultMessage signals failure, else None."""
        text = (result.result or "") if isinstance(getattr(result, "result", None), str) else ""
        errors = getattr(result, "errors", None) or []
        if not result.is_error and result.subtype == "success":
            return None
        joined = text or "; ".join(str(x) for x in errors) or result.subtype
        low = joined.lower()
        if "authenticate" in low or "login" in low or "oauth" in low:
            return ("auth", joined)
        if "rate limit" in low or "429" in low:
            return ("rate_limit", joined)
        if "overloaded" in low or "529" in low:
            return ("overloaded", joined)
        if result.subtype in ("error_max_turns",):
            return ("bad_request", f"Claude Code hit max_turns: {joined}")
        if result.subtype == "error_max_budget_usd":
            return ("bad_request", joined)
        return ("transport", joined)

    async def chat(self, tools: list[Callable] = None, think: bool = None,
                   _retry_attempt: int = 0, tool_choice: dict | None = None) -> AnthropicAIChatMessage:
        """Run one turn on Claude Code. Returns the FINAL assistant text with
        tool_calls=[] (tools already ran inside the session). Intermediate
        rounds (text + tool use + results) are mirrored into self.messages."""
        from claude_agent_sdk import (
            AssistantMessage, UserMessage, SystemMessage, ResultMessage,
            TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock,
        )
        turn = self._cc_turn
        final_text_parts: list[str] = []
        thinking_parts: list[str] = []
        last_round_usage: dict | None = None
        last_model = self.model
        stop_reason: str | None = None
        result: Any = None
        error: FrameworkErrorEvent | None = None
        pending_round: dict | None = None   # assistant round awaiting its tool results
        mirrored_tools: dict[str, dict] = {}

        def _flush_round():
            nonlocal pending_round
            if pending_round is None:
                return
            text = "".join(pending_round["text"])
            calls = pending_round["calls"]
            if calls:
                am = AnthropicAIChatMessage(text)
                am.tool_calls = calls
                self.messages.append(am)
                for tc in calls:
                    r = mirrored_tools.pop(tc.tool_use_id, None)
                    tm = ChatMessage("tool", (r or {}).get("content", "(no result)"))
                    tm["tool_use_id"] = tc.tool_use_id
                    self.messages.append(tm)
            pending_round = None

        try:
            async for m in self._run_query(tools, think):
                if isinstance(m, SystemMessage):
                    if m.subtype == "init":
                        sid = (m.data or {}).get("session_id")
                        if sid and sid != self._cc_session_id:
                            self._cc_session_id = str(sid)
                    continue
                if isinstance(m, AssistantMessage):
                    if m.usage:
                        last_round_usage = dict(m.usage)
                    last_model = m.model or last_model
                    texts, calls = [], []
                    for b in m.content:
                        if isinstance(b, TextBlock):
                            texts.append(b.text)
                        elif isinstance(b, ThinkingBlock):
                            thinking_parts.append(b.thinking)
                        elif isinstance(b, ToolUseBlock):
                            calls.append(AnthropicToolCall(bare_tool_name(b.name), dict(b.input or {}), tool_use_id=b.id))
                    if calls:
                        # Round with tool use: text that accompanied the calls
                        # posts now (runtime's intra-loop post), mirror the round
                        # once its results arrive.
                        _flush_round()
                        pending_round = {"text": texts, "calls": calls}
                        joined = "".join(texts).strip()
                        if joined and turn is not None and turn.auto_post_final_text and not turn.silent:
                            if await self._post_text(turn, joined):
                                turn.posted_text = True
                        # Any text from earlier tool-free rounds belongs before
                        # this round in the mirror, not in the final reply.
                        if final_text_parts:
                            self.messages.append(AnthropicAIChatMessage("".join(final_text_parts)))
                            final_text_parts = []
                    else:
                        final_text_parts.extend(texts)
                    continue
                if isinstance(m, UserMessage):
                    if isinstance(m.content, list):
                        for b in m.content:
                            if isinstance(b, ToolResultBlock):
                                mirrored_tools[b.tool_use_id] = {"content": self._tool_result_text(b.content)}
                    _flush_round()
                    continue
                if isinstance(m, ResultMessage):
                    result = m
                    stop_reason = m.stop_reason
                    sid = m.session_id
                    if sid and sid != self._cc_session_id:
                        self._cc_session_id = str(sid)
                    err = self._result_error(m)
                    if err:
                        error = FrameworkErrorEvent(message=err[1], kind=err[0])
                    continue
        except asyncio.CancelledError:
            _flush_round()
            raise
        except Exception as e:
            print_ts(f"{COLOR_RED}claude_code: {type(e).__name__}: {e}{COLOR_END}", error=True, agent=self.agent.id)
            await self._disconnect()
            error = FrameworkErrorEvent(message=f"claude_code: {type(e).__name__}: {e}", kind=self._error_kind(e))
        _flush_round()
        self._cc_synced = len(self.messages)

        self._record_usage(result, last_round_usage, last_model, failed=error is not None)
        try:
            self._save_meta()
        except Exception:
            pass

        if error is not None:
            msg = AnthropicAIChatMessage(error.message, is_framework_error=True)
            msg.framework_error_kind = error.kind
            msg.done_reason = stop_reason or "error"
            return msg
        out = AnthropicAIChatMessage("".join(final_text_parts))
        out.thinking = "".join(thinking_parts) or None
        out.done_reason = stop_reason or "end_turn"
        out.raw_response = {"session_id": self._cc_session_id, "stop_reason": stop_reason,
                            "num_turns": getattr(result, "num_turns", None)}
        return out

    @staticmethod
    def _tool_result_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "".join(out)

    async def _post_text(self, turn: TurnContext, text: str) -> bool:
        """Post intermediate assistant text (text that accompanied tool calls)
        the way runtime posts it between rounds."""
        try:
            from .tool_executor import _send_text
            from .registry import RUNNERS
            runner = RUNNERS.get(self.agent.id)
            transport = getattr(runner, "transport", None) if runner else None
            try:
                from .discord_io import split_for_discord as _split
            except Exception:
                _split = None
            chunks = _split(text) if callable(_split) else [text]
            for chunk in chunks:
                await _send_text(transport, turn.session_id, turn.channel, chunk)
            return True
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}claude_code: intermediate text post failed: {e}{COLOR_END}", agent=self.agent.id)
            return False

    def _record_usage(self, result, last_round_usage: dict | None, model: str, *, failed: bool):
        usage = dict(getattr(result, "usage", None) or {}) if result is not None else {}
        if not usage and last_round_usage:
            usage = dict(last_round_usage)
        if not usage:
            return
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
        # Context size = the LAST request's input (what the next turn will carry).
        ctx = last_round_usage or usage
        total_input = (int(ctx.get("input_tokens", 0) or 0) + int(ctx.get("cache_read_input_tokens", 0) or 0)
                       + int(ctx.get("cache_creation_input_tokens", 0) or 0))
        self.last_usage = {
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_create,
            "total_input": total_input, "partial": failed,
        }
        print_ts(
            f"claude_code usage: in={in_tok} out={out_tok} cache_create={cache_create} "
            f"cache_read={cache_read} ctx={total_input}{' (failed)' if failed else ''}",
            agent=self.agent.id,
        )
        try:
            from . import usage_ledger
            conv_id = self.conversation_id or ""
            if ":" in conv_id:
                transport, channel_id = conv_id.split(":", 1)
            else:
                transport, channel_id = None, (conv_id or None)
            outcome = "error" if failed else ("compaction" if self.compacted_this_turn else "ok")
            raw = dict(usage)
            raw["model_usage"] = getattr(result, "model_usage", None) if result is not None else None
            raw["total_cost_usd"] = getattr(result, "total_cost_usd", None) if result is not None else None
            usage_ledger.record_usage(
                agent_id=self.agent.id, transport=transport, channel_id=channel_id,
                session_id=conv_id or None, user_id=None, user_handle=None,
                model=model or self.model, usage=raw, outcome=outcome,
            )
        except Exception as e:
            print_ts(f"usage_ledger record failed (continuing): {e}", agent=self.agent.id, error=True)


# ---------------------------------------------------------------- recall selector
async def call_selector(model: str, candidates_text: str, query: str) -> str:
    """Memory-recall selector as a one-shot Claude Code query (no tools, one
    turn). Same prompt pieces as memory_recall_api.call_selector; the
    credential is Claude Code's own."""
    from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions, AssistantMessage, TextBlock
    from .memory_recall_prompt import SELECTOR_SYSTEM, SELECTOR_TIMEOUT_S
    prompt = (f"Available memory files:\n{candidates_text}\n\n"
              f"Select memories relevant to:\n{query}\n\n"
              "Answer with JSON only: {\"selected_memories\": [\"<filename>\", ...]}")
    opts = ClaudeAgentOptions(
        model=model or None, tools=[], max_turns=1, setting_sources=[],
        system_prompt=SELECTOR_SYSTEM, env=dict(_CC_BUILTIN_CONTEXT_OFF),
        settings=_CC_SETTINGS, permission_mode="bypassPermissions",
    )
    cli_path = str(_cc_config().get("cli_path") or "").strip()
    if cli_path:
        opts.cli_path = cli_path
    texts: list[str] = []

    async def _go():
        async for m in sdk_query(prompt=prompt, options=opts):
            if isinstance(m, AssistantMessage):
                texts.extend(b.text for b in m.content if isinstance(b, TextBlock))
    await asyncio.wait_for(_go(), timeout=SELECTOR_TIMEOUT_S)
    return "".join(texts)
