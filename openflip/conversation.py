"""Per-channel conversation state. Thin wrapper on ollama_api.Conversation that adds:
- per-channel agent reference
- attachment-aware preprocessing (turns Discord attachments into URLs the model can pass to tools)
- persistent conversations (save/load to disk, full history on disk)

Persistence helpers shared with AnthropicConversation live in `_conversation_io`.
"""
from __future__ import annotations
import os
import sys
import time

# ollama_api is vendored at openflip/ollama_api/. Used to be an external
# sibling repo with sys.path manipulation; now it ships in-tree so the public
# repo + Mac deployment work without a separate install.
from .ollama_api import Conversation as _BaseConversation, Options
from .agent import Agent
from .utils import print_ts, COLOR_YELLOW, COLOR_END
from . import _conversation_io as _cio
from .config_global import get_model_context_window


def _ollama_options_with_context(agent: Agent) -> dict:
    """Merge agent.ollama_options with the model's configured context window.

    config.json's `models.<name>.context_window` is the single source of
    truth for context size. If the agent already set num_ctx in its
    ollama_options, that wins (per-agent override). Otherwise we inject
    the model's configured window.
    """
    opts = dict(agent.ollama_options or {})
    if "num_ctx" not in opts:
        cw = get_model_context_window(agent.model, "ollama")
        if cw and cw > 0:
            opts["num_ctx"] = cw
    return opts


def _msg_role(m):
    return m.get("role") if hasattr(m, "get") else getattr(m, "role", None)


def _msg_content(m):
    ct = getattr(m, "content_text", None)
    if ct is not None:
        return ct
    return m.get("content", "") if hasattr(m, "get") else getattr(m, "content", "")


class DiscordConversation(_BaseConversation):
    def __init__(self, conversation_id: str, agent: Agent):
        super().__init__(
            conversation_id=conversation_id,
            model=agent.model,
            system_message=agent.system_message,
            options=Options(_ollama_options_with_context(agent)),
        )
        self.agent = agent
        self._persisted_count = 0
        # Latest per-turn token usage from ollama_api response. Populated
        # after every chat() call. /status reads this; field names match
        # the anthropic provider's `last_usage` shape so the UI is uniform.
        self.last_usage: dict | None = None

    async def chat(self, *args, **kwargs):
        """Wrap super().chat() to capture prompt_eval_count / eval_count
        from ollama's response into self.last_usage for /status."""
        # DELIBERATE divergence from AnthropicConversation (trigger timing):
        # we trim EVERY turn, pre-flight. The anthropic provider only trims on
        # a 400-retry (anthropic_conversation.py:1621/2439, gated on
        # `_retry_budget is not None`) because Anthropic runs server-side
        # compaction that bounds context in healthy operation, leaving its
        # local trim as a last-resort backstop. Ollama has NO server-side
        # compaction, so this local trim IS the primary context-bounding
        # mechanism and must run on every request. See _trim_to_fit_window
        # below for the matching estimator/target divergences. Intentional —
        # do not "align" this to the anthropic gate.
        dropped = self._trim_to_fit_window()
        if dropped:
            print_ts(
                f"{COLOR_YELLOW}pre-flight trim: dropped {dropped} oldest message(s) to fit context window{COLOR_END}",
                agent=self.agent.id,
            )
        response = await super().chat(*args, **kwargs)
        if response is not None:
            prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
            output_tokens = getattr(response, "eval_count", 0) or 0
            self.last_usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,  # ollama has no caching
                "cache_read_input_tokens": 0,
                "total_input": prompt_tokens,
                "ts": time.time(),
            }
        return response

    def _agent_dir(self) -> str:
        return os.path.dirname(self.agent.path)

    def _conversation_path(self) -> str:
        return _cio.conversation_path(self._agent_dir(), self.conversation_id)

    def load(self):
        """Load conversation history from disk into memory."""
        _cio.migrate_legacy_to_jsonl(
            self._agent_dir(), self.conversation_id,
            log_agent_id=self.agent.id,
        )
        msgs = _cio.read_all_messages(self._conversation_path())
        if not msgs:
            return
        from .ollama_api import ChatMessage
        # NO count-cap on load — operator directive. Full disk history goes
        # into memory; context is bounded by token-budget compaction only.
        # The _trim_to_fit_window safety net handles real overflow.
        for entry in msgs:
            self.messages.append(ChatMessage(entry["role"], entry.get("content", "")))
        self._persisted_count = len(msgs)
        print_ts(f"Loaded {len(msgs)} messages from disk", agent=self.agent.id)

    def save(self):
        """Append new messages to disk in JSONL — no full-file rewrite."""
        non_system = [m for m in self.messages if _msg_role(m) != "system"]
        new_count = len(non_system) - self._persisted_count
        if new_count <= 0:
            self._persisted_count = len(non_system)
            return
        _cio.append_messages(
            self._conversation_path(),
            non_system[-new_count:],
            content_extractor=_msg_content,
        )
        self._persisted_count = len(non_system)

    def _trim_to_fit_window(self) -> int:
        """Pre-flight: drop oldest non-system messages until estimated input
        fits in (num_ctx - 10k). Ollama silently truncates oversized inputs
        so this isn't strictly needed for correctness — but keeping the head
        intact avoids the model losing recent context to its own internal
        eviction.

        DELIBERATE sibling divergence from AnthropicConversation._trim_to_fit_window
        (anthropic_conversation.py:1225). The two trims differ in three ways,
        ALL intentional and justified by ollama lacking three Anthropic
        features — do NOT "fix" one to match the other:

          1. Estimator //4 (here) vs //2 (anthropic). Anthropic hard-400s when
             the prompt overflows the window, so it OVER-estimates (//2) to
             trim conservatively and stay clear of the 400. Ollama instead
             SILENTLY truncates an oversized input (no error), so under-
             estimating with //4 is harmless: the worst case is ollama evicting
             a little internally, not a failed request. //4 ≈ English-prose
             chars-per-token; //2 ≈ dense code/JSON. We can afford the looser
             estimate precisely because the downside here is benign.

          2. Trims down to `budget` (here) vs `budget*0.8` (anthropic). The
             anthropic 20% headroom exists to keep its cached prompt prefix
             stable across turns (trimming to the exact budget would rotate the
             head every turn and bust the cache). Ollama has NO prompt caching
             (see last_usage: cache_* fields hardcoded 0 above), so headroom
             buys nothing — trimming to the exact budget is correct here.

          3. Fires EVERY turn (call site in chat() above) vs only on 400-retry
             (anthropic). Justified there because Anthropic has server-side
             compaction; we have none. See the call-site comment above.
        """
        window = get_model_context_window(self.model, "ollama")
        if not window:
            return 0
        budget = window - 10_000
        if budget <= 0:
            return 0

        def _est(s: str) -> int:
            # //4 (not anthropic's //2) — deliberate: ollama silently truncates
            # overflow rather than 400'ing, so an under-estimate is benign here.
            # See class-level divergence note in this method's docstring.
            return len(s or "") // 4

        sys_cost = _est(self.system_message)
        msg_cost = sum(_est(_msg_content(m)) for m in self.messages)
        total = sys_cost + msg_cost

        if total <= budget:
            return 0

        original = len(self.messages)
        i = 0
        # Trim down to the exact `budget` (anthropic trims to budget*0.8 for
        # cache-prefix stability — irrelevant here, ollama has no caching).
        while total > budget and i < len(self.messages):
            if _msg_role(self.messages[i]) == "system":
                i += 1
                continue
            total -= _est(_msg_content(self.messages[i]))
            del self.messages[i]
        dropped = original - len(self.messages)
        # Keep persisted_count consistent with the shorter in-memory list, so
        # the next save() doesn't fall into its `new_count <= 0` early-return
        # and silently skip persisting this turn's user/assistant messages.
        self._persisted_count = max(0, self._persisted_count - dropped)
        return dropped

    def clear_history(self):
        """Delete the on-disk conversation file."""
        _cio.delete_conversation_files(self._agent_dir(), self.conversation_id, backup_tag="pre_reset")
        self._persisted_count = 0

    def reapply_agent(self):
        """Re-pull the system message + model + options from the agent (e.g. after reload)."""
        self.model = self.agent.model
        self.system_message = self.agent.system_message
        self.options = Options(_ollama_options_with_context(self.agent))
        if self.messages and self.messages[0].role == 'system':
            self.messages[0]['content'] = self.system_message
        else:
            self.append_system_message(self.system_message)
