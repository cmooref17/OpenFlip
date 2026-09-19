"""Per-conversation (session) settings overrides.

One conversation can carry its own model, context window, compaction
trigger, output cap, reasoning effort, memory-tools switch and (ollama)
sampling options. Every provider conversation class mixes this in, so the
override layer has ONE implementation and the siblings can't drift.

Precedence for every setting (highest wins):

  1. per-turn override      — external ingress token `default_model` / body
                              `model` (model id only; set_model_override)
  2. session override       — this module; `/session set …`, `/effort`, or
                              an ingress token's `session_overrides` block;
                              persisted in the conversation's `.meta.json`
                              under `"overrides"`
  3. per-model config       — config.json `models.<bare>.{context_window,
                              compaction_trigger, max_tokens, effort}`
                              looked up with the EFFECTIVE model (so a
                              session on a different model gets that
                              model's window / trigger / cap automatically)
  4. agent.json             — `model`, `ollama_options`, `memory_enabled`
  5. provider default       — config_global fallbacks

Storage: `self.overrides` is a plain dict of validated, normalized values.
Only keys that are set are present; the meta payload omits the block
entirely when empty so untouched conversations' meta files stay
byte-identical.
"""
from __future__ import annotations

import json
from typing import Any

from .config_global import (
    get_config,
    get_effort,
    get_model_context_window,
    _MAX_TOKENS_CEILING,
    _VALID_EFFORT_LEVELS,
    _VALID_OPENAI_EFFORT_LEVELS,
)

# Anthropic's server-side compaction floor; matches get_compaction_trigger.
_COMPACTION_FLOOR = 50_000
# A window below this is unusable: the local trim budget is `window - 10k`.
_MIN_CONTEXT_WINDOW = 16_000

# Effort vocabularies per provider. OpenAI accepts the union of the Chat
# Completions set and the codex (subscription) set; the request builders map
# the value into whichever vocabulary the active auth path speaks.
_OPENAI_SESSION_EFFORT = ("minimal", "low", "medium", "high", "xhigh")

# Which override keys each provider honors. Anything else is rejected at set
# time with a message naming the valid keys, so a typo can't silently persist.
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("model", "context_window", "compaction_trigger", "max_tokens", "effort", "memory"),
    "openai": ("model", "context_window", "max_tokens", "effort", "memory"),
    "ollama": ("model", "context_window", "options", "memory"),
}

# Meta sidecar key. Legacy `effort_override` (pre-overrides layout) is still
# read on load and migrated into `overrides.effort`; it is never written back.
META_KEY = "overrides"
_LEGACY_EFFORT_KEY = "effort_override"

_TRUE_WORDS = {"true", "on", "yes", "1", "enabled", "enable"}
_FALSE_WORDS = {"false", "off", "no", "0", "disabled", "disable"}


def provider_for_model(model: str) -> str:
    """Infer the provider from a model name. Mirrors agent_ui._provider_for_model
    without importing the nextcord-backed UI module: `openai/…` → openai,
    `claude-…` or `anthropic/…` → anthropic, anything else → ollama."""
    m = (model or "").strip()
    if m.startswith("openai/"):
        return "openai"
    if m.startswith("claude-") or m.startswith("anthropic/"):
        return "anthropic"
    return "ollama"


def _coerce_int(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip().replace(",", "").replace("_", "")
        # Accept 200k / 1m shorthand — the same units operators use in chat.
        mult = 1
        if s[-1:].lower() == "k":
            mult, s = 1_000, s[:-1]
        elif s[-1:].lower() == "m":
            mult, s = 1_000_000, s[:-1]
        try:
            return int(float(s) * mult)
        except ValueError:
            return None
    return None


def _coerce_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _TRUE_WORDS:
            return True
        if s in _FALSE_WORDS:
            return False
    return None


def _coerce_option_value(raw: str) -> Any:
    """`k=v` pair values from a text command: int → float → bool → str."""
    s = raw.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    b = _coerce_bool(s)
    if b is not None:
        return b
    return s


def _coerce_options(raw: Any) -> dict | None:
    """Ollama sampling options: a dict (token JSON) or a `k=v k2=v2` string
    (text command). Keys must be non-empty strings; values scalar."""
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k.strip():
                return None
            if isinstance(v, (dict, list)):
                return None
            out[k.strip()] = v
        return out
    if isinstance(raw, str):
        out = {}
        for pair in raw.split():
            if "=" not in pair:
                return None
            k, _, v = pair.partition("=")
            if not k.strip() or not v.strip():
                return None
            out[k.strip()] = _coerce_option_value(v)
        return out if out else None
    return None


def normalize_override(provider: str, key: str, raw: Any) -> tuple[Any, str]:
    """Validate + normalize one override value for `provider`.

    Returns (value, "") on success or (None, "<reason>") on failure. Values
    arrive as strings from text/slash commands and as typed JSON from ingress
    tokens; both shapes are accepted for every key.
    """
    keys = PROVIDER_KEYS.get(provider) or ()
    if key not in keys:
        return None, f"`{key}` is not a session setting for the {provider} provider (valid: {', '.join(keys)})"

    if key == "model":
        if not isinstance(raw, str) or not raw.strip():
            return None, "model must be a non-empty model name"
        model = raw.strip()
        inferred = provider_for_model(model)
        if inferred != provider:
            return None, (
                f"`{model}` looks like a {inferred} model but this agent runs {provider}; "
                f"a session override can't change provider (histories aren't compatible) — use `/model` for that"
            )
        return model, ""

    if key == "context_window":
        n = _coerce_int(raw)
        if n is None or n < _MIN_CONTEXT_WINDOW:
            return None, f"context_window must be an integer ≥ {_MIN_CONTEXT_WINDOW:,} (e.g. 200000 or 200k)"
        return n, ""

    if key == "compaction_trigger":
        n = _coerce_int(raw)
        if n is None or n < _COMPACTION_FLOOR:
            return None, f"compaction_trigger must be an integer ≥ {_COMPACTION_FLOOR:,} (Anthropic's floor)"
        return n, ""

    if key == "max_tokens":
        n = _coerce_int(raw)
        if n is None or n <= 0:
            return None, "max_tokens must be a positive integer"
        if provider == "anthropic" and n > _MAX_TOKENS_CEILING:
            return None, f"max_tokens must be ≤ {_MAX_TOKENS_CEILING:,} for the anthropic provider"
        return n, ""

    if key == "effort":
        if not isinstance(raw, str):
            return None, "effort must be a level name"
        level = raw.strip().lower()
        valid = _VALID_EFFORT_LEVELS if provider == "anthropic" else _OPENAI_SESSION_EFFORT
        if level not in valid:
            return None, f"effort must be one of: {', '.join(valid)}"
        return level, ""

    if key == "memory":
        b = _coerce_bool(raw)
        if b is None:
            return None, "memory must be on/off (true/false)"
        return b, ""

    if key == "options":
        opts = _coerce_options(raw)
        if not opts:
            return None, "options must be `key=value` pairs (e.g. temperature=0.7 num_predict=512) or a JSON object"
        return opts, ""

    return None, f"unhandled session setting `{key}`"


def format_override_value(key: str, value: Any) -> str:
    if key == "memory":
        return "on" if value else "off"
    if key == "options":
        return " ".join(f"{k}={json.dumps(v)}" for k, v in value.items())
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


class SessionOverridesMixin:
    """Mixed into every provider conversation class. Subclasses must:

      * set the class attribute `_provider_name` ("anthropic"/"openai"/"ollama"),
      * call `_init_overrides()` in `__init__`,
      * implement `_save_meta()` (persist `overrides_meta_payload()`) and call
        `load_overrides_from_meta(meta)` in `load()`,
      * expose `self.agent`.
    """

    _provider_name: str = ""

    # ---------------------------------------------------------------- state

    def _init_overrides(self) -> None:
        self.overrides: dict[str, Any] = {}
        # Per-TURN model override (external ingress). Highest precedence, model
        # id only, cleared by the caller in a `finally` after the turn.
        self._model_override: str | None = None

    def set_model_override(self, raw_model: str | None) -> None:
        """Override the model used for the NEXT turn only (per-turn).

        `raw_model` is the raw, un-normalized model string as it would appear
        in agent.json (provider prefix / `-1m` suffix allowed). Pass None to
        clear. The caller is responsible for clearing it (typically in a
        `finally`) so it does not bleed into later turns on the same
        conversation.
        """
        self._model_override = raw_model or None

    # ------------------------------------------------------------ effective

    def _effective_raw_model(self) -> str:
        """Raw model string for THIS turn: per-turn override, else the session
        override, else the agent's configured model."""
        return self._model_override or self.overrides.get("model") or self.agent.model

    def effective_context_window(self) -> int:
        ov = self.overrides.get("context_window")
        if isinstance(ov, int) and ov > 0:
            return ov
        return get_model_context_window(self._effective_raw_model(), self._provider_name)

    def effective_effort(self) -> str | None:
        """Session effort override, else the effective model's config knob,
        else None (omit the field). Provider-gated by get_effort."""
        ov = self.overrides.get("effort")
        if isinstance(ov, str) and ov:
            return ov
        return get_effort(self._effective_raw_model(), self._provider_name)

    def effective_memory_enabled(self) -> bool:
        ov = self.overrides.get("memory")
        if isinstance(ov, bool):
            return ov
        return bool(getattr(self.agent, "memory_enabled", True))

    # `effort_override` compatibility surface — /effort (slash + text) and the
    # legacy meta key all speak this attribute. Backed by overrides["effort"].
    @property
    def effort_override(self) -> str | None:
        ov = self.overrides.get("effort")
        return ov if isinstance(ov, str) and ov else None

    @effort_override.setter
    def effort_override(self, level: str | None) -> None:
        if level is None:
            self.overrides.pop("effort", None)
            return
        value, err = normalize_override(self._provider_name, "effort", level)
        if err:
            self.overrides.pop("effort", None)
            return
        self.overrides["effort"] = value

    # --------------------------------------------------------------- mutate

    def set_override(self, key: str, raw: Any, *, persist: bool = True) -> tuple[bool, str]:
        """Validate + store one override. Returns (ok, message)."""
        value, err = normalize_override(self._provider_name, key, raw)
        if err:
            return False, err
        self.overrides[key] = value
        if persist:
            self._save_meta()
        return True, format_override_value(key, value)

    def unset_override(self, key: str, *, persist: bool = True) -> bool:
        """Remove one override. Returns True if it was set."""
        had = key in self.overrides
        self.overrides.pop(key, None)
        if persist and had:
            self._save_meta()
        return had

    def clear_overrides(self, *, persist: bool = True) -> int:
        n = len(self.overrides)
        self.overrides.clear()
        if persist and n:
            self._save_meta()
        return n

    def apply_overrides(self, mapping: Any, *, persist: bool = True) -> tuple[bool, list[str]]:
        """Bulk-apply a `{key: value}` block (ingress token `session_overrides`).

        Invalid entries are skipped and reported; valid ones are stored.
        Returns (changed, errors). Persists only when something changed so a
        per-request re-apply of an unchanged token block is a no-op on disk.
        """
        if not isinstance(mapping, dict):
            return False, ["session_overrides must be a JSON object"]
        errors: list[str] = []
        changed = False
        for key, raw in mapping.items():
            if not isinstance(key, str):
                errors.append(f"non-string key {key!r}")
                continue
            value, err = normalize_override(self._provider_name, key, raw)
            if err:
                errors.append(err)
                continue
            if self.overrides.get(key) != value:
                self.overrides[key] = value
                changed = True
        if changed and persist:
            self._save_meta()
        return changed, errors

    # -------------------------------------------------------------- persist

    def overrides_meta_payload(self) -> dict | None:
        """The `overrides` block for the meta sidecar, or None when empty."""
        return dict(self.overrides) if self.overrides else None

    def load_overrides_from_meta(self, meta: dict) -> None:
        """Restore overrides from a meta dict. Re-validates every stored value
        (a hand-edited or stale file can't smuggle in junk) and migrates the
        legacy top-level `effort_override` key."""
        self.overrides = {}
        block = meta.get(META_KEY) if isinstance(meta, dict) else None
        if isinstance(block, dict):
            for key, raw in block.items():
                if not isinstance(key, str):
                    continue
                value, err = normalize_override(self._provider_name, key, raw)
                if not err:
                    self.overrides[key] = value
        legacy = meta.get(_LEGACY_EFFORT_KEY) if isinstance(meta, dict) else None
        if "effort" not in self.overrides and isinstance(legacy, str):
            value, err = normalize_override(self._provider_name, "effort", legacy)
            if not err:
                self.overrides["effort"] = value

    # --------------------------------------------------------------- report

    def overrides_report(self) -> list[tuple[str, str, str]]:
        """[(setting, effective value, source)] for every setting this
        provider honors — what `/session show` prints."""
        provider = self._provider_name
        rows: list[tuple[str, str, str]] = []
        raw_model = self._effective_raw_model()
        if self._model_override:
            src = "per-turn override"
        elif "model" in self.overrides:
            src = "session"
        else:
            src = "agent.json"
        rows.append(("model", raw_model, src))

        cw = self.effective_context_window()
        rows.append(("context_window", f"{cw:,}", "session" if "context_window" in self.overrides else f"config.json ({raw_model})"))

        if "compaction_trigger" in PROVIDER_KEYS[provider]:
            trig = self.effective_compaction_trigger()  # type: ignore[attr-defined]
            if "compaction_trigger" in self.overrides:
                src = "session"
            elif "context_window" in self.overrides:
                src = "derived from session context_window"
            else:
                src = f"config.json ({raw_model})"
            rows.append(("compaction_trigger", f"{trig:,}", src))

        if "max_tokens" in PROVIDER_KEYS[provider]:
            mt = self.effective_max_tokens()  # type: ignore[attr-defined]
            rows.append(("max_tokens", f"{mt:,}" if mt else "provider default", "session" if "max_tokens" in self.overrides else f"config.json ({raw_model})"))

        if "effort" in PROVIDER_KEYS[provider]:
            eff = self.effective_effort()
            rows.append(("effort", eff or "default", "session" if "effort" in self.overrides else (f"config.json ({raw_model})" if eff else "API default")))

        if "options" in PROVIDER_KEYS[provider]:
            opts = self.overrides.get("options")
            rows.append(("options", format_override_value("options", opts) if opts else "(agent.json only)", "session" if opts else "agent.json"))

        rows.append(("memory", "on" if self.effective_memory_enabled() else "off", "session" if "memory" in self.overrides else "agent.json"))
        return rows


def compaction_trigger_for_window(window: int) -> int:
    """`window - reserve`, floored — the same arithmetic get_compaction_trigger
    uses, applied to a session-overridden window."""
    reserve = get_config().get("compaction_reserve_tokens", 20_000)
    return max(int(window) - int(reserve), _COMPACTION_FLOOR)


# ------------------------------------------------------------- /session UI

_SESSION_USAGE = (
    "Usage: `/session` (show) · `/session set <key> <value>` · "
    "`/session unset <key>` · `/session clear`"
)


def session_command_text(conv: Any, action: str, key: str = "", value: str = "") -> str:
    """ONE implementation of the `/session` command body, shared by the
    Discord slash command and the cross-transport text-prefix mirror so the
    two can't drift. Returns the reply text. `conv` must be a conversation
    that mixes in SessionOverridesMixin; None → explanatory error."""
    if conv is None or not hasattr(conv, "overrides_report"):
        return "No active conversation here — send a message first, then try `/session` again."
    provider = getattr(conv, "_provider_name", "") or "?"
    keys = PROVIDER_KEYS.get(provider) or ()
    action = (action or "show").strip().lower()
    key = (key or "").strip().lower()

    if action in ("show", "", "status", "list"):
        agent_name = getattr(getattr(conv, "agent", None), "display_name", "") or getattr(getattr(conv, "agent", None), "id", "")
        lines = [f"**Session settings for THIS conversation** ({agent_name}, {provider})"]
        for setting, val, src in conv.overrides_report():
            marker = " ← **session override**" if src == "session" else f"  _({src})_"
            lines.append(f"• `{setting}`: {val}{marker}")
        set_keys = ", ".join(f"`{k}`" for k in conv.overrides) if conv.overrides else "none"
        lines.append(f"Overrides set: {set_keys}")
        lines.append(f"Keys for this provider: {', '.join(f'`{k}`' for k in keys)}")
        lines.append(_SESSION_USAGE)
        return "\n".join(lines)

    if action == "set":
        if not key:
            return f"⚠️ `/session set` needs a key. {_SESSION_USAGE}"
        if value is None or not str(value).strip():
            return f"⚠️ `/session set {key}` needs a value."
        ok, msg = conv.set_override(key, value)
        if not ok:
            return f"⚠️ {msg}"
        note = ""
        if key == "model":
            note = " Takes effect on the next turn; the per-turn ingress model (if any) still wins."
        elif key == "memory" and not conv.overrides.get("memory"):
            note = " Memory tools are hidden from and blocked for this conversation."
        elif key == "context_window" and provider == "anthropic" and "compaction_trigger" not in conv.overrides:
            note = f" Compaction now triggers at {conv.effective_compaction_trigger():,} (derived)."
        return f"⚙️ `{key}` for THIS conversation set to {msg}.{note} Use `/session unset {key}` to clear."

    if action == "unset":
        if not key:
            return f"⚠️ `/session unset` needs a key. {_SESSION_USAGE}"
        if key not in keys:
            return f"⚠️ `{key}` is not a session setting for the {provider} provider (valid: {', '.join(keys)})"
        if conv.unset_override(key):
            return f"⚙️ `{key}` override cleared for THIS conversation — falling back to the model/agent default."
        return f"`{key}` had no session override here."

    if action == "clear":
        n = conv.clear_overrides()
        return (f"⚙️ Cleared {n} session override(s) for THIS conversation."
                if n else "No session overrides were set here.")

    return f"⚠️ Unknown `/session` action `{action}`. {_SESSION_USAGE}"


def parse_session_args(arg: str) -> tuple[str, str, str]:
    """Text-prefix arg parser: `"set model foo"` → ("set", "model", "foo");
    `"unset model"` → ("unset", "model", ""); `""` → ("show", "", "").
    The value keeps its internal whitespace (ollama `options` take several
    `k=v` pairs)."""
    parts = (arg or "").strip().split(None, 2)
    if not parts:
        return "show", "", ""
    action = parts[0].lower()
    key = parts[1] if len(parts) > 1 else ""
    value = parts[2] if len(parts) > 2 else ""
    return action, key, value
