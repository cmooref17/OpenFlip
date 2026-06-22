"""Global config singleton, loaded once at startup. Tools and runtime read it via get_config()."""
from __future__ import annotations
import os
from .utils import load_json

_config: dict = {}


def load_config(path: str | None = None) -> dict:
    global _config
    path = path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
    _config = load_json(path, default={})
    return _config


def get_config() -> dict:
    if not _config:
        load_config()
    return _config


def reload_config() -> dict:
    return load_config()


# ── Per-integration accessors ──
#
# New canonical shape (post-2026-05-15): per-transport identity + secrets live
# under `integrations.<name>.*` in config.json. Discord-specific owner_id and
# bot tokens used to live as flat top-level keys (`owner_id`) and in a
# separate file (`api_config.json`). Both old shapes are still honored as
# fallback for backward compatibility with existing deployments.

def get_owner_id(integration: str = "discord") -> int:
    """Return the operator's user ID for the given integration.

    Lookup order:
      1. `integrations.<integration>.owner_id` (canonical)
      2. legacy top-level `owner_id` (when integration == "discord")
      3. 0 (no owner)
    """
    cfg = get_config()
    integrations = cfg.get("integrations") or {}
    entry = integrations.get(integration) or {}
    val = entry.get("owner_id")
    if val:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    if integration == "discord":
        legacy = cfg.get("owner_id")
        if legacy:
            try:
                return int(legacy)
            except (TypeError, ValueError):
                pass
    return 0


def get_admin_ids(integration: str = "discord") -> list[int]:
    """Return the elevated-admin user IDs for the given integration.

    Admins are a privilege tier ABOVE normal users but BELOW the owner: the
    owner alone keeps the dangerous powers (shell, restart). The owner is
    always implicitly an admin, so callers can check membership without
    special-casing the owner.

    Lookup order for the list:
      1. `integrations.<integration>.admin_ids` (canonical)
      2. legacy top-level `admin_ids` (when integration == "discord")
      3. [] (no extra admins)

    The owner_id is always appended (deduped).
    """
    cfg = get_config()
    integrations = cfg.get("integrations") or {}
    entry = integrations.get(integration) or {}
    raw = entry.get("admin_ids")
    if not raw and integration == "discord":
        raw = cfg.get("admin_ids")
    ids: list[int] = []
    for v in (raw or []):
        try:
            ids.append(int(v))
        except (TypeError, ValueError):
            continue
    owner = get_owner_id(integration)
    if owner and owner not in ids:
        ids.append(owner)
    return ids


# ── Cross-transport identity links ──
#
# Top-level `identity_links` in config.json maps a per-transport identity to a
# canonical id so one person's 1:1 conversations on different transports share
# ONE conversation history:
#
#   "identity_links": {
#     "discord:139243578504249344": "flip",
#     "imessage:+15551234567": "flip"
#   }
#
# Keys are "<transport>:<native_id>" (Discord: numeric user id; iMessage: the
# raw handle). Values are an arbitrary canonical string. A linked speaker's
# 1:1 sessions get conversation_id "linked:<canonical>" instead of the
# transport-native id (see make_discord_session / make_imessage_session).
#
# SECURITY: links rewrite conversation ROUTING ONLY (which history file +
# in-memory conversation a session resolves to). They confer NO privilege:
# owner/admin/tool ACLs are evaluated per-turn from the session's transport +
# native handle/id (_acl_transport/_acl_speaker/_acl_handle in _run_turn) and
# never consult identity_links. Being owner on Discord does not make the
# linked iMessage handle owner, and vice versa.

LINKED_CONV_PREFIX = "linked:"


def get_identity_links() -> dict[str, str]:
    """Return the identity_links map with normalized keys/values.

    Keys are case-folded + stripped ("<transport>:<native_id>") to match the
    normalization iMessage applies to sender handles; Discord ids are digits
    and unaffected. Malformed entries (no colon, empty value) are skipped.
    """
    cfg = get_config()
    raw = cfg.get("identity_links")
    if not isinstance(raw, dict):
        return {}
    links: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip().lower()
        val = str(v).strip()
        if ":" not in key or not key.split(":", 1)[1] or not val:
            continue
        links[key] = val
    return links


def resolve_linked_conversation_id(transport: str, native_id) -> str:
    """Return "linked:<canonical>" if "<transport>:<native_id>" is linked, else "".

    `native_id` is the transport-native speaker identity: the int user id for
    Discord, the raw handle string for iMessage. Lookup is case-folded to
    match get_identity_links' key normalization.
    """
    if not transport or native_id is None or native_id == "":
        return ""
    links = get_identity_links()
    if not links:
        return ""
    key = f"{transport}:{native_id}".strip().lower()
    canonical = links.get(key, "")
    return f"{LINKED_CONV_PREFIX}{canonical}" if canonical else ""


# ── Handle-based identity accessors (iMessage, future SMS/email) ──
#
# Handle-based transports identify senders by an email/phone STRING, not a
# numeric ID. Owner/admin auth on those transports MUST compare the raw handle
# (case-folded), never a hash of it: Python salts str hashing per process
# (PYTHONHASHSEED), so any hashed-int identity changes every restart and can
# never match a fixed config value. These are the string-returning siblings of
# get_owner_id / get_admin_ids (which stay int-only for Discord).

def get_owner_handle(integration: str = "imessage") -> str:
    """Return the operator's raw handle STRING for a handle-based transport.

    Returns the case-folded (`.strip().lower()`) `integrations.<integration>.
    owner_id` string, or "" if absent. NO int() coercion — the value is a
    handle like "user@example.com" or "+15551234567".
    """
    cfg = get_config()
    integrations = cfg.get("integrations") or {}
    entry = integrations.get(integration) or {}
    val = entry.get("owner_id")
    if val is None or val == "":
        return ""
    return str(val).strip().lower()


def get_admin_handles(integration: str = "imessage") -> list[str]:
    """Return elevated-admin handle STRINGS for a handle-based transport.

    Handle-based sibling of get_admin_ids. Returns the case-folded
    `integrations.<integration>.admin_ids` strings, with the owner handle
    appended (deduped) — mirroring how get_admin_ids always includes the
    owner so callers can check membership without special-casing the owner.
    Empty handles are skipped. NO int() coercion.
    """
    cfg = get_config()
    integrations = cfg.get("integrations") or {}
    entry = integrations.get(integration) or {}
    raw = entry.get("admin_ids") or []
    handles: list[str] = []
    for v in raw:
        if v is None:
            continue
        h = str(v).strip().lower()
        if h and h not in handles:
            handles.append(h)
    owner = get_owner_handle(integration)
    if owner and owner not in handles:
        handles.append(owner)
    return handles


def get_integration_tokens(integration: str = "discord") -> dict[str, str]:
    """Return per-agent secrets/tokens for the given integration.

    Lookup order:
      1. `integrations.<integration>.tokens`
      2. legacy `api_config.json` `tokens` block (Discord only)
      3. {}
    """
    cfg = get_config()
    integrations = cfg.get("integrations") or {}
    entry = integrations.get(integration) or {}
    tokens = entry.get("tokens")
    if isinstance(tokens, dict) and tokens:
        return {k: str(v) for k, v in tokens.items() if v}
    if integration == "discord":
        import os as _os
        from .utils import load_json as _load_json, project_root as _project_root
        api_path = _os.path.join(_project_root(), "api_config.json")
        data = _load_json(api_path, default={})
        api_tokens = (data or {}).get("tokens") or {}
        return {k: str(v) for k, v in api_tokens.items() if v}
    return {}


# ── OpenAI provider accessors ──
#
# The "openai" provider (OpenAIConversation) has two auth paths, checked in
# order:
#   1. ChatGPT/Codex subscription OAuth (preferred): `codex login` writes
#      `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`); openflip reads
#      and refreshes those tokens (see _codex_auth.py) and talks to the
#      Responses API at chatgpt.com/backend-api/codex.
#   2. Plain API key (fallback): `integrations.openai.api_key` in config.json,
#      with the standard OPENAI_API_KEY environment variable as a secondary
#      fallback, against the Chat Completions API.


def get_codex_home() -> str:
    """Return the Codex home directory (where `codex login` keeps auth.json).

    `CODEX_HOME` environment variable overrides; default is `~/.codex`.
    Resolved with os.path.expanduser, which handles the user home correctly
    on Linux, macOS, and Windows — never a hardcoded path.
    """
    env = os.environ.get("CODEX_HOME", "").strip()
    if env:
        return os.path.expanduser(env)
    return os.path.expanduser(os.path.join("~", ".codex"))

def get_openai_api_key() -> str:
    """Return the OpenAI API key, or "" if unconfigured.

    Lookup order:
      1. `integrations.openai.api_key` in config.json (canonical)
      2. OPENAI_API_KEY environment variable
      3. "" (provider unusable — surfaces as an auth error at chat time)
    """
    cfg = get_config()
    entry = (cfg.get("integrations") or {}).get("openai") or {}
    key = entry.get("api_key")
    if key:
        return str(key).strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def get_openai_base_url() -> str:
    """API base for the openai provider. `integrations.openai.base_url`
    overrides (e.g. for a compatible proxy/gateway); default is the
    official endpoint. Trailing slash is stripped so callers can append
    `/v1/chat/completions` uniformly."""
    cfg = get_config()
    entry = (cfg.get("integrations") or {}).get("openai") or {}
    base = str(entry.get("base_url") or "").strip()
    return (base or "https://api.openai.com").rstrip("/")


def get_openai_default_model() -> str:
    """Model used when an openai-provider agent has an empty `model` field.

    `integrations.openai.default_model` overrides; falls back to "gpt-5.1".
    Agents normally set their model explicitly in agent.json.
    """
    cfg = get_config()
    entry = (cfg.get("integrations") or {}).get("openai") or {}
    model = str(entry.get("default_model") or "").strip()
    return model or "gpt-5.1"


# ── Model-context lookup ──
#
# Single source of truth for each model's context window. All providers
# (anthropic + openai + ollama) read from here. Compaction trigger is derived
# as `context_window - compaction_reserve_tokens` so we never have to
# maintain two separate numbers.

# Sensible defaults if the model isn't in config.json's `models` block.
# Anthropic models default to 200k (their published default for sonnet/opus
# without the 1M beta header). OpenAI models default to a conservative 128k —
# bigger-window models (gpt-5 family) should declare their real window in
# config.json's models block. Ollama models default to 32k (a common llama
# default). Bumping a specific model's window means adding it to config.json.
_DEFAULT_CONTEXT_WINDOW_BY_PROVIDER = {
    "anthropic": 200_000,
    "openai": 128_000,
    "ollama": 32_000,
}


def get_model_context_window(model_name: str, provider: str = "") -> int:
    """Return the context window (in tokens) for a model.

    Strips `provider/` prefix from model_name before lookup (config.json
    keys are bare model ids, agent.json values use `provider/model` form).

    Lookup order:
      1. config.json's `models.<bare_model_name>.context_window`
      2. provider default (anthropic=200k, ollama=32k)
      3. final hard fallback of 32k
    """
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    cfg = get_config()
    models = cfg.get("models") or {}
    entry = models.get(bare) or {}
    cw = entry.get("context_window")
    if isinstance(cw, int) and cw > 0:
        return cw
    return _DEFAULT_CONTEXT_WINDOW_BY_PROVIDER.get(provider, 32_000)


def get_compaction_trigger(model_name: str, provider: str = "") -> int:
    """Compaction trigger = context_window - reserve. Provider-side
    compaction (e.g. Anthropic's `compact_20260112`) needs headroom
    inside the window for the summary itself plus the model's response.

    Per-model override: if `models.<bare>.compaction_trigger` is set in
    config.json, use that value directly (still floored at Anthropic's
    50k minimum). Set this when you want compaction to fire at a
    specific point on a given model — e.g. to cap input below a
    pricing-tier boundary or to compact more aggressively than the
    default `window - reserve`.

    Re-measure Anthropic's actual auto-compact behavior on your own
    logs before assuming a threshold — server-side behavior changes.
    """
    cfg = get_config()
    # Strip provider prefix for the per-model lookup; matches get_model_context_window.
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    models = cfg.get("models") or {}
    entry = models.get(bare) or {}
    override = entry.get("compaction_trigger")
    if isinstance(override, int) and override > 0:
        return max(override, 50_000)
    reserve = cfg.get("compaction_reserve_tokens", 20_000)
    window = get_model_context_window(model_name, provider)
    trigger = window - reserve
    # Anthropic requires trigger >= 50k. Floor it.
    return max(trigger, 50_000)


# Valid Anthropic `output_config.effort` reasoning-depth levels. The field is
# Anthropic-only and entirely optional: anything not in this tuple (wrong type,
# junk string, or absent) resolves to None, which means the request omits
# output_config altogether and the API falls back to its default ("high").
# Model-gating caveat: `xhigh` is opus-4-7/opus-4-8 only and `max` is opus-4.6+
# only — the API may 400 on a level the configured model doesn't support, so the
# operator must only set a level valid for the agent's model. See get_effort,
# AnthropicConversation._effort_level, and agents/_shared/MANUAL.md.
_VALID_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# OpenAI `reasoning_effort` levels (Chat Completions, reasoning-capable models
# only — o-series / gpt-5 family). Distinct vocabulary from Anthropic's set:
# no xhigh/max, adds minimal. Only set `models.<bare>.effort` on a model that
# actually supports the parameter — the API 400s otherwise.
_VALID_OPENAI_EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")


def get_effort(model_name: str, provider: str = "") -> str | None:
    """Return the reasoning-effort level for a model → Anthropic
    `output_config.effort` / OpenAI `reasoning_effort`, or None to omit the
    field entirely.

    Per-model override: reads `models.<bare>.effort` from config.json using the
    same bare-name resolution as get_compaction_trigger / get_model_context_window
    (strip the `provider/` prefix). Validates against the provider's valid
    levels (anthropic: low/medium/high/xhigh/max; openai: minimal/low/medium/
    high); anything invalid or absent → None, which the provider treats as
    "omit the field" (API default).

    Provider-gated: a non-empty provider that isn't "anthropic" or "openai"
    returns None — effort is meaningless for ollama and the field would 400.
    An empty provider keeps the historical anthropic validation.
    """
    if provider == "openai":
        valid = _VALID_OPENAI_EFFORT_LEVELS
    elif provider and provider != "anthropic":
        return None
    else:
        valid = _VALID_EFFORT_LEVELS
    cfg = get_config()
    # Strip provider prefix for the per-model lookup; matches get_compaction_trigger.
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    models = cfg.get("models") or {}
    entry = models.get(bare) or {}
    level = entry.get("effort")
    if isinstance(level, str) and level.strip().lower() in valid:
        return level.strip().lower()
    return None


# Default Anthropic output token cap when a model has no explicit
# `models.<bare>.max_tokens` entry. Matches the historical hardcoded literal
# so nothing regresses for models without a config override.
_DEFAULT_MAX_TOKENS = 64_000
# Sanity ceiling for a configured max_tokens. Anthropic's output cap tops out
# at 128k (sonnet with the output-128k beta); anything beyond that is a config
# typo, not a real cap, so we reject it and fall back to the default rather
# than let the API 400 on an absurd value.
_MAX_TOKENS_CEILING = 128_000


def get_max_tokens(model_name: str, provider: str = "") -> int:
    """Return the Anthropic output token cap (request `max_tokens`) for a model.

    Per-model override: reads `models.<bare>.max_tokens` from config.json using
    the same bare-name resolution as get_effort / get_compaction_trigger /
    get_model_context_window (strip the `provider/` prefix). Anthropic-only —
    the ollama provider has its own `num_predict` knob in agent.json.

    Validates that the configured value is a positive int within Anthropic's
    allowed output range (1.._MAX_TOKENS_CEILING). Anything invalid — non-int,
    <= 0, or absurdly large — falls back to _DEFAULT_MAX_TOKENS (64000) so the
    request body stays valid and behavior is unchanged for any model without an
    explicit, sane entry.
    """
    cfg = get_config()
    # Strip provider prefix for the per-model lookup; matches get_effort.
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    models = cfg.get("models") or {}
    entry = models.get(bare) or {}
    override = entry.get("max_tokens")
    if isinstance(override, int) and not isinstance(override, bool):
        if 0 < override <= _MAX_TOKENS_CEILING:
            return override
    return _DEFAULT_MAX_TOKENS
