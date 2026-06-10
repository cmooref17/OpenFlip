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


# ── Model-context lookup ──
#
# Single source of truth for each model's context window. Both providers
# (anthropic + ollama) read from here. Compaction trigger is derived as
# `context_window - compaction_reserve_tokens` so we never have to
# maintain two separate numbers.

# Sensible defaults if the model isn't in config.json's `models` block.
# Anthropic models default to 200k (their published default for sonnet/opus
# without the 1M beta header). Ollama models default to 32k (a common llama
# default). Bumping a specific model's window means adding it to config.json.
_DEFAULT_CONTEXT_WINDOW_BY_PROVIDER = {
    "anthropic": 200_000,
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


def get_effort(model_name: str, provider: str = "") -> str | None:
    """Return the reasoning-effort level for a model → Anthropic
    `output_config.effort`, or None to omit the field entirely.

    Per-model override: reads `models.<bare>.effort` from config.json using the
    same bare-name resolution as get_compaction_trigger / get_model_context_window
    (strip the `provider/` prefix). Validates against the five valid levels
    (low/medium/high/xhigh/max); anything invalid or absent → None, which the
    provider treats as "omit output_config" (API default "high").

    Anthropic-only: if a non-empty provider is passed that isn't "anthropic",
    return None — effort is meaningless for ollama and the field would 400.
    """
    if provider and provider != "anthropic":
        return None
    cfg = get_config()
    # Strip provider prefix for the per-model lookup; matches get_compaction_trigger.
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    models = cfg.get("models") or {}
    entry = models.get(bare) or {}
    level = entry.get("effort")
    if isinstance(level, str) and level.strip().lower() in _VALID_EFFORT_LEVELS:
        return level.strip().lower()
    return None
