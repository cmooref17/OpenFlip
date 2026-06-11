"""Interactive Discord UIs for changing an agent's model and Ollama options.

Two panels, two slash commands:
    /model    — see current model, pick a new one from a live Ollama list
    /options  — see current Ollama options, edit any one via a Modal

Both panels share the layout pattern from toolset_ui.py:
    [Embed showing current state]
    [Agent ▾]      <- StringSelect to switch which agent the panel operates on
    [Picker ▾]     <- StringSelect to pick what to change (model / option key)
    [↺ Reset / Refresh / Close]   <- action buttons

Owner-only — verified by interaction_check.

When a model or option is changed, the agent's JSON is rewritten via
agent.save(), and any LIVE Conversation objects for that agent are updated
in-place (conv.model + conv.options) so existing chat history is preserved.
"""
from __future__ import annotations
import asyncio
from typing import Optional

import nextcord

from .acl import is_owner


_VIEW_TIMEOUT_S = 15 * 60


# Ollama options the panel exposes — type, min, max, fallback default.
# Tuple form: (type, min_value, max_value, default).
OLLAMA_OPTIONS_SCHEMA: dict[str, tuple[str, float, float, float]] = {
    "temperature":   ("float", 0.0, 2.0,    0.8),
    "num_ctx":       ("int",   256, 131072, 4096),
    "num_predict":   ("int",   -1,  32768,  -1),
    "repeat_penalty":("float", 0.0, 2.0,    1.1),
    "repeat_last_n": ("int",   -1,  8192,   64),
    "top_k":         ("int",   0,   1000,   40),
    "top_p":         ("float", 0.0, 1.0,    0.9),
    "min_p":         ("float", 0.0, 1.0,    0.0),
    "mirostat":      ("int",   0,   2,      0),
    "mirostat_eta":  ("float", 0.0, 1.0,    0.1),
    "mirostat_tau":  ("float", 0.0, 10.0,   5.0),
    "seed":          ("int",   -1,  2**31,  -1),
}


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _claude_models() -> list[str]:
    """Anthropic model names, read from config.json's `models` block.

    Single source of truth — the picker reflects whatever is configured.
    No hardcoded list to drift out of sync (that bug froze the dropdown at
    4.7 while config already had 4.8). Falls back to a minimal default only
    if config is missing/empty.
    """
    try:
        from .config_global import get_config
        models = (get_config() or {}).get("models", {}) or {}
        claude = [
            name for name, meta in models.items()
            if (meta or {}).get("provider") == "anthropic"
        ]
        if claude:
            return claude
    except Exception:
        pass
    return ["claude-opus-4-8", "claude-sonnet-4-6"]


def _is_claude_model(model: str) -> bool:
    """Return True if the model name looks like a Claude/Anthropic model."""
    return (
        model.startswith("claude-")
        or model.startswith("anthropic/")
    )


def _openai_models() -> list[str]:
    """OpenAI model names from config.json's `models` block (provider ==
    "openai"). Same single-source-of-truth pattern as _claude_models. No
    fallback list — an operator who hasn't configured any openai models
    gets none in the picker (the provider needs an API key configured
    anyway)."""
    try:
        from .config_global import get_config
        models = (get_config() or {}).get("models", {}) or {}
        return [
            name for name, meta in models.items()
            if (meta or {}).get("provider") == "openai"
        ]
    except Exception:
        return []


def _is_openai_model(model: str) -> bool:
    """Return True only for explicitly-prefixed `openai/...` names. A bare
    "gpt-" prefix is deliberately NOT enough — ollama hosts models named
    gpt-oss etc., and mis-inferring openai there would silently flip the
    agent's provider."""
    return model.startswith("openai/")


def _provider_for_model(model: str) -> str:
    """Infer the provider from a model name."""
    if _is_openai_model(model):
        return "openai"
    return "anthropic" if _is_claude_model(model) else "ollama"


async def _fetch_ollama_models() -> list[str]:
    try:
        from openflip import ollama_api
        models = await asyncio.wait_for(ollama_api.ollama_list(), timeout=30)
        names = []
        for m in models:
            n = m.get("model") if isinstance(m, dict) else None
            if n:
                names.append(n)
        names.sort()
        return names
    except Exception:
        return []


async def _fetch_all_models() -> list[str]:
    """Fetch Ollama models and append Anthropic + OpenAI API models."""
    ollama = await _fetch_ollama_models()
    # Prefix API-provider models to distinguish them in the picker (the
    # prefix also drives _provider_for_model on selection).
    claude = [f"anthropic/{m}" for m in _claude_models()]
    openai = [f"openai/{m}" for m in _openai_models()]
    return ollama + claude + openai


def _runner_for(agent_id: str):
    from . import registry
    return registry.RUNNERS.get(agent_id)


def _propagate_model_to_live_conversations(agent_id: str) -> int:
    """Push agent.model into every existing Conversation object for this
    agent's runner. Returns count updated."""
    runner = _runner_for(agent_id)
    if not runner:
        return 0
    new_model = runner.agent.model
    n = 0
    for conv in runner.conversations.values():
        try:
            conv.model = new_model
            n += 1
        except Exception:
            pass
    return n


def _propagate_options_to_live_conversations(agent_id: str) -> int:
    runner = _runner_for(agent_id)
    if not runner:
        return 0
    new_opts = dict(runner.agent.ollama_options)
    n = 0
    for conv in runner.conversations.values():
        try:
            # ollama_api.Conversation.options is an Options dict-like;
            # mutating in place keeps any in-flight references valid.
            for k, v in new_opts.items():
                conv.options[k] = v
            n += 1
        except Exception:
            pass
    return n


def _list_agents() -> list[tuple[str, str]]:
    """Returns [(id, display_name), ...] for every discovered agent."""
    from . import registry
    return [(a.id, a.display_name) for a in sorted(registry.ALL_AGENTS.values(), key=lambda a: a.id)]


def _get_agent(agent_id: str):
    from . import registry
    return registry.ALL_AGENTS.get(agent_id)


# ──────────────────────────────────────────────────────────────────────
# /model panel
# ──────────────────────────────────────────────────────────────────────

def _build_model_embed(agent_id: Optional[str]) -> nextcord.Embed:
    e = nextcord.Embed(title="🤖 Agent Model", color=0x5865F2)
    if not agent_id:
        e.description = "Pick an agent below."
        return e
    agent = _get_agent(agent_id)
    if not agent:
        e.description = f"Unknown agent: `{agent_id}`."
        e.color = 0xED4245
        return e
    e.description = f"**{agent.display_name}** (`{agent.id}`)"
    prov = getattr(agent, "provider", "ollama")
    e.add_field(name="Current model", value=f"`{agent.model}`", inline=True)
    e.add_field(name="Provider", value=f"`{prov}`", inline=True)
    e.set_footer(text="Pick a model below to switch. Claude models auto-set provider to anthropic. Existing conversations keep history; the new model takes effect on the next message.")
    return e


class _AgentPicker(nextcord.ui.StringSelect):
    """Shared agent-picker used by both /model and /options panels."""
    def __init__(self, current: Optional[str]):
        opts = []
        for aid, dname in _list_agents():
            opts.append(nextcord.SelectOption(
                label=dname or aid, value=aid, description=f"id: {aid}",
                default=(aid == current),
            ))
        if not opts:
            opts = [nextcord.SelectOption(label="(no agents)", value="__none__")]
        super().__init__(placeholder="Pick an agent…", options=opts, min_values=1, max_values=1, row=0)

    async def callback(self, interaction: nextcord.Interaction):
        view = self.view
        if self.values[0] == "__none__":
            await interaction.response.defer()
            return
        view.agent_id = self.values[0]
        view.selected_key = None
        await view.refresh(interaction)


class _ModelPicker(nextcord.ui.StringSelect):
    def __init__(self, model_list: list[str], current: Optional[str], disabled: bool):
        # 25-option Discord cap. Claude models are the ones the operator
        # actually switches between, so guarantee they always appear and let
        # Ollama models fill the remaining slots — Claude is never the thing
        # that gets cut.
        choices: list[str] = list(model_list)
        if current and current not in choices:
            choices.insert(0, current)
        # Keep `current` at the front, then never let the API-provider
        # models (Claude/OpenAI) get sliced off — Ollama fills what's left.
        head = [current] if current else []
        api_models = [c for c in choices
                      if (_is_claude_model(c) or _is_openai_model(c)) and c not in head]
        ollama = [c for c in choices
                  if not (_is_claude_model(c) or _is_openai_model(c)) and c not in head]
        choices = (head + api_models + ollama)[:25]
        opts = []
        for name in choices:
            if _is_claude_model(name):
                desc = "Anthropic (Max subscription)"
            elif _is_openai_model(name):
                desc = "OpenAI (API key)"
            else:
                desc = "Ollama"
            opts.append(nextcord.SelectOption(
                label=_truncate(name, 100), value=name,
                description=desc,
                default=(name == current),
            ))
        if not opts:
            opts = [nextcord.SelectOption(label="(no models found)", value="__none__")]
            disabled = True
        super().__init__(
            placeholder="Pick a model to switch to…",
            options=opts, min_values=1, max_values=1, row=1,
            disabled=disabled,
        )

    async def callback(self, interaction: nextcord.Interaction):
        view: "ModelView" = self.view
        if self.values[0] == "__none__" or not view.agent_id:
            await interaction.response.defer()
            return
        agent = _get_agent(view.agent_id)
        if not agent:
            await interaction.response.send_message("Agent not found.", ephemeral=True)
            return
        new_model = self.values[0]
        if new_model == agent.model:
            await interaction.response.defer()
            return
        # Auto-set provider based on model type
        new_provider = _provider_for_model(new_model)
        old_provider = getattr(agent, "provider", "ollama")
        agent.model = new_model
        agent.provider = new_provider
        # When switching providers, clear existing conversations for this
        # agent — they're incompatible between Ollama and Claude CLI.
        if new_provider != old_provider:
            runner = _runner_for(view.agent_id)
            if runner:
                runner.conversations.clear()
        ok = agent.save()
        if not ok:
            await interaction.response.send_message("❌ Failed to save agent JSON.", ephemeral=True)
            return
        # Push to live conversations so the next chat call uses the new model.
        n = _propagate_model_to_live_conversations(view.agent_id)
        prov_note = f" (provider → `{new_provider}`)" if new_provider != old_provider else ""
        await view.refresh(interaction)
        try:
            await interaction.followup.send(
                f"✅ `{view.agent_id}` model → `{new_model}`{prov_note} (updated {n} live conversation(s)).",
                ephemeral=True,
            )
        except Exception:
            pass


class ModelView(nextcord.ui.View):
    def __init__(self, owner_id: int, *, agent_id: Optional[str] = None, models: list[str] | None = None):
        super().__init__(timeout=_VIEW_TIMEOUT_S)
        self.owner_id = owner_id
        self.agent_id = agent_id
        self.models = models or []
        self.message: Optional[nextcord.Message] = None
        self._build_components()

    def _build_components(self):
        self.clear_items()
        self.add_item(_AgentPicker(self.agent_id))
        agent = _get_agent(self.agent_id) if self.agent_id else None
        self.add_item(_ModelPicker(self.models, agent.model if agent else None, disabled=(agent is None)))
        self.add_item(_RefreshModelsButton())
        self.add_item(_CloseButton())

    async def interaction_check(self, interaction: nextcord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("This panel belongs to someone else.", ephemeral=True)
            return False
        return True

    async def refresh(self, interaction: nextcord.Interaction):
        # Re-fetch model list each time the panel updates so newly-pulled
        # models in Ollama appear without re-opening.
        self.models = await _fetch_all_models()
        self._build_components()
        await interaction.response.edit_message(embed=_build_model_embed(self.agent_id), view=self)

    async def refresh_message(self):
        if not self.message:
            return
        self.models = await _fetch_all_models()
        self._build_components()
        try:
            await self.message.edit(embed=_build_model_embed(self.agent_id), view=self)
        except Exception:
            pass

    async def on_timeout(self):
        for c in self.children:
            try: c.disabled = True
            except Exception: pass
        if self.message:
            try: await self.message.edit(view=self)
            except Exception: pass

    def close_message(self) -> str:
        agent = _get_agent(self.agent_id) if self.agent_id else None
        if agent:
            return f"ℹ️ Model kept as `{agent.model}` for **{agent.display_name}**."
        return "✖ Closed."


class _RefreshModelsButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Refresh", style=nextcord.ButtonStyle.secondary, emoji="🔄", row=2)
    async def callback(self, interaction: nextcord.Interaction):
        await self.view.refresh(interaction)


class _CloseButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Close", style=nextcord.ButtonStyle.secondary, emoji="✖", row=2)
    async def callback(self, interaction: nextcord.Interaction):
        view = self.view
        view.stop()
        # Each view supplies its own close message confirming current state.
        msg = view.close_message() if hasattr(view, "close_message") else "✖ Closed."
        await interaction.response.edit_message(content=msg, embed=None, view=None)


async def open_model_panel(interaction: nextcord.Interaction, *, runner_agent_id: str):
    if not is_owner(interaction.user.id):
        await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
        return
    models = await _fetch_all_models()
    view = ModelView(owner_id=interaction.user.id, agent_id=runner_agent_id, models=models)
    await interaction.response.send_message(embed=_build_model_embed(view.agent_id), view=view, ephemeral=True)
    view.message = await interaction.original_message()


# ──────────────────────────────────────────────────────────────────────
# /options panel
# ──────────────────────────────────────────────────────────────────────

def _format_option(v) -> str:
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "…"


def _build_options_embed(agent_id: Optional[str], selected_key: Optional[str] = None) -> nextcord.Embed:
    e = nextcord.Embed(title="⚙️ Agent Ollama Options", color=0x5865F2)
    if not agent_id:
        e.description = "Pick an agent below."
        return e
    agent = _get_agent(agent_id)
    if not agent:
        e.description = f"Unknown agent: `{agent_id}`."
        e.color = 0xED4245
        return e
    e.description = f"**{agent.display_name}** (`{agent.id}`) on `{agent.model}`"
    opts = dict(agent.ollama_options or {})
    # Show every key in the schema PLUS any extras the JSON has.
    keys = list(OLLAMA_OPTIONS_SCHEMA.keys())
    for k in opts:
        if k not in keys:
            keys.append(k)
    for k in keys:
        in_schema = k in OLLAMA_OPTIONS_SCHEMA
        if in_schema:
            t, lo, hi, default = OLLAMA_OPTIONS_SCHEMA[k]
            type_hint = f"{t} [{lo}..{hi}]"
        else:
            type_hint = "(custom)"
        present = k in opts
        current = opts.get(k, OLLAMA_OPTIONS_SCHEMA.get(k, ("", "", "", "(unset)"))[3] if in_schema else "(unset)")
        marker = "🔹 " if k == selected_key else ("" if present else "○ ")
        e.add_field(
            name=f"{marker}{k}  _({type_hint})_",
            value=f"`{_format_option(current)}`" + ("" if present else "  _(using default — not in JSON)_"),
            inline=False,
        )
    e.set_footer(text="Pick a key below to edit. Empty values use Ollama's default.")
    return e


class _OptionPicker(nextcord.ui.StringSelect):
    def __init__(self, agent_id: Optional[str], selected_key: Optional[str]):
        if agent_id:
            agent = _get_agent(agent_id)
            present = set((agent.ollama_options or {}).keys()) if agent else set()
            opts = []
            for k, (t, lo, hi, default) in OLLAMA_OPTIONS_SCHEMA.items():
                desc = f"{t} [{lo}..{hi}]" + ("  (set in JSON)" if k in present else "  (default)")
                opts.append(nextcord.SelectOption(
                    label=k, value=k, description=_truncate(desc, 100),
                    default=(k == selected_key),
                ))
            disabled = False
        else:
            opts = [nextcord.SelectOption(label="(pick an agent first)", value="__none__")]
            disabled = True
        super().__init__(
            placeholder="Pick an option to edit…",
            options=opts, min_values=1, max_values=1, row=1,
            disabled=disabled,
        )

    async def callback(self, interaction: nextcord.Interaction):
        view: "OptionsView" = self.view
        if self.values[0] == "__none__" or not view.agent_id:
            await interaction.response.defer()
            return
        view.selected_key = self.values[0]
        await interaction.response.send_modal(_EditOptionModal(view, view.selected_key))


class _EditOptionModal(nextcord.ui.Modal):
    def __init__(self, parent: "OptionsView", key: str):
        super().__init__(title=f"Edit {key}", timeout=300)
        self.parent_view = parent
        self.key = key
        agent = _get_agent(parent.agent_id)
        current = (agent.ollama_options or {}).get(key) if agent else ""
        schema = OLLAMA_OPTIONS_SCHEMA.get(key)
        hint = ""
        if schema:
            t, lo, hi, default = schema
            hint = f"{t} {lo}..{hi}  (default {default})"
        self.field = nextcord.ui.TextInput(
            label=_truncate(f"{key}", 45),
            default_value="" if current is None else str(current),
            placeholder=_truncate(hint, 100) or "value",
            required=False,  # blank = remove from JSON, fall back to Ollama default
            style=nextcord.TextInputStyle.short,
        )
        self.add_item(self.field)

    async def callback(self, interaction: nextcord.Interaction):
        agent = _get_agent(self.parent_view.agent_id)
        if not agent:
            await interaction.response.send_message("Agent not found.", ephemeral=True)
            return
        raw = (self.field.value or "").strip()
        opts = dict(agent.ollama_options or {})
        if raw == "":
            # Removing the override.
            opts.pop(self.key, None)
            agent.ollama_options = opts
            ok = agent.save()
            if not ok:
                await interaction.response.send_message("❌ Save failed.", ephemeral=True)
                return
            _propagate_options_to_live_conversations(agent.id)
            await self.parent_view.refresh_message()
            await interaction.response.send_message(f"✅ Removed `{self.key}` override (will use default).", ephemeral=True)
            return
        # Coerce to type.
        schema = OLLAMA_OPTIONS_SCHEMA.get(self.key)
        try:
            if schema is None or schema[0] == "float":
                value = float(raw) if (schema and schema[0] == "float") else (int(raw) if raw.lstrip("-").isdigit() else raw)
            elif schema[0] == "int":
                value = int(raw)
            else:
                value = raw
        except (TypeError, ValueError) as e:
            await interaction.response.send_message(f"❌ Couldn't parse value: {e}", ephemeral=True)
            return
        if schema:
            _, lo, hi, _ = schema
            if value < lo or value > hi:
                await interaction.response.send_message(f"❌ Out of range. Must be {lo}..{hi}.", ephemeral=True)
                return
        opts[self.key] = value
        agent.ollama_options = opts
        ok = agent.save()
        if not ok:
            await interaction.response.send_message("❌ Save failed.", ephemeral=True)
            return
        _propagate_options_to_live_conversations(agent.id)
        await self.parent_view.refresh_message()
        await interaction.response.send_message(f"✅ `{agent.id}.{self.key}` = `{value}`", ephemeral=True)


class OptionsView(nextcord.ui.View):
    def __init__(self, owner_id: int, *, agent_id: Optional[str] = None):
        super().__init__(timeout=_VIEW_TIMEOUT_S)
        self.owner_id = owner_id
        self.agent_id = agent_id
        self.selected_key: Optional[str] = None
        self.message: Optional[nextcord.Message] = None
        self._build_components()

    def _build_components(self):
        self.clear_items()
        self.add_item(_AgentPicker(self.agent_id))
        self.add_item(_OptionPicker(self.agent_id, self.selected_key))
        self.add_item(_RefreshOptionsButton())
        self.add_item(_CloseButton())

    async def interaction_check(self, interaction: nextcord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("This panel belongs to someone else.", ephemeral=True)
            return False
        return True

    async def refresh(self, interaction: nextcord.Interaction):
        self._build_components()
        await interaction.response.edit_message(embed=_build_options_embed(self.agent_id, self.selected_key), view=self)

    async def refresh_message(self):
        if not self.message:
            return
        self._build_components()
        try:
            await self.message.edit(embed=_build_options_embed(self.agent_id, self.selected_key), view=self)
        except Exception:
            pass

    async def on_timeout(self):
        for c in self.children:
            try: c.disabled = True
            except Exception: pass
        if self.message:
            try: await self.message.edit(view=self)
            except Exception: pass

    def close_message(self) -> str:
        agent = _get_agent(self.agent_id) if self.agent_id else None
        if not agent:
            return "✖ Closed."
        present = list((agent.ollama_options or {}).keys())
        if present:
            preview = ", ".join(f"{k}={agent.ollama_options[k]}" for k in present[:4])
            if len(present) > 4:
                preview += f", +{len(present) - 4} more"
            return f"ℹ️ Options unchanged for **{agent.display_name}** ({preview})."
        return f"ℹ️ Options unchanged for **{agent.display_name}** (using all defaults)."


class _RefreshOptionsButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Refresh", style=nextcord.ButtonStyle.secondary, emoji="🔄", row=2)
    async def callback(self, interaction: nextcord.Interaction):
        await self.view.refresh(interaction)


async def open_options_panel(interaction: nextcord.Interaction, *, runner_agent_id: str):
    if not is_owner(interaction.user.id):
        await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
        return
    view = OptionsView(owner_id=interaction.user.id, agent_id=runner_agent_id)
    await interaction.response.send_message(embed=_build_options_embed(view.agent_id), view=view, ephemeral=True)
    view.message = await interaction.original_message()
