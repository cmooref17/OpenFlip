"""Provider registry — the single place that maps `agent.provider` to a
conversation implementation.

Before this existed, every call site branched `if provider == "anthropic":
... else: ...` inline, which meant adding a provider was N scattered edits.
Now the routing lives here; runtime.py / commands.py / tools call these
helpers and adding a fourth provider is a one-file change (plus the new
conversation class itself).

Routing contract (must stay in sync with agent._VALID_PROVIDERS):
  - "anthropic" → AnthropicConversation (OAuth via Claude Code creds)
  - "openai"    → OpenAIConversation (API key, Chat Completions)
  - anything else (including the "ollama" default) → DiscordConversation.
    Falling through to ollama for unknown values preserves the historical
    behavior; agent.py's normalizer already warns on unknown providers at
    load time.

Imports are lazy so importing this module never drags in a provider's
dependency chain the process doesn't use.
"""
from __future__ import annotations

from typing import Any


def conversation_class(provider: str) -> type:
    """Return the conversation class for a provider string."""
    if provider == "anthropic":
        from .anthropic_conversation import AnthropicConversation
        return AnthropicConversation
    if provider == "openai":
        from .openai_conversation import OpenAIConversation
        return OpenAIConversation
    from .conversation import DiscordConversation
    return DiscordConversation


def chat_message_class(provider: str) -> type:
    """Return the ChatMessage class used to append messages to a provider's
    conversation history.

    anthropic + openai share the dict-backed ChatMessage defined in
    anthropic_conversation (OpenAIConversation reuses it — see that module's
    import note); ollama uses its own from the vendored ollama_api package.
    """
    if provider in ("anthropic", "openai"):
        from .anthropic_conversation import ChatMessage
        return ChatMessage
    from .ollama_api import ChatMessage
    return ChatMessage


def make_conversation(agent: Any, conversation_id: str):
    """Construct the right conversation object for an agent.

    All conversation classes share the (conversation_id, agent) constructor
    signature. Caller is responsible for .load() — matching the historical
    inline-branch behavior in runtime.get_conversation.
    """
    return conversation_class(agent.provider)(conversation_id, agent)
