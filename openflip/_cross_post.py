"""Record a message an agent posted into ANOTHER conversation into that conversation's own history.

Why (Flip 2026-10-22: "WHY WOULD A DM TO SOMEONE NOT GO INTO THEIR SESSION?"): every conversation
keeps its own history. When an agent posts into a different chat than the one its turn runs in
(discord_post send/reply to another channel or a DM, send_message to another session), the text
reached the person but NOT the agent's history of that chat. When the person answered, the agent
in that chat had never seen its own question ("Yes what, Zane??").

Fix: after a successful cross-conversation post, append the posted text to the target
conversation's history as the agent's own (assistant) message, plus a short note saying it was
sent from another chat, using the same safe write path inject_context uses:
  * target conversation loaded in memory -> append to the live list + save();
  * target has a turn in flight -> defer through the runner's _pending_inject buffer (lands after
    that turn, never wedged into a tool_use/tool_result pair; arrives as an [INJECTED CONTEXT] note);
  * not loaded -> append to its JSONL on disk (loaded fresh on the next turn there).
Posts into the CURRENT conversation are skipped: the normal turn already records those.
Never raises: a failure here must not turn a delivered message into a failed tool call.
"""
from __future__ import annotations

import os

from .utils import print_ts


def _current_conversation_id() -> str:
    from .tool_executor import CURRENT_SESSION
    try:
        sess = CURRENT_SESSION.get(None)
    except Exception:
        sess = None
    return (getattr(sess, "conversation_id", "") or "") if sess is not None else ""


def record_cross_post(conversation_id: str, text: str, origin_note: str = "") -> str:
    """Append `text` (what this agent just posted) to the history of `conversation_id`.

    Returns a short status for logs/tool feedback ('' when skipped because it is the current
    conversation or there is nothing to record)."""
    try:
        from . import registry, _conversation_io as _cio
        from .tool_executor import CURRENT_AGENT
        text = (text or "").strip()
        if not text or not conversation_id:
            return ""
        conv_id = _cio._safe_conversation_id(conversation_id)
        current = _current_conversation_id()
        if conv_id == current:
            return ""
        agent = CURRENT_AGENT.get()
        runner = registry.RUNNERS.get(agent.id)
        note = origin_note or f"(sent from my other conversation {current or 'a background turn'}; "\
                              f"the person here saw this message)"
        body = f"{text}\n\n[{note}]"

        live_conv, live_key = None, None
        if runner is not None:
            for k, c in runner.conversations.items():
                if getattr(c, "conversation_id", None) == conv_id:
                    live_conv, live_key = c, k
                    break

        if live_conv is not None:
            active = runner._active_turns.get(live_key)
            if active is not None and not active.done():
                # mid-turn there: the pending buffer only carries user-role text
                runner._pending_inject.setdefault(live_key, []).append(
                    f"[INJECTED CONTEXT]: I (this agent) posted this into this chat from another "
                    f"conversation: \"{text}\"")
                print_ts(f"cross-post record: {conv_id} mid-turn -> queued", agent=agent.id)
                return f"recorded in {conv_id} (queued, it was mid-turn)"
            from .providers import chat_message_class
            Msg = chat_message_class(agent.provider)
            live_conv.messages.append(Msg("assistant", body))
            live_conv.save()
            print_ts(f"cross-post record: appended to live {conv_id}", agent=agent.id)
            return f"recorded in {conv_id}"

        path = _cio.conversation_path(os.path.dirname(agent.path), conv_id)
        _cio.append_messages(path, [{"role": "assistant", "content": body}],
                             content_extractor=lambda m: m.get("content", ""))
        print_ts(f"cross-post record: appended to {conv_id} JSONL", agent=agent.id)
        return f"recorded in {conv_id}"
    except Exception as e:  # never break the send itself
        try:
            print_ts(f"cross-post record FAILED for {conversation_id}: {e}")
        except Exception:
            pass
        return f"NOT recorded in {conversation_id}: {e}"
