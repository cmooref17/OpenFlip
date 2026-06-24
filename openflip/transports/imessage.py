"""IMessageTransport — drives iMessage I/O via the `imsg` CLI.

macOS-only. Wraps `~/.local/bin/imsg` (or whatever path config specifies) —
the same tool OpenClaw uses. `imsg watch --json` streams inbound events
from `~/Library/Messages/chat.db` via FSEvents; `imsg send` writes
outbound messages through Messages.app via AppleScript.

Mirrors DiscordTransport's protocol surface so AgentRunner can pick this
transport via agent config without changes to its own interface.

This transport builds a transport-agnostic `InboundMessage` directly
and calls `AgentRunner._handle_inbound` (the iMessage-native sibling of
`_handle_message`). Both methods share the same downstream queue/worker;
the only difference is how the inbound payload is constructed from the
platform-specific event.

Required `Agent` / config fields (per agent):
  * agent.json `transport: "imessage"` — picks this transport
  * `integrations.imessage.agents.<agent_id>` block in config.json:
      handle           — iMessage email/phone this agent receives as
      imsg_path        — optional, defaults to ~/.local/bin/imsg
      allowlist_chats  — optional list[int], chat rowids to listen on
      respond_in       — inherits from agent.json `respond_in`

Permissions required on the host:
  * Full Disk Access — to read `~/Library/Messages/chat.db`
  * Automation: Messages — for `imsg send` AppleScript path

Coexistence with OpenClaw's iMessage channel: both will receive the
same incoming events. Either disable OpenClaw's iMessage adapter
(`openclaw channels remove --channel imessage`) or partition by
recipient handle (different agents respond to different addresses).
"""
from __future__ import annotations
from datetime import datetime, timezone
import asyncio
import contextlib
import json
import os
import sys
from typing import Optional, TYPE_CHECKING

from ..session import Session, InboundMessage, Attachment
from ..utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END

if TYPE_CHECKING:
    from ..runtime import AgentRunner


_DEFAULT_IMSG_PATH = os.path.expanduser("~/.local/bin/imsg")


def make_imessage_session(
    *,
    chat_id: int,
    speaker_handle: str,
    is_owner: bool,
    is_dm: bool,
    display_name: str,
) -> Session:
    """Build a Session for an iMessage chat.

    iMessage has no numeric user IDs — senders are email or phone
    handles (strings). We hash the handle into an int for `speaker_id`
    so it fits Session's int field, but the source-of-truth handle
    stays in display_name + the chat-level transport_id.

    Centralizes the `imessage:<chat_id>` prefix.
    """
    # Split identity: transport_id is the chat.db ROWID (numeric, the
    # routing key imsg send needs; what Session.channel_id_int returns).
    # conversation_id uses the participant handle for 1:1 chats (human-
    # readable file names, stable across chat.db ROWID resets) and falls
    # back to chat_id for group chats (no single handle for the group).
    if is_dm and speaker_handle:
        conv_suffix = speaker_handle
    else:
        conv_suffix = str(chat_id)
    conversation_id = f"imessage:{conv_suffix}"
    # Identity links: a 1:1 chat with a speaker listed in config.json's
    # `identity_links` ("imessage:<handle>" → canonical) shares its history
    # with the same person's linked sessions on other transports via
    # conversation_id "linked:<canonical>". DM-only, and routing/auth fields
    # (transport_id, handle, is_owner) are untouched — the link affects
    # conversation identity ONLY, never privilege.
    if is_dm and speaker_handle:
        from ..config_global import resolve_linked_conversation_id
        linked = resolve_linked_conversation_id("imessage", speaker_handle)
        if linked:
            conversation_id = linked
    return Session(
        transport="imessage",
        transport_id=str(chat_id),
        conversation_id=conversation_id,
        # Stable but non-cryptographic int from the handle. Used for
        # bot-self-echo filtering and owner-id matching only.
        speaker_id=abs(hash(speaker_handle)) % (2**31),
        speaker_role_ids=[],  # iMessage has no role concept
        is_owner=is_owner,
        is_dm=is_dm,
        display_name=display_name,
        # Raw handle — the auth source of truth (acl.is_owner/is_admin compare
        # this case-folded), distinct from the per-process-unstable speaker_id hash.
        handle=speaker_handle,
    )


class IMessageTransport:
    """iMessage Transport. macOS-only. Wraps the `imsg` CLI."""

    name: str = "imessage"

    def __init__(
        self,
        *,
        handle: str,
        imsg_path: str = _DEFAULT_IMSG_PATH,
        allowlist_chats: Optional[list[int]] = None,
        allowlist_senders: Optional[list[str]] = None,
    ):
        if sys.platform != "darwin":
            raise RuntimeError(
                "IMessageTransport requires macOS — imsg/Messages.app aren't "
                "available on this platform."
            )
        if not os.path.isfile(imsg_path) or not os.access(imsg_path, os.X_OK):
            raise RuntimeError(
                f"imsg CLI not found or not executable at {imsg_path}. "
                "Install from your operator's imsg distribution or adjust "
                "integrations.imessage.agents.<id>.imsg_path in config.json."
            )
        self.handle = handle
        self.imsg_path = imsg_path
        self.allowlist_chats = set(allowlist_chats or [])
        # Sender-handle allowlist (phone numbers, emails). If non-empty,
        # only messages from these handles are processed. If empty, ALL
        # senders pass through — that's an open-door config and the
        # operator should know they're running it. iMessage has no
        # auth/handshake (any handle that knows your number can text it),
        # so leaving this empty exposes every tool the agent has to any
        # passing stranger. The transport logs a loud warning at startup
        # in that case (see start()).
        self.allowlist_senders = set(s.lower().strip() for s in (allowlist_senders or []) if s)
        self._runner: Optional["AgentRunner"] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stop_requested: bool = False

    def attach_runner(self, runner: "AgentRunner") -> None:
        """Wire the owning AgentRunner. Called by AgentRunner.__init__."""
        self._runner = runner

    @property
    def bot_user_id(self) -> int:
        """Stable int derived from this agent's handle.

        Used for self-echo filtering. iMessage sender field is a string
        (email or phone) so we hash the handle the same way as in
        make_imessage_session — `is_from_me` from the imsg event is the
        primary self-filter, this is the secondary.
        """
        return abs(hash(self.handle)) % (2**31)

    # ---- Transport protocol methods ----

    # macOS Messages stores chat.db in SQLite WAL mode. Inserts go to
    # chat.db-wal, not chat.db, so FSEvents on chat.db never fire on new
    # messages and `imsg watch` is silent. We poll the SQLite DB directly
    # instead. The Python process needs Full Disk Access (inherited from
    # whatever launched it — Terminal.app or a launchd agent with FDA).

    _CHAT_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
    _POLL_INTERVAL_SEC = 2.0
    # macOS Messages stores dates as nanoseconds since 2001-01-01 (Mac
    # epoch). Add this to convert to Unix epoch seconds.
    _MAC_EPOCH_OFFSET = 978307200

    async def start(self) -> None:
        """Open chat.db read-only and poll for new messages every 2s.

        We're not spawning a subprocess anymore — the previous design used
        `imsg watch --json` but FSEvents don't fire on WAL-mode SQLite
        inserts so it never emitted events on this Mac. Direct polling
        works regardless of WAL mode.
        """
        if self._runner is None:
            print_ts(
                f"{COLOR_RED}IMessageTransport.start: no runner attached{COLOR_END}",
                error=True,
            )
            return
        agent_id = self._runner.agent.id

        # Verify we can actually read chat.db before declaring "online".
        # If FDA isn't granted to this process, sqlite3.connect will
        # succeed but the first query raises sqlite3.OperationalError.
        import sqlite3
        try:
            conn = sqlite3.connect(f"file:{self._CHAT_DB_PATH}?mode=ro", uri=True, timeout=5)
            cur = conn.execute("SELECT MAX(ROWID) FROM message")
            last_rowid = int(cur.fetchone()[0] or 0)
            conn.close()
        except Exception as e:
            print_ts(
                f"{COLOR_RED}IMessageTransport can't read {self._CHAT_DB_PATH}: {e}. "
                f"This process needs Full Disk Access. Grant it to whatever "
                f"launched openflip (Terminal.app, launchd plist, etc.) in "
                f"System Settings → Privacy & Security → Full Disk Access."
                f"{COLOR_END}",
                agent=agent_id, error=True,
            )
            return

        print_ts(
            f"IMessageTransport online (handle={self.handle}, "
            f"starting after rowid={last_rowid}, poll={self._POLL_INTERVAL_SEC}s)",
            agent=agent_id,
        )

        # Loud warning if running without a sender allowlist. iMessage has
        # no handshake; an empty allowlist means any stranger who knows
        # this number can text the agent and invoke every tool it has.
        # This is the only authentication layer; operators should know
        # they're running open.
        if not self.allowlist_senders:
            print_ts(
                f"{COLOR_RED}IMessageTransport SECURITY WARNING: no "
                f"allowlist_senders configured for handle={self.handle}. "
                f"ANY iMessage sender can reach this agent. Set "
                f"integrations.imessage.agents.{agent_id}.allowlist_senders "
                f"in config.json to a list of phone numbers/emails "
                f"(e.g. ['+15551234567', 'you@example.com']).{COLOR_END}",
                agent=agent_id, error=True,
            )
        else:
            print_ts(
                f"IMessageTransport: sender allowlist active "
                f"({len(self.allowlist_senders)} entries)",
                agent=agent_id,
            )

        while not self._stop_requested:
            try:
                new_events = await asyncio.to_thread(self._fetch_new_messages, last_rowid)
                for ev in new_events:
                    last_rowid = max(last_rowid, ev.get("id") or 0)
                    try:
                        await self._dispatch(ev)
                    except Exception as e:
                        print_ts(
                            f"{COLOR_YELLOW}imsg dispatch failed: {e}{COLOR_END}",
                            agent=agent_id, error=True,
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print_ts(
                    f"{COLOR_YELLOW}IMessageTransport poll error (continuing): {e}{COLOR_END}",
                    agent=agent_id, error=True,
                )
            try:
                await asyncio.sleep(self._POLL_INTERVAL_SEC)
            except asyncio.CancelledError:
                break

    def _fetch_new_messages(self, last_rowid: int) -> list[dict]:
        """Sync DB read — run inside asyncio.to_thread to avoid blocking
        the event loop. Returns events shaped like imsg watch JSON:
        {id, guid, text, sender, chat_id, is_from_me, created_at, attachments}.
        """
        import sqlite3
        events: list[dict] = []
        try:
            conn = sqlite3.connect(f"file:{self._CHAT_DB_PATH}?mode=ro", uri=True, timeout=5)
            # Schema notes:
            #   message.handle_id → handle.ROWID; handle.id is the email/phone string.
            #   chat_message_join links chat ↔ message.
            #   message.text may be NULL for attachment-only / tapbacks.
            #   message.date is nanoseconds since 2001-01-01 in modern macOS.
            rows = conn.execute(
                """
                SELECT m.ROWID, m.guid, m.text, h.id, m.is_from_me, cmj.chat_id, m.date
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                WHERE m.ROWID > ?
                ORDER BY m.ROWID ASC
                LIMIT 200
                """,
                (last_rowid,),
            ).fetchall()
            conn.close()
            for rowid, guid, text, sender, is_from_me, chat_id, date_ns in rows:
                ts = ""
                if date_ns:
                    try:
                        unix_s = int(date_ns) / 1_000_000_000 + self._MAC_EPOCH_OFFSET
                        ts = datetime.fromtimestamp(unix_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    except Exception:
                        ts = ""
                events.append({
                    "id": int(rowid),
                    "guid": guid or "",
                    "text": text or "",
                    "sender": sender or "",
                    "chat_id": int(chat_id) if chat_id is not None else None,
                    "is_from_me": bool(is_from_me),
                    "created_at": ts,
                    "attachments": [],
                })
        except sqlite3.OperationalError as e:
            # Bubble up so the outer loop logs and retries — typically
            # transient (DB locked during Messages.app's own write) or
            # an FDA revocation.
            raise
        return events

    async def _dispatch(self, event: dict) -> None:
        """Translate an imsg watch event into InboundMessage, dispatch.

        Event shape (sample, observed from `imsg watch --json`):
          {"guid": "...", "text": "...", "reactions": [], "id": 18149,
           "created_at": "2026-05-15T21:52:35.453Z",
           "sender": "you@example.com", "chat_id": 1,
           "attachments": [], "is_from_me": false}
        """
        if self._runner is None:
            return
        # Filter: self-echo
        if event.get("is_from_me"):
            return
        # Filter: chat allowlist (if configured)
        chat_id = event.get("chat_id")
        if self.allowlist_chats and chat_id not in self.allowlist_chats:
            return
        sender = event.get("sender") or ""
        text = event.get("text") or ""
        if not text:
            return  # ignore reactions, tapbacks, empty bodies
        # Filter: sender allowlist (if configured). This is the AUTH layer
        # for iMessage — no handshake exists, so any handle that knows the
        # agent's number can text it. The allowlist is the only thing
        # standing between a random stranger and full tool access. Drop
        # unlisted senders silently (no reply, no log noise) so probing
        # strangers don't get feedback.
        if self.allowlist_senders:
            sender_norm = sender.lower().strip()
            if sender_norm not in self.allowlist_senders:
                # Log at info level so the operator can see drops in audit,
                # but don't spam — one line per drop.
                print_ts(
                    f"{COLOR_YELLOW}imessage: dropped message from "
                    f"non-allowlisted sender '{sender}' (chat_id={chat_id}){COLOR_END}",
                    agent=getattr(self._runner, "agent", None) and self._runner.agent.id,
                )
                return
        # Determine DM vs group. imsg's chat_id 1-participant means DM
        # in practice — but the event doesn't carry that directly. For
        # v1, assume non-allowlisted chats are DMs (since the operator
        # is the only one likely to talk to the bot 1:1). Group support
        # is a future extension once `imsg chats` participant count is
        # plumbed through.
        is_dm = True

        # Mention detection: does the text contain this agent's handle
        # or display_name? Cheap substring check — good enough for v1.
        display = self._runner.agent.display_name or self._runner.agent.id
        mentions_us = (
            self.handle.split("@")[0].lower() in text.lower()
            or display.lower() in text.lower()
            or f"@{self._runner.agent.id}".lower() in text.lower()
        )

        # Owner check — iMessage identities are handle STRINGS (email/phone),
        # not numeric IDs. Compare the case-folded handle directly against the
        # configured owner handle, using the SAME normalization the sender
        # allowlist uses above (.lower().strip()).
        #
        # The previous implementation hashed the handle to an int and compared
        # it against a config int. Python salts str hashing per process
        # (PYTHONHASHSEED), so that hash changed every restart and could never
        # match any fixed config value — owner resolution was non-functional.
        from ..config_global import get_owner_handle
        try:
            owner_handle = get_owner_handle("imessage")
        except Exception:
            owner_handle = ""
        sender_norm = sender.lower().strip()
        is_owner = bool(owner_handle) and sender_norm == owner_handle
        # Stable-ish int derived from the handle, for INTERNAL keying only
        # (Session.speaker_id field, self-echo labels). NEVER used for
        # owner/admin auth — that goes through the handle comparison above.
        # NOTE: this hash is per-process unstable (PYTHONHASHSEED); fine for
        # in-process keying, but it must not back anything that has to survive
        # a restart. Conversation files are keyed by chat_id (see
        # make_imessage_session / conversation_id), not by this value.
        speaker_id_int = abs(hash(sender)) % (2**31)

        session = make_imessage_session(
            chat_id=int(chat_id) if chat_id is not None else 0,
            speaker_handle=sender,
            is_owner=is_owner,
            is_dm=is_dm,
            display_name=sender,
        )

        attachments = []
        for a in event.get("attachments") or []:
            attachments.append(Attachment(
                content_type=a.get("mime_type") or "application/octet-stream",
                filename=a.get("filename") or "attachment",
                local_path=a.get("path"),
            ))

        inbound = InboundMessage(
            session=session,
            text=text,
            sender_id=speaker_id_int,
            sender_display_name=sender,
            is_dm=is_dm,
            mentions_us=mentions_us,
            attachments=attachments,
        )

        # INTEGRATION POINT — see module docstring.
        # AgentRunner._handle_message is Discord-shaped; iMessage needs
        # a transport-agnostic dispatch path. Until that lands, log loudly.
        handler = getattr(self._runner, "_handle_inbound", None)
        if handler is None:
            print_ts(
                f"{COLOR_RED}AgentRunner has no _handle_inbound method yet — "
                f"iMessage event dropped. Finish Phase-2 decouple in runtime.py "
                f"(see openflip/transports/imessage.py docstring).{COLOR_END}",
                agent=self._runner.agent.id, error=True,
            )
            return
        await handler(inbound, transport=self)

    async def stop(self) -> None:
        """Stop the watch subprocess and reader task."""
        self._stop_requested = True
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._proc.kill()
            except ProcessLookupError:
                pass
            self._proc = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None

    @staticmethod
    def _addr_args(session_id) -> list[str] | None:
        """imsg addressing args for a target. A pure-digit value is a chat.db
        rowid → `--chat-id`; anything else is a participant handle (email or
        +phone) → `--to`. The handle is the STABLE identifier for a 1:1 chat:
        chat rowids reset across chat.db rebuilds, so handle-addressing is what
        makes sends survive a restart (the chat-id captured before the restart
        may no longer resolve). Returns None for an empty/unusable target."""
        s = str(session_id).strip()
        if not s:
            return None
        return ["--chat-id", s] if s.isdigit() else ["--to", s]

    async def send(self, session_id: str, text: str) -> None:
        """Send text to an iMessage chat via `imsg send` (handle or chat-id)."""
        addr = self._addr_args(session_id)
        if addr is None:
            print_ts(
                f"{COLOR_RED}IMessageTransport.send: empty session_id{COLOR_END}",
                error=True,
            )
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                self.imsg_path, "send",
                *addr,
                "--text", text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                print_ts(f"imsg send timed out after 30s", error=True)
                return
            if proc.returncode != 0:
                print_ts(
                    f"imsg send rc={proc.returncode}: {stderr.decode('utf-8', 'replace')[:300]}",
                    error=True,
                )
        except Exception as e:
            print_ts(f"imsg send failed: {e}", error=True)

    async def send_file(self, session_id: str, path: str, content: str = "") -> None:
        """Send a file attachment via `imsg send --file` (handle or chat-id)."""
        addr = self._addr_args(session_id)
        if addr is None:
            return
        args = [self.imsg_path, "send", *addr, "--file", path]
        if content:
            args.extend(["--text", content])
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                return
            if proc.returncode != 0:
                print_ts(
                    f"imsg send (file) rc={proc.returncode}: {stderr.decode('utf-8', 'replace')[:300]}",
                    error=True,
                )
        except Exception as e:
            print_ts(f"imsg send (file) failed: {e}", error=True)

    @contextlib.asynccontextmanager
    async def typing(self, session_id: str):
        """iMessage has no programmatic typing-indicator API. No-op."""
        yield

    async def resolve_session_for_user(self, user_id: int) -> Optional[Session]:
        """iMessage doesn't expose user→DM resolution easily — no-op for v1.

        A future implementation could shell out to `imsg chats --json`,
        filter for a 1-participant chat with the target handle, and
        synthesize a Session. Not worth the complexity until a caller
        actually needs it.
        """
        return None

    async def fetch_message(self, session_id: str, message_id: str) -> Optional[InboundMessage]:
        """Fetch a historical message via `imsg history`. Best-effort."""
        try:
            chat_id = int(session_id)
        except (ValueError, TypeError):
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                self.imsg_path, "history",
                "--chat-id", str(chat_id),
                "--limit", "50",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        except Exception:
            return None
        for line in stdout.decode("utf-8", "replace").splitlines():
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if str(evt.get("id")) != str(message_id) and evt.get("guid") != message_id:
                continue
            sender = evt.get("sender") or ""
            text = evt.get("text") or ""
            speaker_id_int = abs(hash(sender)) % (2**31)
            session = make_imessage_session(
                chat_id=chat_id,
                speaker_handle=sender,
                is_owner=False,
                is_dm=True,
                display_name=sender,
            )
            return InboundMessage(
                session=session,
                text=text,
                sender_id=speaker_id_int,
                sender_display_name=sender,
                is_dm=True,
                mentions_us=False,
                attachments=[],
            )
        return None
