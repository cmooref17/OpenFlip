"""ExternalTransport — authenticated HTTPS ingress for external programs.

Lets an off-box program (a game mod, a webhook, a CLI) POST a single message
to one agent over HTTPS and get the agent's reply back synchronously in the
response body. This transport is built for reach-out-from-the-internet use:

  - Binds 0.0.0.0 (the configured port is open / off-network reachable).
  - TLS with a self-signed cert auto-generated on first start (10-year life).
  - Per-token config in a gitignored token file: each bearer token pins the
    agent, a FIXED session name, a default model, optional model-choice, and a
    per-token rate limit. Tokens never choose their session — it is operator-
    assigned, so a caller can only ever talk into the conversation the operator
    bound their token to.

Endpoint:  POST /<agent_id>            (path agent_id must equal this agent)
Auth:      Authorization: Bearer <token>
Request:   {"message": <str>, "sender_label": <str?>, "model": <str?>}
Response:  200 {"reply": "<agent text>"}   |   4xx/5xx {"error": "..."}

Security posture (fail closed everywhere):
  - Missing/empty token file → every request 401. No token → 401.
  - Token compared in constant time (hmac.compare_digest), no early-out.
  - The session is keyed "external:<name>" and is_owner is hard-False
    (make_external_session), so owner-only tools/disclosure never unlock and a
    tool needs an explicit `auth.external` ACL block to be callable at all.
  - Body capped at 16 KiB → 413. Per-token sliding-window rate limit → 429.
  - Per-session lock serializes concurrent same-token requests so two callers
    can't interleave into one conversation's turn.

Reply capture: the runtime fires a turn fire-and-forget (queue → worker →
`_serialized_turn`) and posts the final assistant text by chunking it through
`channel.send()` → `transport.send()`. We accumulate every chunk for the
session into a per-request buffer and await the turn task to completion, then
join the buffer. This captures the WHOLE reply (multi-chunk included), unlike a
resolve-on-first-send future. See `_handle_request` for the mechanism detail.
"""
from __future__ import annotations
import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import ssl
import time
from typing import Optional, TYPE_CHECKING, AsyncContextManager, Any

from aiohttp import web

from ..session import Session, InboundMessage, make_external_session
from ..utils import print_ts, COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_END, project_root

if TYPE_CHECKING:
    from ..runtime import AgentRunner


# Hard cap on request body size. aiohttp enforces this at read time and raises
# HTTPRequestEntityTooLarge, which we translate to a clean 413 JSON response.
_MAX_BODY_BYTES = 16 * 1024


class ExternalTransport:
    """Authenticated HTTPS transport for external callers. One per agent."""

    name: str = "external"

    def __init__(
        self,
        *,
        port: int = 1780,
        bind_host: str = "0.0.0.0",
        cert_dir: str = "",
        cert_path: str = "",
        key_path: str = "",
        token_path: str = "",
        request_timeout: float = 120.0,
    ):
        self.port = int(port)
        self.bind_host = bind_host or "0.0.0.0"
        self.request_timeout = float(request_timeout) if request_timeout else 120.0

        root = project_root()
        cert_dir = cert_dir or os.path.join(root, "external_cert")
        self.cert_dir = cert_dir
        # Cert/key default into the cert_dir; explicit paths override.
        self.cert_path = cert_path or os.path.join(cert_dir, "external_cert.pem")
        self.key_path = key_path or os.path.join(cert_dir, "external_key.pem")
        self.token_path = token_path or os.path.join(root, "external_tokens.json")

        self._runner: Optional["AgentRunner"] = None
        self._app: Optional[web.Application] = None
        self._site: Optional[web.TCPSite] = None
        self._app_runner: Optional[web.AppRunner] = None

        # Token table: {token_string: entry_dict}. Reloaded from disk whenever
        # the file's mtime changes (per-request stat-check — no SIGHUP needed,
        # so the operator can edit the token file at runtime and the next
        # request picks it up). Empty table = fail closed (every request 401).
        self._tokens: dict[str, dict] = {}
        self._tokens_mtime: float = -1.0

        # Per-token sliding-window request timestamps for rate limiting.
        self._rate_hits: dict[str, list[float]] = {}

        # Per-request reply capture: {session_name: [chunk, ...]}. send()
        # appends; the handler joins and clears.
        self._captures: dict[str, list[str]] = {}
        # Per-session async lock so two requests for the same bound session
        # serialize (one turn at a time per conversation).
        self._session_locks: dict[str, asyncio.Lock] = {}

    def attach_runner(self, runner: "AgentRunner") -> None:
        """Wire the owning AgentRunner. Called by AgentRunner.__init__."""
        self._runner = runner

    @property
    def bot_user_id(self) -> int:
        """Stable int for this transport. HTTPS ingress has no bot user."""
        return 0

    # ------------------------------------------------------------------ tokens

    def _agent_id(self) -> str:
        return self._runner.agent.id if self._runner else "external"

    def _load_tokens(self, *, force: bool = False) -> None:
        """(Re)load the token table if the file changed.

        Fail-closed: any error (missing file, bad JSON, wrong shape) empties the
        table so every request is rejected with 401 until the file is fixed.
        """
        try:
            st = os.stat(self.token_path)
        except FileNotFoundError:
            if self._tokens:
                print_ts(
                    f"{COLOR_RED}ExternalTransport: token file vanished "
                    f"({self.token_path}); rejecting all requests.{COLOR_END}",
                    agent=self._agent_id(), error=True,
                )
            self._tokens = {}
            self._tokens_mtime = -1.0
            return
        except Exception as e:
            print_ts(
                f"{COLOR_RED}ExternalTransport: cannot stat token file "
                f"{self.token_path}: {e}; rejecting all requests.{COLOR_END}",
                agent=self._agent_id(), error=True,
            )
            self._tokens = {}
            self._tokens_mtime = -1.0
            return

        if not force and st.st_mtime == self._tokens_mtime:
            return

        try:
            with open(self.token_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print_ts(
                f"{COLOR_RED}ExternalTransport: token file {self.token_path} is "
                f"unparseable ({e}); rejecting all requests until fixed.{COLOR_END}",
                agent=self._agent_id(), error=True,
            )
            self._tokens = {}
            self._tokens_mtime = st.st_mtime
            return

        # Accept either a flat {token: entry} map or {"tokens": {token: entry}}.
        table = raw.get("tokens") if isinstance(raw, dict) and "tokens" in raw else raw
        parsed: dict[str, dict] = {}
        if isinstance(table, dict):
            for tok, entry in table.items():
                if not isinstance(tok, str) or len(tok) < 32:
                    print_ts(
                        f"{COLOR_YELLOW}ExternalTransport: skipping token with "
                        f"<32 chars (weak/typo) in {self.token_path}.{COLOR_END}",
                        agent=self._agent_id(),
                    )
                    continue
                if not isinstance(entry, dict):
                    continue
                parsed[tok] = entry

        self._tokens = parsed
        self._tokens_mtime = st.st_mtime
        if parsed:
            print_ts(
                f"ExternalTransport: loaded {len(parsed)} token(s) from "
                f"{self.token_path}",
                agent=self._agent_id(),
            )
        else:
            print_ts(
                f"{COLOR_RED}ExternalTransport: token file {self.token_path} has "
                f"no usable tokens; rejecting all requests.{COLOR_END}",
                agent=self._agent_id(), error=True,
            )

    def _match_token(self, provided: str) -> Optional[dict]:
        """Constant-time lookup of a bearer token. Returns its entry or None.

        Compares against EVERY known token with hmac.compare_digest and never
        breaks early, so the time taken doesn't leak which (if any) token
        matched. Only entries pinned to THIS agent are eligible.
        """
        match: Optional[dict] = None
        agent_id = self._agent_id()
        provided_b = provided.encode("utf-8")
        for tok, entry in self._tokens.items():
            same = hmac.compare_digest(tok.encode("utf-8"), provided_b)
            if same and entry.get("agent") == agent_id:
                match = entry
        return match

    def _check_rate_limit(self, token_key: str, limit_per_min: int) -> bool:
        """Sliding-window rate limit. True if the request is allowed."""
        now = time.time()
        cutoff = now - 60.0
        hits = self._rate_hits.setdefault(token_key, [])
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= max(1, int(limit_per_min)):
            return False
        hits.append(now)
        return True

    # --------------------------------------------------------------- handler

    async def _handle_request(self, request: web.Request) -> web.Response:
        """POST /<agent_id> — validate, dispatch one turn, return the reply."""
        if not self._runner:
            return web.json_response({"error": "transport not initialized"}, status=500)
        agent_id = self._runner.agent.id

        # Path agent must match this transport's agent — no cross-agent ingress.
        if request.match_info.get("agent_id", "") != agent_id:
            return web.json_response({"error": "not found"}, status=404)

        # Agent must be enabled to take a turn (operator can disable it without
        # tearing down the transport).
        try:
            from ..persistence import is_enabled
            if not is_enabled(agent_id):
                return web.json_response({"error": "agent unavailable"}, status=503)
        except Exception:
            pass

        # Auth — Bearer token, fail closed. Reload table if the file changed.
        self._load_tokens()
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return web.json_response({"error": "missing bearer token"}, status=401)
        provided = auth[7:].strip()
        if not provided:
            return web.json_response({"error": "missing bearer token"}, status=401)
        entry = self._match_token(provided)
        if entry is None:
            return web.json_response({"error": "invalid token"}, status=401)

        # Per-token rate limit. Key by a hash of the token (never store/echo it).
        token_key = hashlib.sha256(provided.encode("utf-8")).hexdigest()
        rate_limit = int(entry.get("rate_limit", 20) or 20)
        if not self._check_rate_limit(token_key, rate_limit):
            return web.json_response(
                {"error": f"rate limit exceeded ({rate_limit}/min)"}, status=429,
            )

        # Body — read raw bytes (16 KiB cap) and parse JSON ourselves so a
        # wrong/absent Content-Type still works; only malformed JSON is a 400.
        try:
            body_bytes = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "request body too large"}, status=413)
        except Exception:
            return web.json_response({"error": "could not read body"}, status=400)
        if len(body_bytes) > _MAX_BODY_BYTES:
            return web.json_response({"error": "request body too large"}, status=413)
        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            return web.json_response({"error": "malformed JSON body"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)

        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            return web.json_response({"error": "missing or empty 'message'"}, status=400)
        message = message.strip()

        sender_label = body.get("sender_label")
        if not isinstance(sender_label, str):
            sender_label = ""
        sender_label = sender_label.strip()[:120]

        # Model resolution — operator policy on the token decides.
        default_model = entry.get("default_model")
        if not isinstance(default_model, str) or not default_model.strip():
            print_ts(
                f"{COLOR_RED}ExternalTransport: token for {agent_id} has no "
                f"default_model; rejecting.{COLOR_END}",
                agent=agent_id, error=True,
            )
            return web.json_response({"error": "server misconfigured"}, status=500)
        chosen_model = default_model
        requested_model = body.get("model")
        if isinstance(requested_model, str) and requested_model.strip():
            requested_model = requested_model.strip()
            if entry.get("allow_model_choice"):
                allowed = entry.get("allowed_models") or []
                if requested_model in allowed:
                    chosen_model = requested_model
                else:
                    return web.json_response(
                        {"error": f"model '{requested_model}' not allowed",
                         "allowed": list(allowed)},
                        status=400,
                    )
            # allow_model_choice false → silently ignore the requested model and
            # use default_model (the token's pinned policy wins).

        # Session is keyed by the OPERATOR-assigned name on the token, never by
        # anything in the body. All turns on this token land in one conversation.
        session_name = entry.get("session")
        if not isinstance(session_name, str) or not session_name.strip():
            print_ts(
                f"{COLOR_RED}ExternalTransport: token for {agent_id} has no "
                f"'session' name; rejecting.{COLOR_END}",
                agent=agent_id, error=True,
            )
            return web.json_response({"error": "server misconfigured"}, status=500)
        session_name = session_name.strip()

        session = make_external_session(session_name, speaker_label=sender_label)
        text = f"[{sender_label}]: {message}" if sender_label else message

        inbound = InboundMessage(
            session=session,
            text=text,
            sender_id=session.speaker_id,
            sender_display_name=sender_label or f"external:{session_name}",
            is_dm=True,
            mentions_us=False,
            attachments=[],
        )

        # Serialize same-session requests so concurrent callers on one token
        # don't interleave into the same conversation's turn.
        lock = self._session_locks.setdefault(session_name, asyncio.Lock())
        async with lock:
            return await self._dispatch_and_capture(
                inbound=inbound,
                session_name=session_name,
                chosen_model=chosen_model,
                agent_id=agent_id,
            )

    async def _dispatch_and_capture(
        self, *, inbound: InboundMessage, session_name: str,
        chosen_model: str, agent_id: str,
    ) -> web.Response:
        """Run one turn and capture its full final text.

        Mechanism: pre-create the conversation so we can set a per-turn model
        override on it (cleared in `finally` so it never bleeds into a later
        turn), reset a per-session chunk buffer that send() appends to, fire the
        turn through the normal runtime path, then await the turn task and join
        the buffer. shield() keeps a wait_for timeout from cancelling the real
        turn.
        """
        runner = self._runner
        # Conversation key == TransportChannel.id for this session: a non-numeric
        # transport_id hashes to this stable int (channel_shim), and an external
        # conversation_id isn't "linked:" so conv_key_for_session returns it
        # unchanged. This matches the key the worker registers the turn under.
        conv_key = abs(hash(session_name)) % (2**31)
        conv_id = inbound.session.conversation_id  # "external:<name>"

        # Per-turn model override on the (get-or-create) conversation object.
        # AnthropicConversation honors it at its every-turn model resync; other
        # providers simply lack the hook (guarded).
        conv = None
        override_set = False
        try:
            conv = runner.get_conversation(conv_key, conversation_id=conv_id)
            if hasattr(conv, "set_model_override"):
                conv.set_model_override(chosen_model)
                override_set = True
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}ExternalTransport: could not pre-set model "
                f"override ({e}); proceeding on agent default.{COLOR_END}",
                agent=agent_id,
            )

        # Reset the capture buffer for this session BEFORE dispatch.
        self._captures[session_name] = []

        try:
            prev = runner._active_turns.get(conv_key)
            await runner._handle_inbound(inbound, transport=self)

            # Capture the freshly-registered turn task. The worker registers the
            # slot synchronously the instant it dequeues our item (no await
            # between get() and the write), so a single event-loop yield lands
            # it; we spin briefly to be safe. `is not prev` skips a stale slot.
            loop = asyncio.get_event_loop()
            deadline = loop.time() + self.request_timeout
            turn_task: Optional[asyncio.Task] = None
            while loop.time() < deadline:
                cand = runner._active_turns.get(conv_key)
                if cand is not None and cand is not prev:
                    turn_task = cand
                    break
                await asyncio.sleep(0.01)

            if turn_task is None:
                # Never observed a turn. Either the worker is wedged or the turn
                # finished inside the capture window with empty output — return
                # whatever the buffer holds rather than a misleading 504.
                buffered = "".join(self._captures.get(session_name, []))
                if buffered.strip():
                    return web.json_response({"reply": buffered})
                return web.json_response(
                    {"error": f"agent did not start a turn within "
                              f"{int(self.request_timeout)}s"},
                    status=504,
                )

            remaining = max(1.0, deadline - loop.time())
            try:
                await asyncio.wait_for(asyncio.shield(turn_task), timeout=remaining)
            except asyncio.TimeoutError:
                return web.json_response(
                    {"error": f"agent response timeout "
                              f"({int(self.request_timeout)}s)"},
                    status=504,
                )

            reply = "".join(self._captures.get(session_name, []))
            return web.json_response({"reply": reply})
        except Exception as e:
            print_ts(
                f"{COLOR_RED}ExternalTransport handler error for {agent_id}: "
                f"{e}{COLOR_END}",
                agent=agent_id, error=True,
            )
            return web.json_response({"error": "internal error"}, status=500)
        finally:
            self._captures.pop(session_name, None)
            if override_set and conv is not None:
                with contextlib.suppress(Exception):
                    conv.set_model_override(None)

    # --------------------------------------------------------------- TLS cert

    def _ensure_cert(self) -> bool:
        """Ensure a TLS cert+key exist at the configured paths.

        Loads them if present. Otherwise auto-generates a self-signed cert with
        a 10-year validity (cryptography lib preferred, openssl CLI fallback),
        writing the key with 0600. Logs the cert's SHA-256 fingerprint. Returns
        False if no cert could be made available (caller fails startup loudly).
        """
        if os.path.isfile(self.cert_path) and os.path.isfile(self.key_path):
            self._log_cert_fingerprint()
            return True

        os.makedirs(self.cert_dir, exist_ok=True)
        if self._gen_cert_cryptography() or self._gen_cert_openssl():
            try:
                os.chmod(self.key_path, 0o600)
            except Exception:
                pass
            self._log_cert_fingerprint()
            return True
        return False

    def _gen_cert_cryptography(self) -> bool:
        """Generate a self-signed cert via the `cryptography` lib. False if the
        lib is unavailable or generation fails."""
        try:
            import datetime
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
        except Exception:
            return False
        try:
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, f"openflip-external-{self._agent_id()}"),
            ])
            now = datetime.datetime.utcnow()
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=3650))
                .sign(key, hashes.SHA256())
            )
            with open(self.key_path, "wb") as f:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ))
            with open(self.cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            print_ts(
                f"ExternalTransport: generated self-signed cert (cryptography) "
                f"→ {self.cert_path}",
                agent=self._agent_id(),
            )
            return True
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}ExternalTransport: cryptography cert-gen failed "
                f"({e}); trying openssl.{COLOR_END}",
                agent=self._agent_id(),
            )
            return False

    def _gen_cert_openssl(self) -> bool:
        """Generate a self-signed cert by shelling out to openssl. False on
        failure (openssl missing, nonzero exit)."""
        import shutil
        import subprocess
        if not shutil.which("openssl"):
            return False
        try:
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", self.key_path, "-out", self.cert_path,
                    "-days", "3650",
                    "-subj", f"/CN=openflip-external-{self._agent_id()}",
                ],
                check=True, capture_output=True, timeout=60,
            )
            print_ts(
                f"ExternalTransport: generated self-signed cert (openssl) "
                f"→ {self.cert_path}",
                agent=self._agent_id(),
            )
            return True
        except Exception as e:
            print_ts(
                f"{COLOR_RED}ExternalTransport: openssl cert-gen failed: "
                f"{e}{COLOR_END}",
                agent=self._agent_id(), error=True,
            )
            return False

    def _log_cert_fingerprint(self) -> None:
        """Log the SHA-256 fingerprint of the cert so the operator can pin it."""
        try:
            with open(self.cert_path, "rb") as f:
                pem = f.read()
            der = ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
            fp = hashlib.sha256(der).hexdigest()
            fp_colon = ":".join(fp[i:i + 2] for i in range(0, len(fp), 2)).upper()
            print_ts(
                f"{COLOR_GREEN}ExternalTransport cert SHA-256: {fp_colon}{COLOR_END}",
                agent=self._agent_id(),
            )
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}ExternalTransport: could not compute cert "
                f"fingerprint: {e}{COLOR_END}",
                agent=self._agent_id(),
            )

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the HTTPS server. Fails closed on missing tokens or cert."""
        if not self._runner:
            print_ts(
                f"{COLOR_RED}ExternalTransport.start: no runner attached{COLOR_END}",
                error=True,
            )
            return
        agent_id = self._runner.agent.id

        # Load tokens up front. No tokens → bind nothing, idle forever, so the
        # port never accepts unauthenticated traffic.
        self._load_tokens(force=True)
        if not self._tokens:
            print_ts(
                f"{COLOR_RED}ExternalTransport: startup BLOCKED for {agent_id} — "
                f"no valid tokens in {self.token_path}. Add a token to enable.{COLOR_END}",
                agent=agent_id, error=True,
            )
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            return

        # Cert is mandatory — fail loudly if we can't make one available.
        if not self._ensure_cert():
            print_ts(
                f"{COLOR_RED}ExternalTransport: startup FAILED for {agent_id} — "
                f"could not load or generate a TLS cert at {self.cert_path}. "
                f"Install `cryptography` or `openssl`.{COLOR_END}",
                agent=agent_id, error=True,
            )
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            return

        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
        except Exception as e:
            print_ts(
                f"{COLOR_RED}ExternalTransport: TLS context build failed for "
                f"{agent_id} ({e}); cert unreadable. Startup blocked.{COLOR_END}",
                agent=agent_id, error=True,
            )
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            return

        self._app = web.Application(client_max_size=_MAX_BODY_BYTES)
        # ONE route: POST /{agent_id} (dynamic). The handler verifies the path
        # segment equals THIS agent's id via request.match_info["agent_id"] and
        # 404s otherwise. Registering it dynamic (not the literal f"/{agent_id}")
        # is REQUIRED: a static path populates no match_info, so the handler's
        # agent-id check would read "" and 404 every request.
        self._app.router.add_post("/{agent_id}", self._handle_request)

        self._app_runner = web.AppRunner(self._app)
        await self._app_runner.setup()
        self._site = web.TCPSite(
            self._app_runner, self.bind_host, self.port, ssl_context=ssl_ctx,
        )
        await self._site.start()

        print_ts(
            f"{COLOR_GREEN}ExternalTransport online: https://{self.bind_host}:"
            f"{self.port}/{agent_id} (0.0.0.0 bind, TLS, bearer-token auth, "
            f"{len(self._tokens)} token(s)){COLOR_END}",
            agent=agent_id,
        )

    async def stop(self) -> None:
        """Stop the HTTPS server and clean up."""
        if self._site:
            with contextlib.suppress(Exception):
                await self._site.stop()
            self._site = None
        if self._app_runner:
            with contextlib.suppress(Exception):
                await self._app_runner.cleanup()
            self._app_runner = None
        self._app = None

    # ----------------------------------------------------------- Transport API

    async def send(self, session_id: str, text: str) -> None:
        """Capture an outbound chunk for an in-flight external request.

        The runtime posts a final reply by chunking it through channel.send →
        this method; we append each chunk to the session's capture buffer so the
        handler can join them into the full reply. With no in-flight request for
        the session (buffer absent), the chunk is dropped — HTTPS ingress has no
        push channel for unsolicited messages.
        """
        buf = self._captures.get(session_id)
        if buf is not None:
            buf.append(text)
        # else: no waiting request — nothing to deliver to. Drop silently.

    async def send_file(self, session_id: str, path: str, content: str = "") -> Optional[str]:
        """No file delivery over a synchronous HTTPS reply. No-op."""
        return None

    @contextlib.asynccontextmanager
    async def typing(self, session_id: str) -> AsyncContextManager[Any]:
        """No typing indicator for HTTPS. No-op."""
        yield

    async def resolve_session_for_user(self, user_id: int) -> Optional[Session]:
        """No user-id session resolution for external ingress. No-op."""
        return None

    async def fetch_message(self, session_id: str, message_id: str) -> Optional[InboundMessage]:
        """External ingress stores no messages. No-op."""
        return None
