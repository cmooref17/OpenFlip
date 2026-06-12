"""Fetch a URL and return its content as text.

Uses aiohttp to GET a URL and returns the response body. HTML is converted
to readable text by stripping tags. Binary content is rejected.

Sends realistic browser headers so Cloudflare-protected sites (and other
bot-screening WAFs) don't reject the request outright. Some endpoints will
still block based on JS challenges or fingerprinting; nothing we can do
about those without a headless browser.

SSRF guard: fetching PRIVATE / INTERNAL / link-local / loopback / cloud-
metadata addresses is restricted to the owner/admins. The host is DNS-
resolved ONCE, EVERY resolved address is checked, and the connection is
PINNED to the vetted address (the request URL carries the IP; Host header +
TLS SNI carry the original hostname) — so the client never re-resolves the
name, closing the resolve-twice DNS-rebinding race as well as the plain
rebinding-to-localhost case. The check+pin is re-run on each redirect hop
(closes a public URL 302-ing to http://169.254.169.254/). Public addresses
stay open to everyone. Owner/admins retain full access so the owner can
still hit localhost services (webapp, etc.) — privileged fetches skip
the guard AND the pin entirely.

For non-privileged callers resolution failure fails CLOSED (no pin → no
fetch), so every non-privileged connection goes to a guard-vetted address —
there is no unpinned fallback path.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
from urllib.parse import urljoin, urlparse

import aiohttp

from ._base import tool, ToolResult, TOOL_REGISTRY
from ..utils import print_ts, http_session, COLOR_YELLOW, COLOR_END


_MAX_BYTES = 100_000  # ~100 KB text cap
_TIMEOUT = 20  # seconds
_MAX_REDIRECTS = 5  # match the previous allow_redirects cap

# Statuses aiohttp would normally treat as redirects.
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

# Realistic Chrome-on-Linux fingerprint. Matched headers (UA + sec-ch-ua) so
# Cloudflare/Akamai don't flag the mismatch.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def _strip_html(html: str) -> str:
    """Crude HTML-to-text: strip tags, collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- SSRF guard helpers ----------------------------------------------------

def _is_internal_ip(ip_str: str) -> bool:
    """True if `ip_str` is a private/internal/reserved address.

    Covers loopback (127/8, ::1), private (10/8, 172.16/12, 192.168/16,
    fc00::/7), link-local (169.254/16 incl. the 169.254.169.254 cloud-metadata
    IP, fe80::/10), reserved blocks, and 0.0.0.0/unspecified. Non-IP strings
    return False (they're not an internal literal; DNS resolution handles
    hostnames).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


async def _resolve_and_vet_host(host: str) -> tuple[bool, str]:
    """DNS-resolve `host` ONCE and return (internal, pinned_ip).

    `internal` is True if ANY resolved address is private/internal — the
    caller must block the fetch. `pinned_ip` is the first vetted (public)
    address; the caller must CONNECT TO THAT ADDRESS, not re-resolve the
    hostname. Resolving once for the check and letting the HTTP client
    resolve again for the connect is a DNS-rebinding TOCTOU: an attacker-
    controlled DNS server answers the guard's lookup with a public IP and
    the client's lookup with 127.0.0.1 / 169.254.169.254. Pinning closes it.

    IP literals resolve to themselves. Fail-open ONLY on resolution failure —
    (False, "") lets the normal request path surface the connect error; we
    fail CLOSED whenever resolution succeeds and yields an internal IP. Uses
    the loop's threadpool resolver so we never block the event loop.
    """
    if not host:
        return (False, "")
    host = host.strip("[]")  # IPv6 literals arrive bracketed from urlparse
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except Exception:
        return (False, "")  # resolution failed entirely → don't special-case
    # Pinning collapses the address list to ONE — prefer IPv4 so a host with
    # broken/absent IPv6 routing doesn't lose the v4 fallback aiohttp's own
    # resolver iteration used to provide.
    pinned = ""
    pinned_v4 = ""
    for info in infos:
        ip_str = info[4][0]  # sockaddr is (address, port[, flowinfo, scopeid])
        if _is_internal_ip(ip_str):
            return (True, "")
        if not pinned:
            pinned = ip_str
        if not pinned_v4 and ":" not in ip_str:
            pinned_v4 = ip_str
    return (False, pinned_v4 or pinned)


def _caller_is_owner_or_admin() -> bool:
    """Whether the current tool caller is the owner or an admin.

    Mirrors the ACL-context derivation in runtime.py (~1375-1396): the Session
    (when present) is the source of truth for transport + handle on handle-based
    transports (iMessage), while Discord uses the numeric speaker_id. We then
    defer to acl.is_admin (owner + admins, transport-aware) rather than hand-
    rolling an owner_id compare, so owner+admins both pass and the transport
    logic stays correct. Fail closed: any error / unknown caller → False.
    """
    from ..acl import is_admin
    from ..tool_executor import CURRENT_SPEAKER_ID, CURRENT_SESSION

    try:
        speaker_id = int(CURRENT_SPEAKER_ID.get(None) or 0)
    except Exception:
        speaker_id = 0

    transport = "discord"
    handle = ""
    sess = CURRENT_SESSION.get(None)
    if sess is not None:
        transport = getattr(sess, "transport", "discord") or "discord"
        if transport != "discord":
            # Handle-based transports: the raw handle is the auth key (the int
            # speaker_id is a per-process-unstable hash and never matches).
            handle = getattr(sess, "handle", "") or ""

    try:
        return is_admin(speaker_id, transport, handle=handle)
    except Exception:
        return False


@tool
async def fetch_url(url: str) -> ToolResult:
    """Fetch the content of a URL and return it as text. Useful for reading web pages, APIs, or any HTTP resource. HTML is automatically converted to readable text.

    Args:
        url: The full URL to fetch (must start with http:// or https://).
    """
    url = (url or "").strip()
    if not url:
        return ToolResult.fail("No URL provided.")
    if not url.startswith(("http://", "https://")):
        return ToolResult.fail("URL must start with http:// or https://")

    print_ts(f"{COLOR_YELLOW}fetch_url: {url}{COLOR_END}")

    # Owner/admins bypass the SSRF guard (so the owner can fetch localhost services).
    # Determined once; the per-hop guard below only blocks non-privileged callers.
    privileged = _caller_is_owner_or_admin()

    try:
        session = await http_session()

        # Follow redirects MANUALLY so the private-IP guard runs on every hop —
        # a public URL can 302 to http://169.254.169.254/. We re-check the host
        # of each URL (initial + each redirect Location) before fetching it.
        current_url = url
        for _hop in range(_MAX_REDIRECTS + 1):  # 1 initial fetch + up to N redirects
            parsed = urlparse(current_url)
            host = parsed.hostname or ""

            # The URL we actually request. For non-privileged callers the
            # netloc is rewritten to the guard-vetted IP (DNS-rebinding pin,
            # see _resolve_and_vet_host); `current_url` itself keeps the
            # hostname so redirect resolution (urljoin) stays host-relative.
            request_url = current_url
            request_headers = _BROWSER_HEADERS
            request_kwargs: dict = {}

            # SSRF guard: block internal/private targets for non-owner/admin,
            # and pin the connection to the vetted address so aiohttp can't
            # re-resolve the hostname to something the guard never saw.
            if not privileged:
                internal, pinned_ip = await _resolve_and_vet_host(host)
                if internal:
                    return ToolResult.fail(
                        f"Refusing to fetch internal/private address {host} — "
                        f"that's restricted to the owner/admins."
                    )
                if not pinned_ip:
                    # Resolution failed → fail CLOSED rather than letting
                    # aiohttp re-resolve unguarded (a selectively-failing
                    # resolver would otherwise slip an unvetted address past
                    # the guard). A genuine DNS outage fails here instead of
                    # at connect time — same outcome for the caller.
                    return ToolResult.fail(f"Could not resolve host {host}.")
                ip_for_url = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
                userinfo = ""
                if parsed.username is not None:
                    userinfo = parsed.username
                    if parsed.password is not None:
                        userinfo += f":{parsed.password}"
                    userinfo += "@"
                port = parsed.port  # None → scheme default
                netloc = f"{userinfo}{ip_for_url}" + (f":{port}" if port else "")
                request_url = parsed._replace(netloc=netloc).geturl()
                # Virtual hosting still needs the real name: Host header
                # for HTTP routing, server_hostname for TLS SNI (certs
                # aren't verified here — ssl=False below — but vhost
                # selection on the server side keys off SNI).
                request_headers = dict(_BROWSER_HEADERS)
                host_hdr = f"[{host}]" if ":" in host else host
                request_headers["Host"] = host_hdr + (f":{port}" if port else "")
                if parsed.scheme == "https":
                    request_kwargs["server_hostname"] = host

            async with session.get(
                request_url,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT),
                allow_redirects=False,  # manual — so we can guard each hop
                ssl=False,  # don't trip on self-signed certs (e.g. webapp localhost)
                **request_kwargs,
            ) as resp:
                # Redirect → resolve the next hop and loop to re-run the guard.
                location = resp.headers.get("Location")
                if resp.status in _REDIRECT_STATUSES and location:
                    current_url = urljoin(current_url, location)  # handles relative
                    continue

                ct = (resp.content_type or "").lower()

                # Reject binary content
                if ct.startswith(("image/", "audio/", "video/", "application/octet-stream")):
                    return ToolResult.fail(f"URL returned binary content ({ct}), cannot display as text.")

                if resp.status >= 400:
                    body = await resp.text(errors="replace")
                    return ToolResult.fail(f"HTTP {resp.status}: {body[:500]}")

                raw = await resp.read()
                if len(raw) > _MAX_BYTES:
                    raw = raw[:_MAX_BYTES]

                text = raw.decode("utf-8", errors="replace")

                # Convert HTML to text
                if "html" in ct:
                    text = _strip_html(text)

                if not text.strip():
                    return ToolResult(model_feedback=f"Fetched {url} — response was empty.")

                # Truncate if still too long after stripping
                if len(text) > _MAX_BYTES:
                    text = text[:_MAX_BYTES] + "\n\n[truncated]"

                return ToolResult(model_feedback=f"Content from {url}:\n\n{text}")

        # Exhausted the redirect budget without reaching a terminal response.
        return ToolResult.fail(f"Too many redirects (>{_MAX_REDIRECTS}).")

    except aiohttp.ClientError as e:
        return ToolResult.fail(f"Request failed: {e}")
    except Exception as e:
        return ToolResult.fail(f"Unexpected error fetching URL: {e}")


# Don't post raw fetched content to Discord
if "fetch_url" in TOOL_REGISTRY:
    TOOL_REGISTRY["fetch_url"].silent_to_discord = True
