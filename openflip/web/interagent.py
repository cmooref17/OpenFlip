"""Inter-agent threaded view: merges two (or more) agents' per-channel
conversation jsonl files into one timeline ordered by timestamp, and
marks messages that crossed the bridge between them.

How talk_to_agent traffic appears in jsonl:
- The CALLER's file logs the tool invocation as role='tool' (the tool
  result confirming dispatch).
- The CALLEE's file gets an injected user message whose content starts
  with "<sender_id>: <body>".

We don't need exact pair matching — timestamps + sender prefix is enough
to render an intuitive thread."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from . import openflip_data as _data


PEER_PREFIX_RE = re.compile(r"^([a-z][a-z0-9_]*): ")


def _detect_peer_sender(content: str, known_agents: Set[str],
                         self_id: str) -> str | None:
    """Locate the peer sender id in a user-role message body.

    Discord user messages get wrapped with a tool-config block + "---\\n\\n"
    separator before the actual sender's text. We split on the last
    "---" and check the trailing body for the "<peer_id>: " prefix that
    talk_to_agent prepends."""
    if not isinstance(content, str):
        return None
    # The split-on-last-"---" handles both wrapped-by-framework AND
    # bare-prefix cases.
    if "---" in content:
        body = content.rsplit("---", 1)[1].lstrip()
    else:
        body = content
    m = PEER_PREFIX_RE.match(body)
    if not m:
        return None
    sender = m.group(1)
    if sender in known_agents and sender != self_id:
        return sender
    return None


def merged_timeline(agent_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Build a unified timeline from (agent_id, channel_id) pairs.
    Each agent uses their OWN channel — inter-agent traffic does not
    share a channel between participants.

    Each output row:
        - agent: agent_id whose file it came from
        - channel: channel_id for that agent
        - role, content, ts: passthrough
        - peer_sender: detected sender id when this is an inbound from
          another known agent (None otherwise)
    """
    known_agents: Set[str] = {a["id"] for a in _data.list_agents()}
    all_rows: List[Dict[str, Any]] = []
    for aid, cid in agent_pairs:
        rows = _data.read_conversation(aid, cid)
        for r in rows:
            content = r.get("content") or ""
            peer = None
            if r.get("role") == "user":
                peer = _detect_peer_sender(content, known_agents, aid)
            all_rows.append({
                "agent": aid,
                "channel": cid,
                "role": r.get("role"),
                "content": content,
                "ts": r.get("ts") or 0,
                "peer_sender": peer,
            })
    all_rows.sort(key=lambda r: r["ts"])
    return all_rows


def find_interagent_links(focal_agent: str) -> List[Dict[str, Any]]:
    """Scan focal_agent's conversation files looking for peer-sender
    prefixes (signals that another agent has spoken into this channel
    via talk_to_agent). Returns a list of {peer, focal_channel,
    focal_msg_count, last_seen}.

    The 'focal_channel' here is the channel as seen from focal_agent's
    side. When we render a thread, we pair this with the peer agent's
    OWN conversation channel — discovered by looking at THEIR files for
    a focal_agent prefix."""
    focal = _data.get_agent(focal_agent)
    if not focal:
        return []
    known_agents: Set[str] = {a["id"] for a in _data.list_agents()}
    out_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in focal["conversations"]:
        rows = _data.read_conversation(focal_agent, c["channel_id"])
        for r in rows:
            if r.get("role") != "user":
                continue
            peer = _detect_peer_sender(
                r.get("content") or "", known_agents, focal_agent
            )
            if not peer:
                continue
            key = (peer, c["channel_id"])
            slot = out_by_pair.setdefault(key, {
                "peer": peer,
                "focal_channel": c["channel_id"],
                "count": 0,
                "last_seen": 0,
            })
            slot["count"] += 1
            slot["last_seen"] = max(slot["last_seen"], r.get("ts") or 0)
    return sorted(out_by_pair.values(),
                  key=lambda x: x["last_seen"], reverse=True)


def find_peer_channel_for(focal_agent: str, peer_agent: str,
                            focal_channel: str) -> str | None:
    """Given that focal_agent saw peer_agent speak in focal_channel,
    find peer_agent's OWN channel where it received the prompt that
    triggered that reply.

    Heuristic: scan peer_agent's conversation files for tool-result
    rows mentioning the focal channel id, OR for user-role rows whose
    body starts with 'focal_agent: '. We pick whichever channel has the
    closest timestamp overlap with focal_agent's activity in
    focal_channel."""
    peer = _data.get_agent(peer_agent)
    if not peer:
        return None
    known_agents: Set[str] = {a["id"] for a in _data.list_agents()}
    # Collect candidate channels in peer's data where focal_agent sent
    # them something.
    best_channel: str | None = None
    best_score: int = 0
    for c in peer["conversations"]:
        rows = _data.read_conversation(peer_agent, c["channel_id"])
        score = 0
        for r in rows:
            if r.get("role") != "user":
                continue
            sender = _detect_peer_sender(
                r.get("content") or "", known_agents, peer_agent
            )
            if sender == focal_agent:
                score += 1
        if score > best_score:
            best_score = score
            best_channel = c["channel_id"]
    return best_channel
