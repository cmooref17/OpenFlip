"""LIVE check (not part of tests/run.sh: it calls the real Anthropic API, one short request).

The 400 "Tool definitions/code execution functions cannot be mixed with other content" came from a tool_search
tool_result that held tool_reference blocks AND a text note (a select: query naming one deferred tool plus one
already-loaded tool). This builds exactly that conversation, runs it through openflip's own converter
(_openflip_msgs_to_anthropic) and sends it, with one deferred tool and one loaded tool, the way a real turn does.
Pass = HTTP 200. Also sends the OLD shape (refs + text in the same tool_result) to prove that's the 400 cause.

Run: .lvenv/bin/python tests/live_tool_reference_check.py
"""
import asyncio, json, os, sys
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from openflip import anthropic_conversation as ac

MODEL = os.environ.get("LIVE_MODEL", "claude-haiku-4-5")

def tools():
    def tool(name, desc, deferred):
        t = {"name": name, "description": desc, "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}
        if deferred: t["defer_loading"] = True
        return t
    return [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search"} if os.environ.get("LIVE_SERVER_SEARCH") else
        tool("tool_search", "Load deferred tools by name.", False),
        tool("end_chain", "End the chain.", False),               # always loaded
        tool("generate_image", "Make an image.", True),           # deferred
    ]

def conversation(note):
    return [
        {"role": "user", "content": "Load generate_image, then say ok."},
        {"role": "assistant", "content": "", "tool_calls": [NS(tool_use_id="toolu_live1", function_name="tool_search",
                                                                   args={"query": "select:generate_image,end_chain"})]},
        {"role": "tool", "content": note, "tool_use_id": "toolu_live1", "tool_references": ["generate_image"]},
    ]

async def send(api_msgs, system):
    token = await ac._load_oauth_access_token()
    headers, sys_blocks = ac.build_auth("oauth", access_token=token)
    body = {"model": MODEL, "max_tokens": 32, "system": sys_blocks + [{"type": "text", "text": system or "Be brief."}],
            "messages": api_msgs, "tools": tools()}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{ac._DEFAULT_API_BASE}/v1/messages", json=body, headers=headers)
    return r.status_code, r.text[:300]

async def main():
    note = "Already loaded (call directly): end_chain."
    system, api = ac._openflip_msgs_to_anthropic(conversation(note), "Be brief.")
    res = api[-1]["content"][0]
    kinds = [b["type"] for b in res["content"]]
    print("openflip's tool_result content:", kinds)
    code, txt = await send(api, system)
    print("openflip's own shape  ->", code, "" if code == 200 else txt)
    old = json.loads(json.dumps(api))
    old[-1]["content"][0]["content"].append({"type": "text", "text": note})     # the pre-fix shape
    code2, txt2 = await send(old, system)
    print("old shape (refs+text) ->", code2, txt2[:160] if code2 != 200 else "")
    ok = code == 200 and all(k == "tool_reference" for k in kinds)
    print("LIVE PASS" if ok else "LIVE FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
