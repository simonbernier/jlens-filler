"""
Shared OpenRouter plumbing for the stage-1 (Figure 2) scripts, 10 and 11.

One place for everything both scripts must agree on:
  * the model id, base URL and PROVIDER pin (the knowledge filter in 10 must be
    computed on the SAME serving stack the sweep in 11 uses);
  * reasoning disabled ({"effort": "none"}) — V4 Flash is a hybrid reasoning
    model, and filler-token uplift is only meaningful with reasoning OFF;
  * client construction (.env via python-dotenv), retries, answer parsing.

Key: export OPENROUTER_API_KEY=sk-or-...  (or put it in a .env in the repo root)
"""
from __future__ import annotations

import os
import re
import time

API_MODEL = "deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"

# Pin one OpenRouter provider so every call (and every k) hits the same serving
# stack — with ~17 providers behind this model id, routing drift is a bigger
# consistency risk than temperature. An endpoint label like "deepinfra/fp8"
# splits into two fields: the provider slug goes in `order`, the quantization in
# `quantizations`; allow_fallbacks=False makes calls FAIL rather than silently
# reroute if that endpoint is down. A pin that matches NO live endpoint fails
# with 404 "No endpoints found" — run list_endpoints() (a cell in 11 does) to
# see what is live and pick a slug from there. Set to None to route freely.
PROVIDER: dict | None = {
    "order": ["deepinfra"],
    "allow_fallbacks": False,
    "quantizations": ["fp8"],
}

# V4 Flash is reasoning-capable; the paper's setting is answer-immediately with
# NO chain of thought, so reasoning must be explicitly off. Recorded in
# data/fig2_meta.json so the datasets are auditable.
REASONING: dict = {"effort": "none"}

MAX_TOKENS = 24
WORKERS = 8


def parse_answer(text: str) -> int | None:
    """First integer in the reply ('Answer: 138' -> 138)."""
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def make_client():
    from openai import OpenAI  # pip install openai
    try:  # pick up a local .env if python-dotenv is installed
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, "Set OPENROUTER_API_KEY (env var or .env file)."
    return OpenAI(api_key=key, base_url=BASE_URL)


def _extra_body() -> dict:
    body: dict = {"reasoning": REASONING}
    if PROVIDER:
        body["provider"] = PROVIDER
    return body


def query(client, messages, temperature: float = 0.0, max_retries: int = 5) -> str:
    """One chat completion with exponential-backoff retries."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL, messages=messages, temperature=temperature,
                max_tokens=MAX_TOKENS, extra_body=_extra_body())
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return ""


def list_endpoints() -> None:
    """Print the providers currently serving API_MODEL (free, no key needed)."""
    import requests
    eps = requests.get(f"{BASE_URL}/models/{API_MODEL}/endpoints",
                       timeout=30).json()["data"]["endpoints"]
    for e in eps:
        print(f"  {e.get('tag', '?'):>22}   quant={e.get('quantization')}   "
              f"({e.get('provider_name')})")


def smoke_call(client) -> None:
    """One cheap call proving key + model id + provider pin + parsing."""
    resp = client.chat.completions.create(
        model=API_MODEL, temperature=0.0, max_tokens=MAX_TOKENS,
        messages=[{"role": "user",
                   "content": "What is 2+2? Answer with just the number."}],
        extra_body=_extra_body())
    txt = resp.choices[0].message.content
    print("model served:   ", resp.model)
    print("provider served:", getattr(resp, "provider", "(not reported)"))
    print("reply:", repr(txt), "-> parsed:", parse_answer(txt))
    assert parse_answer(txt) == 4, "API smoke call failed to parse '4'"
