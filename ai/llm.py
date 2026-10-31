"""LLM client for Grok (xAI) through its OpenAI-compatible API, with a response cache.

Only two places call the LLM: resolution generation (resolve.py) and the close-call
classification second opinion (classify.py). Both send masked text only.

The cache (data/llm_cache.sqlite) is keyed by a hash of model + messages, so a repeated
analysis of the same ticket returns the same output even if the API is down (risk table,
section 18). Disable with LLM_CACHE=0.

Check the connection and list available models: python -m ai.llm --check
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import threading
import time

from ai import config


class LLMUnavailable(RuntimeError):
    """Raised when the API cannot be reached (no key, network, auth or rate-limit errors)."""


_client = None
_lock = threading.Lock()


def available() -> bool:
    return config.LLM_PROVIDER != "offline" and bool(config.LLM_API_KEY)


def _get_client():
    global _client
    if _client is None:
        if not available():
            raise LLMUnavailable("no LLM API key configured (set GROK_API_KEY in .env)")
        from openai import OpenAI
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                         timeout=config.LLM_TIMEOUT_S, max_retries=2)
    return _client


# ----------------------------------------------------------------------------- cache
def _cache_conn():
    config.LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.LLM_CACHE_PATH, timeout=30)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT, created REAL)")
    return conn


def _cache_key(messages, max_tokens) -> str:
    raw = json.dumps({"model": config.LLM_MODEL, "messages": messages, "max_tokens": max_tokens,
                      "temperature": config.LLM_TEMPERATURE}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key):
    if not config.LLM_CACHE_ENABLED:
        return None
    with _lock, _cache_conn() as conn:
        row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key, text):
    if not config.LLM_CACHE_ENABLED:
        return
    with _lock, _cache_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (key, config.LLM_MODEL, text, time.time()))


# ----------------------------------------------------------------------------- calls
def chat(messages: list[dict], max_tokens: int | None = None, json_mode: bool = True) -> tuple[str, dict]:
    """Return (text, meta). meta has cache_hit and latency_ms. Raises LLMUnavailable."""
    if max_tokens and config.LLM_IS_REASONING:
        max_tokens += 400  # hidden reasoning tokens count against the budget
    max_tokens = max_tokens or config.LLM_MAX_TOKENS
    key = _cache_key(messages, max_tokens)
    cached = _cache_get(key)
    if cached is not None:
        return cached, {"cache_hit": True, "latency_ms": 0}
    client = _get_client()
    kwargs = dict(model=config.LLM_MODEL, messages=messages, temperature=config.LLM_TEMPERATURE,
                  max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if config.LLM_IS_REASONING and "gpt-oss" in config.LLM_MODEL:
        kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # openai raises typed errors; all mean "not usable right now"
        if json_mode and "response_format" in str(exc):
            kwargs.pop("response_format")
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc2:
                raise LLMUnavailable(f"{type(exc2).__name__}: {exc2}") from exc2
        else:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
    text = resp.choices[0].message.content or ""
    finish = resp.choices[0].finish_reason
    if finish != "length" and text.strip():  # never cache truncated or empty outputs
        _cache_put(key, text)
    return text, {"cache_hit": False, "finish_reason": finish,
                  "latency_ms": int((time.perf_counter() - t0) * 1000)}


def parse_json(text: str):
    """Parse a JSON object from model output, tolerating code fences or leading prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def chat_json(system: str, user: str, max_tokens: int | None = None):
    """Convenience wrapper returning a dict, or None when the LLM is unavailable/unparseable."""
    try:
        text, _ = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
        return parse_json(text)
    except (LLMUnavailable, json.JSONDecodeError):
        return None


def check():
    print(f"provider={config.LLM_PROVIDER} base_url={config.LLM_BASE_URL} model={config.LLM_MODEL}")
    if not available():
        print("No API key configured: the pipeline will use the offline extractive generator.")
        return
    client = _get_client()
    try:
        models = sorted(m.id for m in client.models.list().data)
        print("available models:", ", ".join(models))
        if config.LLM_MODEL not in models:
            print(f"WARNING: LLM_MODEL={config.LLM_MODEL} is not in the list; set LLM_MODEL in .env")
    except Exception as exc:
        print("could not list models:", exc)
    try:
        text, meta = chat([{"role": "system", "content": "Return JSON only."},
                           {"role": "user", "content": 'Return {"ok": true}'}], max_tokens=20)
        print("test call:", text.strip(), meta)
    except LLMUnavailable as exc:
        print("test call failed:", exc)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        print(__doc__)
