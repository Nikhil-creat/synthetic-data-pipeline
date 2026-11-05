"""
assistant.py - LLM router for free-tier Groq and Gemini API keys.

Auto-detection:
    * Provider: Groq keys start with 'gsk_', Gemini keys with 'AIza'. Either environment
      variable may hold either kind of key; the provider is inferred from the key itself.
    * Model: the router lists the models each key can use, drops non-chat models
      (speech, embeddings, image, guard models), ranks the rest, and caches the result.
    * Self-healing: if a call fails (rate limit, retired model, 5xx), it tries the next
      ranked model, then the other provider.

Keys come from the environment only. They are never logged or returned by the API.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("sdp.assistant")

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_GROQ_SKIP = re.compile(r"whisper|tts|guard|embed|playai|orpheus|safeguard|compound|vision-preview", re.I)
_GEMINI_SKIP = re.compile(
    r"tts|image|embed|aqa|vision|live|audio|robotics|computer|imagen|veo|gemma|learnlm|thinking-exp", re.I
)
_DEFAULTS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "gemini": ["gemini-flash-latest", "gemini-2.0-flash"],
}


class LLMUnavailable(Exception):
    """Raised when no key is configured or every provider/model failed."""


def detect_provider(key: str) -> Optional[str]:
    """Infer the provider from the key prefix."""
    key = (key or "").strip()
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("AIza"):
        return "gemini"
    return None


def rank_groq_models(ids: List[str]) -> List[str]:
    """Best-first chat models: prefer Llama, larger sizes, newer versions, 'versatile' variants."""
    scored = []
    for mid in ids:
        low = mid.lower()
        if _GROQ_SKIP.search(low):
            continue
        size = re.search(r"(\d+)b", low)
        ver = re.search(r"llama-?(\d+(?:\.\d+)?)", low)
        score = (int(size.group(1)) if size else 10)
        score += 30 if "versatile" in low else 0
        score += 25 if "llama" in low else 0
        score += float(ver.group(1)) * 10 if ver else 0
        scored.append((score, mid))
    return [m for _, m in sorted(scored, reverse=True)]


def rank_gemini_models(models: List[Dict[str, Any]]) -> List[str]:
    """Best-first generateContent models. Flash first (free-tier friendly), then newer versions."""
    scored = []
    for m in models:
        name = str(m.get("name", "")).split("/")[-1]
        if "generateContent" not in m.get("supportedGenerationMethods", []) or _GEMINI_SKIP.search(name):
            continue
        versioned = re.match(r"gemini-(\d+(?:\.\d+)?)-(flash-lite|flash|pro)", name)
        alias = re.match(r"gemini-(flash-lite|flash|pro)-latest", name)
        if versioned:
            ver, fam = float(versioned.group(1)), versioned.group(2)
        elif alias:
            ver, fam = 2.5, alias.group(1)  # '-latest' aliases track the newest release
        else:
            continue
        score = {"flash": 3, "flash-lite": 2, "pro": 1}[fam] * 100 + ver * 10
        score -= 5 if re.search(r"preview|exp", name) else 0
        scored.append((score, name))
    return [n for _, n in sorted(scored, reverse=True)]


@dataclass
class Provider:
    name: str
    key: str
    models: List[str] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str = ""


class LLMRouter:
    """Discovers models per key and answers chat requests with automatic fallback."""

    TTL = 3600.0

    def __init__(self, groq_key: Optional[str] = None, gemini_key: Optional[str] = None,
                 timeout: float = 25.0) -> None:
        self.timeout = timeout
        self.providers: Dict[str, Provider] = {}
        keys = [
            groq_key or os.getenv("GROQ_API_KEY", ""),
            gemini_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        ]
        for key in keys:
            name = detect_provider(key)
            if name and name not in self.providers:
                self.providers[name] = Provider(name, key.strip())

    # ---- discovery ----
    async def discover(self, force: bool = False) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for p in self.providers.values():
                if force or not p.models or time.time() - p.fetched_at > self.TTL:
                    await self._fetch_models(client, p)

    async def _fetch_models(self, client: httpx.AsyncClient, p: Provider) -> None:
        try:
            if p.name == "groq":
                r = await client.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {p.key}"})
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
                p.models = rank_groq_models(ids)
            else:
                r = await client.get(f"{GEMINI_BASE}/models", params={"pageSize": 200},
                                     headers={"x-goog-api-key": p.key})
                r.raise_for_status()
                p.models = rank_gemini_models(r.json().get("models", []))
            p.error = ""
        except httpx.HTTPError as exc:
            p.error = f"model discovery failed: {type(exc).__name__}"
            logger.warning("%s %s", p.name, p.error)
        if not p.models:
            p.models = list(_DEFAULTS[p.name])  # last resort when listing fails
        p.fetched_at = time.time()

    async def status(self) -> Dict[str, Any]:
        await self.discover()
        return {
            "configured": sorted(self.providers),
            "providers": {n: {"selected": p.models[0] if p.models else None, "candidates": p.models[:3],
                              "error": p.error} for n, p in self.providers.items()},
        }

    # ---- chat ----
    def _order(self) -> List[Provider]:
        return [self.providers[n] for n in ("groq", "gemini") if n in self.providers]

    async def chat(self, messages: List[Dict[str, str]], system: str = "", max_tokens: int = 700,
                   temperature: float = 0.3) -> Dict[str, Any]:
        if not self.providers:
            raise LLMUnavailable("No GROQ_API_KEY or GEMINI_API_KEY is configured.")
        await self.discover()
        tried: List[str] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for p in self._order():
                for model in p.models[:3]:
                    call = self._groq if p.name == "groq" else self._gemini
                    try:
                        text = await call(client, p, model, messages, system, max_tokens, temperature)
                    except httpx.HTTPStatusError as exc:
                        tried.append(f"{p.name}/{model}: HTTP {exc.response.status_code}")
                        continue
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        tried.append(f"{p.name}/{model}: {type(exc).__name__}")
                        continue
                    if text:
                        return {"text": text, "provider": p.name, "model": model, "tried": tried}
                    tried.append(f"{p.name}/{model}: empty reply")
        raise LLMUnavailable("All providers failed: " + "; ".join(tried))

    @staticmethod
    async def _groq(client, p, model, messages, system, max_tokens, temperature) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        r = await client.post(
            f"{GROQ_BASE}/chat/completions", headers={"Authorization": f"Bearer {p.key}"},
            json={"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature},
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    @staticmethod
    async def _gemini(client, p, model, messages, system, max_tokens, temperature) -> str:
        body: Dict[str, Any] = {
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = await client.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                              headers={"x-goog-api-key": p.key}, json=body)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(x.get("text", "") for x in parts).strip()
