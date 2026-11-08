"""Minimal LLM clients. Anything with a `complete(system, user) -> str` method works.

Clients are injected into judges, so tests and offline runs never need an API key.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Claude via the Anthropic SDK (`pip install sixdim-eval[anthropic]`)."""

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1500):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            from anthropic import Anthropic  # optional dependency
        except ImportError:
            raise RuntimeError("the anthropic package is not installed: pip install 'sixdim-eval[anthropic]'") from None
        self.model = model or os.getenv("SIXDIM_MODEL", "claude-sonnet-4-6")
        self.name = f"anthropic:{self.model}"
        self._client = Anthropic(api_key=key)
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class OpenAICompatibleClient:
    """Any OpenAI-compatible `/chat/completions` endpoint: OpenAI, Ollama, vLLM, LM Studio, OpenRouter.

    Configure with OPENAI_API_KEY and optionally OPENAI_BASE_URL (default https://api.openai.com/v1).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or os.getenv("SIXDIM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.name = f"openai:{self.model}"
        self._http = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
