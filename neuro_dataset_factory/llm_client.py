"""Minimal OpenAI-compatible JSON client using API_BASE/API_KEY.

The factory intentionally keeps this client dependency-free. Secrets are read from
the environment (or an explicitly requested env file) and are never written to
cache entries, reports, prompts, or provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from neuro_dataset_factory.storage import read_json, write_json


def load_env_file(path: Path) -> list[str]:
    """Load simple KEY=VALUE entries without overriding ambient variables."""
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _chat_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _extract_json(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 3:
            text = parts[1].lstrip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = json.loads(text[start:index + 1])
                    if isinstance(value, dict):
                        return value
                    break
    raise ValueError("LLM response does not contain a JSON object")


class OpenAICompatibleJSONClient:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        timeout: int = 180,
        max_retries: int = 3,
        use_env_proxy: bool = False,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_env_proxy = use_env_proxy
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler() if use_env_proxy else urllib.request.ProxyHandler({})
        )
        self._calls = 0
        self._cache_hits = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        return self._calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 12000,
        prompt_version: str,
        refresh_cache: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload_key = {
            "model": self.model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_version": prompt_version,
        }
        cache_key = hashlib.sha256(
            json.dumps(payload_key, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file() and not refresh_cache:
            cached = read_json(cache_path)
            if isinstance(cached, dict) and isinstance(cached.get("response"), dict):
                with self._lock:
                    self._cache_hits += 1
                return cached["response"], {"cache_key": cache_key, "cache_hit": True}

        api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("API_BASE")
        if not api_key or not api_base:
            raise RuntimeError("API_KEY and API_BASE must be set in the environment")
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            _chat_endpoint(api_base),
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                parsed = _extract_json(str(content))
                write_json(cache_path, {
                    "model": self.model,
                    "prompt_version": prompt_version,
                    "response": parsed,
                })
                with self._lock:
                    self._calls += 1
                return parsed, {"cache_key": cache_key, "cache_hit": False}
            except (OSError, KeyError, IndexError, TypeError, ValueError, urllib.error.HTTPError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep((1.5 ** attempt) + random.random() * 0.25)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")
