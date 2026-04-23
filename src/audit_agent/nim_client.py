"""Thin wrapper around the OpenAI SDK pointed at NVIDIA NIM.

Provides structured-output helpers, retry logic, per-call cost/latency logging,
and an on-disk response cache so repeat runs cost nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 45.0

RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)
FATAL_EXCEPTIONS = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class NIMClient:
    """OpenAI-compatible client that talks to NVIDIA NIM."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        self.api_key = api_key or _env("NIM_API_KEY")
        self.base_url = base_url or _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.cache_dir = Path(cache_dir or _env("CACHE_DIR", ".cache/nim_responses"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        configured_timeout = (
            timeout_seconds if timeout_seconds is not None else float(_env("NIM_TIMEOUT_SECONDS", "45"))
        )
        self.timeout_seconds = max(configured_timeout, DEFAULT_TIMEOUT_SECONDS)
        self.max_retries = (
            max_retries if max_retries is not None else int(_env("NIM_MAX_RETRIES", "2"))
        )
        self._disabled_reason: str | None = None

        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            self._client = None
            self._disabled_reason = f"client initialization failed: {exc}"
            logger.warning("NIM client disabled: %s", self._disabled_reason)

        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def _normalize_models(self, model: str | Sequence[str]) -> list[str]:
        if isinstance(model, str):
            candidates = model.split(",")
        else:
            candidates = list(model)
        return [candidate.strip() for candidate in candidates if candidate.strip()]

    def _cache_key(self, model: str | Sequence[str], messages: list[dict], **kwargs: Any) -> str:
        blob = json.dumps(
            {"models": self._normalize_models(model), "messages": messages, **kwargs},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _write_cache(self, key: str, data: dict) -> None:
        self._cache_path(key).write_text(json.dumps(data, ensure_ascii=False))

    def _create_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        extra: dict[str, Any],
    ) -> Any:
        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        for attempt in retryer:
            with attempt:
                return self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                )

    def _should_disable_after_failure(self, exc: Exception) -> bool:
        return isinstance(exc, RETRYABLE_EXCEPTIONS + FATAL_EXCEPTIONS)

    def _message_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("Model returned an empty message payload")

    def _coerce_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            fence_lines = stripped.splitlines()
            if len(fence_lines) >= 3:
                stripped = "\n".join(fence_lines[1:-1]).strip()
                if stripped.lower().startswith("json\n"):
                    stripped = stripped[5:].strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                candidate = stripped[start:end + 1]
                json.loads(candidate)
                return candidate
            raise

    def chat(
        self,
        model: str | Sequence[str],
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Send a chat completion and return the raw response dict.

        Caches by (model, messages, kwargs) hash so repeat calls are free.
        """
        if self._disabled_reason:
            raise RuntimeError(f"NIM unavailable: {self._disabled_reason}")

        if self._client is None:
            raise RuntimeError("NIM client is not initialized")

        extra: dict[str, Any] = {}
        if response_format:
            extra["response_format"] = response_format
        if logprobs:
            extra["logprobs"] = True
            if top_logprobs is not None:
                extra["top_logprobs"] = top_logprobs

        candidate_models = self._normalize_models(model)
        if not candidate_models:
            raise ValueError("At least one NIM model must be configured")

        cache_key = self._cache_key(model, messages, temperature=temperature, **extra)
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", cache_key[:12])
                return cached

        errors: list[str] = []
        last_exc: Exception | None = None
        for candidate_model in candidate_models:
            t0 = time.perf_counter()
            try:
                resp = self._create_completion(
                    candidate_model,
                    messages,
                    temperature,
                    max_tokens,
                    extra,
                )
            except Exception as exc:
                last_exc = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                errors.append(f"{candidate_model}: {type(exc).__name__}: {exc}")
                if isinstance(exc, FATAL_EXCEPTIONS):
                    self._disabled_reason = f"{type(exc).__name__}: {exc}"
                    break
                if len(candidate_models) > 1:
                    logger.warning(
                        "NIM model %s failed, trying next candidate: %s: %s",
                        candidate_model,
                        type(exc).__name__,
                        exc,
                    )
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = resp.model_dump()

            self.total_calls += 1
            usage = result.get("usage") or {}
            self.total_tokens += usage.get("total_tokens", 0)
            self.total_latency_ms += elapsed_ms
            logger.info(
                "NIM call model=%s tokens=%s latency=%.0fms",
                candidate_model,
                usage.get("total_tokens", "?"),
                elapsed_ms,
            )

            if use_cache:
                self._write_cache(cache_key, result)
            return result

        if last_exc is not None and self._should_disable_after_failure(last_exc):
            self._disabled_reason = f"{type(last_exc).__name__}: {last_exc}"
        detail = " | ".join(errors) if errors else "unknown failure"
        raise RuntimeError(f"NIM request failed across models: {detail}") from last_exc

    def chat_json(
        self,
        model: str | Sequence[str],
        messages: list[dict],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> dict:
        """Chat with JSON-mode; optionally validate against a pydantic schema."""
        rf: dict | None = {"type": "json_object"}
        raw = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=rf,
            use_cache=use_cache,
        )
        message = raw["choices"][0]["message"]
        content = self._coerce_json_text(self._message_text(message))
        parsed = json.loads(content)
        if schema:
            schema.model_validate(parsed)  # raises on bad shape
        return parsed

    def chat_text(
        self,
        model: str | Sequence[str],
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> str:
        raw = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        return self._message_text(raw["choices"][0]["message"])

    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "disabled_reason": self._disabled_reason,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "avg_latency_ms": (
                round(self.total_latency_ms / self.total_calls, 1)
                if self.total_calls
                else 0
            ),
        }
