from __future__ import annotations

import os
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types
from groq import Groq
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import BackendSpec, get_gemini_api_keys, get_groq_api_keys, get_openai_api_keys

_GEMINI_SESSION_DISABLED = os.getenv("GEMINI_DISABLED", "").lower() in ("1", "true", "yes")
_GROQ_SESSION_DISABLED = os.getenv("GROQ_DISABLED", "").lower() in ("1", "true", "yes")


class ApiKeyPool:
    """Rotate through numbered API keys before falling back to OpenAI."""

    def __init__(self, provider: str, keys: list[str]) -> None:
        self.provider = provider
        self.keys = keys
        self._index = 0
        self._exhausted: set[int] = set()

    def __len__(self) -> int:
        return len(self.keys)

    def available_count(self) -> int:
        return sum(1 for i in range(len(self.keys)) if i not in self._exhausted)

    def current_index(self) -> int | None:
        for i in range(len(self.keys)):
            idx = (self._index + i) % len(self.keys)
            if idx not in self._exhausted:
                return idx
        return None

    def current_key(self) -> str | None:
        idx = self.current_index()
        if idx is None:
            return None
        self._index = idx
        return self.keys[idx]

    def mark_current_exhausted(self) -> None:
        idx = self.current_index()
        if idx is None:
            return
        self._exhausted.add(idx)
        print(
            f"  [{self.provider}] key #{idx + 1}/{len(self.keys)} quota exhausted; trying next key...",
            flush=True,
        )
        self._index = (idx + 1) % len(self.keys)

    def rotate_on_rate_limit(self) -> bool:
        if len(self.keys) <= 1:
            return False
        idx = self.current_index()
        if idx is None:
            return False
        next_idx = (idx + 1) % len(self.keys)
        if next_idx in self._exhausted:
            return False
        print(
            f"  [{self.provider}] rate limited on key #{idx + 1}; rotating to key #{next_idx + 1}...",
            flush=True,
        )
        self._index = next_idx
        return True


_GROQ_KEY_POOL: ApiKeyPool | None = None
_GEMINI_KEY_POOL: ApiKeyPool | None = None
_OPENAI_KEY_POOL: ApiKeyPool | None = None


def _groq_pool() -> ApiKeyPool:
    global _GROQ_KEY_POOL
    if _GROQ_KEY_POOL is None:
        _GROQ_KEY_POOL = ApiKeyPool("groq", get_groq_api_keys())
    return _GROQ_KEY_POOL


def _gemini_pool() -> ApiKeyPool:
    global _GEMINI_KEY_POOL
    if _GEMINI_KEY_POOL is None:
        _GEMINI_KEY_POOL = ApiKeyPool("gemini", get_gemini_api_keys())
    return _GEMINI_KEY_POOL


def _openai_pool() -> ApiKeyPool:
    global _OPENAI_KEY_POOL
    if _OPENAI_KEY_POOL is None:
        _OPENAI_KEY_POOL = ApiKeyPool("openai", get_openai_api_keys())
    return _OPENAI_KEY_POOL


class LLMClient:
    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self._openai: OpenAI | None = None
        self._openai_key: str | None = None
        self._groq: Groq | None = None
        self._groq_key: str | None = None
        self._gemini: genai.Client | None = None
        self._gemini_key: str | None = None

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=30))
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int | None = 2048,
    ) -> str:
        if self.spec.provider == "openai":
            return self._openai_complete(system_prompt, user_prompt, temperature, max_output_tokens)
        if self.spec.provider == "gemini":
            return self._gemini_complete(system_prompt, user_prompt, temperature, max_output_tokens)
        if self.spec.provider == "groq":
            return self._groq_complete(system_prompt, user_prompt, temperature, max_output_tokens)
        raise ValueError(f"Unknown provider: {self.spec.provider}")

    def _get_openai_client(self, api_key: str) -> OpenAI:
        if self._openai is None or self._openai_key != api_key:
            self._openai = OpenAI(api_key=api_key, timeout=120.0)
            self._openai_key = api_key
        return self._openai

    def _openai_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int | None,
        model: str | None = None,
    ) -> str:
        pool = _openai_pool()
        if not pool.keys:
            raise RuntimeError("No OpenAI API keys configured (OPENAI_API_KEY)")
        kwargs: dict = dict(
            model=model or self.spec.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        last_error: Exception | None = None
        keys_tried = 0
        while keys_tried < len(pool):
            api_key = pool.current_key()
            if api_key is None:
                break
            keys_tried += 1
            openai = self._get_openai_client(api_key)
            for attempt in range(2):
                try:
                    response = openai.chat.completions.create(**kwargs)
                    return response.choices[0].message.content or ""
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    if _is_openai_quota_exhausted(msg):
                        pool.mark_current_exhausted()
                        break
                    if _is_rate_limited(msg):
                        if attempt == 0:
                            delay = min(_parse_retry_delay(msg), 20.0)
                            print(f"  [openai] rate limited, waiting {delay:.0f}s...", flush=True)
                            time.sleep(delay)
                            continue
                        if pool.rotate_on_rate_limit():
                            break
                        pool.mark_current_exhausted()
                        break
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("OpenAI unavailable after trying all configured keys.")

    def _groq_fallback_complete(
        self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int | None
    ) -> str:
        fallback_model = os.getenv(
            "GROQ_FALLBACK_MODEL",
            os.getenv("OPENAI_MODEL_SOLVER_MINI", "gpt-4o-mini"),
        )
        print(
            f"  [groq] all keys exhausted; using OpenAI fallback ({fallback_model})",
            flush=True,
        )
        return self._openai_complete(
            system_prompt, user_prompt, temperature, max_output_tokens, model=fallback_model
        )

    def _get_groq_client(self, api_key: str) -> Groq:
        if self._groq is None or self._groq_key != api_key:
            self._groq = Groq(api_key=api_key, timeout=120.0)
            self._groq_key = api_key
        return self._groq

    def _groq_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int | None,
        model: str | None = None,
    ) -> str:
        if _GROQ_SESSION_DISABLED:
            return self._groq_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)

        pool = _groq_pool()
        if not pool.keys:
            return self._groq_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)

        kwargs: dict = dict(
            model=model or self.spec.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        last_error: Exception | None = None
        keys_tried = 0
        while keys_tried < len(pool):
            api_key = pool.current_key()
            if api_key is None:
                break
            keys_tried += 1
            groq = self._get_groq_client(api_key)

            for attempt in range(2):
                try:
                    response = groq.chat.completions.create(**kwargs)
                    return response.choices[0].message.content or ""
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    if _is_groq_daily_exhausted(msg):
                        pool.mark_current_exhausted()
                        break
                    if _is_rate_limited(msg):
                        if attempt == 0:
                            delay = min(_parse_groq_retry_delay(msg), 30.0)
                            print(f"  [groq] rate limited, waiting {delay:.0f}s...", flush=True)
                            time.sleep(delay)
                            continue
                        if pool.rotate_on_rate_limit():
                            break
                        pool.mark_current_exhausted()
                        break
                    raise

        print("  [groq] unavailable after trying all keys; switching to OpenAI fallback.", flush=True)
        if last_error:
            print(f"  [groq] last error: {last_error}", flush=True)
        return self._groq_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)

    def _gemini_fallback_complete(
        self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int | None
    ) -> str:
        fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL",
            os.getenv("OPENAI_MODEL_SOLVER_MINI", "gpt-4o-mini"),
        )
        print(
            f"  [gemini] all keys exhausted; using OpenAI fallback ({fallback_model})",
            flush=True,
        )
        return self._openai_complete(
            system_prompt, user_prompt, temperature, max_output_tokens, model=fallback_model
        )

    def _get_gemini_client(self, api_key: str) -> genai.Client:
        if self._gemini is None or self._gemini_key != api_key:
            self._gemini = genai.Client(api_key=api_key)
            self._gemini_key = api_key
        return self._gemini

    def _gemini_complete(
        self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int | None
    ) -> str:
        if _GEMINI_SESSION_DISABLED:
            return self._gemini_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)

        pool = _gemini_pool()
        if not pool.keys:
            return self._gemini_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)

        config_kwargs: dict = dict(
            system_instruction=system_prompt,
            temperature=temperature,
        )
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        models_to_try = [self.spec.model]
        if self.spec.model != "gemini-2.5-flash":
            models_to_try.append("gemini-2.5-flash")

        last_error: Exception | None = None
        keys_tried = 0
        while keys_tried < len(pool):
            api_key = pool.current_key()
            if api_key is None:
                break
            keys_tried += 1
            gemini = self._get_gemini_client(api_key)

            for model in models_to_try:
                for attempt in range(2):
                    try:
                        response = gemini.models.generate_content(
                            model=model,
                            contents=user_prompt,
                            config=genai_types.GenerateContentConfig(**config_kwargs),
                        )
                        return response.text or ""
                    except Exception as exc:
                        last_error = exc
                        msg = str(exc)
                        if _is_daily_quota_exhausted(msg):
                            pool.mark_current_exhausted()
                            break
                        if _is_rate_limited(msg):
                            if attempt == 0:
                                delay = min(_parse_retry_delay(msg), 15.0)
                                print(f"  [gemini/{model}] rate limited, waiting {delay:.0f}s...", flush=True)
                                time.sleep(delay)
                                continue
                            if pool.rotate_on_rate_limit():
                                break
                            pool.mark_current_exhausted()
                            break
                        raise
                else:
                    continue
                break

        print("  [gemini] unavailable after trying all keys; switching to OpenAI fallback.", flush=True)
        if last_error:
            print(f"  [gemini] last error: {last_error}", flush=True)
        return self._gemini_fallback_complete(system_prompt, user_prompt, temperature, max_output_tokens)


def build_clients(backends: list[BackendSpec]) -> dict[str, LLMClient]:
    return {b.backend_id: LLMClient(b) for b in backends}


def _is_rate_limited(message: str) -> bool:
    lowered = message.lower()
    return "429" in message or "rate_limit" in lowered or "resource_exhausted" in lowered


def _parse_retry_delay(message: str, default: float = 10.0) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.I)
    if match:
        return float(match.group(1)) + 1.0
    return default


def _is_daily_quota_exhausted(message: str) -> bool:
    lowered = message.lower()
    return (
        "perday" in lowered
        or "per_day" in lowered
        or "per day" in lowered
        or "permodelperday" in lowered
        or "generatecontentinputtokenspermodelperday" in lowered
        or "tokens per day" in lowered
        or ("limit 100000" in lowered and "used" in lowered)
    )


def _is_openai_quota_exhausted(message: str) -> bool:
    lowered = message.lower()
    return "insufficient_quota" in lowered or "quota" in lowered


def _is_groq_daily_exhausted(message: str) -> bool:
    return _is_daily_quota_exhausted(message) or "tpd" in message.lower()


def _parse_groq_retry_delay(message: str, default: float = 10.0) -> float:
    match = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", message, re.I)
    if match:
        return float(match.group(1)) * 60 + float(match.group(2)) + 1.0
    return _parse_retry_delay(message, default)
