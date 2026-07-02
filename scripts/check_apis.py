#!/usr/bin/env python3
"""Quick connectivity check for all API providers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_backends, get_gemini_api_keys, get_groq_api_keys, get_openai_api_keys, require_api_keys
from src.llm import build_clients


def main() -> None:
    require_api_keys()
    openai_keys = get_openai_api_keys()
    groq_keys = get_groq_api_keys()
    gemini_keys = get_gemini_api_keys()
    print(f"Configured keys: openai={len(openai_keys)}, groq={len(groq_keys)}, gemini={len(gemini_keys)}")
    backends = get_backends()
    clients = build_clients(backends)
    ok = True
    for backend_id, client in clients.items():
        provider = client.spec.provider
        if provider == "gemini" and os.getenv("GEMINI_DISABLED", "").lower() in ("1", "true", "yes"):
            print(f"[SKIP] {backend_id}: GEMINI_DISABLED=1")
            continue
        if provider == "groq" and os.getenv("GROQ_DISABLED", "").lower() in ("1", "true", "yes"):
            print(f"[SKIP] {backend_id}: GROQ_DISABLED=1")
            continue
        try:
            text = client.complete("You are a test assistant.", 'Reply with JSON: {"status": "ok"}', temperature=0)
            print(f"[OK] {backend_id} ({client.spec.model}): {text[:80]}...")
        except Exception as exc:
            ok = False
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(f"[FAIL] {backend_id}: quota/rate limit — try GEMINI_MODEL=gemini-2.5-flash in .env")
            else:
                print(f"[FAIL] {backend_id}: {exc}")
    if not ok:
        sys.exit(1)
    print("All providers reachable.")


if __name__ == "__main__":
    main()
