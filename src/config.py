from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    provider: str
    model: str
    display_name: str


def get_backends() -> list[BackendSpec]:
    return [
        BackendSpec(
            backend_id="openai_mini",
            provider="openai",
            model=os.getenv("OPENAI_MODEL_SOLVER_MINI", "gpt-4o-mini"),
            display_name="OpenAI GPT-4o-mini",
        ),
        BackendSpec(
            backend_id="openai_strong",
            provider="openai",
            model=os.getenv("OPENAI_MODEL_SOLVER_STRONG", "gpt-4o"),
            display_name="OpenAI GPT-4o",
        ),
        BackendSpec(
            backend_id="gemini",
            provider="gemini",
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            display_name="Google Gemini",
        ),
        BackendSpec(
            backend_id="groq",
            provider="groq",
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            display_name="Groq Llama",
        ),
    ]


def load_numbered_api_keys(primary_names: tuple[str, ...], numbered_prefix: str, *, max_slots: int = 20) -> list[str]:
    """Load primary key(s) plus numbered backups like PREFIX_2, PREFIX_3, ..."""
    keys: list[str] = []
    seen: set[str] = set()
    for name in primary_names:
        value = (os.getenv(name) or "").strip()
        if value and value not in seen:
            keys.append(value)
            seen.add(value)
    for slot in range(2, max_slots + 1):
        value = (os.getenv(f"{numbered_prefix}_{slot}") or "").strip()
        if value and value not in seen:
            keys.append(value)
            seen.add(value)
    return keys


def get_groq_api_keys() -> list[str]:
    return load_numbered_api_keys(("GROQ_API_KEY",), "GROQ_API_KEY")


def get_gemini_api_keys() -> list[str]:
    return load_numbered_api_keys(("GOOGLE_API_KEY", "GEMINI_API_KEY"), "GOOGLE_API_KEY")


def get_openai_api_keys() -> list[str]:
    return load_numbered_api_keys(("OPENAI_API_KEY",), "OPENAI_API_KEY")


def require_api_keys() -> None:
    missing = []
    openai_keys = get_openai_api_keys()
    if not openai_keys:
        missing.append("OPENAI_API_KEY")

    gemini_disabled = os.getenv("GEMINI_DISABLED", "").lower() in ("1", "true", "yes")
    groq_disabled = os.getenv("GROQ_DISABLED", "").lower() in ("1", "true", "yes")

    gemini_keys = get_gemini_api_keys()
    groq_keys = get_groq_api_keys()
    if not gemini_disabled and not gemini_keys:
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    if not groq_disabled and not groq_keys:
        missing.append("GROQ_API_KEY")

    if missing:
        raise RuntimeError(f"Missing API keys in .env: {', '.join(missing)}")

    g = (gemini_keys[0] if gemini_keys else "").strip()
    if g and not gemini_disabled and not (g.startswith("AIza") or g.startswith("AQ.")):
        import warnings

        warnings.warn(
            "GOOGLE_API_KEY should be from https://aistudio.google.com/apikey "
            "(starts with AIza or AQ.).",
            stacklevel=2,
        )


def structure_repair_enabled() -> bool:
    if os.getenv("STRUCTURE_REPAIR_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.getenv("OPENAI_API_KEY"))
