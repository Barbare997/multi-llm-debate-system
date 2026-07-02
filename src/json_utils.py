from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    text = _strip_markdown_fence(text.strip())
    candidates = _json_candidates(text)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        for repaired in (candidate, _repair_json_escapes(candidate)):
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

    if last_error:
        raise last_error
    raise ValueError("No JSON object found in model response")


def _strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    if text:
        candidates.append(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _repair_json_escapes(text: str) -> str:
    """Fix invalid JSON escapes common in LaTeX (e.g. \\leq, \\left)."""
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\" or i + 1 >= len(text):
            result.append(ch)
            i += 1
            continue

        nxt = text[i + 1]
        if nxt == "u" and i + 5 < len(text):
            hex_part = text[i + 2 : i + 6]
            if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                result.append(text[i : i + 6])
                i += 6
                continue

        if nxt in '"\\/bfnrt':
            result.append(ch)
            result.append(nxt)
            i += 2
            continue

        result.append("\\\\")
        i += 1

    return "".join(result)
