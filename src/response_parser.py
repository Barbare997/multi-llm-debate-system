from __future__ import annotations

import json
import re
from typing import Any

from src.json_utils import extract_json
from src.role_assignment import SOLVER_ROLES, validate_ballot


class ParseError(ValueError):
    """Raised when model output cannot be turned into the expected stage dict."""


def parse_stage_response(
    stage: str,
    raw: str,
    *,
    backend_ids: set[str] | None = None,
    solution_id: str | None = None,
) -> dict[str, Any]:
    """Turn model text into the dict shape the pipeline expects."""
    raw = (raw or "").strip()
    if not raw:
        raise ParseError("empty model response")

    if _looks_like_json(raw):
        try:
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                data = parsed
            else:
                raise ParseError("JSON response was not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc
    else:
        parsers = {
            "stage0": _parse_stage0,
            "stage0_5": lambda text: _parse_stage0_5(text, backend_ids or set()),
            "stage1": _parse_stage1,
            "stage2": lambda text: _parse_stage2(text, solution_id or "solver_1"),
            "stage3": _parse_stage3,
            "stage4": _parse_stage4,
        }
        parser = parsers.get(stage)
        if parser is None:
            raise ParseError(f"unknown stage: {stage}")
        data = parser(raw)

    validate_stage_response(stage, data, backend_ids=backend_ids, solution_id=solution_id)
    return data


def validate_stage_response(
    stage: str,
    data: dict[str, Any],
    *,
    backend_ids: set[str] | None = None,
    solution_id: str | None = None,
) -> None:
    if stage == "stage0":
        conf = data.get("confidence_by_role", {})
        if not isinstance(conf, dict):
            raise ParseError("stage0 missing confidence_by_role")
        return

    if stage == "stage0_5":
        if not backend_ids:
            raise ParseError("stage0_5 requires backend_ids")
        if not validate_ballot(data, backend_ids):
            raise ParseError("stage0_5 ballot failed validation")
        return

    if stage == "stage1":
        if not as_text(data.get("solution")):
            raise ParseError("stage1 missing solution")
        return

    if stage == "stage2":
        if data.get("solution_id") != (solution_id or data.get("solution_id")):
            data["solution_id"] = solution_id or data.get("solution_id", "solver_1")
        if not as_text(data.get("overall_assessment")):
            raise ParseError("stage2 missing overall_assessment")
        return

    if stage == "stage3":
        if not as_text(data.get("refined_solution")):
            raise ParseError("stage3 missing refined_solution")
        return

    if stage == "stage4":
        winner = as_text(data.get("winner"))
        if winner not in SOLVER_ROLES:
            raise ParseError(f"stage4 invalid winner: {winner!r}")
        return


def repair_template(stage: str, *, backend_ids: set[str] | None = None) -> str:
    templates = {
        "stage0": (
            "PREFERRED_ROLE: Solver|Judge\n"
            "SOLVER_CONFIDENCE: 0.0-1.0\n"
            "JUDGE_CONFIDENCE: 0.0-1.0\n"
            "REASONING:\n..."
        ),
        "stage0_5": (
            "judge=<backend_id>\n"
            "solver_1=<backend_id>\n"
            "solver_2=<backend_id>\n"
            "solver_3=<backend_id>\n"
            "REASONING:\n..."
            + (f"\nValid backend_ids: {', '.join(sorted(backend_ids or []))}" if backend_ids else "")
        ),
        "stage1": "SOLUTION:\n...\n\nFINAL_ANSWER:\n...",
        "stage2": (
            "STRENGTHS:\n- ...\n\n"
            "WEAKNESSES:\n- ...\n\n"
            "ERRORS:\n- location | type | description | severity\n\n"
            "SUGGESTED_CHANGES:\n- ...\n\n"
            "OVERALL_ASSESSMENT: promising_but_flawed|fundamentally_wrong|correct|incomplete"
        ),
        "stage3": "CHANGES_MADE:\n- critique | response | accepted=true/false\n\nREFINED_SOLUTION:\n...\n\nFINAL_ANSWER:\n...\n\nCONFIDENCE: 0.0-1.0",
        "stage4": "WINNER: solver_1|solver_2|solver_3\nCONFIDENCE: 0.0-1.0\nREASONING:\n...",
    }
    return templates[stage]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("```")


def _section(text: str, header: str) -> str:
    pattern = rf"(?is)^{re.escape(header)}\s*:?\s*\n?(.*?)(?=^\s*[A-Z][A-Z0-9_ ]+\s*:|\Z)"
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _line_value(text: str, key: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


def _bullet_items(block: str) -> list[str]:
    items: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•")):
            items.append(line.lstrip("-*• ").strip())
        elif items:
            items[-1] = f"{items[-1]} {line}"
        else:
            items.append(line)
    return [item for item in items if item]


def _parse_stage0(text: str) -> dict[str, Any]:
    preferred = _line_value(text, "PREFERRED_ROLE").lower()
    solver_conf = _float_or_default(_line_value(text, "SOLVER_CONFIDENCE"), 0.5)
    judge_conf = _float_or_default(_line_value(text, "JUDGE_CONFIDENCE"), 0.5)
    reasoning = _section(text, "REASONING") or text

    if preferred.startswith("judge"):
        prefs = ["Judge", "Solver"]
    elif preferred.startswith("solver"):
        prefs = ["Solver", "Judge"]
    elif "judge" in text.lower() and "solver" in text.lower():
        prefs = ["Solver", "Judge"] if text.lower().find("solver") <= text.lower().find("judge") else ["Judge", "Solver"]
    elif "judge" in text.lower():
        prefs = ["Judge", "Solver"]
    else:
        prefs = ["Solver", "Judge"]

    return {
        "role_preferences": prefs,
        "confidence_by_role": {"Solver": solver_conf, "Judge": judge_conf},
        "reasoning": reasoning,
    }


def _parse_stage0_5(text: str, backend_ids: set[str]) -> dict[str, Any]:
    fields: dict[str, str] = {"reasoning": _section(text, "REASONING") or text}
    for key in ("judge", *SOLVER_ROLES):
        value = _line_value(text, key)
        if not value:
            match = re.search(rf"(?im)^{re.escape(key)}\s*=\s*(\S+)", text)
            value = match.group(1).strip() if match else ""
        fields[key] = value

    if backend_ids and not validate_ballot(fields, backend_ids):
        fields = _repair_ballot_from_text(text, backend_ids, fields)

    return fields


def _repair_ballot_from_text(text: str, backend_ids: set[str], fields: dict[str, str]) -> dict[str, str]:
    ordered = [bid for bid in sorted(backend_ids) if re.search(rf"\b{re.escape(bid)}\b", text)]
    judge = fields.get("judge", "")
    if judge not in backend_ids:
        judge = ""
        for bid in ordered:
            if re.search(rf"judge[^.\n]*{re.escape(bid)}|{re.escape(bid)}[^.\n]*judge", text, re.I):
                judge = bid
                break
        if not judge and ordered:
            judge = ordered[0]
        if not judge and backend_ids:
            judge = sorted(backend_ids)[0]

    solvers = [bid for bid in ordered if bid != judge]
    for bid in sorted(backend_ids):
        if bid not in solvers and bid != judge:
            solvers.append(bid)

    repaired = {"judge": judge, "reasoning": fields.get("reasoning", text)}
    for idx, role in enumerate(SOLVER_ROLES):
        repaired[role] = solvers[idx] if idx < len(solvers) else ""
    return repaired


def _parse_stage1(text: str) -> dict[str, Any]:
    solution = _section(text, "SOLUTION") or text
    answer = _section(text, "FINAL_ANSWER") or extract_short_answer(solution)
    return {"solution": solution, "final_answer": answer}


def _parse_stage2(text: str, solution_id: str) -> dict[str, Any]:
    strengths = _bullet_items(_section(text, "STRENGTHS"))
    weaknesses = _bullet_items(_section(text, "WEAKNESSES"))
    suggested = _bullet_items(_section(text, "SUGGESTED_CHANGES"))
    errors_block = _section(text, "ERRORS")
    errors: list[dict[str, str]] = []
    for line in _bullet_items(errors_block):
        parts = [part.strip() for part in re.split(r"\s*\|\s*", line)]
        if len(parts) >= 4:
            errors.append(
                {
                    "location": parts[0],
                    "error_type": parts[1],
                    "description": parts[2],
                    "severity": parts[3],
                }
            )
        elif line:
            errors.append(
                {
                    "location": "unspecified",
                    "error_type": "logic",
                    "description": line,
                    "severity": "minor",
                }
            )

    overall = _line_value(text, "OVERALL_ASSESSMENT").lower().replace(" ", "_")
    if overall not in {"promising_but_flawed", "fundamentally_wrong", "correct", "incomplete"}:
        lowered = text.lower()
        if "fundamentally wrong" in lowered or "completely wrong" in lowered:
            overall = "fundamentally_wrong"
        elif "correct" in lowered and "incorrect" not in lowered:
            overall = "correct"
        elif "incomplete" in lowered:
            overall = "incomplete"
        else:
            overall = "promising_but_flawed"

    return {
        "solution_id": solution_id,
        "evaluation": {
            "strengths": strengths,
            "weaknesses": weaknesses or ([text] if text else []),
            "errors": errors,
            "suggested_changes": suggested,
        },
        "overall_assessment": overall,
    }


def _parse_stage3(text: str) -> dict[str, Any]:
    refined_solution = _section(text, "REFINED_SOLUTION") or _section(text, "SOLUTION") or text
    refined_answer = _section(text, "FINAL_ANSWER") or extract_short_answer(refined_solution)
    confidence = _float_or_default(_line_value(text, "CONFIDENCE"), 0.5)
    changes_made = _parse_changes_made(_section(text, "CHANGES_MADE"))
    return {
        "changes_made": changes_made,
        "refined_solution": refined_solution,
        "refined_answer": refined_answer,
        "confidence": confidence,
    }


def _parse_changes_made(block: str) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for line in _bullet_items(block):
        parts = [part.strip() for part in re.split(r"\s*\|\s*", line)]
        if len(parts) >= 3:
            accepted_raw = parts[2].lower()
            accepted = accepted_raw in {"true", "yes", "accepted"}
            changes.append(
                {
                    "critique": parts[0],
                    "response": parts[1],
                    "accepted": accepted,
                }
            )
        elif line:
            changes.append({"critique": line, "response": "", "accepted": False})
    return changes


def _parse_stage4(text: str) -> dict[str, Any]:
    winner = _line_value(text, "WINNER").lower().replace(" ", "_")
    match = re.search(r"solver[_\s-]?([123])", winner or text, re.IGNORECASE)
    if match:
        winner = f"solver_{match.group(1)}"
    if winner not in SOLVER_ROLES:
        winner = "solver_1"
    confidence = _float_or_default(_line_value(text, "CONFIDENCE"), 0.5)
    reasoning = _section(text, "REASONING") or text
    return {"winner": winner, "confidence": confidence, "reasoning": reasoning}


def _coerce_numeric_tail(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"-?\d+", compact) or re.fullmatch(r"\d+\+\d+", compact):
        return compact
    labeled = re.search(
        r"(?:final answer|answer is)\s*:?\s*(-?\d+(?:\s*\+\s*\d+)?)",
        text,
        re.IGNORECASE,
    )
    if labeled:
        return re.sub(r"\s+", "", labeled.group(1))
    return ""


def extract_short_answer(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        match = re.search(
            r"(?:final answer|answer is|therefore|thus|hence|we get|equals?)\s*:?\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip().rstrip(".")
            numeric = _coerce_numeric_tail(candidate)
            if numeric:
                return numeric
            if not _is_weak_fragment(candidate) and len(candidate) <= 40:
                return candidate
    for line in reversed(lines):
        numeric = _coerce_numeric_tail(line)
        if numeric:
            return numeric
    return ""


def _is_weak_fragment(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[,;:\s.]+", t):
        return True
    if re.match(r"^[,;:\s]*(we conclude|therefore|thus|hence)\s*:?\s*$", t, re.IGNORECASE):
        return True
    alnum = re.sub(r"\W", "", t)
    return len(alnum) < 2


def _float_or_default(raw: str, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))
