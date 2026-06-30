from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SOLVER_ROLES = ("solver_1", "solver_2", "solver_3")
JUDGE_ROLE = "judge"


def normalize_preferences(raw: Any) -> dict[str, Any]:
    """Coerce model output into the expected preference dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    return {
        "role_preferences": ["Solver", "Judge"],
        "confidence_by_role": {"Solver": 0.5, "Judge": 0.5},
        "reasoning": "Fallback: model returned unexpected JSON shape.",
    }


def format_preferences_for_assignment(preferences: dict[str, dict[str, Any]]) -> str:
    lines = ["Participant self-assessments (Stage 0):"]
    for backend_id, pref in sorted(preferences.items()):
        lines.append(f"\nbackend_id: {backend_id}")
        lines.append(f"role_preferences: {pref.get('role_preferences', [])}")
        lines.append(f"confidence_by_role: {pref.get('confidence_by_role', {})}")
        lines.append(f"reasoning: {pref.get('reasoning', '')}")
    return "\n".join(lines)


def validate_ballot(ballot: dict[str, Any], backend_ids: set[str]) -> bool:
    judge = ballot.get("judge")
    if judge not in backend_ids:
        return False
    solvers = [ballot.get(role) for role in SOLVER_ROLES]
    if len(set(solvers)) != 3:
        return False
    if any(s not in backend_ids for s in solvers):
        return False
    if judge in solvers:
        return False
    if set(solvers) | {judge} != backend_ids:
        return False
    return True


def normalize_ballot(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "judge": raw.get("judge", ""),
        "solver_1": raw.get("solver_1", ""),
        "solver_2": raw.get("solver_2", ""),
        "solver_3": raw.get("solver_3", ""),
        "reasoning": raw.get("reasoning", ""),
    }


def aggregate_role_ballots(
    ballots: list[dict[str, Any]],
    preferences: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """
    Combine Stage 0.5 ballots from all LLMs into one backend_id -> role map.
    Judge is chosen by majority vote; solver slots by averaged rank.
    """
    backend_ids = set(preferences.keys())
    valid = [b for b in ballots if validate_ballot(b, backend_ids)]
    if not valid:
        raise ValueError("No valid role ballots from LLMs")

    judge_votes = Counter(b["judge"] for b in valid)
    top_count = judge_votes.most_common(1)[0][1]
    judge_candidates = [bid for bid, count in judge_votes.items() if count == top_count]
    judge_backend = _tiebreak_judge(judge_candidates, preferences)

    remaining = sorted(backend_ids - {judge_backend})
    rank_scores: dict[str, float] = defaultdict(float)
    for ballot in valid:
        for idx, role in enumerate(SOLVER_ROLES):
            backend_id = ballot[role]
            if backend_id in remaining:
                rank_scores[backend_id] += idx

    ordered_solvers = sorted(remaining, key=lambda bid: (rank_scores[bid], bid))
    assignment: dict[str, str] = {judge_backend: JUDGE_ROLE}
    for role, backend_id in zip(SOLVER_ROLES, ordered_solvers):
        assignment[backend_id] = role
    return assignment


def _tiebreak_judge(candidates: list[str], preferences: dict[str, dict[str, Any]]) -> str:
    def judge_conf(backend_id: str) -> float:
        conf = preferences[backend_id].get("confidence_by_role", {})
        return float(conf.get("Judge", conf.get("judge", 0.0)))

    return sorted(candidates, key=lambda bid: (-judge_conf(bid), bid))[0]


def invert_assignment(assignment: dict[str, str]) -> dict[str, str]:
    return {role: backend for backend, role in assignment.items()}
