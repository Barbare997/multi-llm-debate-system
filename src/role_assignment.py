from __future__ import annotations

import os
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


def _judge_confidence(backend_id: str, preferences: dict[str, dict[str, Any]]) -> float:
    conf = preferences[backend_id].get("confidence_by_role", {})
    try:
        value = float(conf.get("Judge", conf.get("judge", 0.5)))
    except (TypeError, ValueError):
        value = 0.5
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _vote_weight() -> float:
    raw = os.getenv("ROLE_JUDGE_VOTE_WEIGHT", "1.0")
    try:
        return float(raw)
    except ValueError:
        return 1.0


def _confidence_weight() -> float:
    raw = os.getenv("ROLE_JUDGE_CONF_WEIGHT", "1.0")
    try:
        return float(raw)
    except ValueError:
        return 1.0


def hybrid_judge_scores(
    ballots: list[dict[str, Any]],
    preferences: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """
    Hybrid Stage 0.5 score per backend:
      score = vote_weight * (judge votes) + conf_weight * (Stage 0 judge self-confidence)
    """
    backend_ids = set(preferences.keys())
    valid = [b for b in ballots if validate_ballot(b, backend_ids)]
    judge_votes = Counter(b["judge"] for b in valid) if valid else Counter()

    vote_w = _vote_weight()
    conf_w = _confidence_weight()
    scores: dict[str, float] = {}
    for backend_id in backend_ids:
        scores[backend_id] = vote_w * judge_votes.get(backend_id, 0) + conf_w * _judge_confidence(
            backend_id, preferences
        )
    return scores


def aggregate_role_ballots(
    ballots: list[dict[str, Any]],
    preferences: dict[str, dict[str, Any]],
    cloud_backend_ids: set[str] | None = None,
) -> dict[str, str]:
    """
    Stage 0.5 hybrid algorithm:
    1. Each LLM submits a role ballot (who should judge + solver ranking).
    2. Judge = argmax hybrid score (votes + Stage 0 judge confidence).
    3. Ties broken by preferring a cloud/API backend, then higher judge confidence.
    4. Remaining three backends fill solver slots by averaged ballot rank.
    """
    backend_ids = set(preferences.keys())
    cloud_backend_ids = cloud_backend_ids or set()
    valid = [b for b in ballots if validate_ballot(b, backend_ids)]
    if not valid:
        raise ValueError("No valid role ballots from LLMs")

    scores = hybrid_judge_scores(valid, preferences)
    max_score = max(scores.values())
    judge_candidates = [bid for bid, score in scores.items() if score == max_score]
    judge_backend = _tiebreak_judge(judge_candidates, preferences, cloud_backend_ids)

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


def _tiebreak_judge(
    candidates: list[str],
    preferences: dict[str, dict[str, Any]],
    cloud_backend_ids: set[str],
) -> str:
    if len(candidates) == 1:
        return candidates[0]

    cloud_tied = sorted(bid for bid in candidates if bid in cloud_backend_ids)
    if cloud_tied:
        return cloud_tied[0]

    return sorted(
        candidates,
        key=lambda bid: (-_judge_confidence(bid, preferences), bid),
    )[0]


def invert_assignment(assignment: dict[str, str]) -> dict[str, str]:
    return {role: backend for backend, role in assignment.items()}
