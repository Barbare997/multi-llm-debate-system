from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import structure_repair_enabled
from src.grader import is_correct_answer, sanitize_final_answer
from src.llm import LLMClient
from src.response_parser import ParseError, parse_stage_response, repair_template


SINGLE_SYSTEM = """Solve this olympiad mathematics problem independently from the statement only — you do not have the official answer.

For answer-type problems: full reasoning in SOLUTION. FINAL_ANSWER must be one integer only (no units, words, or sentences). Answer exactly what the problem asks for (e.g. after the bus arrived, not before).
For proof-type problems: full proof in SOLUTION, main claim in FINAL_ANSWER.

Reply in plain text using exactly this format:
SOLUTION:
...

FINAL_ANSWER:
one integer for answer-type problems, or main claim for proofs"""


REPAIR_SYSTEM = """You are a strict formatter. Extract the intended math content and return ONLY:
SOLUTION:
...

FINAL_ANSWER:
one integer only for answer-type problems, or main claim for proofs"""


def _problem_statement(problem: dict[str, Any]) -> str:
    return (
        f"Type: {problem['question_type']}\n\n"
        f"{problem['statement']}"
    )


def _complete_solution(
    client: LLMClient,
    problem: dict[str, Any],
    repair_client: LLMClient | None = None,
) -> dict[str, Any]:
    user = _problem_statement(problem)
    raw = client.complete(SINGLE_SYSTEM, user, max_output_tokens=4096)
    try:
        return parse_stage_response("stage1", raw)
    except ParseError:
        if repair_client is not None and structure_repair_enabled():
            repaired = repair_client.complete(
                REPAIR_SYSTEM,
                f"Required format:\n{repair_template('stage1')}\n\nBroken response:\n{raw}",
                temperature=0.0,
                max_output_tokens=4096,
            )
            return parse_stage_response("stage1", repaired)
        raw = client.complete(SINGLE_SYSTEM, user, max_output_tokens=4096)
        return parse_stage_response("stage1", raw)


def run_single_llm_baseline(
    client: LLMClient,
    problem: dict[str, Any],
    grader: LLMClient,
    repair_client: LLMClient | None = None,
) -> dict[str, Any]:
    parsed = _complete_solution(client, problem, repair_client=repair_client)
    answer = sanitize_final_answer(parsed.get("final_answer", ""), problem)
    solution = parsed.get("solution", "")
    correct = is_correct_answer(grader, problem, answer, solution)
    return {"final_answer": answer, "solution": solution, "correct": correct}


def run_voting_baseline(
    clients: list[LLMClient],
    problem: dict[str, Any],
    grader: LLMClient,
    repair_client: LLMClient | None = None,
) -> dict[str, Any]:
    answers: list[str] = []
    details: list[dict[str, Any]] = []
    for client in clients:
        parsed = _complete_solution(client, problem, repair_client=repair_client)
        answer = sanitize_final_answer(parsed.get("final_answer", ""), problem)
        solution = parsed.get("solution", "")
        answers.append(answer)
        details.append(
            {
                "backend": client.spec.backend_id,
                "final_answer": answer,
                "solution": solution,
                "correct": is_correct_answer(grader, problem, answer, solution),
            }
        )

    winner = _pick_voting_winner(details, problem)
    winner_row = next((d for d in details if d["final_answer"] == winner), {})
    winner_solution = winner_row.get("solution", "")
    correct = is_correct_answer(grader, problem, winner, winner_solution)
    return {"final_answer": winner, "votes": answers, "details": details, "correct": correct}


def _pick_voting_winner(details: list[dict[str, Any]], problem: dict[str, Any]) -> str:
    if not details:
        return ""
    counts = Counter(d["final_answer"] for d in details)
    max_count = max(counts.values())
    tied_answers = [answer for answer, count in counts.items() if count == max_count]
    if len(tied_answers) == 1:
        return tied_answers[0]

    candidates = [d for d in details if d["final_answer"] in tied_answers]
    correct_rows = [d for d in candidates if d.get("correct")]
    if len(correct_rows) == 1:
        return correct_rows[0]["final_answer"]
    if correct_rows:
        candidates = correct_rows

    backend_priority = {"openai_strong": 0, "openai_mini": 1, "gemini": 2, "groq": 3}
    candidates.sort(key=lambda row: backend_priority.get(row["backend"], 99))
    return candidates[0]["final_answer"]


def save_baseline(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
