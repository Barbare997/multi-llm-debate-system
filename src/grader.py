from __future__ import annotations

import json
import re
from typing import Any

from src.json_utils import extract_json
from src.llm import LLMClient


def as_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def grade_answer(
    client: LLMClient,
    problem: dict[str, Any],
    candidate_answer: Any,
    candidate_solution: Any = "",
) -> dict[str, Any]:
    """Grade a candidate answer against ground truth (numeric or proof)."""
    candidate_answer = as_answer_text(candidate_answer)
    candidate_solution = as_answer_text(candidate_solution)
    system = (
        "You are an expert olympiad grader. Compare the candidate answer to the official "
        "ground truth. For answer-type problems, accept mathematically equivalent forms "
        "(e.g. 180 vs 'the answer is 180', or k=2 vs k = 2). "
        "For proof problems, check whether the main claim is established with "
        "valid reasoning (full rigor not required, but no fatal errors). "
        "Respond ONLY with JSON: "
        '{"correct": true/false, "score": 0.0-1.0, "reasoning": "brief explanation"}'
    )
    user = (
        f"Subject: {problem['subject']}\n"
        f"Question type: {problem['question_type']}\n"
        f"Problem:\n{problem['statement']}\n\n"
        f"Official ground truth:\n{problem['ground_truth_answer']}\n\n"
        f"Official solution reference:\n{problem.get('solution_reference', '')[:4000]}\n\n"
        f"Candidate final answer:\n{candidate_answer}\n\n"
        f"Candidate solution excerpt:\n{candidate_solution[:4000]}"
    )
    raw = client.complete(system, user, temperature=0.0, max_output_tokens=512)
    result = extract_json(raw)
    result["correct"] = bool(result.get("correct", False))
    result["score"] = float(result.get("score", 0.0))
    return result


def normalize_answer(text: Any) -> str:
    text = as_answer_text(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("$", "")
    return text


def quick_match(candidate: Any, ground_truth: Any) -> bool:
    candidate = as_answer_text(candidate)
    ground_truth = as_answer_text(ground_truth)
    if not candidate or not ground_truth:
        return False
    c = normalize_answer(candidate)
    g = normalize_answer(ground_truth)
    if c == g:
        return True
    # Allow "answer: 180" style when ground truth is a short token
    if len(g) <= 40 and g in c:
        return True
    nums_c = set(re.findall(r"-?\d+(?:\.\d+)?", c))
    nums_g = set(re.findall(r"-?\d+(?:\.\d+)?", g))
    return bool(nums_g) and nums_g == nums_c
