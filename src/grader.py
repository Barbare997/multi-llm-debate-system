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


def is_valid_final_answer(answer: Any, question_type: str = "answer") -> bool:
    """For answer-type problems, require a short numeric (or m+n) final answer."""
    if question_type != "answer":
        return True
    text = as_answer_text(answer).strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"-?\d+", compact):
        return True
    if re.fullmatch(r"\d+\+\d+", compact):
        return True
    labeled = re.search(
        r"(?:final answer|answer is)\s*:?\s*(-?\d+(?:\s*\+\s*\d+)?)",
        text,
        re.IGNORECASE,
    )
    return labeled is not None


def sanitize_final_answer(answer: Any, problem: dict[str, Any]) -> str:
    """Normalize accepted answer forms; return empty string for invalid answer-type outputs."""
    text = as_answer_text(answer).strip()
    if problem.get("question_type") != "answer":
        return text
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"-?\d+", compact):
        return compact
    if re.fullmatch(r"\d+\+\d+", compact):
        return compact
    labeled = re.search(
        r"(?:final answer|answer is)\s*:?\s*(-?\d+(?:\s*\+\s*\d+)?)",
        text,
        re.IGNORECASE,
    )
    if labeled:
        return re.sub(r"\s+", "", labeled.group(1))
    return ""


def quick_match(
    candidate: Any,
    ground_truth: Any,
    *,
    question_type: str | None = None,
) -> bool:
    candidate = as_answer_text(candidate)
    ground_truth = as_answer_text(ground_truth)
    if not candidate or not ground_truth:
        return False
    c = normalize_answer(candidate)
    g = normalize_answer(ground_truth)
    if c == g:
        return True
    gt_compact = re.sub(r"\s+", "", g)
    if question_type == "answer" and re.fullmatch(r"-?\d+", gt_compact):
        cand_compact = re.sub(r"\s+", "", c)
        if re.fullmatch(r"-?\d+", cand_compact):
            return cand_compact == gt_compact
        return False
    # Allow "answer: 180" style when ground truth is a short token
    if len(g) <= 40 and g in c:
        return True
    nums_c = set(re.findall(r"-?\d+(?:\.\d+)?", c))
    nums_g = set(re.findall(r"-?\d+(?:\.\d+)?", g))
    return bool(nums_g) and nums_g == nums_c


def is_correct_answer(
    grader: LLMClient,
    problem: dict[str, Any],
    answer: Any,
    solution: Any = "",
) -> bool:
    answer = sanitize_final_answer(answer, problem)
    if problem.get("question_type") == "answer" and not is_valid_final_answer(
        answer, "answer"
    ):
        return False
    if quick_match(answer, problem["ground_truth_answer"], question_type=problem.get("question_type")):
        return True
    graded = grade_answer(grader, problem, answer, solution)
    return bool(graded.get("correct"))
