from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config import BackendSpec, RESULTS_DIR, structure_repair_enabled
from src.grader import as_answer_text, is_correct_answer, sanitize_final_answer
from src.llm import LLMClient, build_clients
from src.response_parser import ParseError, parse_stage_response, repair_template
from src.role_assignment import (
    JUDGE_ROLE,
    SOLVER_ROLES,
    aggregate_role_ballots,
    format_preferences_for_assignment,
    hybrid_judge_scores,
    invert_assignment,
    normalize_ballot,
    normalize_preferences,
)


STAGE0_SYSTEM = """You are participating in a multi-LLM olympiad debate system.
Given a problem, self-assess which role suits you best: Solver or Judge.

Reply in plain text using exactly this format:
PREFERRED_ROLE: Solver|Judge
SOLVER_CONFIDENCE: 0.0-1.0
JUDGE_CONFIDENCE: 0.0-1.0
REASONING:
brief explanation"""


STAGE0_5_SYSTEM = """You are assigning roles for a multi-LLM olympiad debate.
You will see each participant's Stage 0 self-assessment (your backend_id is given below).
Choose exactly one Judge and rank the other three as solvers.

Reply in plain text using exactly this format:
judge=<backend_id>
solver_1=<backend_id>
solver_2=<backend_id>
solver_3=<backend_id>
REASONING:
brief explanation

Rules:
- Use each backend_id exactly once.
- Pick the participant best suited to judge (not necessarily yourself).
- solver_1 is strongest, solver_3 is weakest."""


STAGE1_SYSTEM = """You are a Solver in an olympiad mathematics debate. Solve the problem independently from the statement only — you do not have access to the official answer or solution.

For answer-type problems: full reasoning in SOLUTION. FINAL_ANSWER must be one integer only (no units, words, or sentences). Answer exactly what the problem asks for (e.g. after the bus arrived, not before).
For proof-type problems: full proof in SOLUTION, main claim in FINAL_ANSWER.

Reply in plain text using exactly this format:
SOLUTION:
full step-by-step reasoning

FINAL_ANSWER:
one integer for answer-type problems, or main claim for proofs"""


STAGE2_SYSTEM = """You are a peer reviewer in an olympiad debate. Critique the given solution rigorously.
Do not solve the problem yourself — only evaluate the solution provided.

Be specific and mathematical. Only flag real errors, not stylistic preferences.
If a step is correct, say so. Do not invent mistakes.
Distinguish critical errors from minor presentation issues.

Reply in plain text using exactly this format:
STRENGTHS:
- ...

WEAKNESSES:
- ...

ERRORS:
- location | error_type | description | severity

SUGGESTED_CHANGES:
- ...

OVERALL_ASSESSMENT: promising_but_flawed|fundamentally_wrong|correct|incomplete"""


STAGE3_SYSTEM = """You are a Solver revising your solution after peer review. You still do not have the official answer — rely on your reasoning and the critiques.

Rules:
- Address each critique explicitly.
- If a critique is mathematically wrong, defend your original step and keep it.
- Only change parts where the critique identifies a real logical or calculation error.
- Do not change a correct answer just because peers were skeptical.
- If no critique is valid, keep your solution and final answer unchanged.
- For answer-type problems: FINAL_ANSWER must be one integer only (no units, words, or sentences). Answer exactly what the problem asks for (e.g. after the bus arrived, not before).

Reply in plain text using exactly this format:
CHANGES_MADE:
- critique | response | accepted=true/false

REFINED_SOLUTION:
updated reasoning or proof (or unchanged if no valid critiques)

FINAL_ANSWER:
one integer for answer-type problems, or main claim for proofs

CONFIDENCE: 0.0-1.0"""


STAGE4_SYSTEM = """You are the final Judge in an olympiad debate. Pick the best refined solution.
You do not have the official answer — judge only by mathematical correctness, completeness, and logical validity.

You receive each solver's original solution, peer reviews, and refined solution.
Compare original vs refined for each solver:
- Prefer refinement only when it clearly fixes a real error.
- If refinement made a solution worse, do not reward that solver.
- Prefer correctness over verbosity or presentation.

Reply in plain text using exactly this format:
WINNER: solver_1|solver_2|solver_3
CONFIDENCE: 0.0-1.0
REASONING:
brief explanation comparing all three refined solutions"""


REPAIR_SYSTEM = """You are a strict formatter for a multi-LLM debate pipeline.
You receive a model response that failed parsing.
Extract the intended content without changing mathematical meaning.
Return ONLY the required plain-text template — no markdown fences, no JSON."""


class DebatePipeline:
    def __init__(
        self,
        backends: list[BackendSpec],
        grader_client: LLMClient | None = None,
        verbose: bool = True,
    ) -> None:
        self.backends = {b.backend_id: b for b in backends}
        self.clients = build_clients(backends)
        self.cloud_backend_ids = {bid for bid, spec in self.backends.items() if spec.provider == "openai"}
        self.grader_client = grader_client or self.clients[backends[0].backend_id]
        self.repair_client = self._pick_repair_client()
        self.verbose = verbose

    def _pick_repair_client(self) -> LLMClient | None:
        if not structure_repair_enabled():
            return None
        for backend_id in ("openai_mini", "openai_strong"):
            client = self.clients.get(backend_id)
            if client and client.spec.provider == "openai":
                return client
        for client in self.clients.values():
            if client.spec.provider == "openai":
                return client
        return None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _call(self, client: LLMClient, system: str, user: str, max_tokens: int, label: str) -> str:
        self._log(f"{label} - waiting for API...")
        t0 = time.time()
        raw = client.complete(system, user, temperature=0.2, max_output_tokens=max_tokens)
        self._log(f"{label} - done ({time.time() - t0:.1f}s)")
        return raw

    def _repair_with_openai(
        self,
        stage: str,
        raw: str,
        *,
        backend_ids: set[str] | None = None,
        label: str,
    ) -> str:
        if self.repair_client is None:
            raise ParseError("structure repair requested but no OpenAI client is available")
        template = repair_template(stage, backend_ids=backend_ids)
        user = (
            f"Required format:\n{template}\n\n"
            f"Broken model response:\n{raw}\n\n"
            "Reformat the broken response into the required format."
        )
        self._log(f"{label} - OpenAI structure repair...")
        return self.repair_client.complete(REPAIR_SYSTEM, user, temperature=0.0, max_output_tokens=2048)

    def _call_structured(
        self,
        stage: str,
        client: LLMClient,
        system: str,
        user: str,
        max_tokens: int,
        label: str,
        *,
        backend_ids: set[str] | None = None,
        solution_id: str | None = None,
    ) -> Any:
        raw = self._call(client, system, user, max_tokens, label)
        try:
            return parse_stage_response(
                stage,
                raw,
                backend_ids=backend_ids,
                solution_id=solution_id,
            )
        except ParseError as exc:
            self._log(f"{label} - parse failed ({exc})")
            if self.repair_client is not None:
                try:
                    repaired_raw = self._repair_with_openai(
                        stage,
                        raw,
                        backend_ids=backend_ids,
                        label=label,
                    )
                    return parse_stage_response(
                        stage,
                        repaired_raw,
                        backend_ids=backend_ids,
                        solution_id=solution_id,
                    )
                except ParseError as repair_exc:
                    self._log(f"{label} - OpenAI repair failed ({repair_exc}); retrying model once...")
            else:
                self._log(f"{label} - retrying model once...")

            raw = self._call(client, system, user, max_tokens, f"{label} (retry)")
            try:
                return parse_stage_response(
                    stage,
                    raw,
                    backend_ids=backend_ids,
                    solution_id=solution_id,
                )
            except ParseError:
                if self.repair_client is None:
                    raise
                repaired_raw = self._repair_with_openai(
                    stage,
                    raw,
                    backend_ids=backend_ids,
                    label=f"{label} (retry repair)",
                )
                return parse_stage_response(
                    stage,
                    repaired_raw,
                    backend_ids=backend_ids,
                    solution_id=solution_id,
                )

    def run_problem(self, problem: dict[str, Any], save_dir: Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"problem_id": problem["id"], "stages": {}}
        pid = problem["id"]
        backend_ids = set(self.backends.keys())
        self._log(f"\n[{pid}] Starting debate pipeline...")

        # Stage 0
        prefs: dict[str, dict[str, Any]] = {}
        for backend_id, client in self.clients.items():
            self._log(f"[{pid}] Stage 0: role preference ({backend_id})...")
            prefs[backend_id] = normalize_preferences(
                self._call_structured(
                    "stage0",
                    client,
                    STAGE0_SYSTEM,
                    self._problem_prompt(problem),
                    512,
                    f"[{pid}] Stage 0 ({backend_id})",
                )
            )
        result["stages"]["stage0_preferences"] = prefs

        # Stage 0.5 — each LLM votes on role assignment (majority decides judge)
        pref_text = format_preferences_for_assignment(prefs)
        ballots: list[dict[str, Any]] = []
        for backend_id, client in self.clients.items():
            self._log(f"[{pid}] Stage 0.5: role ballot ({backend_id})...")
            prompt = (
                f"{pref_text}\n\n"
                f"Your backend_id: {backend_id}\n"
                f"Valid backend_ids: {', '.join(sorted(backend_ids))}\n"
                "Assign roles for all four participants."
            )
            raw_ballot = self._call_structured(
                "stage0_5",
                client,
                STAGE0_5_SYSTEM,
                prompt,
                512,
                f"[{pid}] Stage 0.5 ({backend_id})",
                backend_ids=backend_ids,
            )
            ballot = normalize_ballot(raw_ballot)
            ballot["voter"] = backend_id
            ballots.append(ballot)
        result["stages"]["stage0_5_ballots"] = ballots
        result["stages"]["stage0_5_hybrid_scores"] = hybrid_judge_scores(ballots, prefs)
        assignment = aggregate_role_ballots(ballots, prefs, cloud_backend_ids=self.cloud_backend_ids)
        role_to_backend = invert_assignment(assignment)
        result["stages"]["stage0_5_assignment"] = assignment
        result["stages"]["role_to_backend"] = role_to_backend
        self._log(f"[{pid}] Stage 0.5: roles assigned by LLM vote -> {assignment}")

        # Stage 1
        solutions: dict[str, dict[str, Any]] = {}
        for role in SOLVER_ROLES:
            backend_id = role_to_backend[role]
            self._log(f"[{pid}] Stage 1: solving ({role} via {backend_id})...")
            solutions[role] = self._call_structured(
                "stage1",
                self.clients[backend_id],
                STAGE1_SYSTEM,
                self._problem_prompt(problem),
                4096,
                f"[{pid}] Stage 1 ({role})",
            )
            solutions[role]["final_answer"] = sanitize_final_answer(
                solutions[role].get("final_answer", ""), problem
            )
        result["stages"]["stage1_solutions"] = solutions

        # Stage 2
        reviews: dict[str, dict[str, Any]] = {}
        for reviewer_role in SOLVER_ROLES:
            for target_role in SOLVER_ROLES:
                if reviewer_role == target_role:
                    continue
                backend_id = role_to_backend[reviewer_role]
                self._log(f"[{pid}] Stage 2: {reviewer_role} reviews {target_role}...")
                prompt = (
                    f"{self._problem_prompt(problem)}\n\n"
                    f"Review solution_id={target_role}:\n"
                    f"solution:\n{solutions[target_role].get('solution', '')}\n"
                    f"final_answer:\n{solutions[target_role].get('final_answer', '')}"
                )
                review = self._call_structured(
                    "stage2",
                    self.clients[backend_id],
                    STAGE2_SYSTEM,
                    prompt,
                    2048,
                    f"[{pid}] Stage 2 ({reviewer_role}->{target_role})",
                    solution_id=target_role,
                )
                review["solution_id"] = target_role
                reviews[f"{reviewer_role}_reviews_{target_role}"] = review
        result["stages"]["stage2_reviews"] = reviews

        # Stage 3
        refined: dict[str, dict[str, Any]] = {}
        for role in SOLVER_ROLES:
            backend_id = role_to_backend[role]
            self._log(f"[{pid}] Stage 3: refining ({role})...")
            peer_reviews = [
                reviews[f"{r}_reviews_{role}"]
                for r in SOLVER_ROLES
                if r != role
            ]
            prompt = (
                f"{self._problem_prompt(problem)}\n\n"
                f"Your original solution:\n{solutions[role].get('solution', '')}\n"
                f"Your original final answer:\n{solutions[role].get('final_answer', '')}\n\n"
                f"Peer reviews:\n{json.dumps(peer_reviews, indent=2)}"
            )
            refined[role] = self._call_structured(
                "stage3",
                self.clients[backend_id],
                STAGE3_SYSTEM,
                prompt,
                4096,
                f"[{pid}] Stage 3 ({role})",
            )
            refined[role]["refined_answer"] = sanitize_final_answer(
                refined[role].get("refined_answer", ""), problem
            )
        result["stages"]["stage3_refined"] = refined

        # Stage 4
        judge_backend = role_to_backend[JUDGE_ROLE]
        self._log(f"[{pid}] Stage 4: judging (via {judge_backend})...")
        judge_prompt = (
            f"{self._problem_prompt(problem)}\n\n"
            f"Original solutions:\n{json.dumps(solutions, indent=2)}\n\n"
            f"Peer reviews:\n{json.dumps(reviews, indent=2)}\n\n"
            f"Refined solutions:\n{json.dumps(refined, indent=2)}"
        )
        judgment = self._call_structured(
            "stage4",
            self.clients[judge_backend],
            STAGE4_SYSTEM,
            judge_prompt,
            1024,
            f"[{pid}] Stage 4 (judge)",
        )
        result["stages"]["stage4_judgment"] = judgment

        winner = judgment.get("winner", "solver_1")
        final_answer = sanitize_final_answer(
            refined.get(winner, {}).get("refined_answer", ""),
            problem,
        )
        final_solution = as_answer_text(refined.get(winner, {}).get("refined_solution", ""))
        result["final_answer"] = final_answer
        result["final_solution"] = final_solution
        result["winner"] = winner

        # Grade outcomes
        self._log(f"[{pid}] Grading final answer...")
        result["grading"] = self._grade_problem(
            problem, solutions, refined, final_answer, final_solution, winner
        )

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "debate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        self._log(f"[{pid}] Done. Winner: {winner}")
        return result

    def _grade_problem(
        self,
        problem: dict[str, Any],
        solutions: dict[str, dict[str, Any]],
        refined: dict[str, dict[str, Any]],
        final_answer: str,
        final_solution: str,
        winner: str,
    ) -> dict[str, Any]:
        gt = problem["ground_truth_answer"]
        initial_answers = {r: as_answer_text(solutions[r].get("final_answer", "")) for r in SOLVER_ROLES}
        refined_answers = {r: as_answer_text(refined[r].get("refined_answer", "")) for r in SOLVER_ROLES}

        def is_correct(answer: Any, solution: Any) -> bool:
            answer = sanitize_final_answer(answer, problem)
            solution = as_answer_text(solution)
            return is_correct_answer(self.grader_client, problem, answer, solution)

        initial_correct = {
            r: is_correct(solutions[r].get("final_answer", ""), solutions[r].get("solution", ""))
            for r in SOLVER_ROLES
        }
        refined_correct = {
            r: is_correct(refined[r].get("refined_answer", ""), refined[r].get("refined_solution", ""))
            for r in SOLVER_ROLES
        }
        final_correct = is_correct(final_answer, final_solution)

        refinement_hurt = any(
            initial_correct[r] and not refined_correct[r]
            for r in SOLVER_ROLES
        )
        any_refined_correct = any(refined_correct.values())
        judge_missed_best = any_refined_correct and not final_correct

        return {
            "ground_truth": gt,
            "initial_correct": initial_correct,
            "refined_correct": refined_correct,
            "final_correct": final_correct,
            "consensus_initial": len(set(initial_answers.values())) == 1,
            "consensus_refined": len(set(refined_answers.values())) == 1,
            "refinement_hurt": refinement_hurt,
            "judge_missed_best": judge_missed_best,
            "winner_was_refined_correct": refined_correct.get(winner, False),
            "improved": (not any(initial_correct.values())) and final_correct
            or (any(refined_correct[r] and not initial_correct[r] for r in SOLVER_ROLES))
            or (final_correct and not any(initial_correct.values())),
        }

    @staticmethod
    def _problem_prompt(problem: dict[str, Any]) -> str:
        """Problem text for solvers/judge — statement only, no official answer or URL."""
        return (
            f"Subject: {problem['subject']}\n"
            f"Type: {problem['question_type']}\n\n"
            f"{problem['statement']}"
        )


def load_debate_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
