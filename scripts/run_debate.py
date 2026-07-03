#!/usr/bin/env python3
"""Run the multi-LLM debate pipeline on IZhO problems."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import run_single_llm_baseline, run_voting_baseline, save_baseline
from src.config import DATA_DIR, RESULTS_DIR, get_backends, require_api_keys
from src.llm import build_clients
from src.pipeline import DebatePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run debate pipeline on IZhO problems")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run")
    parser.add_argument("--ids", nargs="*", help="Specific problem IDs")
    parser.add_argument(
        "--problems-file",
        default="problems.json",
        help="Problem list under data/ (e.g. problems_medium.json)",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-debate", action="store_true")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between problems")
    parser.add_argument("--quiet", action="store_true", help="Hide per-stage progress logs")
    args = parser.parse_args()

    require_api_keys()
    problems_path = DATA_DIR / args.problems_file
    if not problems_path.exists():
        raise SystemExit(f"Problems file not found: {problems_path}")
    problems = json.loads(problems_path.read_text(encoding="utf-8"))

    if args.ids:
        problems = [p for p in problems if p["id"] in args.ids]
    if args.limit:
        problems = problems[: args.limit]

    backends = get_backends()
    clients = build_clients(backends)
    pipeline = DebatePipeline(
        backends,
        grader_client=clients["openai_strong"],
        verbose=not args.quiet,
    )
    grader = clients["openai_strong"]
    solver_clients = [clients[b.backend_id] for b in backends[:3]]

    for problem in tqdm(problems, desc="Problems"):
        out_dir = RESULTS_DIR / problem["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_debate:
            debate_path = out_dir / "debate.json"
            if not debate_path.exists():
                pipeline.run_problem(problem, save_dir=out_dir)

        if not args.skip_baselines:
            baseline_path = out_dir / "baselines.json"
            if not baseline_path.exists():
                single = run_single_llm_baseline(clients["groq"], problem, grader)
                voting = run_voting_baseline(solver_clients, problem, grader)
                save_baseline(baseline_path, {"single": single, "voting": voting})

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
