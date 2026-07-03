from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOTS_DIR, RESULTS_DIR
from src.pipeline import load_debate_result


def collect_debate_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/debate.json")):
        data = load_debate_result(path)
        grading = data.get("grading", {})
        rows.append(
            {
                "problem_id": data["problem_id"],
                "final_correct": grading.get("final_correct", False),
                "consensus_initial": grading.get("consensus_initial", False),
                "consensus_refined": grading.get("consensus_refined", False),
                "improved": grading.get("improved", False),
                "any_initial_correct": any(grading.get("initial_correct", {}).values()),
                "any_refined_correct": any(grading.get("refined_correct", {}).values()),
                "refinement_hurt": grading.get("refinement_hurt", False),
                "judge_missed_best": grading.get("judge_missed_best", False),
                "winner": data.get("winner"),
            }
        )
    return pd.DataFrame(rows)


def collect_baseline_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/baselines.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = path.parent.name
        rows.append(
            {
                "problem_id": pid,
                "single_correct": data.get("single", {}).get("correct", False),
                "voting_correct": data.get("voting", {}).get("correct", False),
            }
        )
    return pd.DataFrame(rows)


def compute_summary(debate_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict[str, float]:
    n = max(len(debate_df), 1)
    summary = {
        "overall_accuracy": debate_df["final_correct"].mean() if len(debate_df) else 0.0,
        "improvement_rate": debate_df["improved"].mean() if len(debate_df) else 0.0,
        "consensus_rate_initial": debate_df["consensus_initial"].mean() if len(debate_df) else 0.0,
        "consensus_rate_refined": debate_df["consensus_refined"].mean() if len(debate_df) else 0.0,
        "refinement_hurt_rate": debate_df["refinement_hurt"].mean() if len(debate_df) else 0.0,
        "judge_missed_best_rate": debate_df["judge_missed_best"].mean() if len(debate_df) else 0.0,
    }

    if len(baseline_df):
        merged = debate_df.merge(baseline_df, on="problem_id", how="left")
        disagree = merged[~merged["consensus_initial"].fillna(False)]
        if len(disagree):
            summary["judge_accuracy_when_disagree"] = disagree["final_correct"].mean()
        else:
            summary["judge_accuracy_when_disagree"] = float("nan")
        summary["single_llm_accuracy"] = baseline_df["single_correct"].mean()
        summary["voting_accuracy"] = baseline_df["voting_correct"].mean()
    return summary


def plot_results(debate_df: pd.DataFrame, baseline_df: pd.DataFrame, summary: dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: system comparison
    labels = ["Debate System", "Single LLM", "Majority Vote"]
    values = [
        summary.get("overall_accuracy", 0) * 100,
        summary.get("single_llm_accuracy", 0) * 100,
        summary.get("voting_accuracy", 0) * 100,
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#2563eb", "#94a3b8", "#64748b"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("System Comparison on Evaluated Problems")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    # Debate metrics
    metric_labels = [
        "Overall Accuracy",
        "Improvement Rate",
        "Initial Consensus",
        "Refined Consensus",
        "Refinement Hurt",
        "Judge Missed Best",
    ]
    metric_values = [
        summary.get("overall_accuracy", 0) * 100,
        summary.get("improvement_rate", 0) * 100,
        summary.get("consensus_rate_initial", 0) * 100,
        summary.get("consensus_rate_refined", 0) * 100,
        summary.get("refinement_hurt_rate", 0) * 100,
        summary.get("judge_missed_best_rate", 0) * 100,
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metric_labels, metric_values, color="#7c3aed")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Debate System Metrics")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "debate_metrics.png", dpi=150)
    plt.close(fig)

    # Per-problem correctness
    if len(debate_df):
        merged = debate_df.copy()
        if len(baseline_df):
            merged = merged.merge(baseline_df, on="problem_id", how="left")
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(merged))
        ax.plot(x, merged["final_correct"].fillna(False).astype(int), marker="o", label="Debate")
        if "single_correct" in merged.columns:
            ax.plot(
                x,
                merged["single_correct"].fillna(False).astype(int),
                marker="s",
                label="Single LLM",
            )
        if "voting_correct" in merged.columns:
            ax.plot(
                x,
                merged["voting_correct"].fillna(False).astype(int),
                marker="^",
                label="Voting",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(merged["problem_id"], rotation=45, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.set_title("Per-Problem Correctness")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "per_problem_correctness.png", dpi=150)
        plt.close(fig)

    # Save summary JSON
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_evaluation(results_dir: Path | None = None, plots_dir: Path | None = None) -> dict[str, float]:
    results_dir = results_dir or RESULTS_DIR
    plots_dir = plots_dir or PLOTS_DIR
    debate_df = collect_debate_metrics(results_dir)
    baseline_df = collect_baseline_metrics(results_dir)
    summary = compute_summary(debate_df, baseline_df)
    plot_results(debate_df, baseline_df, summary, plots_dir)
    debate_df.to_csv(plots_dir / "debate_results.csv", index=False)
    if len(baseline_df):
        baseline_df.to_csv(plots_dir / "baseline_results.csv", index=False)
    return summary
