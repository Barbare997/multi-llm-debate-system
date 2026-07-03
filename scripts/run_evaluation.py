#!/usr/bin/env python3
"""Generate evaluation metrics and plots from debate results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PLOTS_DIR, RESULTS_DIR
from src.evaluation import run_evaluation


def main() -> None:
    summary = run_evaluation(RESULTS_DIR, PLOTS_DIR)
    print("Evaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
