"""Rebuild the paper's result tables from the raw experiment CSVs.

    python paper/reproduce.py

Checks that every reported average matches the per-dimension scores, so the
published numbers can be verified from the data in this repository.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if any(v and v.strip() for v in row.values())]


def experiment1() -> list[tuple[str, float, float]]:
    """Experiment 1: CV generation from a casual prompt. Returns (model, reported, recomputed)."""
    dims = ["Completeness", "Contextual_Utilisation", "Factual_Accuracy", "Confidence_Calibration"]
    out = []
    for row in load("experiment1_model_comparison.csv"):
        recomputed = statistics.fmean(float(row[d]) for d in dims)
        out.append((row["Model"], float(row["Average"]), round(recomputed, 3)))
    return sorted(out, key=lambda r: -r[1])


def experiment2() -> dict[str, dict[str, float]]:
    """Experiment 2: factual accuracy per prompt wording and model."""
    table: dict[str, dict[str, float]] = {}
    for row in load("experiment2_prompt_variants.csv"):
        table.setdefault(row["Prompt"], {})[row["Model"]] = float(row["Factual_Accuracy"])
    return table


def experiment3() -> list[tuple[str, list[int], float]]:
    """Experiment 3: ChatGPT summary scored in three separate sessions."""
    out = []
    for row in load("experiment3_consistency.csv"):
        sessions = [(row.get(f"Session_{i}") or "").strip() for i in (1, 2, 3)]
        if not all(sessions) or not row.get("Dimension"):
            continue
        scores = [int(s) for s in sessions]
        out.append((row["Dimension"], scores, round(statistics.fmean(scores), 2)))
    return out


def main() -> None:
    print("Experiment 1 - CV generation (mean of 4 dimensions, 1-5)")
    for model, reported, recomputed in experiment1():
        status = "ok" if abs(reported - recomputed) < 0.01 else "MISMATCH"
        print(f"  {model:<20} reported {reported:<6} recomputed {recomputed:<6} {status}")

    print("\nExperiment 2 - factual accuracy by prompt wording")
    for prompt, models in experiment2().items():
        cells = "  ".join(f"{m}={v}" for m, v in models.items())
        print(f"  {prompt}: {cells}")

    print("\nExperiment 3 - same prompt, three sessions")
    for dim, scores, mean in experiment3():
        print(f"  {dim:<26} {scores} mean {mean}")


if __name__ == "__main__":
    main()
