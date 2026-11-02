"""Summaries and quality gates over a set of evaluations."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .agreement import DimensionAgreement
from .dimensions import ALL_DIMENSIONS, Dimension
from .models import Evaluation


def dimension_means(evaluations: list[Evaluation]) -> dict[str, float]:
    out = {}
    for dim in ALL_DIMENSIONS:
        vals = [e.scores[dim] for e in evaluations if dim in e.scores]
        if vals:
            out[dim.value] = round(statistics.fmean(vals), 2)
    return out


def summary_markdown(evaluations: list[Evaluation]) -> str:
    if not evaluations:
        return "No evaluations.\n"
    means = dimension_means(evaluations)
    overall = round(statistics.fmean(e.overall for e in evaluations), 2)
    lines = [
        f"### Six-dimension evaluation ({len(evaluations)} cases, judge `{evaluations[0].judge}`)",
        "",
        "| Dimension | Mean (1-5) |",
        "|---|---|",
        *[f"| {d.replace('_', ' ').capitalize()} | {v:.2f} |" for d, v in means.items()],
        f"| **Overall** | **{overall:.2f}** |",
        "",
    ]
    worst = sorted(evaluations, key=lambda e: e.overall)[:5]
    flagged = [e for e in worst if e.flags]
    if flagged:
        lines += ["**Lowest-scoring cases**", ""]
        for e in flagged:
            top = e.flags[0]
            lines.append(f"- `{e.case_id}` ({e.overall:.2f}): {top.dimension.value} - {top.issue}")
        lines.append("")
    return "\n".join(lines)


def agreement_markdown(rows: list[DimensionAgreement], threshold: int) -> str:
    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    lines = [
        "| Dimension | n | Exact | Within 1 | MAE | Kappa (quadratic) | TPR | TNR (fails found) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.dimension} | {r.n} | {fmt(r.exact)} | {fmt(r.within_one)} | {fmt(r.mae)} | "
            f"{fmt(r.kappa_quadratic)} | {fmt(r.tpr)} | {fmt(r.tnr)} ({r.human_fail_count} human fails) |"
        )
    lines.append(f"\nPass = score >= {threshold}. TPR/TNR are n/a when humans gave no passes/fails.")
    return "\n".join(lines)


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def quality_gate(
    evaluations: list[Evaluation],
    min_overall: float | None = None,
    min_dimension: dict[Dimension, float] | None = None,
) -> GateResult:
    failures = []
    if not evaluations:
        return GateResult(False, ["no evaluations to check"])
    overall = statistics.fmean(e.overall for e in evaluations)
    if min_overall is not None and overall < min_overall:
        failures.append(f"mean overall {overall:.2f} < {min_overall}")
    means = dimension_means(evaluations)
    for dim, threshold in (min_dimension or {}).items():
        got = means.get(dim.value)
        if got is None:
            failures.append(f"{dim.value}: no scores")
        elif got < threshold:
            failures.append(f"{dim.value}: mean {got:.2f} < {threshold}")
    return GateResult(not failures, failures)
