"""How far can you trust a judge? Compare its scores with human labels.

An LLM-as-judge is itself a model that can be wrong. Before using its scores
to make decisions, measure agreement with human labels on a sample:

* exact / within-one agreement and mean absolute error on the 1-5 scale
* quadratic-weighted Cohen's kappa (chance-corrected agreement for ordinal scores)
* pass/fail view: true-positive rate (judge passes what humans pass) and
  true-negative rate (judge fails what humans fail). A judge with high accuracy
  but low TNR is missing the failures you care about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .dimensions import ALL_DIMENSIONS, Dimension
from .models import Case, Evaluation


@dataclass
class DimensionAgreement:
    dimension: str
    n: int
    exact: float
    within_one: float
    mae: float
    kappa_quadratic: float | None
    tpr: float | None
    tnr: float | None
    human_fail_count: int


def _div(a: float, b: float) -> float | None:
    return round(a / b, 3) if b else None


def quadratic_weighted_kappa(human: list[int], judge: list[int], k: int = 5) -> float | None:
    """Cohen's kappa with quadratic weights for scores in 1..k. Returns None when undefined."""
    n = len(human)
    if n == 0 or n != len(judge):
        return None
    observed = [[0] * k for _ in range(k)]
    for h, j in zip(human, judge, strict=True):
        observed[h - 1][j - 1] += 1
    hist_h = [sum(row) for row in observed]
    hist_j = [sum(observed[i][c] for i in range(k)) for c in range(k)]
    num = den = 0.0
    for i in range(k):
        for c in range(k):
            w = (i - c) ** 2 / (k - 1) ** 2
            num += w * observed[i][c]
            den += w * hist_h[i] * hist_j[c] / n
    if den == 0:
        return None
    return round(1 - num / den, 3)


def dimension_agreement(
    dimension: Dimension, human: list[int], judge: list[int], threshold: int = 4
) -> DimensionAgreement:
    n = len(human)
    diffs = [abs(h - j) for h, j in zip(human, judge, strict=True)]
    h_pass = [h >= threshold for h in human]
    j_pass = [j >= threshold for j in judge]
    tp = sum(h and j for h, j in zip(h_pass, j_pass, strict=True))
    tn = sum((not h) and (not j) for h, j in zip(h_pass, j_pass, strict=True))
    pos, neg = sum(h_pass), n - sum(h_pass)
    return DimensionAgreement(
        dimension=dimension.value,
        n=n,
        exact=round(sum(d == 0 for d in diffs) / n, 3) if n else 0.0,
        within_one=round(sum(d <= 1 for d in diffs) / n, 3) if n else 0.0,
        mae=round(sum(diffs) / n, 3) if n else 0.0,
        kappa_quadratic=quadratic_weighted_kappa(human, judge),
        tpr=_div(tp, pos),
        tnr=_div(tn, neg),
        human_fail_count=neg,
    )


def judge_agreement(cases: list[Case], evaluations: list[Evaluation], threshold: int = 4) -> list[DimensionAgreement]:
    """Agreement per dimension over every case that has both a human and a judge score."""
    by_id = {e.case_id: e for e in evaluations}
    results = []
    for dim in ALL_DIMENSIONS:
        human, judge = [], []
        for case in cases:
            ev = by_id.get(case.id)
            if case.human and ev and dim in case.human and dim in ev.scores:
                human.append(case.human[dim])
                judge.append(ev.scores[dim])
        if human:
            results.append(dimension_agreement(dim, human, judge, threshold))
    return results


def as_dicts(rows: list[DimensionAgreement]) -> list[dict]:
    return [asdict(r) for r in rows]
