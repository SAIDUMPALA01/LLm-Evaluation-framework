"""Consistency: does the same input give the same answer twice?

Experiment 3 in the paper ran one summarisation prompt in three separate
sessions and found the meaning shifted between runs. These helpers measure
that automatically, for model outputs and for the judge's own scores.
"""

from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher
from itertools import combinations

from .dimensions import Dimension
from .models import Evaluation


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def text_similarity(a: str, b: str) -> float:
    """Token-level similarity in [0, 1] (1 = identical wording and order)."""
    return SequenceMatcher(None, _tokens(a), _tokens(b), autojunk=False).ratio()


def output_consistency(outputs: list[str]) -> dict:
    """Pairwise similarity across repeated outputs for the same prompt, mapped to a 1-5 score."""
    if len(outputs) < 2:
        raise ValueError("need at least two outputs to measure consistency")
    sims = [text_similarity(a, b) for a, b in combinations(outputs, 2)]
    mean = statistics.fmean(sims)
    score = 1 + sum(mean >= c for c in (0.4, 0.6, 0.75, 0.9))
    return {
        "runs": len(outputs),
        "mean_pairwise_similarity": round(mean, 3),
        "min_pairwise_similarity": round(min(sims), 3),
        "score": score,
    }


def score_stability(evaluations: list[Evaluation]) -> dict[str, dict[str, float]]:
    """For repeated judge runs on one case: mean, standard deviation and range per dimension."""
    if len(evaluations) < 2:
        raise ValueError("need at least two evaluations of the same case")
    out: dict[str, dict[str, float]] = {}
    dims: set[Dimension] = set().union(*(e.scores.keys() for e in evaluations))
    for dim in sorted(dims, key=lambda d: d.value):
        vals = [e.scores[dim] for e in evaluations if dim in e.scores]
        if len(vals) >= 2:
            out[dim.value] = {
                "mean": round(statistics.fmean(vals), 2),
                "stdev": round(statistics.stdev(vals), 3),
                "range": max(vals) - min(vals),
            }
    return out
