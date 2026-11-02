"""The six evaluation dimensions from the paper, with scoring anchors.

Reference: Dumpala, S. (2026). "AI Model Performance in CV Generation and
Consistency Testing: A Practical Six-Dimension Evaluation Framework".
Zenodo. https://doi.org/10.5281/zenodo.20102941
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    CONTEXTUAL_UTILISATION = "contextual_utilisation"
    FACTUAL_ACCURACY = "factual_accuracy"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    CONSISTENCY = "consistency"
    SPECIFICATION_ADHERENCE = "specification_adherence"


@dataclass(frozen=True)
class DimensionSpec:
    dimension: Dimension
    question: str
    fail_signals: str


SPECS: dict[Dimension, DimensionSpec] = {
    Dimension.COMPLETENESS: DimensionSpec(
        Dimension.COMPLETENESS,
        "Does the output cover everything the source and the request call for?",
        "Missing sections, empty headings, placeholder text, silently dropped facts.",
    ),
    Dimension.CONTEXTUAL_UTILISATION: DimensionSpec(
        Dimension.CONTEXTUAL_UTILISATION,
        "Does the output actually use the information in the source, rather than generic filler?",
        "Boilerplate that would fit anyone; source details ignored.",
    ),
    Dimension.FACTUAL_ACCURACY: DimensionSpec(
        Dimension.FACTUAL_ACCURACY,
        "Is every claim correct and supported by the source?",
        "Invented names, numbers, dates, contact details, projects or citations.",
    ),
    Dimension.CONFIDENCE_CALIBRATION: DimensionSpec(
        Dimension.CONFIDENCE_CALIBRATION,
        "Can a reader tell what is uncertain, missing or assumed?",
        "Unsupported content stated with full confidence; no warnings about gaps.",
    ),
    Dimension.CONSISTENCY: DimensionSpec(
        Dimension.CONSISTENCY,
        "Is the output free of internal contradictions (and stable across runs)?",
        "Contradictory statements; meaning changes between repeated runs.",
    ),
    Dimension.SPECIFICATION_ADHERENCE: DimensionSpec(
        Dimension.SPECIFICATION_ADHERENCE,
        "Does the output follow the requested format, length and structure?",
        "Wrong format, ignored constraints (e.g. 'exactly 5 bullets').",
    ),
}

SCALE = {
    5: "no problems",
    4: "minor issues",
    3: "noticeable problems but usable",
    2: "mostly failed",
    1: "complete failure",
}

ALL_DIMENSIONS: tuple[Dimension, ...] = tuple(Dimension)


def parse_dimension(name: str | Dimension) -> Dimension:
    """Accept a Dimension or 'factual_accuracy', 'Factual Accuracy', 'Factual-Accuracy', etc."""
    if isinstance(name, Dimension):
        return name
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return Dimension(key)
    except ValueError:
        valid = ", ".join(d.value for d in Dimension)
        raise ValueError(f"Unknown dimension {name!r}. Valid: {valid}") from None
