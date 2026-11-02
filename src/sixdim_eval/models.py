"""Typed data structures shared across the library."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .dimensions import ALL_DIMENSIONS, Dimension, parse_dimension

Score = int  # 1..5


class Case(BaseModel):
    """One thing to evaluate: the source material and the model output built from it."""

    id: str
    source: str
    output: str
    spec: str | None = Field(None, description="The instruction/specification the output had to follow.")
    human: dict[Dimension, Score] | None = Field(
        None, description="Optional human scores (1-5) used to check judge agreement."
    )

    @field_validator("source", "output")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("human", mode="before")
    @classmethod
    def _normalise_human(cls, v: object) -> object:
        if isinstance(v, dict):
            out: dict[Dimension, int] = {}
            for k, score in v.items():
                if score is None:
                    continue
                score = int(score)
                if not 1 <= score <= 5:
                    raise ValueError(f"human score for {k} must be 1-5, got {score}")
                out[parse_dimension(k)] = score
            return out
        return v


class Flag(BaseModel):
    dimension: Dimension
    issue: str
    severity: Literal["high", "medium", "low"] = "medium"

    @field_validator("dimension", mode="before")
    @classmethod
    def _dim(cls, v: object) -> object:
        return parse_dimension(v)


class Evaluation(BaseModel):
    """The judge's verdict on one case. `overall` is always computed locally, never trusted from the model."""

    case_id: str
    scores: dict[Dimension, Score]
    flags: list[Flag] = []
    summary: str = ""
    judge: str = ""
    latency_ms: float | None = None

    @field_validator("scores", mode="before")
    @classmethod
    def _normalise_scores(cls, v: object) -> object:
        if isinstance(v, dict):
            out = {}
            for k, s in v.items():
                s = int(round(float(s)))
                if not 1 <= s <= 5:
                    raise ValueError(f"score for {k} must be 1-5, got {s}")
                out[parse_dimension(k)] = s
            return out
        return v

    @model_validator(mode="after")
    def _at_least_one(self) -> Evaluation:
        if not self.scores:
            raise ValueError("evaluation has no scores")
        return self

    @property
    def overall(self) -> float:
        return round(sum(self.scores.values()) / len(self.scores), 2)

    def to_record(self) -> dict:
        data = self.model_dump(mode="json")
        data["overall"] = self.overall
        return data


def missing_dimensions(scores: dict[Dimension, int]) -> list[Dimension]:
    return [d for d in ALL_DIMENSIONS if d not in scores]
