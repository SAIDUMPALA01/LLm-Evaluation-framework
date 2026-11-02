"""Judges turn a `Case` into an `Evaluation`.

* `LLMJudge` asks a language model to score all six dimensions, using the
  "MISSING instruction" from the paper so the judge admits what it cannot
  ground instead of inventing a justification.
* `HeuristicJudge` is a free, offline, deterministic baseline. It only scores
  the dimensions it can measure with simple rules and leaves the rest out
  rather than guessing.
"""

from __future__ import annotations

import re
import time
from typing import Protocol

from pydantic import ValidationError

from .clients import LLMClient
from .dimensions import ALL_DIMENSIONS, SCALE, SPECS, Dimension
from .models import Case, Evaluation, Flag
from .parsing import JudgeOutputError, extract_json_object


class Judge(Protocol):
    name: str

    def evaluate(self, case: Case) -> Evaluation: ...


# --------------------------------------------------------------------------- LLM judge


def build_system_prompt(dimensions: tuple[Dimension, ...] = ALL_DIMENSIONS) -> str:
    dims = "\n".join(f"- {d.value}: {SPECS[d].question} Failure signals: {SPECS[d].fail_signals}" for d in dimensions)
    scale = ", ".join(f"{k} = {v}" for k, v in SCALE.items())
    keys = ", ".join(f'"{d.value}": <int 1-5>' for d in dimensions)
    return f"""You are a strict, honest evaluator of AI-generated output.
You receive SOURCE (the only trusted information), optionally SPEC (what was requested),
and OUTPUT (what the model produced). Score OUTPUT on each dimension from 1 to 5 ({scale}).

Dimensions:
{dims}

Rules:
- Judge ONLY against SOURCE and SPEC. Do not use outside knowledge to "fill in" facts.
- If a claim in OUTPUT cannot be found in SOURCE, treat it as unsupported and flag it.
  Write "MISSING" in your reasoning rather than inventing a justification.
- Flags must name the dimension and the exact problem.

Reply with ONLY a JSON object, no markdown fences:
{{"scores": {{{keys}}},
 "flags": [{{"dimension": "<dimension>", "issue": "<what is wrong>", "severity": "high|medium|low"}}],
 "summary": "<2-3 sentence verdict>"}}"""


def build_user_prompt(case: Case) -> str:
    parts = [f"SOURCE:\n{case.source.strip()}"]
    if case.spec:
        parts.append(f"SPEC:\n{case.spec.strip()}")
    parts.append(f"OUTPUT:\n{case.output.strip()}")
    return "\n\n".join(parts)


class LLMJudge:
    def __init__(self, client: LLMClient, retries: int = 1, dimensions: tuple[Dimension, ...] = ALL_DIMENSIONS):
        self.client = client
        self.retries = retries
        self.dimensions = dimensions
        self.name = f"llm:{getattr(client, 'name', type(client).__name__)}"
        self._system = build_system_prompt(dimensions)

    def evaluate(self, case: Case) -> Evaluation:
        user = build_user_prompt(case)
        last: Exception | None = None
        for _ in range(self.retries + 1):
            start = time.perf_counter()
            raw = self.client.complete(self._system, user)
            latency = (time.perf_counter() - start) * 1000
            try:
                data = extract_json_object(raw)
                evaluation = Evaluation(
                    case_id=case.id,
                    scores=data.get("scores", {}),
                    flags=data.get("flags", []) or [],
                    summary=str(data.get("summary", "")),
                    judge=self.name,
                    latency_ms=round(latency, 1),
                )
                missing = [d.value for d in self.dimensions if d not in evaluation.scores]
                if missing:
                    raise JudgeOutputError(f"judge omitted dimensions: {', '.join(missing)}")
                return evaluation
            except (JudgeOutputError, ValidationError, TypeError, ValueError) as exc:
                last = exc
        raise JudgeOutputError(f"could not get a valid evaluation for case {case.id!r}: {last}")


# --------------------------------------------------------------------------- heuristic judge

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"(?:https?://|www\.)[^\s)\]>,]+|\b[\w-]+\.(?:com|io|org|net|dev|ai|app|co\.uk|uk)/[^\s)\]>,]+", re.I)
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?")
_PLACEHOLDER = re.compile(r"lorem ipsum|\[(?:your|insert|add)[^\]]*\]|\bTBD\b|\bXXX+\b|<[^>]*here>", re.I)
_MISSING_MARK = re.compile(r"\bMISSING\b|\bnot provided\b|\bunknown\b|\bnot (?:stated|available)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.M)
_EXACT_BULLETS = re.compile(r"exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bullet", re.I)
_WORDNUM = {
    w: i for i, w in enumerate(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"], start=1)
}
_STOP = set(  # noqa: SIM905 - a word list reads better as a string
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "will",
        "would",
        "into",
        "your",
        "you",
        "our",
        "their",
        "they",
        "them",
        "about",
        "which",
        "when",
        "what",
        "where",
        "while",
        "been",
        "being",
        "also",
        "than",
        "then",
        "there",
        "these",
        "those",
        "over",
        "under",
        "such",
        "only",
        "very",
        "more",
        "most",
        "some",
        "many",
        "much",
        "other",
        "each",
        "both",
    ]
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9@.+%]", "", text.lower())


def unsupported_facts(source: str, output: str) -> list[str]:
    """Emails, URLs, phone numbers and numbers in `output` that do not appear in `source`."""
    src = _norm(source)
    src_digits = re.sub(r"\D", "", source)
    found: list[str] = []
    for rx in (_EMAIL, _URL, _PHONE):
        for m in rx.findall(output):
            key = re.sub(r"\D", "", m) if rx is _PHONE else _norm(m)
            haystack = src_digits if rx is _PHONE else src
            if key and key not in haystack and m not in found:
                found.append(m.strip())
    scrubbed = _PHONE.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", output)))
    for m in _NUMBER.findall(scrubbed):
        if len(m.strip("%")) < 2:  # single digits are usually list markers or trivia
            continue
        if _norm(m) not in src and m not in found:
            found.append(m)
    return found


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()) if w not in _STOP}


def _band(value: float, cuts: tuple[float, float, float, float]) -> int:
    """Map a 0..1 value to 1..5 using four ascending cut points."""
    return 1 + sum(value >= c for c in cuts)


class HeuristicJudge:
    """Rule-based, offline baseline. Scores 5 of 6 dimensions; consistency needs repeated runs."""

    name = "heuristic:v1"

    def evaluate(self, case: Case) -> Evaluation:
        start = time.perf_counter()
        scores: dict[Dimension, int] = {}
        flags: list[Flag] = []

        bad = unsupported_facts(case.source, case.output)
        scores[Dimension.FACTUAL_ACCURACY] = max(1, 5 - len(bad))
        for fact in bad[:10]:
            flags.append(
                Flag(dimension=Dimension.FACTUAL_ACCURACY, issue=f"'{fact}' is not in the source", severity="high")
            )

        placeholders = _PLACEHOLDER.findall(case.output)
        scores[Dimension.COMPLETENESS] = max(1, 5 - 2 * len(placeholders))
        for p in placeholders[:5]:
            flags.append(Flag(dimension=Dimension.COMPLETENESS, issue=f"placeholder text: {p!r}", severity="medium"))

        src_words, out_words = _content_words(case.source), _content_words(case.output)
        coverage = len(src_words & out_words) / len(src_words) if src_words else 1.0
        scores[Dimension.CONTEXTUAL_UTILISATION] = _band(coverage, (0.15, 0.3, 0.45, 0.6))
        if coverage < 0.3:
            flags.append(
                Flag(
                    dimension=Dimension.CONTEXTUAL_UTILISATION,
                    issue=f"only {coverage:.0%} of the source's key terms appear in the output",
                    severity="medium",
                )
            )

        marks_gaps = bool(_MISSING_MARK.search(case.output))
        if bad and not marks_gaps:
            scores[Dimension.CONFIDENCE_CALIBRATION] = 1 if len(bad) >= 3 else 2
            flags.append(
                Flag(
                    dimension=Dimension.CONFIDENCE_CALIBRATION,
                    issue="unsupported details are presented confidently, with no warning to the reader",
                    severity="high",
                )
            )
        elif bad:
            scores[Dimension.CONFIDENCE_CALIBRATION] = 3
        else:
            scores[Dimension.CONFIDENCE_CALIBRATION] = 5

        if case.spec and (m := _EXACT_BULLETS.search(case.spec)):
            want = int(m.group(1)) if m.group(1).isdigit() else _WORDNUM[m.group(1).lower()]
            got = len(_BULLET.findall(case.output))
            scores[Dimension.SPECIFICATION_ADHERENCE] = 5 if got == want else max(1, 5 - 2 * abs(got - want))
            if got != want:
                flags.append(
                    Flag(
                        dimension=Dimension.SPECIFICATION_ADHERENCE,
                        issue=f"asked for exactly {want} bullet points, got {got}",
                        severity="medium",
                    )
                )

        return Evaluation(
            case_id=case.id,
            scores=scores,
            flags=flags,
            summary=f"{len(bad)} unsupported fact(s), {len(placeholders)} placeholder(s), "
            f"{coverage:.0%} source-term coverage.",
            judge=self.name,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
