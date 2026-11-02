"""Read cases and write results as JSON Lines (one JSON object per line)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import Case, Evaluation


def read_cases(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            cases.append(Case.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid case: {exc}") from exc
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate case ids: {', '.join(sorted(dupes))}")
    return cases


def write_results(path: str | Path, evaluations: Iterable[Evaluation]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for ev in evaluations:
            fh.write(json.dumps(ev.to_record(), ensure_ascii=False) + "\n")


def read_results(path: str | Path) -> list[Evaluation]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            data.pop("overall", None)
            out.append(Evaluation.model_validate(data))
    return out
