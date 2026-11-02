"""Defensive parsing of JSON returned by language models.

Models sometimes wrap JSON in markdown fences or add prose around it even when
told not to. These helpers recover the object when possible and raise a clear
error when not.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JudgeOutputError(ValueError):
    """The model's reply could not be turned into a valid evaluation."""


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise JudgeOutputError("no JSON object found in model output") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeOutputError(f"invalid JSON in model output: {exc.msg}") from None
    if not isinstance(value, dict):
        raise JudgeOutputError("expected a JSON object at the top level")
    return value
