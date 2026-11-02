import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    """Returns queued replies in order, recording every prompt it was sent."""

    name = "fake"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


ALL_FIVES = json.dumps(
    {
        "scores": {
            "completeness": 5,
            "contextual_utilisation": 5,
            "factual_accuracy": 5,
            "confidence_calibration": 4,
            "consistency": 5,
            "specification_adherence": 5,
        },
        "flags": [],
        "summary": "Grounded and complete.",
    }
)


@pytest.fixture
def root() -> Path:
    return ROOT
