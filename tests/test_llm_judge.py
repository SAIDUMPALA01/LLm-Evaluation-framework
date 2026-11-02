import json

import pytest

from sixdim_eval import Case, Dimension, LLMJudge
from sixdim_eval.parsing import JudgeOutputError

from .conftest import ALL_FIVES, FakeClient

CASE = Case(id="c1", source="Name: Ada. Skill: Python.", output="Ada - Python developer.", spec="One line.")


def test_scores_all_dimensions_and_sends_source_spec_output():
    client = FakeClient(ALL_FIVES)
    ev = LLMJudge(client).evaluate(CASE)
    assert len(ev.scores) == 6
    assert ev.scores[Dimension.CONFIDENCE_CALIBRATION] == 4
    system, user = client.calls[0]
    assert "MISSING" in system
    assert user.index("SOURCE:") < user.index("SPEC:") < user.index("OUTPUT:")


def test_retries_once_after_garbage():
    client = FakeClient("sorry, I cannot do JSON", ALL_FIVES)
    ev = LLMJudge(client).evaluate(CASE)
    assert ev.overall == pytest.approx(29 / 6, abs=0.01)
    assert len(client.calls) == 2


def test_missing_dimension_counts_as_failure():
    partial = json.dumps({"scores": {"completeness": 5}, "flags": [], "summary": ""})
    with pytest.raises(JudgeOutputError, match="omitted dimensions"):
        LLMJudge(FakeClient(partial, partial)).evaluate(CASE)


def test_out_of_range_score_is_rejected():
    bad = json.loads(ALL_FIVES)
    bad["scores"]["completeness"] = 9
    with pytest.raises(JudgeOutputError):
        LLMJudge(FakeClient(json.dumps(bad), json.dumps(bad))).evaluate(CASE)
