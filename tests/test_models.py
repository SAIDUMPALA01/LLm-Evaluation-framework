import pytest
from pydantic import ValidationError

from sixdim_eval import Case, Dimension, Evaluation


def test_case_normalises_human_labels():
    case = Case(id="x", source="s", output="o", human={"Factual Accuracy": 4, "completeness": None})
    assert case.human == {Dimension.FACTUAL_ACCURACY: 4}


def test_case_rejects_out_of_range_label():
    with pytest.raises(ValidationError):
        Case(id="x", source="s", output="o", human={"completeness": 6})


def test_case_rejects_blank_output():
    with pytest.raises(ValidationError):
        Case(id="x", source="s", output="   ")


def test_overall_is_computed_locally():
    ev = Evaluation(case_id="x", scores={"completeness": 5, "factual_accuracy": 2})
    assert ev.overall == 3.5
    assert ev.to_record()["overall"] == 3.5


def test_unknown_dimension_rejected():
    with pytest.raises(ValidationError):
        Evaluation(case_id="x", scores={"vibes": 5})
