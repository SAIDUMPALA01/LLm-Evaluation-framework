import pytest

from sixdim_eval import Case, Dimension, Evaluation, judge_agreement, quadratic_weighted_kappa
from sixdim_eval.agreement import dimension_agreement


def test_kappa_perfect_and_undefined():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]) is None
    assert quadratic_weighted_kappa([], []) is None


def test_kappa_matches_reference_value():
    # Reference value from sklearn.metrics.cohen_kappa_score(h, j, weights="quadratic")
    h = [1, 2, 3, 4, 5, 5, 4, 3]
    j = [1, 3, 3, 4, 4, 5, 2, 3]
    assert quadratic_weighted_kappa(h, j) == pytest.approx(0.7624, abs=1e-3)


def test_tpr_tnr_on_pass_fail():
    # humans: pass, pass, fail, fail ; judge: pass, fail, fail, pass
    row = dimension_agreement(Dimension.FACTUAL_ACCURACY, [5, 4, 2, 1], [5, 3, 2, 4], threshold=4)
    assert row.tpr == 0.5
    assert row.tnr == 0.5
    assert row.exact == 0.5
    assert row.within_one == 0.75
    assert row.mae == 1.0


def test_judge_agreement_only_uses_overlap():
    cases = [
        Case(id="a", source="s", output="o", human={"completeness": 5}),
        Case(id="b", source="s", output="o", human={"completeness": 2}),
        Case(id="c", source="s", output="o"),
    ]
    evs = [
        Evaluation(case_id="a", scores={"completeness": 5}),
        Evaluation(case_id="b", scores={"completeness": 5}),
        Evaluation(case_id="c", scores={"completeness": 1}),
    ]
    (row,) = judge_agreement(cases, evs)
    assert row.n == 2
    assert row.tnr == 0.0  # the one human failure was missed by the judge
