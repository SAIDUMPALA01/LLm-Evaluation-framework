import pytest

from sixdim_eval import Evaluation, output_consistency, score_stability, text_similarity


def test_similarity_bounds():
    assert text_similarity("the cat sat", "the cat sat") == 1.0
    assert text_similarity("alpha beta", "gamma delta") == 0.0


def test_identical_outputs_score_five():
    assert output_consistency(["a b c"] * 3)["score"] == 5


def test_divergent_outputs_score_low():
    report = output_consistency(["AI boosts productivity", "jobs are at risk from automation", "experts disagree"])
    assert report["score"] == 1


def test_needs_two_runs():
    with pytest.raises(ValueError):
        output_consistency(["only one"])


def test_score_stability():
    runs = [Evaluation(case_id="x", scores={"completeness": s}) for s in (4, 5, 3)]
    stats = score_stability(runs)["completeness"]
    assert stats["mean"] == 4.0 and stats["range"] == 2
