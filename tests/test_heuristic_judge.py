from sixdim_eval import Case, Dimension, HeuristicJudge
from sixdim_eval.judges import unsupported_facts

SOURCE = "Ada Lovelace. Email ada@example.com. Built a churn model with 0.81 ROC-AUC in 2024."


def test_detects_invented_contact_details_and_numbers():
    out = (
        "Ada Lovelace | ada@example.com | +44 7700 900123 | github.com/ada-fake\n"
        "Churn model, 0.81 ROC-AUC, 2024, 40% faster."
    )
    bad = unsupported_facts(SOURCE, out)
    assert "+44 7700 900123" in bad
    assert any("github.com/ada-fake" in b for b in bad)
    assert "40%" in bad
    assert "0.81" not in bad and "2024" not in bad and "ada@example.com" not in bad


def test_grounded_output_scores_high():
    ev = HeuristicJudge().evaluate(
        Case(id="g", source=SOURCE, output="Ada Lovelace, ada@example.com. Churn model, 0.81 ROC-AUC (2024).")
    )
    assert ev.scores[Dimension.FACTUAL_ACCURACY] == 5
    assert ev.scores[Dimension.CONFIDENCE_CALIBRATION] == 5


def test_confident_fabrication_hurts_calibration_but_marked_gaps_do_not():
    fabricated = HeuristicJudge().evaluate(Case(id="f", source=SOURCE, output="Phone +44 7700 900123. Salary 55000."))
    hedged = HeuristicJudge().evaluate(Case(id="h", source=SOURCE, output="Phone: MISSING. Estimated salary 55000."))
    assert fabricated.scores[Dimension.CONFIDENCE_CALIBRATION] <= 2
    assert hedged.scores[Dimension.CONFIDENCE_CALIBRATION] == 3


def test_placeholders_hurt_completeness():
    ev = HeuristicJudge().evaluate(Case(id="p", source=SOURCE, output="[Your Name]\n[Insert email]\nLorem ipsum"))
    assert ev.scores[Dimension.COMPLETENESS] == 1


def test_bullet_count_spec():
    case = Case(id="b", source="x y z", output="- a\n- b\n- c\n- d", spec="Summarise in exactly five bullet points.")
    ev = HeuristicJudge().evaluate(case)
    assert ev.scores[Dimension.SPECIFICATION_ADHERENCE] == 3
    assert any("got 4" in f.issue for f in ev.flags)


def test_heuristic_does_not_guess_consistency():
    ev = HeuristicJudge().evaluate(Case(id="c", source=SOURCE, output="Ada Lovelace"))
    assert Dimension.CONSISTENCY not in ev.scores
