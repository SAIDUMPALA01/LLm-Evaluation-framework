import importlib.util


def _load(root):
    spec = importlib.util.spec_from_file_location("reproduce", root / "paper" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_experiment1_averages_match_published_values(root):
    for model, reported, recomputed in _load(root).experiment1():
        assert abs(reported - recomputed) < 0.01, model


def test_experiment3_confidence_calibration_failed_every_session(root):
    rows = {dim: scores for dim, scores, _ in _load(root).experiment3()}
    assert rows["Confidence_Calibration"] == [2, 2, 2]
