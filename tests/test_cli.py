import json

from sixdim_eval.cli import main


def test_end_to_end_with_heuristic_judge(root, tmp_path, capsys):
    cases = root / "examples" / "labelled_sample.jsonl"
    out = tmp_path / "results.jsonl"
    assert main(["evaluate", str(cases), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6 and all("overall" in r for r in rows)

    assert main(["agreement", str(out), "--cases", str(cases), "--json"]) == 0
    agreement = json.loads(capsys.readouterr().out)
    assert {r["dimension"] for r in agreement} >= {"factual_accuracy", "completeness"}

    assert main(["report", str(out)]) == 0
    assert "Six-dimension evaluation" in capsys.readouterr().out


def test_gate_fails_below_threshold(root, tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    main(["evaluate", str(root / "examples" / "cases.jsonl"), "--out", str(out)])
    assert main(["gate", str(out), "--min", "factual_accuracy=5"]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert main(["gate", str(out), "--min-overall", "1"]) == 0


def test_missing_api_key_is_a_clean_error(root, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(
        ["evaluate", str(root / "examples" / "cases.jsonl"), "--judge", "anthropic", "--out", str(tmp_path / "x")]
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
