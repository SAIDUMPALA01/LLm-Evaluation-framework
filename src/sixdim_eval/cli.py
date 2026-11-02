"""Command-line interface: `sixdim evaluate | agreement | report | gate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agreement import as_dicts, judge_agreement
from .consistency import score_stability
from .dimensions import parse_dimension
from .io import read_cases, read_results, write_results
from .judges import HeuristicJudge, Judge, LLMJudge
from .parsing import JudgeOutputError
from .report import agreement_markdown, quality_gate, summary_markdown


def make_judge(kind: str, model: str | None) -> Judge:
    if kind == "heuristic":
        return HeuristicJudge()
    if kind == "anthropic":
        from .clients import AnthropicClient

        return LLMJudge(AnthropicClient(model=model))
    if kind == "openai":
        from .clients import OpenAICompatibleClient

        return LLMJudge(OpenAICompatibleClient(model=model))
    raise SystemExit(f"unknown judge {kind!r}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    cases = read_cases(args.cases)
    try:
        judge = make_judge(args.judge, args.model)
    except RuntimeError as exc:  # e.g. missing API key
        print(f"error: {exc}", file=sys.stderr)
        return 2
    results, errors, stability = [], 0, {}
    for case in cases:
        runs = []
        for _ in range(args.repeat):
            try:
                runs.append(judge.evaluate(case))
            except JudgeOutputError as exc:
                errors += 1
                print(f"warning: {exc}", file=sys.stderr)
        if runs:
            results.append(runs[0])
            if len(runs) > 1:
                stability[case.id] = score_stability(runs)
        print(f"{case.id}: {runs[0].overall:.2f}" if runs else f"{case.id}: FAILED", file=sys.stderr)
    write_results(args.out, results)
    print(f"wrote {len(results)} evaluations to {args.out} ({errors} judge errors)", file=sys.stderr)
    if stability:
        path = Path(args.out).with_suffix(".stability.json")
        path.write_text(json.dumps(stability, indent=2))
        print(f"judge stability across {args.repeat} runs written to {path}", file=sys.stderr)
    return 1 if errors and not results else 0


def cmd_agreement(args: argparse.Namespace) -> int:
    cases = read_cases(args.cases)
    rows = judge_agreement(cases, read_results(args.results), threshold=args.threshold)
    if not rows:
        print("no overlapping human labels and judge scores found", file=sys.stderr)
        return 1
    print(json.dumps(as_dicts(rows), indent=2) if args.json else agreement_markdown(rows, args.threshold))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print(summary_markdown(read_results(args.results)))
    return 0


def _parse_min(values: list[str]) -> dict:
    out = {}
    for item in values:
        name, _, num = item.partition("=")
        if not num:
            raise SystemExit(f"--min expects dimension=value, got {item!r}")
        out[parse_dimension(name)] = float(num)
    return out


def cmd_gate(args: argparse.Namespace) -> int:
    result = quality_gate(read_results(args.results), args.min_overall, _parse_min(args.min))
    if result.passed:
        print("quality gate: PASS")
        return 0
    print("quality gate: FAIL")
    for f in result.failures:
        print(f"  - {f}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sixdim", description="Six-dimension evaluation for LLM outputs.")
    p.add_argument("--version", action="version", version=f"sixdim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("evaluate", help="score every case in a JSONL file")
    e.add_argument("cases")
    e.add_argument("--judge", choices=["heuristic", "anthropic", "openai"], default="heuristic")
    e.add_argument("--model", help="model name for LLM judges")
    e.add_argument("--out", default="results.jsonl")
    e.add_argument("--repeat", type=int, default=1, help="run the judge N times per case to measure its stability")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("agreement", help="compare judge scores with human labels")
    a.add_argument("results")
    a.add_argument("--cases", required=True, help="the cases file containing human labels")
    a.add_argument("--threshold", type=int, default=4, help="score counted as a pass (default 4)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_agreement)

    r = sub.add_parser("report", help="markdown summary of results")
    r.add_argument("results")
    r.set_defaults(func=cmd_report)

    g = sub.add_parser("gate", help="exit 1 if scores are below thresholds (for CI)")
    g.add_argument("results")
    g.add_argument("--min-overall", type=float)
    g.add_argument("--min", action="append", default=[], metavar="DIM=VALUE")
    g.set_defaults(func=cmd_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
