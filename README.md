# sixdim-eval

[![CI](https://github.com/SAIDUMPALA01/LLm-Evaluation-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIDUMPALA01/LLm-Evaluation-framework/actions/workflows/ci.yml)
[![Paper DOI](https://img.shields.io/badge/paper-10.5281%2Fzenodo.20102941-blue)](https://doi.org/10.5281/zenodo.20102941)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Catch LLM outputs that look finished but are missing, invented or overconfident — and measure how far you can trust the judge that scores them.**

`sixdim-eval` is the open-source implementation of my paper
[*AI Model Performance in CV Generation and Consistency Testing: A Practical Six-Dimension Evaluation Framework*](https://doi.org/10.5281/zenodo.20102941) (Zenodo, 2026).
It gives you a Python library, a CLI and a GitHub Action that:

- **score** any LLM output against its source material on six dimensions, with an LLM judge (Claude or any OpenAI-compatible model) or a free offline heuristic judge;
- **flag** invented contact details, numbers and links, placeholder text and broken format rules;
- **check the judge against humans** — agreement, quadratic-weighted kappa, and true-positive / true-negative rates;
- **gate CI** so a prompt or model change that makes outputs worse fails the build.

---

## Why

While testing four models on a simple task — "turn my degree transcripts into a CV" — I found the most dangerous failures were invisible: sections quietly left out, contact details and projects invented, and no model ever warning the reader that anything might be wrong. Accuracy alone does not capture that, so the paper proposes six dimensions:

| Dimension | Question it answers |
|---|---|
| Completeness | Does the output cover everything the source and request call for? |
| Contextual utilisation | Did it use the source, or write generic filler? |
| Factual accuracy | Is every claim supported by the source? |
| Confidence calibration | Can the reader tell what is uncertain or missing? |
| Consistency | Is it free of contradictions, and stable across runs? |
| Specification adherence | Does it follow the requested format and constraints? |

Each is scored 1 (complete failure) to 5 (no problems).

---

## Quick start

```bash
git clone https://github.com/SAIDUMPALA01/LLm-Evaluation-framework.git
cd LLm-Evaluation-framework
pip install -e ".[anthropic]"

# Free, offline baseline judge
sixdim evaluate examples/cases.jsonl --out results.jsonl
sixdim report results.jsonl

# LLM judge (Claude); use --judge openai for any OpenAI-compatible endpoint, including Ollama
export ANTHROPIC_API_KEY=...
sixdim evaluate examples/cases.jsonl --judge anthropic --out results.jsonl
```

Example report from the heuristic judge on the bundled cases:

```
| Dimension               | Mean (1-5) |
|-------------------------|------------|
| Completeness            | 4.33       |
| Contextual utilisation  | 3.50       |
| Factual accuracy        | 4.17       |
| Confidence calibration  | 3.83       |
| Specification adherence | 4.33       |

Lowest-scoring cases
- cv-generic (3.00): completeness - placeholder text: '[Your Name]'
- cv-fabricated-contacts (3.25): factual_accuracy - '+44 7700 900123' is not in the source
- summary-invented-stat (3.60): factual_accuracy - '37%' is not in the source
```

### Input format

One JSON object per line. `spec` and `human` are optional.

```json
{"id": "cv-01", "source": "Name: Alex Morgan. MSc Data Science...", "spec": "Use ONLY the information provided...", "output": "ALEX MORGAN ...", "human": {"factual_accuracy": 2, "completeness": 4}}
```

---

## Can you trust the judge?

An LLM judge is a model too, and it can be wrong. Before acting on its scores, label a sample yourself (around 100 cases is a good start), then compare:

```bash
sixdim evaluate my_labelled_cases.jsonl --judge anthropic --out judge.jsonl
sixdim agreement judge.jsonl --cases my_labelled_cases.jsonl
```

```
| Dimension        | n | Exact | Within 1 | MAE  | Kappa (quadratic) | TPR  | TNR (fails found)    |
|------------------|---|-------|----------|------|-------------------|------|----------------------|
| factual_accuracy | 6 | 0.33  | 0.83     | 0.83 | 0.73              | 1.00 | 0.67 (3 human fails) |
```

**TNR** is the one to watch: the share of outputs *humans* failed that the judge also failed. A judge with high accuracy but low TNR passes the bad outputs you were trying to catch.
(This sample comes from the heuristic judge on `examples/labelled_sample.jsonl`. The labels in that file are illustrative, written to show the format; they are not a benchmark.)

To check whether the judge agrees *with itself*, run it several times per case:

```bash
sixdim evaluate cases.jsonl --judge anthropic --repeat 3 --out results.jsonl
# -> results.stability.json: mean, standard deviation and range of each dimension across runs
```

---

## Use it in CI

Add a quality gate to any repository that generates text with an LLM:

```yaml
# .github/workflows/llm-quality.yml
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python generate_outputs.py > evals/cases.jsonl   # your app writes {id, source, output}
      - uses: SAIDUMPALA01/LLm-Evaluation-framework@main
        with:
          cases: evals/cases.jsonl
          judge: anthropic
          min-overall: "4.0"
          min: |
            factual_accuracy=4.5
            confidence_calibration=3.5
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The summary table appears on the workflow run page, and the job fails if any threshold is missed. Or use the CLI directly: `sixdim gate results.jsonl --min-overall 4 --min factual_accuracy=4.5`.

---

## Python API

```python
from sixdim_eval import Case, HeuristicJudge, LLMJudge, judge_agreement, output_consistency
from sixdim_eval.clients import AnthropicClient

case = Case(id="1", source="Name: Ada. Skills: Python, SQL.", output="Ada - Python, SQL, Kubernetes.")
evaluation = LLMJudge(AnthropicClient()).evaluate(case)
print(evaluation.overall, evaluation.flags)

# Did three runs of the same prompt say the same thing?
output_consistency([run1, run2, run3])  # {'mean_pairwise_similarity': 0.71, 'score': 3, ...}
```

Any object with `complete(system, user) -> str` can be passed to `LLMJudge`, so you can plug in another provider or a fake for tests.

---

## The two judges

| | `LLMJudge` | `HeuristicJudge` |
|---|---|---|
| Cost | API calls | Free, offline, deterministic |
| Dimensions | All six | Five (consistency needs repeated runs, so it is left out instead of guessed) |
| Catches | Paraphrased fabrications, contradictions, subtle omissions | Invented emails, phone numbers, URLs and numbers; placeholders; low source coverage; wrong bullet counts |
| Misses | Whatever the model misses — check with `sixdim agreement` | Invented words and claims, e.g. a made-up degree classification |

Both apply the paper's **MISSING instruction**: the LLM judge is told to write "MISSING" instead of inventing a justification, and the heuristic judge rewards outputs that mark gaps instead of filling them.

---

## The paper's data

Everything behind the paper is in [`paper/`](paper/): the PDF, the raw CSVs, the exact prompts and the lab notebook. Personal contact details in the prompts were redacted before publishing.

```bash
python paper/reproduce.py
```

recomputes the result tables from the CSVs and checks every published average (CI runs this on every commit):

| Model (Experiment 1) | Completeness | Contextual util. | Factual acc. | Confidence cal. | Average |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 4.8 | 5.0 | 5.0 | 4.0 | **4.70** |
| ChatGPT | 4.0 | 4.0 | 4.0 | 3.5 | 3.88 |
| DeepSeek | 3.5 | 4.0 | 2.5 | 2.0 | 3.00 |
| Blackbox AI | 2.5 | 3.0 | 3.0 | 2.5 | 2.75 |

In Experiment 2, the prompt "Fill in ALL sections completely" dropped DeepSeek's factual accuracy from 5 to 1 (it invented a phone number, email, GitHub profile and three projects), while the MISSING-instruction prompt scored 5 on factual accuracy for all four models. In Experiment 3, confidence calibration scored 2/5 in all three sessions.

**Limitations.** The paper's scores come from a single annotator on a small number of tasks, so they show patterns rather than a statistically robust ranking. The agreement tools in this library exist to address that in follow-up work.

---

## Project layout

```
src/sixdim_eval/
  dimensions.py   the six dimensions and the 1-5 scale
  judges.py       LLMJudge (with retry and strict validation) and HeuristicJudge
  agreement.py    judge-vs-human agreement: kappa, TPR, TNR, MAE
  consistency.py  output and score stability across repeated runs
  report.py       markdown summaries and the CI quality gate
  cli.py          `sixdim` command
paper/            paper PDF, raw data, prompts, notebook, reproduce.py
action.yml        GitHub Action
tests/            36 tests, no API key or network needed
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Roadmap

- [ ] Publish a labelled benchmark of about 100 CV and summary outputs, with judge-agreement results for several models
- [ ] Publish the package to PyPI
- [ ] An `inspect_ai` scorer so the dimensions can be used inside the UK AI Security Institute's Inspect framework

## Citation

```bibtex
@misc{dumpala2026sixdim,
  author = {Dumpala, Sai},
  title  = {AI Model Performance in CV Generation and Consistency Testing: A Practical Six-Dimension Evaluation Framework},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.20102941}
}
```

MIT licence. Built by **Sai Dumpala** — [LinkedIn](https://www.linkedin.com/in/sai-dumpala) · [ORCID](https://orcid.org/0009-0001-0260-0072)
