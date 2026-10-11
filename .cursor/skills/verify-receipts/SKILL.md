---
name: verify-receipts
description: Use before opening or updating a PR, or when README/docs numbers change. Re-runs rag-guard tests and the eval receipt check, compares hit-rate, refusal accuracy, and grounded-rate to the README, and writes a receipts block for the PR body.
---

# Verify receipts (rag-guard)

Published real-eval line in README (artifact `eval/results.json`, command `PYTHONPATH=. python3 bin/eval_real.py`):

- `n` = 20
- `refusal_accuracy` = 0.9
- `retrieval_hit_rate` = 1.0
- `grounded_rate` = 0.8824

The two-case quickstart block that scores 1.0 is an illustration. `tests/test_eval.py` covers it. Do not treat those 1.0s as the published result.

## Steps

1. Read AGENTS.md "Commands". From the repo root:
   - `pip install -e ".[dev]"` and, for the coverage gate, `pip install pytest-cov`
   - Tests: `python -m pytest -q --cov=rag_guard --cov-report=term-missing --cov-fail-under=90`
   - Offline receipt check: `python bin/check_receipts.py`
   There is no ruff or mypy config. Do not invent a lint command. Capture exit codes and the last ~20 lines of each run.
2. `python bin/check_receipts.py` recomputes retrieval hit-rate, refusal accuracy, and grounded-rate from `eval/results.json` joined to the labeled cases in `bin/eval_real.py` (same formulas as `rag_guard.evaluate.evaluate`). It fails if those rates, the stored JSON summary, or the README line disagree. It does not call a model and does not write `eval/results.json`.
3. Live harness, only when `claude` is on PATH: `PYTHONPATH=. python3 bin/eval_real.py`. That command overwrites `eval/results.json`. Compare the printed `n`, `refusal_accuracy`, `retrieval_hit_rate`, and `grounded_rate` to the README line. If they differ, restore the committed `eval/results.json` (`git checkout -- eval/results.json`) and mark the live re-run UNVERIFIED with the delta. Do not commit the new file unless this PR is intentionally updating the published numbers, and then update the README in the same commit. If `claude` is absent, say so. CI does not run the live harness.
4. `git diff origin/main -- README.md docs/ eval/results.json` — list every number that changed. For each, name the command that produces it. If you cannot reproduce it, mark it UNVERIFIED and stop; do not open the PR as ready.
5. Run `python bin/check_no_employer_data.py` (the `no-employer-data` CI job). Employer terms are matched by SHA-256 against `.github/blocked-terms.sha256` across every tracked file; export-control markers are whole words and skip the README `## About` bio. Any hit = stop and report. A scanner error, or a missing or empty hash file, is a failure.
6. Confirm no thresholds were lowered (`--cov-fail-under` stays 90; `MIN_SCORE` / `HOOK_MIN_OVERLAP` unchanged) and no tests were skipped or removed.
7. Output a Markdown block:
   - Commands run + pass/fail
   - Numbers: before → after → command → reproduced? (Y/N)
   - Not verified / known gaps (live `claude -p` re-run, if skipped)
8. Paste that block into the PR's "How verified" section. Never claim a check passed that you did not run.
