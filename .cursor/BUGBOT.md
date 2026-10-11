# .cursor/BUGBOT.md — review rules for rag-guard
# (Bugbot does NOT read .cursor/rules/*.mdc; it reads this file.)

Flag as **blocking**:

1. Any guard, gate, validator, or refusal path that now fails OPEN (returns allow/True/answer on exception, timeout, empty, or malformed input). The UserPromptSubmit injection decision must stay fail-closed on non-interactive entrypoints (`sdk-cli` / `claude -p` stay silent unless `RAG_GUARD_ALLOW_HEADLESS=1`).
2. Lowered thresholds: `--cov-fail-under` (floor is 90), `MIN_SCORE`, `HOOK_MIN_OVERLAP`, or other gate constants, or retry/timeout bounds removed.
3. Tests that were deleted, skipped, xfailed, or had assertions weakened without a stated reason in the PR body.
4. README or docs numbers that changed without a reproducing command or artifact. The published eval line (`n`, `refusal_accuracy`, `retrieval_hit_rate`, `grounded_rate`) must match `eval/results.json`. `python bin/check_receipts.py` is the offline check.
5. Any content that looks like employer or corporate data: company names, internal acronyms, CUI/ITAR markings, real-looking employee records, internal URLs. This repo is personal work. The existing author bio in README is not a place to add more of that.
6. `rag_guard.reindex` (or a caller) that unlinks the index cache before the new index is ready. Bumping chunk id or chunk-text shape without bumping `index.CORPUS_SCHEMA`.
7. Secrets, tokens, `.env` files, or absolute personal paths committed.
8. Unpinned GitHub Actions (must be full SHA) or widened workflow `permissions:`.
9. Live network or model calls added to the default test path. `bin/eval_real.py` is opt-in and must stay out of the pytest suite.

Flag as **non-blocking**:

- New runtime dependency without justification. The core is zero-dependency stdlib.
- Non-deterministic code in eval paths (unseeded RNG, unordered iteration in hashes or output).
- Exceptions swallowed in non-decision code.

Do not comment on naming bikesheds. This repo has no ruff config; do not demand a formatter that is not set up.
