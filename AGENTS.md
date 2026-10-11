# AGENTS.md — rag-guard

> Portable agent instructions. Read by Cursor (local + cloud agents), Claude Code (via the pointer in `CLAUDE.md`), Codex, and Grok Bot cloud-agent runs.
> Keep this file under ~150 lines. Procedures live in `.cursor/skills/`, scoped rules in `.cursor/rules/*.mdc`, review rules in `.cursor/BUGBOT.md`.

## What this repo is

Guarded RAG: a pure-stdlib, zero-runtime-dependency pipeline that refuses to answer when retrieval finds no support, checks the answer against retrieved context for groundedness, and redacts PII, with a trace on every step. The eval harness scores retrieval hit-rate, refusal accuracy, and grounded-rate. Bring your own model (`complete(prompt) -> str`).

Headline recorded run (20 cases, 12-doc corpus, live `claude -p`; artifact `eval/results.json`): `n=20`, `refusal_accuracy=0.9`, `retrieval_hit_rate=1.0`, `grounded_rate=0.8824`. Two refusal misses were out-of-corpus identity questions that the groundedness guard still flagged. Offline check: `python bin/check_receipts.py`. Live re-run: `PYTHONPATH=. python3 bin/eval_real.py`.

- **Status:** active · **License:** MIT · **Package:** PyPI `guarded-rag` (import `rag_guard`)
- **Entry points:** `rag_guard/`, `tests/`, `eval/`, `bin/`, `ops/`, `docs/`
- **Part of:** the governed-agent stack (rag-guard · graph-guard · agent-gate · crucible · oracle-gate · agent-cost-attribution · bow · fleet-mode). Also a Claude Code `UserPromptSubmit` hook.

## Commands (run these; do not guess)

```bash
pip install -e ".[dev]"     # setup. CI also installs pytest-cov for the coverage gate.
# lint: n/a — this repo has no ruff or mypy config. Do not add one to satisfy a template.
# typecheck: n/a
python -m pytest -q         # tests (Python 3.11–3.13, FakeProvider, no keys, no network)
python -m pytest -q --cov=rag_guard --cov-report=term-missing --cov-fail-under=90
python bin/check_receipts.py                 # offline: README rates vs eval/results.json
PYTHONPATH=. python3 bin/eval_real.py        # live harness; needs claude on PATH; writes eval/results.json
python bin/demo.py                           # grounded answer, refusal, PII redaction, tiny eval
PYTHONPATH=. python3 bin/hook_report.py      # join hook fires to session transcripts
```

A task is not done until the tests pass and the output is pasted in the PR. The coverage floor is 90. Do not lower it. README's "currently 99% covered" is a measured point, not a knob.

## Non-negotiables

1. **No employer data.** Personal work only. Never add, paste, paraphrase, or anonymize anything from an employer system, document, ticket, chat, or dataset. No employer names in fixtures, no internal acronyms, no realistic HR records. Synthetic data only. If a prompt looks like it came from work, stop and ask. The author bio in README is the only existing biographical blurb; do not extend it.
2. **Receipts over hype.** Every number in README, docs, or PR text must be reproducible from a committed command. The three eval rates above must match `eval/results.json`. No unmeasured "blazing fast" or "production-grade".
3. **Honest benchmarks.** Report misses at the same prominence as wins. The 0.9 refusal accuracy is the number; the two misses are part of the claim. Never tune on the labeled set. Never delete an inconvenient run; mark it superseded with a reason.
4. **Fail closed.** The injection decision denies on unexpected entrypoints. Error *reporting* may fail open; *decisions* may not. Never weaken a guard to make a test pass. Never catch-and-ignore exceptions in a decision path.
5. **Tests and CI are the definition of done.** New behavior ships with a test that fails without it. Do not lower `--cov-fail-under`, skip, or xfail to get green.
6. **Model never grades its own output.** Prefer mechanical verdicts. A live judge, if used, is a different lineage from the author. Record both.
7. **Irreversible = human-gated.** No force-push, no history rewrite, no PyPI publish, no release tags, no posting. Open a PR; a human merges.

## Code style

- Python 3.11+; stdlib-first; zero runtime dependencies. A new runtime dependency needs a one-line justification in the PR. Dev deps stay in the `dev` extra.
- Type hints on public functions. Small pure functions. No global mutable state.
- Errors: raise specific exceptions; no bare `except` in a decision path.
- Secrets only via env vars. Never print secret values. Tests stay key-free via `FakeProvider`.
- Don't commit `__pycache__/`, `*.pyc`, `.coverage`, `.venv/`, or `.cursor/hook-logs/`.

## How to work in this repo

- Start non-trivial work in Plan mode; save the plan to `.cursor/plans/` and link it in the PR.
- Default to one agent. Fan out only for read-heavy, independent work. Writes stay single-threaded per file.
- Keep PRs small, one intent. Title: `type(scope): summary`.
- PR body: **What / Why / How verified (commands + output) / Numbers changed (before → after, with command) / Risks / Not done**.
- When you learn something durable, propose an edit to this file or a skill in the same PR.

## Directory map

- `rag_guard/` — pipeline, guards, TF-IDF retriever, JSON + sqlite indexes, corpus, web corroboration, rebuild health, hook log
- `tests/` — unittest-style pytest; offline; includes hook-isolation and eval-harness tests
- `eval/` — committed `results.json` from the 20-case live run
- `bin/` — `demo.py`, `eval_real.py`, `hook_userpromptsubmit.py`, `hook_report.py`, `check_receipts.py`
- `ops/` — launchd plist for a scheduled sqlite reindex
- `docs/superpowers/` — guarded-grounding design spec and plan
- `.github/` — CI (3.11–3.13 + coverage comment + receipt and employer-data jobs), CodeQL, Dependabot, tag-triggered publish

## Known traps

- **The shell guard blocks `git pull --rebase`.** `.cursor/hooks/guard-shell.sh` denies any command whose text contains `rebase`, plus force-push, `reset --hard`, `git tag`, twine/uv publish, and `gh release`. It asks before `git push` and before curl/wget. Pipe-to-shell is denied. The hook fails closed. Fetch and merge, or ask a human. Do not loosen the hook to unblock yourself.
- **Two index backends.** `sqlite_index.py` is the default (~0.35s cold hook). `index.py` stores JSON forward vectors (~3.9s). Rankings match. `RAG_GUARD_BACKEND=json` rolls back.
- **Serve-stale is on.** A stale index is served immediately and a detached `python -m rag_guard.reindex --backends sqlite` refreshes it. Lock: `<cache>.rebuild.lock` (one at a time, 300s steal). `RAG_GUARD_SERVE_STALE=0` restores blocking. The child records ok/fail to `<cache>.rebuild.json` because its stderr is DEVNULL. The hook appends one warning after 2 consecutive failures or 6h without a successful rebuild.
- **Reindex must never unlink the cache first.** That opened a window where a hook found no index to serve.
- **Bump `index.CORPUS_SCHEMA`** when chunk id or chunk-text shape changes. It is in the fingerprint; a stale cache would keep serving the old format. Chunk ids are qualified by corpus root, relative to the root's parent (`corpus._add_file`).
- **Hook isolation fails closed.** `bin/hook_userpromptsubmit.py` grounds only interactive entrypoints (`cli`, `vscode`, `jetbrains`). `claude -p` reports `sdk-cli` and gets silence. Opt back in only with `RAG_GUARD_ALLOW_HEADLESS=1`. Error handling stays fail-open. Do not weaken this.
- **Hook log.** `hooklog.py` records fires and silences to `~/.local/state/rag-guard/hook-fires.jsonl` (mode 0600), opt-in via `RAG_GUARD_HOOK_LOG=1`. Change `MIN_SCORE` / `HOOK_MIN_OVERLAP` from `bin/hook_report.py` data, not intuition.
- **Live eval is not the CI gate.** `bin/eval_real.py` overwrites `eval/results.json`. CI runs `bin/check_receipts.py` only. Do not commit a new results file unless the README numbers change in the same PR.
- **No Grok client in this repo.** `.cursor/rules/60-grok-integration.mdc` is intentionally omitted. Keep the `complete(prompt) -> str` seam.
