# rag-guard

Guarded RAG: a pure-stdlib, zero-runtime-dependency pipeline that refuses to answer when retrieval finds no support, checks answers against retrieved context for groundedness, and redacts PII from output — with a trace on every step and an eval harness that scores it.

- **Status:** active, public repo with CI/CodeQL/coverage badges, published to PyPI as `guarded-rag`
- **Entry points:** `rag_guard/` (import name; PyPI package name is `guarded-rag`), `eval/`, `ops/`, `bin/`, `docs/`
- **Run/test:** `pip install guarded-rag`; see README Quickstart for pipeline usage
- **Two index backends:** `sqlite_index.py` (inverted, DEFAULT — ~0.35s cold hook) and `index.py` (JSON forward vectors, ~3.9s). Identical rankings; `RAG_GUARD_BACKEND=json` rolls back. Rebuild is global and costs ~13-15s — `python -m rag_guard.reindex` on a schedule keeps it off the interactive path (`ops/*.plist` exists but is NOT loaded).
- **HOOK ISOLATION (do not weaken):** `bin/hook_userpromptsubmit.py` grounds ONLY on interactive entrypoints (`cli`/`vscode`/`jetbrains`); `claude -p` reports `sdk-cli` and gets silence. This once contaminated a published experiment (535 verdicts re-judged). The injection decision fails CLOSED; error handling stays fail-open. Opt back in only via `RAG_GUARD_ALLOW_HEADLESS=1`.
- **Constraints:** bring-your-own-model, no runtime deps; also powers a UserPromptSubmit hook wired into `~/.claude/settings.json`; one layer of the "cost-governance stack" with bow/fleet-mode/agent-gate/agent-cost-attribution/oracle-gate
