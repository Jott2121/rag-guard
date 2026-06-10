# Contributing to rag-guard

Thanks for taking a look. Small, focused PRs are welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

## Ground rules

- **Zero runtime dependencies.** The core stays pure stdlib. Dev/test dependencies go in the `dev` extra.
- **Tests required.** Every behavior change needs a test. The suite must pass on Python 3.11-3.13 (CI enforces this).
- **Keep the seams.** Providers implement `complete(prompt) -> str`; retrievers implement `retrieve()`. New integrations go behind those interfaces, not into the pipeline.
- **No keys in tests.** Tests and CI run deterministic and key-free via `FakeProvider`.

## Workflow

1. Open an issue first for anything beyond a small fix.
2. Branch, change, add tests, run `python -m pytest -q`.
3. Open a PR with a short description of what changed and why.

## License

By contributing, you agree your contributions are licensed under MIT.
