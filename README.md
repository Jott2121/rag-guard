# rag-guard

[![ci](https://github.com/Jott2121/rag-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Jott2121/rag-guard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

**Guarded RAG — answers that are grounded in retrieved context, refuse when there's no support, and ship with an eval harness that puts a number on it.**

The failure mode of RAG isn't bad retrieval — it's the confident answer with *nothing behind it*. `rag-guard` is a small, runnable pipeline that makes that hard: it refuses when retrieval finds no support, checks the answer against the context, redacts PII from the output, and traces every step. Pure-stdlib core, **zero runtime dependencies**, bring your own model. Receipts over hype.

```text
"how long is shipping?"  → grounded answer, sources=[ship]      ✓
"quantum chromodynamics?" → refuses (no support), model not called ✓
```

## The three guards

1. **Refuse-when-unsupported.** If the top retrieval score is below threshold, the pipeline refuses and **never even calls the model** — no support, no answer.
2. **Groundedness check.** After the model answers, verify the answer is actually backed by the retrieved context; flag it if not. (Lexical-overlap proxy here, swappable for an NLI/LLM judge behind the same interface.)
3. **PII output filter.** Emails, phones, SSNs, and card-like numbers are redacted from whatever the model returns.

Every result carries a **trace** (what was retrieved + scores, refused?, grounded?) so the system is auditable.

## Quickstart

```python
from rag_guard.retriever import Retriever
from rag_guard.pipeline import RagGuard
from rag_guard.providers import FakeProvider   # swap for a real model provider

ret = Retriever([
    {"id": "ship",    "text": "Standard shipping takes 3 to 5 business days."},
    {"id": "returns", "text": "Return any item within 30 days for a full refund."},
])
rag = RagGuard(ret, FakeProvider("Shipping takes 3 to 5 business days."))

print(rag.answer("how long does shipping take"))
# {'answer': 'Shipping takes 3 to 5 business days.', 'refused': False,
#  'grounded': True, 'support': 1.0, 'sources': ['ship'], 'trace': {...}}

print(rag.answer("quantum chromodynamics")["refused"])   # True — refuses, no support
```

## Measure it (the eval harness)

```python
from rag_guard.evaluate import evaluate
cases = [
    {"query": "how long does shipping take", "gold": "ship", "expect_refusal": False},
    {"query": "quantum chromodynamics",                         "expect_refusal": True},
]
print(evaluate(rag, cases))
# {'n': 2, 'refusal_accuracy': 1.0, 'retrieval_hit_rate': 1.0, 'grounded_rate': 1.0, 'cases': [...]}
```

Re-run the eval on any model or config change to catch regressions **before a user does**.

**A real run, not a demo fixture.** The two cases above are an illustration — they score 1.0 across the board, so don't read anything into them. `bin/eval_real.py` runs a 20-case labeled set over a 12-doc corpus through a live model (`claude -p`):

```bash
python bin/eval_real.py
# {'n': 20, 'refusal_accuracy': 0.9, 'retrieval_hit_rate': 1.0, 'grounded_rate': 0.8824}
```

The two refusal misses were out-of-corpus identity questions ("who's the CEO?") that scored just over threshold — but the groundedness guard still flagged both, so nothing unsupported got through unflagged. Full output lands in `eval/results.json`.

## Bring your own model

The model sits behind a one-method seam — `complete(prompt) -> str`. `FakeProvider` keeps tests/CI deterministic and key-free; a real provider (Anthropic Messages API, or a `claude -p` shell-out) drops in without touching the pipeline or guards. Retrieval is the same: the stdlib TF-IDF `Retriever` is a stand-in for real embeddings / a vector DB behind `retrieve()`.

## Run / test

```bash
pip install -e ".[dev]" && python -m pytest -q     # 19 tests
python bin/demo.py                                  # see grounded answer, refusal, PII redaction, eval
```

## About

Built by **Jeff Otterson** ([Jott2121](https://github.com/Jott2121)). Companion to [**agent-gate**](https://github.com/Jott2121/agent-gate) (an MCP gate for agent work), [**bow**](https://github.com/Jott2121/bow), [**fleet-mode**](https://github.com/Jott2121/fleet-mode), and [**agent-cost-attribution**](https://github.com/Jott2121/agent-cost-attribution) — the same discipline: ground it, guard it, measure it, and keep the receipts. MIT.
