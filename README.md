# rag-guard

[![ci](https://github.com/Jott2121/rag-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Jott2121/rag-guard/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Jott2121/rag-guard/actions/workflows/codeql.yml/badge.svg)](https://github.com/Jott2121/rag-guard/actions/workflows/codeql.yml)
[![Coverage](https://raw.githubusercontent.com/Jott2121/rag-guard/python-coverage-comment-action-data/badge.svg)](https://github.com/Jott2121/rag-guard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Jott2121/rag-guard/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

**Guarded RAG: answers grounded in retrieved context, refusal when there's no support, and an eval harness that puts a number on it.**

The failure mode of RAG isn't bad retrieval. It's the confident answer with *nothing behind it*. `rag-guard` is a small, runnable pipeline that makes that hard: it refuses when retrieval finds no support, checks the answer against the context, redacts PII from the output, and traces every step. Pure-stdlib core, **zero runtime dependencies**, bring your own model.

> 🧩 One layer of a five-repo [**cost-governance stack**](https://github.com/Jott2121/bow#the-system-a-cost-governance-stack) for operating AI agents cost-efficiently; [bow](https://github.com/Jott2121/bow) is the flagship that operates the stack in production.

```text
"how long is shipping?"  → grounded answer, sources=[ship]      ✓
"quantum chromodynamics?" → refuses (no support), model not called ✓
```

![rag-guard demo: grounded answer, refusal, PII redaction, eval](assets/demo.gif)

## The three guards

1. **Refuse-when-unsupported.** If the top retrieval score is below threshold, the pipeline refuses and **never even calls the model**. No support, no answer.
2. **Groundedness check.** After the model answers, verify the answer is actually backed by the retrieved context; flag it if not. (Lexical-overlap proxy here, swappable for an NLI/LLM judge behind the same interface.)
3. **PII output filter.** Emails, phones, SSNs, and card-like numbers are redacted from whatever the model returns.

Every result carries a **trace** (what was retrieved + scores, refused?, grounded?) so the system is auditable.

## Install

```bash
pip install guarded-rag
```

Zero runtime dependencies — it's stdlib all the way down. (PyPI name is `guarded-rag` — the import is still `rag_guard`.)

## Quickstart

```python
from rag_guard import Retriever, RagGuard
from rag_guard import FakeProvider   # swap for a real model provider

ret = Retriever([
    {"id": "ship",    "text": "Standard shipping takes 3 to 5 business days."},
    {"id": "returns", "text": "Return any item within 30 days for a full refund."},
])
rag = RagGuard(ret, FakeProvider("Shipping takes 3 to 5 business days."))

print(rag.answer("how long does shipping take"))
# {'answer': 'Shipping takes 3 to 5 business days.', 'refused': False,
#  'grounded': True, 'support': 1.0, 'sources': ['ship', 'returns'], 'trace': {...}}

print(rag.answer("quantum chromodynamics")["refused"])   # True: refuses, no support
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

**A real run, not a demo fixture.** The two cases above are an illustration. They score 1.0 across the board, so don't read anything into them. `bin/eval_real.py` runs a 20-case labeled set over a 12-doc corpus through a live model (`claude -p`):

```bash
PYTHONPATH=. python3 bin/eval_real.py   # requires claude CLI on PATH
# {'n': 20, 'refusal_accuracy': 0.9, 'retrieval_hit_rate': 1.0, 'grounded_rate': 0.8824}
```

The two refusal misses were out-of-corpus identity questions ("who's the CEO?") that scored just over threshold, but the groundedness guard still flagged both, so nothing unsupported got through unflagged. Full output lands in `eval/results.json`.

## Bring your own model

The model sits behind a one-method seam: `complete(prompt) -> str`. `FakeProvider` keeps tests/CI deterministic and key-free; a real provider drops in without touching the pipeline or guards. Retrieval is the same: the stdlib TF-IDF `Retriever` is a stand-in for real embeddings / a vector DB behind `retrieve()`.

### Real provider

Any object with `complete(prompt) -> str` works. Here's an Anthropic provider in stdlib only, no SDK required:

```python
import json, os, urllib.request

class AnthropicProvider:
    def __init__(self, model="claude-sonnet-4-5", max_tokens=512):
        self.model, self.max_tokens = model, max_tokens

    def complete(self, prompt: str) -> str:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)["content"][0]["text"]

rag = RagGuard(ret, AnthropicProvider())
```

## Run / test

```bash
git clone https://github.com/Jott2121/rag-guard && cd rag-guard
pip install -e ".[dev]" && python -m pytest -q     # tests pass on Python 3.11-3.13
python bin/demo.py                                  # see grounded answer, refusal, PII redaction, eval
```

CI (badge above) runs the same suite across Python 3.11, 3.12, and 3.13 on every push.

## The guarded-grounding layer (a corpus → web truth ladder)

The core retriever + guards are surface-agnostic, so the repo also ships the pieces to run
them over a **real knowledge base** with a **three-tier truth ladder**:

1. **Local notes** — `corpus.py` turns a folder of markdown into `{id, text, source, weight}` chunks; `index.py`
   builds a persisted, **fingerprinted** TF-IDF index that rebuilds only when the corpus changes
   (a note is searchable the moment it's saved); `service.py` keeps a warm singleton and `cli.py`
   is a `stdin → JSON` query entry.
2. **The web** — when local notes can't ground the answer, `webverify.py` escalates to web search
   (injected, so the core stays network-free and unit-testable).
3. **Corroboration** — a claim is only `WEB-VERIFIED` when **≥2 independent, authority-weighted**
   sources agree (official > established > social; social-only can't clear the bar).

Answers from the guarded pipeline carry a **confidence stamp** and never come back empty:

```text
✔ GROUNDED         — backed by your notes
✔ WEB-VERIFIED     — ≥2 independent sources agree (cited)
⚠ SINGLE SOURCE    — found on the web, one source only (cited)
⚠ SOURCES CONFLICT — sources disagree; both shown
⚠ UNVERIFIED       — couldn't back it anywhere (best-effort, flagged)
```

**How I actually run it:** a Claude Code `UserPromptSubmit` hook (`bin/hook_userpromptsubmit.py`)
grounds every terminal session against my own notes + wikis — advisory (a hook can't force a
refusal), silent when nothing relevant matches. **That part is live.** A companion assistant-side
wrap (it lives in my chief-of-staff repo, not this one) turns the same core into real
refuse-*teeth* — the returned string *is* the delivered message, with a corrective retry loop and
labeled fallback; it's built and adversarially reviewed but **not yet activated**.
Honest v1 limits: retrieval is lexical TF-IDF (swap embeddings behind the same `retrieve()` seam);
content-level **syndication detection is deferred** (independence is by-publisher for now); the
`contradicts_local` flag is defined but not yet surfaced.

## Where this sits in the landscape (prior art)

None of the *ideas* here are novel — and a guardrail library shouldn't pretend otherwise. What's
uncommon is the packaging: a small, readable, **zero-dependency** implementation you can audit in
one sitting, with an eval harness, rather than a trained model, a heavyweight framework, or a
hosted service. The lineage:

- **Corrective / Self / Agentic RAG.** "Escalate to the web when local retrieval is weak" is
  literally the core of **Corrective RAG (CRAG)** ([arXiv 2401.15884](https://arxiv.org/abs/2401.15884));
  the groundedness self-check echoes **Self-RAG** ([arXiv 2310.11511](https://arxiv.org/abs/2310.11511)).
  Frameworks operationalize the same loops — **LangGraph agentic RAG**
  ([docs](https://docs.langchain.com/oss/python/langgraph/agentic-rag)) and **LlamaIndex agentic RAG**.
- **Guardrails & groundedness eval.** Refuse-when-unsupported and groundedness scoring are commodity:
  **Guardrails AI** ([repo](https://github.com/guardrails-ai/guardrails)),
  **NVIDIA NeMo Guardrails** ([repo](https://github.com/NVIDIA-NeMo/Guardrails)),
  **Amazon Bedrock contextual grounding check**
  ([docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)),
  and the eval libraries **RAGAS faithfulness**
  ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)) and
  **TruLens** ([RAG triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)).
- **Cross-source corroboration** is an *active research area*, not a solved primitive: resolving
  conflicting evidence with source credibility ([arXiv 2505.17762](https://arxiv.org/abs/2505.17762)),
  corroborating-and-refuting evidence retrieval ([arXiv 2503.07937](https://arxiv.org/abs/2503.07937)),
  and multi-source disagreement modeling ([arXiv 2602.18693](https://arxiv.org/abs/2602.18693));
  consumer answer engines like Perplexity productize cited web answers. rag-guard sits *downstream*
  of this literature — it packages authority-weighted corroboration as a small guard tier; it does
  not contribute a new verification method.

**The claim, precisely:** not "I invented self-verifying RAG," but "here's a clean, tested,
dependency-free implementation of the guardrail patterns, wired into a real workflow, with the
confidence and limitations labeled honestly."

## Reliability & security

A guardrail library has to hold itself to its own bar, so the repo is gated:

- **Coverage-gated test matrix** — pytest on Python 3.11–3.13, build fails below the coverage floor (currently 99% covered).
- **CodeQL** — `security-extended` static analysis on every push, PR, and weekly; findings surface in the Security tab.
- **Pinned supply chain** — GitHub Actions pinned to commit SHAs, kept current by **Dependabot**.
- **Branch protection** — `main` requires CI + CodeQL to pass before a merge.
- **Disclosure policy** — see [SECURITY.md](SECURITY.md); private reporting is enabled.

## About

Built by **Jeff Otterson** ([Jott2121](https://github.com/Jott2121)). Companion to [**agent-gate**](https://github.com/Jott2121/agent-gate) (an MCP gate for agent work), [**bow**](https://github.com/Jott2121/bow), [**fleet-mode**](https://github.com/Jott2121/fleet-mode), and [**agent-cost-attribution**](https://github.com/Jott2121/agent-cost-attribution). This one's job is simple: if the context can't back the answer, the answer doesn't ship. MIT.
