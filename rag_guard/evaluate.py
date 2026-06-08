"""Eval harness — run a labeled set through the pipeline and report metrics.

Metrics:
- retrieval_hit_rate: of answered-label cases with a `gold` doc id, how often the
  gold doc was actually retrieved.
- refusal_accuracy: of cases with an `expect_refusal` label, how often the
  system's refuse/answer decision matched.
- grounded_rate: of cases the system answered, how often the answer was grounded.

Case shape: {"query": str, "gold": doc_id?, "expect_refusal": bool?}.
Re-run this on any model/config change to catch regressions before a user does.
"""
from __future__ import annotations


def evaluate(rag, cases: list[dict]) -> dict:
    results = []
    refusal_total = refusal_ok = 0
    retr_total = retr_hit = 0
    answered = grounded = 0

    for case in cases:
        out = rag.answer(case["query"])
        results.append({"query": case["query"], "refused": out["refused"],
                        "grounded": out["grounded"], "sources": out["sources"]})

        if "expect_refusal" in case:
            refusal_total += 1
            if out["refused"] == case["expect_refusal"]:
                refusal_ok += 1

        if case.get("gold") is not None and not case.get("expect_refusal", False):
            retr_total += 1
            if case["gold"] in out["sources"]:
                retr_hit += 1

        if not out["refused"]:
            answered += 1
            if out["grounded"]:
                grounded += 1

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "n": len(cases),
        "refusal_accuracy": rate(refusal_ok, refusal_total),
        "retrieval_hit_rate": rate(retr_hit, retr_total),
        "grounded_rate": rate(grounded, answered),
        "cases": results,
    }
