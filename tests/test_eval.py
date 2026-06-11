"""Tests for the eval harness — measure the system, don't trust it.

Runs a labeled set through the pipeline and reports retrieval hit-rate, refusal
accuracy, and grounded-rate. This is the 'receipts' layer: you ship a number,
not a vibe — and it re-runs on every model/config change to catch regressions.
"""
import unittest

from rag_guard import retriever as R, pipeline as P
from rag_guard.evaluate import evaluate
from rag_guard.providers import FakeProvider


def _rag():
    ret = R.Retriever([
        {"id": "ship", "text": "Standard shipping takes 3 to 5 business days. Express is overnight."},
        {"id": "returns", "text": "You can return any item within 30 days for a full refund."},
    ])
    return P.RagGuard(ret, FakeProvider("Shipping takes 3 to 5 business days."))


class EvalTests(unittest.TestCase):
    def test_metrics_on_a_small_labeled_set(self):
        cases = [
            {"query": "how long does shipping take", "gold": "ship", "expect_refusal": False},
            {"query": "quantum chromodynamics lagrangian", "expect_refusal": True},
        ]
        m = evaluate(_rag(), cases)
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["refusal_accuracy"], 1.0)     # both refusal labels correct
        self.assertEqual(m["retrieval_hit_rate"], 1.0)   # gold doc retrieved for the answered case
        self.assertEqual(m["grounded_rate"], 1.0)        # the answered case is grounded
        self.assertEqual(len(m["cases"]), 2)

    def test_detects_a_wrong_refusal(self):
        # label says it should answer, but it's out-of-corpus -> system refuses -> miss
        cases = [{"query": "totally unrelated zzz", "gold": "ship", "expect_refusal": False}]
        m = evaluate(_rag(), cases)
        self.assertEqual(m["refusal_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
