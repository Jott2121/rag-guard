"""Tests for the guardrails — the part that stops a RAG system lying.

Three guards: (1) refuse when retrieval found no real support (the #1 cause of
confident hallucination), (2) a groundedness check that the answer is actually
backed by the retrieved context, (3) a PII output filter. All deterministic and
stdlib so they're verifiable in CI. (Groundedness here is a lexical-overlap proxy
for an NLI/LLM judge — same interface, swappable.)
"""
import unittest

from rag_guard import guard as Gd


class RefusalTests(unittest.TestCase):
    def test_refuses_when_top_score_below_threshold(self):
        hits = [{"id": "x", "text": "...", "score": 0.0}]
        self.assertTrue(Gd.should_refuse(hits, min_score=0.05))

    def test_does_not_refuse_with_real_support(self):
        hits = [{"id": "x", "text": "...", "score": 0.42}]
        self.assertFalse(Gd.should_refuse(hits, min_score=0.05))

    def test_refuses_on_empty_retrieval(self):
        self.assertTrue(Gd.should_refuse([], min_score=0.05))


class GroundednessTests(unittest.TestCase):
    def test_grounded_answer_passes(self):
        ctx = ["Standard shipping takes 3 to 5 business days."]
        res = Gd.groundedness("Shipping takes 3 to 5 business days.", ctx)
        self.assertTrue(res["grounded"])
        self.assertGreaterEqual(res["support"], 0.5)

    def test_ungrounded_answer_flagged(self):
        ctx = ["Standard shipping takes 3 to 5 business days."]
        res = Gd.groundedness("The moon is made of cheese.", ctx)
        self.assertFalse(res["grounded"])


class PiiTests(unittest.TestCase):
    def test_redacts_email_and_phone(self):
        out = Gd.redact_pii("Reach me at jeff@example.com or 406-555-0142 anytime.")
        self.assertNotIn("jeff@example.com", out)
        self.assertNotIn("406-555-0142", out)
        self.assertIn("[redacted-email]", out)
        self.assertIn("[redacted-phone]", out)

    def test_leaves_clean_text_untouched(self):
        s = "Standard shipping takes 3 to 5 business days."
        self.assertEqual(Gd.redact_pii(s), s)


if __name__ == "__main__":
    unittest.main()
