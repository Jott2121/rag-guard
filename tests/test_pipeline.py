"""Tests for the end-to-end guarded RAG pipeline.

retrieve -> (refuse if unsupported) -> grounded prompt -> model -> groundedness
check + PII redaction -> a structured, observable result. The model is behind a
provider seam (FakeProvider) so the whole pipeline is deterministic in CI.
"""
import unittest

from rag_guard import retriever as R, pipeline as P
from rag_guard.providers import FakeProvider


def _rg(reply: str):
    ret = R.Retriever([
        {"id": "ship", "text": "Standard shipping takes 3 to 5 business days. Express is overnight."},
        {"id": "returns", "text": "You can return any item within 30 days for a full refund."},
        {"id": "pets", "text": "Cats and dogs are allowed with a deposit and monthly pet rent."},
    ])
    return P.RagGuard(ret, FakeProvider(reply))


class PipelineTests(unittest.TestCase):
    def test_refuses_unsupported_query_without_calling_model(self):
        prov = FakeProvider("SHOULD NOT BE USED")
        ret = R.Retriever([{"id": "ship", "text": "Shipping takes 3 to 5 days."}])
        rg = P.RagGuard(ret, prov)
        out = rg.answer("quantum chromodynamics lagrangian")
        self.assertTrue(out["refused"])
        self.assertIsNone(prov.last_prompt)            # model never called on no support
        self.assertEqual(out["sources"], [])

    def test_grounded_answer_cites_sources(self):
        out = _rg("Shipping takes 3 to 5 business days.").answer("how long does shipping take")
        self.assertFalse(out["refused"])
        self.assertTrue(out["grounded"])
        self.assertIn("ship", out["sources"])

    def test_ungrounded_model_output_is_flagged(self):
        out = _rg("The moon is made of cheese.").answer("how long does shipping take")
        self.assertFalse(out["refused"])
        self.assertFalse(out["grounded"])              # answer not supported by context

    def test_pii_in_model_output_is_redacted(self):
        out = _rg("Email jeff@example.com about shipping.").answer("shipping question")
        self.assertNotIn("jeff@example.com", out["answer"])
        self.assertIn("[redacted-email]", out["answer"])

    def test_result_has_observability_trace(self):
        out = _rg("Shipping takes 3 to 5 business days.").answer("shipping time")
        self.assertIn("trace", out)
        self.assertEqual(out["trace"]["query"], "shipping time")
        self.assertTrue(out["trace"]["retrieved"])     # list of {id, score}
        self.assertIn("score", out["trace"]["retrieved"][0])

    def test_grounded_prompt_instructs_context_only(self):
        prov = FakeProvider("Shipping takes 3 to 5 business days.")
        ret = R.Retriever([{"id": "ship", "text": "Standard shipping takes 3 to 5 business days."}])
        P.RagGuard(ret, prov).answer("shipping time")
        # the model was told to answer only from the provided context
        self.assertIn("context", prov.last_prompt.lower())
        self.assertIn("Standard shipping takes", prov.last_prompt)


if __name__ == "__main__":
    unittest.main()
