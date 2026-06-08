"""Tests for the retriever — pure-stdlib TF-IDF cosine over a small corpus.

Real RAG, no heavy deps and no API keys, so CI stays green and fast. The scoring
is the load-bearing part for the GUARDRAIL: when nothing relevant is retrieved
(top score ~0), the pipeline must refuse rather than hallucinate. (Swap in real
embeddings behind the same `retrieve()` interface later.)
"""
import unittest

from rag_guard import retriever as R


def _corpus():
    return R.Retriever([
        {"id": "ship", "text": "Standard shipping takes 3 to 5 business days. Express is overnight."},
        {"id": "returns", "text": "You can return any item within 30 days for a full refund."},
        {"id": "pets", "text": "Cats and dogs are allowed with a deposit and monthly pet rent."},
    ])


class RetrieverTests(unittest.TestCase):
    def test_retrieves_the_most_relevant_doc(self):
        hits = _corpus().retrieve("how long does shipping take", k=1)
        self.assertEqual(hits[0]["id"], "ship")
        self.assertGreater(hits[0]["score"], 0)

    def test_respects_k(self):
        hits = _corpus().retrieve("return policy and shipping", k=2)
        self.assertEqual(len(hits), 2)

    def test_irrelevant_query_scores_zero(self):
        # the guardrail depends on this: no lexical overlap => no support => refuse
        hits = _corpus().retrieve("quantum chromodynamics lagrangian", k=3)
        self.assertTrue(all(h["score"] == 0 for h in hits) or hits == [])

    def test_stopword_only_overlap_scores_zero(self):
        # "is/the/of" must NOT create spurious matches — otherwise a query like
        # "what is the capital of France" gets fake support and the refusal guard fails.
        hits = _corpus().retrieve("what is the of", k=3)
        self.assertTrue(all(h["score"] == 0 for h in hits))

    def test_results_sorted_by_score_desc(self):
        hits = _corpus().retrieve("return a dog within 30 days", k=3)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
