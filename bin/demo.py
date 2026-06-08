"""End-to-end demo: grounded answer, refusal, PII redaction, and an eval run.

Run:  PYTHONPATH=. python3 bin/demo.py     (no API key — uses FakeProvider)
"""
from rag_guard.retriever import Retriever
from rag_guard.pipeline import RagGuard
from rag_guard.providers import FakeProvider
from rag_guard.evaluate import evaluate

CORPUS = [
    {"id": "ship", "text": "Standard shipping takes 3 to 5 business days. Express is overnight."},
    {"id": "returns", "text": "You can return any item within 30 days for a full refund."},
    {"id": "pets", "text": "Cats and dogs are allowed with a $300 deposit and $35/mo pet rent."},
]


def show(title, out):
    print(f"\n[{title}]")
    print(f"  refused : {out['refused']}   grounded: {out['grounded']}   sources: {out['sources']}")
    print(f"  answer  : {out['answer']}")


def main():
    ret = Retriever(CORPUS)

    # 1) grounded answer
    rag = RagGuard(ret, FakeProvider("Standard shipping takes 3 to 5 business days."))
    show("grounded", rag.answer("how long does shipping take?"))

    # 2) refusal — no support in the corpus, model never called
    show("refusal", rag.answer("what is the capital of France?"))

    # 3) PII redaction on model output
    rag_pii = RagGuard(ret, FakeProvider("Email jeff@example.com or call 406-555-0142 about shipping."))
    show("pii-redaction", rag_pii.answer("how do I ask about shipping?"))

    # 4) eval harness
    cases = [
        {"query": "how long does shipping take", "gold": "ship", "expect_refusal": False},
        {"query": "what is your return window", "gold": "returns", "expect_refusal": False},
        {"query": "who won the 1998 world cup", "expect_refusal": True},
    ]
    m = evaluate(RagGuard(ret, FakeProvider("Standard shipping takes 3 to 5 business days.")), cases)
    print("\n[eval]")
    print(f"  n={m['n']}  refusal_accuracy={m['refusal_accuracy']}  "
          f"retrieval_hit_rate={m['retrieval_hit_rate']}  grounded_rate={m['grounded_rate']}")


if __name__ == "__main__":
    main()
