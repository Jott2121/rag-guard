"""Real eval: run rag-guard's harness over a non-trivial labeled set with a live model.

Unlike the bundled demo (3 cases, FakeProvider → trivially 1.0s), this uses a real
model via a `claude -p` shell-out and a 20-case labeled set with paraphrases and
hard "plausible-but-absent" refusals, so the metrics mean something.

Run:  PYTHONPATH=. python3 bin/eval_real.py
Needs the `claude` CLI on PATH (or swap ClaudeProvider for an Anthropic Messages provider).
"""
from __future__ import annotations
import json, subprocess, sys, os

from rag_guard.retriever import Retriever
from rag_guard.pipeline import RagGuard
from rag_guard.evaluate import evaluate


class ClaudeProvider:
    """complete(prompt) -> str via the local `claude -p` CLI (no API key needed on a Max plan)."""
    def complete(self, prompt: str) -> str:
        try:
            r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=120)
            return (r.stdout or "").strip()
        except Exception as e:
            return f"(provider error: {e})"


# --- A small but real knowledge base for a fictional SaaS, "Northwind Analytics" ---
CORPUS = [
    {"id": "shipping",   "text": "Northwind ships hardware sensors via standard ground in 3 to 5 business days. Express shipping is next-business-day if ordered before 1pm Mountain Time."},
    {"id": "returns",    "text": "Hardware can be returned within 30 days of delivery for a full refund. The device must be in its original packaging. Software subscriptions are not returnable."},
    {"id": "refunds",    "text": "Once a returned device is received and inspected, refunds are issued to the original payment method within 7 to 10 business days."},
    {"id": "tiers",      "text": "Northwind has three plans: Free (1 dashboard, community support), Pro at $49 per user per month (unlimited dashboards, email support), and Enterprise with custom pricing (SSO, audit logs, a dedicated success manager)."},
    {"id": "retention",  "text": "After you cancel, Northwind retains your analytics data for 90 days so you can reactivate without loss. After 90 days the data is permanently deleted and cannot be recovered."},
    {"id": "security",   "text": "All data is encrypted in transit with TLS 1.2+ and at rest with AES-256. Northwind is SOC 2 Type II audited. Single sign-on (SSO) via SAML and SCIM provisioning are available on the Enterprise plan."},
    {"id": "ratelimits", "text": "The REST API allows 60 requests per minute on Free, 600 per minute on Pro, and 6,000 per minute on Enterprise. Exceeding the limit returns HTTP 429 with a Retry-After header."},
    {"id": "support",    "text": "Email support operates Monday through Friday, 6am to 6pm Mountain Time. Enterprise customers also get 24/7 phone support and a one-hour response SLA for critical issues."},
    {"id": "sla",        "text": "Northwind guarantees 99.9% monthly uptime for Pro and Enterprise. If uptime falls below that, affected customers receive service credits of 10% of the monthly fee per 0.1% below target."},
    {"id": "export",     "text": "You can export your raw event data at any time as CSV or Parquet from the Settings > Data Export page. Exports over 5 GB are delivered as a downloadable link by email."},
    {"id": "cancel",     "text": "To cancel, go to Billing > Subscription and click Cancel Plan. Cancellation takes effect at the end of the current billing period; you are not charged again after that."},
    {"id": "trial",      "text": "Every new account starts with a 14-day free trial of the Pro plan. No credit card is required to start the trial, and it converts to the Free plan automatically if you do not upgrade."},
]

# query, gold doc id (for answerable), expect_refusal
CASES = [
    # direct answerable
    {"query": "How many days do I have to return a device?", "gold": "returns", "expect_refusal": False},
    {"query": "How fast is express shipping?", "gold": "shipping", "expect_refusal": False},
    {"query": "When will I get my refund after sending a device back?", "gold": "refunds", "expect_refusal": False},
    {"query": "How much does the Pro plan cost per user?", "gold": "tiers", "expect_refusal": False},
    {"query": "What is the API rate limit on the Free plan?", "gold": "ratelimits", "expect_refusal": False},
    {"query": "What uptime does Northwind guarantee?", "gold": "sla", "expect_refusal": False},
    {"query": "What are your email support hours?", "gold": "support", "expect_refusal": False},
    {"query": "In what formats can I export my data?", "gold": "export", "expect_refusal": False},
    {"query": "How long is the free trial?", "gold": "trial", "expect_refusal": False},
    {"query": "Is single sign-on available?", "gold": "security", "expect_refusal": False},
    # paraphrased / indirect answerable (tests retrieval + grounding under rewording)
    {"query": "I changed my mind about the hardware I bought — can I send it back?", "gold": "returns", "expect_refusal": False},
    {"query": "What happens to my analytics data once I stop paying?", "gold": "retention", "expect_refusal": False},
    {"query": "How do I stop my subscription?", "gold": "cancel", "expect_refusal": False},
    {"query": "Is my information encrypted while stored?", "gold": "security", "expect_refusal": False},
    {"query": "What do I get if the service goes down a lot one month?", "gold": "sla", "expect_refusal": False},
    # should refuse — clearly out of corpus
    {"query": "What is the capital of France?", "expect_refusal": True},
    {"query": "Who is Northwind's CEO?", "expect_refusal": True},
    {"query": "What is Northwind's stock ticker symbol?", "expect_refusal": True},
    # should refuse — hard: plausible/on-topic but the answer is genuinely absent
    {"query": "Do you offer a student or nonprofit discount?", "expect_refusal": True},
    {"query": "Can I pay with cryptocurrency?", "expect_refusal": True},
]


def main():
    ret = Retriever(CORPUS)
    rag = RagGuard(ret, ClaudeProvider())
    print(f"Running real eval: {len(CASES)} cases over {len(CORPUS)} docs via claude -p ...\n", flush=True)
    m = evaluate(rag, CASES)

    print("=== rag-guard real eval ===")
    print(f"n                 = {m['n']}")
    print(f"refusal_accuracy  = {m['refusal_accuracy']}")
    print(f"retrieval_hit_rate= {m['retrieval_hit_rate']}")
    print(f"grounded_rate     = {m['grounded_rate']}\n")
    print("per-case:")
    for c, r in zip(CASES, m["cases"]):
        exp = c.get("expect_refusal")
        flag = "" if exp is None else ("  OK" if r["refused"] == exp else "  <-- refusal MISS")
        print(f"  refused={str(r['refused']):5}  grounded={str(r['grounded']):5}  src={r['sources']}{flag}  | {c['query'][:60]}")

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(m, f, indent=2)
    print("\nwrote eval/results.json")


if __name__ == "__main__":
    main()
