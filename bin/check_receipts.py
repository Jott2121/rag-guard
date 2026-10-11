"""Offline receipt check for the published rag-guard eval numbers.

Compares three sources that must agree:

- the real-eval line in README.md
- the summary fields in eval/results.json
- rates recomputed from that file's per-case rows joined to the labeled
  cases in bin/eval_real.py

Recompute uses the same definitions as rag_guard.evaluate.evaluate
(retrieval hit-rate, refusal accuracy, grounded-rate). This does not call
a model and does not write eval/results.json. The live harness remains
`PYTHONPATH=. python3 bin/eval_real.py`.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("n", "refusal_accuracy", "retrieval_hit_rate", "grounded_rate")
README_LINE = re.compile(
    r"\{\s*'n':\s*20,\s*'refusal_accuracy':\s*([0-9.]+),\s*"
    r"'retrieval_hit_rate':\s*([0-9.]+),\s*'grounded_rate':\s*([0-9.]+)\s*\}"
)


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def recompute(labels: list[dict], rows: list[dict]) -> dict:
    """Mirror rag_guard.evaluate.evaluate over a stored result list."""
    if len(labels) != len(rows):
        raise SystemExit(
            f"case count mismatch: labels={len(labels)} results={len(rows)}"
        )
    refusal_total = refusal_ok = 0
    retr_total = retr_hit = 0
    answered = grounded = 0
    for case, out in zip(labels, rows):
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
    return {
        "n": len(rows),
        "refusal_accuracy": rate(refusal_ok, refusal_total),
        "retrieval_hit_rate": rate(retr_hit, retr_total),
        "grounded_rate": rate(grounded, answered),
    }


def load_cases() -> list[dict]:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "rag_guard_eval_real", ROOT / "bin" / "eval_real.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("could not load bin/eval_real.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.CASES)


def readme_metrics(text: str) -> dict:
    match = README_LINE.search(text)
    if match is None:
        raise SystemExit(
            "README.md is missing the published 20-case eval line "
            "({'n': 20, 'refusal_accuracy': ..., 'retrieval_hit_rate': ..., "
            "'grounded_rate': ...})"
        )
    refusal, hit, grounded = match.groups()
    return {
        "n": 20,
        "refusal_accuracy": float(refusal),
        "retrieval_hit_rate": float(hit),
        "grounded_rate": float(grounded),
    }


def same(key: str, left, right) -> bool:
    if key == "n":
        return int(left) == int(right)
    if left is None or right is None:
        return left is right
    return round(float(left), 4) == round(float(right), 4)


def main() -> int:
    readme = readme_metrics((ROOT / "README.md").read_text())
    stored = json.loads((ROOT / "eval" / "results.json").read_text())
    recomputed = recompute(load_cases(), stored["cases"])
    stored_summary = {key: stored[key] for key in KEYS}

    mismatches = []
    for key in KEYS:
        if not same(key, readme[key], stored_summary[key]):
            mismatches.append(
                f"  {key}: README={readme[key]} results.json={stored_summary[key]}"
            )
        if not same(key, stored_summary[key], recomputed[key]):
            mismatches.append(
                f"  {key}: results.json={stored_summary[key]} recomputed={recomputed[key]}"
            )
    if mismatches:
        print("eval receipt mismatch:", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        return 1

    shown = ", ".join(f"{key}={stored_summary[key]}" for key in KEYS)
    print(f"eval receipts match README and recomputed cases: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
