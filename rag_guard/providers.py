"""Model provider seam — keeps the pipeline deterministic and model-agnostic.

`FakeProvider` is for tests/CI (no API key, no network). A real provider
(Anthropic Messages API, or a `claude -p` shell-out) implements the same
`complete(prompt) -> str` and drops in without touching the pipeline or guards.
"""
from __future__ import annotations


class FakeProvider:
    def __init__(self, reply: str = "I don't know."):
        self.reply = reply
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply
