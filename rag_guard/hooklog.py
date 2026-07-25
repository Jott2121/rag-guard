"""Append-only observability log for the grounding hook.

The hook injects notes on ~92% of prompts. Nobody has ever measured whether the injected
passages get USED, which means every proposal to tighten or loosen MIN_SCORE /
HOOK_MIN_OVERLAP is a guess at an unobserved number. This records what actually happened
per prompt -- including the SILENT decisions, since underfiring is the failure mode that
leaves no trace -- so that call can be made on data.

Two hard rules:

1. **Opt-in.** Records carry verbatim prompts, so a public install must never start logging
   because someone imported the package. Set RAG_GUARD_HOOK_LOG=1.
2. **Fail-silent.** Instrumentation that can break grounding is worse than no
   instrumentation. Every path here swallows its own errors and returns False.
"""
from __future__ import annotations

import json
import os
import time

ENABLE_ENV = "RAG_GUARD_HOOK_LOG"
DEFAULT_MAX_BYTES = 20_000_000


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENABLE_ENV, "") == "1"


def log_event(event: dict, *, path: str, env=None, max_bytes: int = DEFAULT_MAX_BYTES) -> bool:
    """Append one JSON object. Returns True only if a line actually landed on disk."""
    if not enabled(env):
        return False
    try:
        # Serialize BEFORE touching the filesystem so an unserializable event cannot
        # leave a truncated line behind.
        if "ts" not in event:
            event = {"ts": time.time(), **event}
        line = json.dumps(event) + "\n"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _rotate_if_needed(path, max_bytes)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def _rotate_if_needed(path: str, max_bytes: int) -> None:
    try:
        if os.path.getsize(path) < max_bytes:
            return
    except OSError:
        return
    # One previous window is kept. Two windows of a size-capped log is plenty of history
    # for this question, and it bounds disk use to 2 * max_bytes.
    os.replace(path, path + ".1")


def read_events(path: str) -> list[dict]:
    """All events, oldest first, rotated window included. Corrupt lines are skipped --
    a partially-flushed final line must not sink the whole analysis."""
    events = []
    for candidate in (path + ".1", path):
        try:
            with open(candidate) as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except (ValueError, TypeError):
                        continue
        except OSError:
            continue
    return events
