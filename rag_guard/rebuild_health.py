"""Did the background rebuild actually succeed? Serving stale requires an answer.

Serving a stale index is a deliberate trade: one prompt against a slightly older corpus
beats a human waiting 15 seconds. That trade holds only while the staleness is bounded and
observable. If the detached rebuild fails every time -- one unparseable note, a full disk --
the lock releases, the next prompt serves stale again, and it repeats forever with nothing
anywhere saying so. The child's stderr goes to DEVNULL because its parent's stdout is a hook
response channel, which means a file is the only place a failure signal can live.

Silent staleness is fine. Unbounded, unobservable staleness is not.
"""
from __future__ import annotations

import json
import os
import time

# Two consecutive failures is a pattern, not a blip: a single failure can be a rebuild
# racing a file that was mid-write.
FAILURE_THRESHOLD = 2
# Rebuilds take 12-16s and are triggered by any note write. Six hours without one landing
# means the mechanism is not running, not that the corpus was quiet.
STALE_WARN_SECONDS = 6 * 60 * 60


def marker_path(cache_path: str) -> str:
    return cache_path + ".rebuild.json"


def _read(cache_path: str) -> dict:
    try:
        with open(marker_path(cache_path)) as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(cache_path: str, payload: dict) -> None:
    try:
        path = marker_path(cache_path)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError:
        pass  # telemetry must never break the thing it observes


def record_success(cache_path: str, *, chunks: int) -> None:
    _write(cache_path, {"state": "ok", "at": time.time(), "chunks": chunks,
                        "consecutive_failures": 0})


def record_failure(cache_path: str, error: str) -> None:
    previous = _read(cache_path)
    _write(cache_path, {"state": "failed", "at": time.time(),
                        "error": str(error)[:500],
                        "consecutive_failures": int(previous.get("consecutive_failures", 0)) + 1,
                        "last_ok": previous.get("at") if previous.get("state") == "ok"
                        else previous.get("last_ok")})


def read_status(cache_path: str) -> dict:
    payload = _read(cache_path)
    state = payload.get("state")
    if state not in ("ok", "failed"):
        return {"state": "unknown", "stale_seconds": None, "consecutive_failures": 0}
    at = payload.get("at")
    return {
        "state": state,
        "stale_seconds": (time.time() - at) if isinstance(at, (int, float)) else None,
        "consecutive_failures": int(payload.get("consecutive_failures", 0)),
        "chunks": payload.get("chunks"),
        "error": payload.get("error"),
    }


def warning(cache_path: str) -> str | None:
    """One line to surface upward, or None when everything is fine.

    Silent on a healthy system on purpose: a caveat printed on every prompt is noise, and
    noise is how a real warning gets ignored."""
    status = read_status(cache_path)
    if status["state"] == "unknown":
        return None
    if status["consecutive_failures"] >= FAILURE_THRESHOLD:
        return (f"NOTE: the notes index has failed to rebuild "
                f"{status['consecutive_failures']} times "
                f"({(status.get('error') or 'unknown error').splitlines()[0][:120]}). "
                f"Passages above may be out of date.")
    stale = status["stale_seconds"]
    if status["state"] == "ok" and stale is not None and stale > STALE_WARN_SECONDS:
        return (f"NOTE: the notes index has not rebuilt in {stale / 3600:.0f}h. "
                f"Passages above may be out of date.")
    return None
