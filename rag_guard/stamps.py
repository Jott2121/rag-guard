"""Confidence-ladder stamps appended to a delivered answer."""
from __future__ import annotations

GROUNDED = "grounded"
WEB_VERIFIED = "web_verified"
SINGLE_SOURCE = "single_source"
CONFLICT = "conflict"
UNVERIFIED = "unverified"
GENERAL_ONLY = "general_only"
GUARD_UNAVAILABLE = "guard_unavailable"
WEB_CHECK_FAILED = "web_check_failed"

_BANNER = {
    GROUNDED: "✔ GROUNDED — from your notes",
    WEB_VERIFIED: "✔ WEB-VERIFIED — corroborated across independent sources",
    SINGLE_SOURCE: "⚠ SINGLE SOURCE — found on the web, one source only",
    CONFLICT: "⚠ SOURCES CONFLICT — sources disagree; see both",
    UNVERIFIED: "⚠ UNVERIFIED — I couldn't back this in your notes or the web",
    GENERAL_ONLY: "⚠ No relevant notes found — answering from general knowledge only",
    GUARD_UNAVAILABLE: "⚠ guard unavailable — grounding check failed; answer unverified",
    WEB_CHECK_FAILED: "⚠ web check failed — could not verify online; answer unverified",
}


def stamp_answer(answer, status, sources) -> str:
    lines = [answer.rstrip(), "", f"[{_BANNER.get(status, _BANNER[UNVERIFIED])}]"]
    if sources:
        lines.append("sources: " + ", ".join(sources))
    return "\n".join(lines)
