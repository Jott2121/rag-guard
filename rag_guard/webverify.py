"""Tier 2/3: search the web and corroborate a claim across independent, authority-
weighted sources. Web access is INJECTED via search_fn (returns [{"url","supports"}]),
so this module is network-free and unit-testable. Syndication detection is future work;
v1 independence = distinct publishers, with social-only agreement blocked from verified."""
from __future__ import annotations

from urllib.parse import urlparse

from rag_guard import stamps

_OFFICIAL = (".gov", ".mil", ".gov.uk")
_SOCIAL = {"twitter.com", "x.com", "reddit.com", "facebook.com", "t.me",
           "instagram.com", "tiktok.com"}


def publisher_of(url):
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def authority_tier(url):
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(_OFFICIAL):
        return 3
    return 1 if publisher_of(url) in _SOCIAL else 2


def independent(sources):
    seen, out = set(), []
    for s in sources:
        pub = publisher_of(s["url"])
        if pub not in seen:
            seen.add(pub)
            out.append(s)
    return out


def verify_claim(query, candidate, search_fn, *, min_sources=2, max_sources=None):
    raw = search_fn(query) or []
    for s in raw:
        s["publisher"] = publisher_of(s["url"])
        s["authority_tier"] = authority_tier(s["url"])
    uniq = independent(raw)
    if max_sources:
        uniq = uniq[:max_sources]
    supporting = [s for s in uniq if s.get("supports")]
    refuting = [s for s in uniq if not s.get("supports")]
    has_credible = any(s["authority_tier"] >= 2 for s in supporting)

    if supporting and refuting:
        status, conflict = stamps.CONFLICT, True
    elif len(supporting) >= min_sources and has_credible:
        status, conflict = stamps.WEB_VERIFIED, False
    elif supporting:
        status, conflict = stamps.SINGLE_SOURCE, False
    else:
        status, conflict = stamps.UNVERIFIED, False

    confidence = min(1.0, len(supporting) / float(min_sources)) if supporting else 0.0
    return {"status": status, "sources": uniq, "confidence": round(confidence, 3),
            "conflict": conflict, "contradicts_local": False}
