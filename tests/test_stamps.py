from rag_guard import stamps

def test_grounded_has_banner():
    assert "GROUNDED" in stamps.stamp_answer("ok", stamps.GROUNDED, ["m/x.md#0"])

def test_unverified_and_error_banners():
    assert "UNVERIFIED" in stamps.stamp_answer("m", stamps.UNVERIFIED, [])
    assert "guard" in stamps.stamp_answer("m", stamps.GUARD_UNAVAILABLE, []).lower()
    assert "web" in stamps.stamp_answer("m", stamps.WEB_CHECK_FAILED, []).lower()

def test_web_verified_lists_sources():
    out = stamps.stamp_answer("x", stamps.WEB_VERIFIED, ["https://a.gov", "https://b.org"])
    assert "a.gov" in out and "b.org" in out
