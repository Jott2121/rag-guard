from rag_guard import webverify, stamps

def test_publisher_and_authority():
    assert webverify.publisher_of("https://www.irs.gov/pub/x") == "irs.gov"
    assert webverify.authority_tier("https://irs.gov/x") == 3
    assert webverify.authority_tier("https://twitter.com/u/1") == 1

def test_independent_dedupes_by_publisher():
    srcs = [{"url": "https://a.gov/1"}, {"url": "https://a.gov/2"}, {"url": "https://b.org/1"}]
    assert len(webverify.independent(srcs)) == 2

def test_two_independent_nonsocial_is_web_verified():
    def sf(q): return [{"url": "https://a.gov/x", "supports": True},
                       {"url": "https://b.org/y", "supports": True}]
    v = webverify.verify_claim("q", "a", sf, min_sources=2)
    assert v["status"] == stamps.WEB_VERIFIED and len(v["sources"]) == 2

def test_social_only_is_not_web_verified():
    def sf(q): return [{"url": "https://twitter.com/a/1", "supports": True},
                       {"url": "https://reddit.com/b/2", "supports": True}]
    v = webverify.verify_claim("q", "a", sf, min_sources=2)
    assert v["status"] != stamps.WEB_VERIFIED

def test_single_supporting_is_single_source():
    v = webverify.verify_claim("q", "a", lambda q: [{"url": "https://a.gov/x", "supports": True}])
    assert v["status"] == stamps.SINGLE_SOURCE

def test_disagreement_is_conflict():
    def sf(q): return [{"url": "https://a.gov/x", "supports": True},
                       {"url": "https://b.org/y", "supports": False}]
    assert webverify.verify_claim("q", "a", sf)["conflict"] is True

def test_no_results_is_unverified():
    assert webverify.verify_claim("q", "a", lambda q: [])["status"] == stamps.UNVERIFIED

def test_max_sources_caps_fetches():
    big = [{"url": f"https://s{i}.org/x", "supports": True} for i in range(20)]
    v = webverify.verify_claim("q", "a", lambda q: big, max_sources=3)
    assert len(v["sources"]) <= 3
