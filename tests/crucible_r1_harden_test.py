import pytest

from rag_guard.guard import _content_tokens, groundedness


def test_content_tokens_none_input_yields_empty_set():
    # `_content_tokens(None)` must fall back to "" (not a placeholder word),
    # so no tokens are extracted from missing text.
    assert _content_tokens(None) == set()


def test_content_tokens_empty_string_yields_empty_set():
    # Same fallback path as None: empty string has no content tokens.
    assert _content_tokens("") == set()


def test_content_tokens_falsy_text_does_not_leak_placeholder_token():
    # Guards against a mutant that substitutes a non-empty placeholder (e.g. "XXXX")
    # for falsy text in `(text or "").lower()` — that would inject a spurious
    # "xxxx" token into the result.
    assert "xxxx" not in _content_tokens(None)
    assert "xxxx" not in _content_tokens("")


def test_groundedness_support_rounded_to_four_decimal_places():
    # answer content tokens: {"apple", "banana", "cherry"} (3 tokens)
    # context content tokens: {"apple", "info"}
    # overlap = {"apple"} -> support = 1/3 = 0.333333... -> round(x, 4) == 0.3333
    # A mutant that rounds to 5 decimals instead would yield 0.33333, which is
    # NOT approx-equal to 0.3333 at the tight tolerance used below.
    result = groundedness("apple banana cherry", ["apple info"])
    expected_support = round(1 / 3, 4)
    assert expected_support == pytest.approx(0.3333, rel=1e-6)
    assert result["support"] == pytest.approx(expected_support, abs=1e-9)
    assert result["grounded"] is False


def test_groundedness_support_exact_value_not_five_decimals():
    # Independent computation: 1/3 rounded to 4 places is 0.3333, which differs
    # from the 5-decimal rounding (0.33333) by 3e-5 -- well outside a 1e-9 tolerance,
    # so this fails against the mutant that rounds to 5 decimal places.
    result = groundedness("apple banana cherry", ["apple info"])
    assert result["support"] != pytest.approx(0.33333, abs=1e-9)
    assert result["support"] == pytest.approx(0.3333, abs=1e-9)
