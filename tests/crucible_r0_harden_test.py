"""Tests for rag_guard.guard: should_refuse, groundedness, redact_pii."""
import pytest

from rag_guard.guard import groundedness, redact_pii, should_refuse


# ---------------------------------------------------------------------------
# should_refuse
# ---------------------------------------------------------------------------

def test_should_refuse_empty_hits_refuses():
    assert should_refuse([]) is True


def test_should_refuse_score_above_default_threshold_does_not_refuse():
    assert should_refuse([{"score": 0.9}]) is False


def test_should_refuse_score_below_default_threshold_refuses():
    assert should_refuse([{"score": 0.01}]) is True


def test_should_refuse_score_exactly_at_default_threshold_does_not_refuse():
    # condition is strictly `max < min_score`, so equality does not refuse
    assert should_refuse([{"score": 0.05}]) is False


def test_should_refuse_missing_score_key_defaults_to_zero_and_refuses():
    assert should_refuse([{}]) is True


def test_should_refuse_uses_max_score_across_multiple_hits():
    hits = [{"score": 0.01}, {"score": 0.2}, {"score": 0.0}]
    assert should_refuse(hits) is False


def test_should_refuse_all_low_scores_refuses():
    hits = [{"score": 0.01}, {"score": 0.02}, {"score": 0.03}]
    assert should_refuse(hits) is True


def test_should_refuse_custom_min_score_below_threshold_refuses():
    assert should_refuse([{"score": 0.5}], min_score=0.6) is True


def test_should_refuse_custom_min_score_equal_does_not_refuse():
    assert should_refuse([{"score": 0.5}], min_score=0.5) is False


# ---------------------------------------------------------------------------
# groundedness
# ---------------------------------------------------------------------------

def test_groundedness_empty_answer_is_not_grounded_and_zero_support():
    result = groundedness("", ["anything here"])
    assert result["grounded"] is False
    assert result["support"] == pytest.approx(0.0)


def test_groundedness_answer_of_only_stopwords_is_not_grounded():
    result = groundedness("the and for", ["anything here"])
    assert result["grounded"] is False
    assert result["support"] == pytest.approx(0.0)


def test_groundedness_half_support_is_grounded_at_default_threshold():
    # answer content tokens: {cats, dogs}; context tokens: {cats, great, pets}
    # ("are" is a stopword) -> intersection {cats} -> support 1/2 = 0.5
    result = groundedness("cats dogs", ["cats are great pets"])
    assert result["support"] == pytest.approx(0.5)
    assert result["grounded"] is True


def test_groundedness_quarter_support_is_not_grounded_at_default_threshold():
    # answer tokens: {cats, dogs, birds, fish}; context tokens: {cats, everywhere}
    # intersection {cats} -> support 1/4 = 0.25
    result = groundedness("cats dogs birds fish", ["cats everywhere"])
    assert result["support"] == pytest.approx(0.25)
    assert result["grounded"] is False


def test_groundedness_empty_contexts_list_gives_zero_support():
    result = groundedness("cats dogs", [])
    assert result["support"] == pytest.approx(0.0)
    assert result["grounded"] is False


def test_groundedness_rounds_support_to_four_decimal_places():
    # answer tokens: {cats, dogs, birds}; context tokens: {cats, only, here}
    # intersection {cats} -> support 1/3 = 0.3333...
    result = groundedness("cats dogs birds", ["cats only here"], threshold=0.3)
    assert result["support"] == pytest.approx(0.3333, abs=1e-4)
    assert result["grounded"] is True


def test_groundedness_merges_tokens_across_multiple_contexts():
    # answer tokens: {cats, dogs, birds}; contexts merge to {cats, dogs}
    # intersection {cats, dogs} -> support 2/3 = 0.6667
    result = groundedness("cats dogs birds", ["cats", "dogs"])
    assert result["support"] == pytest.approx(0.6667, abs=1e-4)
    assert result["grounded"] is True


def test_groundedness_full_support_is_grounded():
    result = groundedness("cats dogs", ["cats dogs everywhere"])
    assert result["support"] == pytest.approx(1.0)
    assert result["grounded"] is True


def test_groundedness_zero_support_is_not_grounded():
    result = groundedness("cats dogs", ["nothing relevant whatsoever"])
    assert result["support"] == pytest.approx(0.0)
    assert result["grounded"] is False


# ---------------------------------------------------------------------------
# redact_pii
# ---------------------------------------------------------------------------

def test_redact_pii_redacts_email():
    text = "Contact me at foo.bar@example.com please"
    assert redact_pii(text) == "Contact me at [redacted-email] please"


def test_redact_pii_redacts_ssn():
    text = "SSN: 123-45-6789"
    assert redact_pii(text) == "SSN: [redacted-ssn]"


def test_redact_pii_redacts_phone_with_dashes():
    text = "Call 555-123-4567 now"
    assert redact_pii(text) == "Call [redacted-phone] now"


def test_redact_pii_redacts_phone_with_dots():
    text = "Phone: 555.123.4567"
    assert redact_pii(text) == "Phone: [redacted-phone]"


def test_redact_pii_redacts_16_digit_card_number():
    text = "Card 4111111111111111 expires"
    assert redact_pii(text) == "Card [redacted-card]expires"


def test_redact_pii_redacts_13_digit_card_number_lower_bound():
    text = "Card 1234567890123 test"
    assert redact_pii(text) == "Card [redacted-card]test"


def test_redact_pii_does_not_redact_12_digit_number_below_card_lower_bound():
    text = "Number 123456789012 test"
    assert redact_pii(text) == "Number 123456789012 test"


def test_redact_pii_does_not_redact_unseparated_10_digit_number():
    text = "Number 5551234567 here"
    assert redact_pii(text) == "Number 5551234567 here"


def test_redact_pii_does_not_redact_space_separated_ssn_like_number():
    # SSN pattern only matches dash-separated groups
    text = "SSN 123 45 6789"
    assert redact_pii(text) == "SSN 123 45 6789"


def test_redact_pii_redacts_multiple_emails():
    text = "a@b.com and c@d.com"
    assert redact_pii(text) == "[redacted-email] and [redacted-email]"


def test_redact_pii_redacts_mixed_pii_types_in_one_string():
    text = "Email me at a@b.com or call 555-123-4567, SSN 123-45-6789"
    expected = "Email me at [redacted-email] or call [redacted-phone], SSN [redacted-ssn]"
    assert redact_pii(text) == expected


def test_redact_pii_leaves_text_without_pii_unchanged():
    text = "Hello world, nothing sensitive here."
    assert redact_pii(text) == text


def test_redact_pii_empty_string_returns_empty_string():
    assert redact_pii("") == ""
