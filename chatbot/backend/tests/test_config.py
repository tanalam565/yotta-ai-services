import pytest

import config


def test_parse_cors_empty_returns_empty_list():
    assert config._parse_cors_allowed_origins("") == []


def test_parse_cors_strips_and_splits_values():
    parsed = config._parse_cors_allowed_origins(" https://a.com , https://b.com ")
    assert parsed == ["https://a.com", "https://b.com"]


def test_parse_cors_wildcard_cannot_mix_with_specific_origins():
    with pytest.raises(ValueError):
        config._parse_cors_allowed_origins("*,https://a.com")
