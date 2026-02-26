"""Tests for endpoint group normalization."""

from __future__ import annotations

import pytest

from ekap_anom.utils.url import (
    normalize_url,
    endpoint_group,
    extract_extension,
    path_depth,
    path_tokens,
    parse_query_string,
)


class TestNormalizeUrl:
    def test_lowercase(self) -> None:
        assert normalize_url("/EKAP/Default.aspx") == "/ekap/default.aspx"

    def test_trailing_slash(self) -> None:
        assert normalize_url("/EKAP/Page/") == "/ekap/page"

    def test_root_not_stripped(self) -> None:
        assert normalize_url("/") == "/"

    def test_numeric_template(self) -> None:
        assert normalize_url("/EKAP/Item/12345/Detail.aspx") == "/ekap/item/{num}/detail.aspx"

    def test_no_numeric_template(self) -> None:
        result = normalize_url("/EKAP/Item/12345/Detail.aspx", numeric_template=False)
        assert "12345" in result

    def test_url_decode(self) -> None:
        result = normalize_url("/EKAP/%44efault.aspx")
        assert "default" in result.lower()


class TestEndpointGroup:
    def test_aspx(self) -> None:
        result = endpoint_group("/EKAP/Default.aspx")
        assert result == "/ekap/default.aspx"

    def test_ashx(self) -> None:
        result = endpoint_group("/EKAP/Ortak/YeniIhaleAramaData.ashx")
        assert result == "/ekap/ortak/yeniihalearamadata.ashx"


class TestExtractExtension:
    def test_aspx(self) -> None:
        assert extract_extension("/EKAP/Default.aspx") == "aspx"

    def test_no_extension(self) -> None:
        assert extract_extension("/api/health") is None

    def test_uppercase(self) -> None:
        assert extract_extension("/Page.ASHX") == "ashx"


class TestPathDepth:
    def test_root(self) -> None:
        assert path_depth("/") == 0

    def test_nested(self) -> None:
        assert path_depth("/a/b/c/d.aspx") == 4


class TestPathTokens:
    def test_basic(self) -> None:
        assert path_tokens("/EKAP/Default.aspx") == ["ekap", "default.aspx"]


class TestParseQueryString:
    def test_empty(self) -> None:
        result = parse_query_string(None)
        assert result["key_count"] == 0

    def test_dash(self) -> None:
        result = parse_query_string("-")
        assert result["key_count"] == 0

    def test_basic(self) -> None:
        result = parse_query_string("id=123&type=test")
        assert result["key_count"] == 2
        assert "id" in result["key_set"]
        assert result["decoded_len"] > 0

    def test_special_chars(self) -> None:
        result = parse_query_string("q=<script>alert(1)</script>")
        assert result["special_char_ratio"] > 0
