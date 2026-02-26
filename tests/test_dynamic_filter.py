"""Tests for the dynamic content filter."""

from __future__ import annotations

import pandas as pd
import pytest

from ekap_anom.config import FilterConfig
from ekap_anom.filters.dynamic_only import filter_dynamic_only, is_dynamic


class TestIsDynamic:
    def test_aspx(self) -> None:
        assert is_dynamic("/EKAP/Default.aspx", {"aspx", "ashx"}) is True

    def test_ashx(self) -> None:
        assert is_dynamic("/EKAP/Ortak/Data.ashx", {"aspx", "ashx"}) is True

    def test_static_css(self) -> None:
        assert is_dynamic("/styles/main.css", {"aspx", "ashx"}) is False

    def test_static_js(self) -> None:
        assert is_dynamic("/scripts/app.js", {"aspx", "ashx"}) is False

    def test_static_image(self) -> None:
        assert is_dynamic("/images/logo.png", {"aspx", "ashx"}) is False

    def test_no_extension(self) -> None:
        assert is_dynamic("/api/health", {"aspx", "ashx"}) is False

    def test_case_insensitive(self) -> None:
        assert is_dynamic("/EKAP/Page.ASPX", {"aspx", "ashx"}) is True

    def test_optional_asmx(self) -> None:
        config = FilterConfig(enable_optional=True)
        assert is_dynamic("/service.asmx", config.allowed_extensions) is True


class TestFilterDynamicOnly:
    def test_filters_correctly(self) -> None:
        df = pd.DataFrame({
            "uri_stem": [
                "/EKAP/Default.aspx",
                "/static/main.css",
                "/EKAP/Data.ashx",
                "/images/logo.png",
                "/scripts/app.js",
            ],
            "status": [200, 200, 200, 200, 200],
        })
        result = filter_dynamic_only(df)
        assert len(result) == 2
        assert set(result["uri_stem"]) == {"/EKAP/Default.aspx", "/EKAP/Data.ashx"}

    def test_empty_input(self) -> None:
        df = pd.DataFrame({"uri_stem": [], "status": []})
        result = filter_dynamic_only(df)
        assert len(result) == 0

    def test_all_static(self) -> None:
        df = pd.DataFrame({
            "uri_stem": ["/a.css", "/b.js", "/c.png"],
            "status": [200, 200, 200],
        })
        result = filter_dynamic_only(df)
        assert len(result) == 0
