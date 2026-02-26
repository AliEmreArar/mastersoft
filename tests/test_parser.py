"""Tests for IIS W3C log parser."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ekap_anom.ingest.parser import parse_fields_header, parse_line, parse_file


class TestParseFieldsHeader:
    def test_basic(self) -> None:
        line = "#Fields: date time c-ip cs-method cs-uri-stem sc-status"
        fields = parse_fields_header(line)
        assert fields == ["date", "time", "c-ip", "cs-method", "cs-uri-stem", "sc-status"]

    def test_with_parentheses(self) -> None:
        line = "#Fields: date time cs(User-Agent) cs(Referer)"
        fields = parse_fields_header(line)
        assert "cs(User-Agent)" in fields
        assert "cs(Referer)" in fields


class TestParseLine:
    def test_valid_line(self) -> None:
        fields = ["date", "time", "c-ip", "cs-method", "cs-uri-stem", "sc-status"]
        raw = "2024-04-22 13:05:42 192.168.1.1 GET /EKAP/Default.aspx 200"
        row = parse_line(raw, fields)
        assert row is not None
        assert row["timestamp"] == dt.datetime(2024, 4, 22, 13, 5, 42, tzinfo=dt.timezone.utc)
        assert row["method"] == "GET"
        assert row["uri_stem"] == "/EKAP/Default.aspx"
        assert row["status"] == 200

    def test_dash_normalization(self) -> None:
        fields = ["date", "time", "cs-username", "cs-method", "cs-uri-stem", "sc-status"]
        raw = "2024-04-22 13:05:42 - GET /test.aspx 200"
        row = parse_line(raw, fields)
        assert row is not None
        assert row["username"] is None
        assert row["username_missing"] == 1
        assert row["method_missing"] == 0

    def test_wrong_column_count(self) -> None:
        fields = ["date", "time", "c-ip"]
        raw = "2024-04-22 13:05:42"  # too few
        assert parse_line(raw, fields) is None

    def test_numeric_coercion(self) -> None:
        fields = ["date", "time", "cs-method", "cs-uri-stem", "sc-status", "time-taken"]
        raw = "2024-04-22 10:00:00 GET /page.aspx 404 1234"
        row = parse_line(raw, fields)
        assert row is not None
        assert row["status"] == 404
        assert row["time_taken"] == 1234


class TestParseFile:
    def test_parse_sample_file(self, tmp_path: Path) -> None:
        content = (
            "#Software: Microsoft Internet Information Services\n"
            "#Fields: date time c-ip cs-method cs-uri-stem sc-status\n"
            "2024-04-22 13:05:42 10.0.0.1 GET /EKAP/Default.aspx 200\n"
            "2024-04-22 13:05:43 10.0.0.2 POST /EKAP/Teklif/Submit.aspx 302\n"
            "# comment line\n"
            "2024-04-22 13:05:44 10.0.0.3 GET /static/logo.png 200\n"
        )
        log_file = tmp_path / "test.log"
        log_file.write_text(content)

        chunks = list(parse_file(log_file, chunk_size=100))
        assert len(chunks) == 1
        df = chunks[0]
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "method" in df.columns

    def test_midfile_fields_change(self, tmp_path: Path) -> None:
        content = (
            "#Fields: date time cs-method cs-uri-stem sc-status\n"
            "2024-04-22 10:00:00 GET /page1.aspx 200\n"
            "#Fields: date time cs-method cs-uri-stem sc-status time-taken\n"
            "2024-04-22 10:00:01 GET /page2.aspx 200 500\n"
        )
        log_file = tmp_path / "test2.log"
        log_file.write_text(content)

        chunks = list(parse_file(log_file, chunk_size=100))
        assert len(chunks) == 1
        df = chunks[0]
        assert len(df) == 2
