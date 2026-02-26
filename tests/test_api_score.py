"""Tests for the FastAPI scoring API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ekap_anom.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestScoreEndpoint:
    def test_score_empty(self, client: TestClient) -> None:
        resp = client.post("/score", json={"records": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["anomalies"] == 0

    def test_score_single(self, client: TestClient) -> None:
        resp = client.post("/score", json={
            "records": [{
                "uri_stem": "/EKAP/Default.aspx",
                "method": "GET",
                "status": 200,
                "time_taken": 500,
                "sc_bytes": 2048,
                "cs_bytes": 128,
                "client_ip": "10.0.0.1",
                "user_agent": "Mozilla/5.0",
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert "final_score" in result
        assert "endpoint_group" in result

    def test_score_batch(self, client: TestClient) -> None:
        records = [
            {
                "uri_stem": f"/EKAP/Page{i}.aspx",
                "method": "GET",
                "status": 200,
                "time_taken": 100 * i,
                "sc_bytes": 1024,
                "cs_bytes": 64,
                "client_ip": f"10.0.0.{i}",
                "user_agent": "Mozilla/5.0",
            }
            for i in range(5)
        ]
        resp = client.post("/score", json={"records": records})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5


class TestMetricsEndpoint:
    def test_metrics_returns_text(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "ekap_" in resp.text or "process_" in resp.text
