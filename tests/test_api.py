import httpx2
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import OK_BODY


def test_health(client):
    assert client.get("/health").status_code == 200


def test_query_returns_response_and_cost(client, fake_llm):
    fake_llm(lambda r: httpx2.Response(200, json=OK_BODY))
    body = client.post(
        "/agent/query", json={"query": "x"}, headers={"X-Trace-ID": "t1"}
    ).json()
    assert body["response"] == "hi"
    assert body["cost_usd"] == pytest.approx(0.00075)
    assert body["trace_id"] == "t1"


def test_query_sends_api_key(client, fake_llm):
    fake_llm(lambda r: httpx2.Response(200, json=OK_BODY))
    client.post("/agent/query", json={"query": "x"})
    assert fake_llm.calls["requests"][0].headers["authorization"] == "Bearer sk-test"


def test_generated_trace_ids_are_unique(client, fake_llm):
    fake_llm(lambda r: httpx2.Response(200, json=OK_BODY))
    first = client.post("/agent/query", json={"query": "x"}).json()["trace_id"]
    second = client.post("/agent/query", json={"query": "x"}).json()["trace_id"]
    assert first.startswith("trace-")
    assert first != second


def test_upstream_connect_error_returns_generic_502(client, fake_llm):
    def fail(request):
        raise httpx2.ConnectError("dns fail for internal-host:8443", request=request)

    fake_llm(fail)
    r = client.post("/agent/query", json={"query": "x"})
    assert r.status_code == 502
    assert r.json() == {"detail": "Upstream LLM service error"}


def test_upstream_status_error_returns_502(client, fake_llm):
    fake_llm(lambda r: httpx2.Response(401, json={"error": "bad key"}))
    r = client.post("/agent/query", json={"query": "x"})
    assert r.status_code == 502


def test_internal_error_returns_generic_500(client, orchestrator, monkeypatch):
    async def broken(*args):
        raise RuntimeError("bug at /srv/secret/path")

    monkeypatch.setattr(orchestrator, "process_query", broken)
    r = client.post("/agent/query", json={"query": "x"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}


def test_missing_query_returns_422(client):
    assert client.post("/agent/query", json={}).status_code == 422


def test_startup_fails_for_model_without_price(env):
    env.setenv("LLM_MODEL", "gpt-typo")
    with pytest.raises(RuntimeError, match="LLM_MODEL 'gpt-typo' has no entry"):
        with TestClient(app):
            pass


def test_startup_fails_without_api_key(env):
    env.delenv("OPENAI_API_KEY")
    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        with TestClient(app):
            pass


def test_shutdown_closes_http_client(env):
    with TestClient(app) as test_client:
        http_client = test_client.app.state.agent_orchestrator.http_client
        assert not http_client.is_closed
    assert http_client.is_closed
