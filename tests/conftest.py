import httpx2
import pytest
from fastapi.testclient import TestClient

from api.main import app

OK_BODY = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
}


@pytest.fixture
def env(monkeypatch):
    """Set a dummy API key and clear optional LLM settings from the real env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)
    return monkeypatch


@pytest.fixture
def client(env):
    """A TestClient with the app's lifespan running."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def orchestrator(client, monkeypatch):
    """The running app's orchestrator with retry waits disabled."""
    orch = client.app.state.agent_orchestrator
    monkeypatch.setattr(orch._call_llm.retry, "wait", lambda *a, **k: 0)
    return orch


@pytest.fixture
def fake_llm(orchestrator):
    """Route LLM calls to a handler; returns a setter and a call counter."""
    calls = {"n": 0, "requests": []}

    def install(handler):
        def counting(request):
            calls["n"] += 1
            calls["requests"].append(request)
            return handler(request)

        orchestrator.http_client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(counting)
        )

    install.calls = calls
    return install
