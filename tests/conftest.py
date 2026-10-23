import os

import httpx2
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as app_module  # noqa: E402

OK_BODY = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
}


@pytest.fixture
def orchestrator(monkeypatch):
    """The app's orchestrator with retry waits disabled."""
    orch = app_module.agent_orchestrator
    monkeypatch.setattr(orch._call_llm.retry, "wait", lambda *a, **k: 0)
    return orch


@pytest.fixture
def fake_llm(orchestrator, monkeypatch):
    """Route LLM calls to a handler; returns a setter and a call counter."""
    calls = {"n": 0, "requests": []}

    def install(handler):
        def counting(request):
            calls["n"] += 1
            calls["requests"].append(request)
            return handler(request)

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(counting))
        monkeypatch.setattr(orchestrator, "http_client", client)

    install.calls = calls
    return install


@pytest.fixture
def client():
    return TestClient(app_module.app)
