import asyncio

import httpx2
import pytest

from orchestrator.core import LLMRetryableError, LLMServiceError
from tests.conftest import OK_BODY


def status(code):
    return lambda request: httpx2.Response(code, json={"error": "x"})


def timeout(request):
    raise httpx2.ReadTimeout("slow", request=request)


def bad_format(request):
    return httpx2.Response(200, json={"unexpected": True})


@pytest.mark.parametrize(
    ("handler", "error", "expected_calls"),
    [
        (status(401), LLMServiceError, 1),
        (status(400), LLMServiceError, 1),
        (bad_format, LLMServiceError, 1),
        (status(429), LLMRetryableError, 3),
        (status(503), LLMRetryableError, 3),
        (timeout, LLMRetryableError, 3),
    ],
)
def test_only_transient_errors_are_retried(
    orchestrator, fake_llm, handler, error, expected_calls
):
    fake_llm(handler)
    with pytest.raises(error):
        asyncio.run(orchestrator.process_query("x", "u", "s", "t"))
    assert fake_llm.calls["n"] == expected_calls


def test_retry_recovers_after_transient_error(orchestrator, fake_llm):
    responses = iter([httpx2.Response(503), httpx2.Response(200, json=OK_BODY)])
    fake_llm(lambda request: next(responses))
    response, cost = asyncio.run(orchestrator.process_query("x", "u", "s", "t"))
    assert response == "hi"
    assert cost == pytest.approx(0.00075)
    assert fake_llm.calls["n"] == 2


def test_error_keeps_original_cause(orchestrator, fake_llm):
    fake_llm(status(401))
    with pytest.raises(LLMServiceError) as exc_info:
        asyncio.run(orchestrator.process_query("x", "u", "s", "t"))
    assert isinstance(exc_info.value.__cause__, httpx2.HTTPStatusError)


def test_request_uses_max_completion_tokens(orchestrator, fake_llm):
    fake_llm(lambda r: httpx2.Response(200, json=OK_BODY))
    asyncio.run(orchestrator.process_query("x", "u", "s", "t"))
    sent = fake_llm.calls["requests"][0].content
    assert b'"max_completion_tokens"' in sent
    assert b'"max_tokens"' not in sent
