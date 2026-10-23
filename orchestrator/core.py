import logging

import httpx2
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from billing.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""


class LLMRetryableError(LLMServiceError):
    """Transient LLM error worth retrying: timeout, network error, 429 or 5xx."""


def _is_retryable_status(status_code: int) -> bool:
    """Return True for rate limiting and server-side errors."""
    return status_code == 429 or status_code >= 500


class AgentOrchestrator:
    """Answer user queries with a single LLM call and track its cost."""

    def __init__(
        self,
        llm_api_key: str,
        llm_endpoint: str,
        llm_model: str,
        cost_tracker: CostTracker,
    ):
        self.llm_api_key = llm_api_key
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.cost_tracker = cost_tracker
        self.http_client = httpx2.AsyncClient(
            timeout=30.0
        )  # Global client with timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(LLMRetryableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_llm(
        self, messages: list[dict[str, str]], trace_id: str
    ) -> tuple[str, int, int]:
        """Calls the LLM API with retry logic and timeout."""
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.llm_model,
            "messages": messages,
            # TODO: OpenAI reasoning models may reject temperature; drop or make it
            # per-model if LLM_MODEL changes from gpt-4o-mini.
            "temperature": 0.7,
            "max_completion_tokens": 500,
        }

        logger.info(f"[{trace_id}] Calling LLM with model {self.llm_model}")

        try:
            response = await self.http_client.post(
                self.llm_endpoint, json=payload, headers=headers
            )
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses

            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"]
            input_tokens = response_data["usage"]["prompt_tokens"]
            output_tokens = response_data["usage"]["completion_tokens"]

            logger.info(
                f"[{trace_id}] LLM call successful. Input tokens: {input_tokens}, Output tokens: {output_tokens}"
            )

            return content, input_tokens, output_tokens

        except httpx2.TimeoutException as e:
            logger.error(f"[{trace_id}] LLM API call timed out.")
            raise LLMRetryableError("LLM API call timed out.") from e
        except httpx2.RequestError as e:
            logger.error(
                f"[{trace_id}] An error occurred while requesting LLM API: {e}"
            )
            raise LLMRetryableError(f"LLM API request failed: {e}") from e
        except httpx2.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(
                f"[{trace_id}] LLM API returned an error status {status_code}: {e.response.text}"
            )
            if _is_retryable_status(status_code):
                raise LLMRetryableError(f"LLM API error: {status_code}") from e
            raise LLMServiceError(f"LLM API error: {status_code}") from e
        except KeyError as e:
            logger.error(
                f"[{trace_id}] Unexpected LLM response format: {e}. Response: {response.text}"
            )
            raise LLMServiceError("Unexpected LLM response format.") from e

    async def process_query(
        self, query: str, user_id: str, session_id: str, trace_id: str
    ) -> tuple[str, float]:
        """Processes a user query using a basic ReAct-like orchestration."""
        total_cost_for_query = 0.0

        # Step 1: Initial planning/response generation
        system_prompt = (
            "You are a helpful AI assistant. Respond concisely and directly to the user's query. "
            "If you don't know the answer, state that you don't know. "
            "Your goal is to provide a single, direct answer or action. "
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        llm_response_content, input_tokens, output_tokens = await self._call_llm(
            messages, trace_id
        )
        total_cost_for_query += self.cost_tracker.record_llm_usage(
            self.llm_model, input_tokens, output_tokens
        )
        # For now, we assume the LLM's response is the final answer.
        # Later this would involve tool parsing, reflection, etc.
        return llm_response_content, total_cost_for_query
