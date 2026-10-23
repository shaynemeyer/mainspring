# ./api/main.py
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from billing.cost_tracker import CostTracker, load_prices
from orchestrator.core import AgentOrchestrator, LLMServiceError

# --- Logging Setup --
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_orchestrator() -> AgentOrchestrator:
    """Create the orchestrator from environment config, failing fast if invalid."""
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prices = load_prices()
    if model not in prices:
        raise RuntimeError(f"LLM_MODEL {model!r} has no entry in billing/prices.json")
    return AgentOrchestrator(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        llm_endpoint=os.getenv(
            "LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
        ),
        llm_model=model,
        cost_tracker=CostTracker(prices),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build shared objects at startup and close the HTTP client at shutdown."""
    app.state.agent_orchestrator = build_orchestrator()
    yield
    await app.state.agent_orchestrator.aclose()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Mainspring API",
    description=(
        "API gateway for Mainspring, an orchestration framework for agentic AI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---Models---
class AgentRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default_session"


class AgentResponse(BaseModel):
    response: str
    cost_usd: float
    duration_ms: float
    trace_id: str


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "message": "Mainspring is operational"}


# --- Main Agent Endpoint ---
@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest, http_request: Request):
    start_time = time.perf_counter()
    trace_id = http_request.headers.get("X-Trace-ID") or f"trace-{uuid.uuid4().hex}"
    logger.info(
        f"[{trace_id}] Received query for user {request.user_id} "
        f"({len(request.query)} chars)"
    )

    try:
        # Orchestrate the agent's response
        orchestrator = http_request.app.state.agent_orchestrator
        agent_response, total_cost = await orchestrator.process_query(
            request.query, request.user_id, request.session_id, trace_id
        )
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"[{trace_id}] Query processed in {duration_ms:.2f}ms "
            f"with total cost ${total_cost:.6f}"
        )
        return AgentResponse(
            response=agent_response,
            cost_usd=total_cost,
            duration_ms=duration_ms,
            trace_id=trace_id,
        )
    except LLMServiceError as e:
        logger.error(f"[{trace_id}] LLM service failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM service error",
        ) from e
    except Exception as e:
        logger.exception(
            f"[{trace_id}] An unexpected error occurred during query processing."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from e
