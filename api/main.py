# ./api/main.py
import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from billing.cost_tracker import CostTracker, load_prices
from orchestrator.core import AgentOrchestrator, LLMServiceError

# --- Configuration ---
LLM_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Logging Setup --
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Mainspring API",
    description="API gateway for Mainspring, an orchestration framework for agentic AI.",
    version="0.1.0",
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


# --- Global Cost Tracker ---
prices = load_prices()
if LLM_MODEL not in prices:
    raise RuntimeError(f"LLM_MODEL {LLM_MODEL!r} has no entry in billing/prices.json")
cost_tracker = CostTracker(prices)

# --- Agent Orchestrator Initialization ---
# This would typically be dependency injected or managed by a factory
agent_orchestrator = AgentOrchestrator(
    llm_api_key=LLM_API_KEY,
    llm_endpoint=LLM_ENDPOINT,
    llm_model=LLM_MODEL,
    cost_tracker=cost_tracker,
)


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "message": "Agentic AI System is operational"}


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
        agent_response, total_cost = await agent_orchestrator.process_query(
            request.query, request.user_id, request.session_id, trace_id
        )
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"[{trace_id}] Query processed in {duration_ms:.2f}ms with total cost ${total_cost:.6f}"
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
