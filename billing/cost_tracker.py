import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PRICES_PATH = Path(__file__).with_name("prices.json")


def load_prices(path: Path = PRICES_PATH) -> dict[str, dict[str, float]]:
    """Load per-model prices in USD per 1M tokens from a JSON file."""
    return json.loads(path.read_text())


class CostTracker:
    """Track LLM spend using per-model prices loaded from a file."""

    def __init__(self, prices: dict[str, dict[str, float]]):
        self.prices = prices
        self.total_cost = 0.0

    def record_llm_usage(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Return the USD cost of one LLM call and add it to the running total."""
        price = self.prices[model]
        cost = (
            input_tokens * price["input_per_m"] + output_tokens * price["output_per_m"]
        ) / 1_000_000
        self.total_cost += cost
        logger.info(
            "LLM usage recorded: model=%s, input=%d, output=%d, cost=$%.6f",
            model, input_tokens, output_tokens, cost,
        )
        return cost
