import pytest

from billing.cost_tracker import CostTracker, load_prices


def test_cost_uses_per_million_prices():
    tracker = CostTracker(load_prices())
    assert tracker.record_llm_usage("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_total_cost_accumulates():
    tracker = CostTracker({"m": {"input_per_m": 1.0, "output_per_m": 2.0}})
    tracker.record_llm_usage("m", 1_000_000, 0)
    tracker.record_llm_usage("m", 0, 1_000_000)
    assert tracker.total_cost == pytest.approx(3.0)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        CostTracker({}).record_llm_usage("nope", 1, 1)
