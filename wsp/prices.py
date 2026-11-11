"""Rough cost figures for API-key runs. Every number shown from here is an estimate; providers change their prices.

Prices are dollars per million tokens, from the providers' pricing pages as read on 2026-11-08. A model that isn't listed gets
no estimate rather than a wrong one. Subscription runs show tokens only: they aren't billed per token.
"""
from __future__ import annotations

# cache_write: Claude at the 1-hour rate the anthropic backend uses; OpenAI at the default 30-minute rate
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 8.0},
    "gpt-6-astra": {"input": 10.0, "output": 50.0, "cache_read": 1.00, "cache_write": 12.5},
    "gpt-6.1-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.10, "cache_write": 2.5},
    "gpt-6.1-terra": {"input": 1.0, "output": 5.0, "cache_read": 0.05, "cache_write": 1.25},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}

# Tokens per forward pass and time per pass in this project's runs (Opus 5.5 at high effort; results/evidence/), for the
# estimate before a run. Time is on an RTX 4070 SUPER; another model will write more or less than this.
TYPICAL = {
    "edge_backdoor": {"output": 4_250, "cache_read": 126_000, "cache_write": 6_700, "minutes": 1},
    "qwen_math": {"output": 7_900, "cache_read": 168_000, "cache_write": 48_800, "minutes": 60},
}


def cost_usd(model: str | None, tokens: dict) -> float | None:
    """Estimated cost of token counts as kept in a run's state.json. None if the model's price isn't known."""
    p = PRICES.get(model or "")
    if not p or not tokens:
        return None
    return sum(tokens.get(k, 0) * p[k] for k in ("input", "output", "cache_read", "cache_write")) / 1e6


def estimate(task: str, budget: int, model: str | None) -> dict:
    """-> {"usd": float or None, "minutes": int}: what a run might take, going by this project's own runs."""
    t = TYPICAL.get(task)
    if not t:
        return {"usd": None, "minutes": None}
    usd = cost_usd(model, {k: t[k] * budget for k in ("output", "cache_read", "cache_write")})
    return {"usd": usd, "minutes": t["minutes"] * budget}
