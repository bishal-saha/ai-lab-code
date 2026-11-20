"""Per-model prices (USD per million tokens) used to cost each episode.

First-party Claude API list prices as of November 2026. Cache writes are billed
at 1.25x input (5-minute TTL) and cache reads at 0.1x input. Check the current
price list before quoting costs: https://platform.claude.com/docs/en/about-claude/pricing
"""

from __future__ import annotations

PRICES = {
    # model: (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "fake": (0.0, 0.0),
}
ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}


def resolve(model: str) -> str:
    return ALIASES.get(model, model)


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0
) -> float:
    inp, out = PRICES[resolve(model)]
    total = (
        input_tokens * inp + output_tokens * out + cache_read * inp * 0.1 + cache_write * inp * 1.25
    ) / 1e6
    return round(total, 6)
