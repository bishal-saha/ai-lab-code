"""The article's planning model, vendored verbatim, plus a fit to measured runs.

Everything from `Run` to `cost_per_ticket` (and `ARTICLE_SCENARIOS`) is copied unchanged from
`ticket_cost.py` in "AI Agent Cost per Task" (https://iambishal.com/resources/articles/ai-agent-cost-per-task),
where every default is an illustrative input. tests/ checks it reproduces the article's printed
output line for line.

Below the copy: `fit()` fills the model's inputs from measured attempts (the article's advice:
"fill the model with ten real tickets before you commit"), and `predict()` maps each experiment
condition onto the model's own levers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean, median

# ---------------------------------------------------------------------------------------------
# Verbatim from the article (ticket_cost.py)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One agent attempt on one ticket. Every default is an illustrative input."""

    steps: int = 12
    prefix_tokens: int = 5_000  # system prompt, tool definitions, policy
    tokens_added_per_step: int = 1_500  # last output plus tool result
    output_tokens_per_step: int = 400
    tool_calls_per_step: int = 1
    tool_cost_per_call: float = 0.01
    price_in: float = 2.00  # $ per million tokens, strong model
    price_out: float = 10.00
    cheap_in: float = 0.40  # $ per million tokens, cheap model
    cheap_out: float = 2.00
    caching: bool = True
    write_mult: float = 1.25  # 5-minute cache write
    read_mult: float = 0.10
    history_cap: int | None = None  # keep only the last N chunks word for word
    summary_tokens: int = 600  # running summary once the cap bites
    compaction_out: int = 300  # output spent keeping the summary current
    cheap_first: int = 0  # phase routing: first N steps on cheap model
    cheap_share: float = 0.0  # per-step switching: share on cheap model


@dataclass(frozen=True)
class Outcomes:
    p_resolve: float = 0.75  # attempt resolves the ticket
    p_handoff: float = 0.05  # attempt hands off cleanly; the rest fail
    fail_cost_ratio: float = 1.5  # assumption: failed attempts run longer
    max_attempts: int = 2
    handoff_cost: float = 6.00  # human time after a clean handoff
    abandon_cost: float = 9.00  # human time starting from scratch


def step_prices(r: Run, i: int) -> tuple[float, float]:
    if r.cheap_share:  # switching every step: blend the two prices
        s = r.cheap_share
        return ((1 - s) * r.price_in + s * r.cheap_in, (1 - s) * r.price_out + s * r.cheap_out)
    if i < r.cheap_first:
        return r.cheap_in, r.cheap_out
    return r.price_in, r.price_out


def attempt_cost(r: Run) -> float:
    # History stays cached only if each call extends that model's previous call.
    append_only = r.caching and r.history_cap is None and not r.cheap_share
    total = 0.0
    for i in range(r.steps):  # step i carries i chunks of history
        kept = i if r.history_cap is None else min(i, r.history_cap)
        history = kept * r.tokens_added_per_step
        out = r.output_tokens_per_step
        if kept < i:  # older chunks folded into a running summary
            history += r.summary_tokens
            out += r.compaction_out
        # A model's first call writes its cache; switching writes the prefix twice.
        fresh = i in (0, r.cheap_first) or (r.cheap_share > 0 and i == 1)
        if append_only and not fresh:  # read last prompt, write the new chunk
            new = r.tokens_added_per_step
            billed_in = (r.prefix_tokens + history - new) * r.read_mult + new * r.write_mult
        elif append_only:  # empty cache on this model: write the whole prompt
            billed_in = (r.prefix_tokens + history) * r.write_mult
        elif r.caching:  # only the prefix is cached; history at full price
            billed_in = r.prefix_tokens * (r.write_mult if fresh else r.read_mult) + history
        else:
            billed_in = r.prefix_tokens + history
        p_in, p_out = step_prices(r, i)
        total += (billed_in * p_in + out * p_out) / 1e6
    return total + r.steps * r.tool_calls_per_step * r.tool_cost_per_call


def cost_per_ticket(attempt: float, o: Outcomes) -> tuple[float, float]:
    p_fail = 1 - o.p_resolve - o.p_handoff
    attempts = sum(p_fail**j for j in range(o.max_attempts))  # expected attempts
    agent = attempts * attempt * ((1 - p_fail) + p_fail * o.fail_cost_ratio)
    human = attempts * o.p_handoff * o.handoff_cost + p_fail**o.max_attempts * o.abandon_cost
    return agent + human, attempts * o.p_resolve  # all-in cost, share agent resolves


_base, _odds = Run(), Outcomes()
ARTICLE_SCENARIOS = [
    ("baseline (append-only cache)", _base, _odds),
    ("no prompt caching", replace(_base, caching=False), _odds),
    ("history cap 4", replace(_base, history_cap=4), _odds),
    ("phase routing: first 6 steps cheap", replace(_base, cheap_first=6), _odds),
    ("per-step switching, 50% cheap", replace(_base, cheap_share=0.5), _odds),
    (
        "per-step switching, 50% cheap, no caching",
        replace(_base, cheap_share=0.5, caching=False),
        _odds,
    ),
    ("18 steps instead of 12", replace(_base, steps=18), _odds),
    ("tool fee $0.02 per call", replace(_base, tool_cost_per_call=0.02), _odds),
    ("resolve 65%, fail 30%", _base, replace(_odds, p_resolve=0.65)),
    ("3 attempts instead of 2", _base, replace(_odds, max_attempts=3)),
]


def article_output() -> list[str]:
    out = []
    for name, run, o in ARTICLE_SCENARIOS:
        attempt = attempt_cost(run)
        per_ticket, by_agent = cost_per_ticket(attempt, o)
        out.append(
            f"attempt ${attempt:.3f}  per ticket ${per_ticket:.3f}  agent resolves {by_agent:.1%}  {name}"
        )
    return out


# ---------------------------------------------------------------------------------------------
# Filling the model from measured attempts
# ---------------------------------------------------------------------------------------------

# This experiment's prices (USD per million tokens): pricing.py's table.
SONNET = (2.00, 10.00)
HAIKU = (0.10, 0.50)


def fit(baseline_attempts: list[dict]) -> Run:
    """Fill the model's inputs from first attempts under the cached baseline. Each attempt row has
    `calls` (one per model call, in order, with window_tokens and output_tokens) and `system_tokens`
    (the size of the tools + system prompt, counted once)."""
    steps = round(median(len(a["calls"]) for a in baseline_attempts))
    first_windows = [a["calls"][0]["window_tokens"] for a in baseline_attempts]
    added = [
        b["window_tokens"] - c["window_tokens"]
        for a in baseline_attempts
        for c, b in zip(a["calls"], a["calls"][1:], strict=False)
    ]
    out = [c["output_tokens"] for a in baseline_attempts for c in a["calls"]]
    return Run(
        steps=steps,
        prefix_tokens=round(mean(first_windows)),  # tools, system and the ticket prompt
        tokens_added_per_step=round(mean(added)),
        output_tokens_per_step=round(mean(out)),
        tool_calls_per_step=1,
        tool_cost_per_call=0.0,  # internal tools
        price_in=SONNET[0],
        price_out=SONNET[1],
        cheap_in=HAIKU[0],
        cheap_out=HAIKU[1],
    )


def predict(condition: str, base: Run) -> float:
    """The model's attempt cost for a condition, using only the model's own levers."""
    if condition == "cache":
        r = base
    elif condition == "no-cache":
        r = replace(base, caching=False)
    elif condition == "cache-break":
        # No lever for it. The article says a break "forces a rewrite at the write premium":
        # model it as no caching with every input token billed at the write multiplier.
        r = replace(base, caching=False, price_in=base.price_in * base.write_mult)
    elif condition == "cap":
        r = replace(base, history_cap=2)
    elif condition == "haiku":
        r = replace(base, price_in=base.cheap_in, price_out=base.cheap_out)
    elif condition == "phase":
        r = replace(base, cheap_first=2)
    elif condition == "switch":
        r = replace(base, cheap_share=0.5)
    else:
        raise ValueError(condition)
    return attempt_cost(r)
