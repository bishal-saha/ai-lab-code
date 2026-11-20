"""Single tool-calling loop (ReAct-style): one model, one context, every tool.

The model reads each result and picks the next call until it decides it is done.
Quickest to build; nothing but the model decides the order of steps.
"""

from __future__ import annotations

from four_shapes.episode import POLICY, Context

SYSTEM = f"""You are a customer-support agent for Kestrel Outfitters.

{POLICY}

Work the ticket with the tools, then finish with a one-line summary of what you did."""


async def run(ctx: Context) -> None:
    await ctx.tool_loop("loop", SYSTEM, ctx.s.prompt)
