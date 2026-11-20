"""Run one episode: one ticket, one architecture, one condition, one model.

Builds a fresh support server and ledger, hands the architecture a `Context`, and
grades the ledger afterwards. An architecture's early stop (a refusal, the LLM-call
budget, the circuit breaker) or a transport error ends the episode, never the run.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict

from mcp import Client

from four_shapes.architectures import graph, loop, multi, planner
from four_shapes.episode import Context, Episode, Model, Stop, anthropic_tool
from four_shapes.grading import grade
from four_shapes.kestrel_api import KestrelAPI
from four_shapes.ledger import Ledger
from four_shapes.pricing import cost_usd
from four_shapes.scenarios import Scenario
from four_shapes.servers.support import build_server

ARCHITECTURES: dict[str, Callable[[Context], Awaitable[None]]] = {
    "loop": loop.run,
    "graph": graph.run,
    "planner": planner.run,
    "multi": multi.run,
}
CONDITIONS = ["normal", "faults"]


async def run_episode(
    s: Scenario,
    arch: str,
    condition: str,
    api: KestrelAPI,
    customers: dict[str, list[str]],
    model: Model,
    repeat: int = 0,
) -> Episode:
    ep = Episode(
        scenario=s.id,
        kind=s.kind,
        outcome=s.outcome,
        arch=arch,
        condition=condition,
        model=model.name,
        repeat=repeat,
    )
    ledger = Ledger()
    server = build_server(api, s, ledger, customers, faults=condition == "faults")
    started = time.perf_counter()
    try:
        async with Client(server) as client:
            tools = [anthropic_tool(t) for t in (await client.list_tools()).tools]
            await ARCHITECTURES[arch](Context(s, model, client, ledger, ep, tools))
    except Stop as stop:
        ep.stop = stop.reason
        ep.error = str(stop)
    except Exception as err:  # an API or transport failure fails the episode, not the run
        ep.stop = "error"
        ep.error = f"{type(err).__name__}: {err}"
    ep.latency_s = round(time.perf_counter() - started, 3)

    g = grade(s, ledger)
    for key, value in asdict(g).items():
        setattr(ep, key, list(value) if key == "sequence" else value)
    ep.calls = ledger.to_list()
    ep.cost_usd = cost_usd(
        model.name, ep.input_tokens, ep.output_tokens, ep.cache_read_tokens, ep.cache_write_tokens
    )
    return ep
