# Four Shapes, One Ticket: Agent Architecture Patterns, Measured

> Code and raw results for **[Four Shapes, One Ticket](https://iambishal.com/ai-lab/projects/agent-architecture-patterns-benchmark)** in the AI Lab on iambishal.com.
> It measures the comparison table in **[AI Agent Architecture Patterns](https://iambishal.com/resources/articles/ai-agent-architecture-patterns)**, whose ratings say: *"my judgement for this scenario, not measurements."*

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.11 · mcp 2.3 · kestrel-outfitters 0.3.0 |
| Run | `uv run four-shapes --dry-run` (offline) · `uv run four-shapes` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

One support assistant, four ways to build it: a single tool-calling loop, a graph workflow, a planner-executor and a multi-agent supervisor. They have the same tools, the same tickets, the same policy text and the same models. **Who owns the next step** (the model or your code) is the only thing that changes. How does each design succeed, fail and cost?

**Hypothesis, written before the first live run** (from the article's table):

1. **The graph workflow is the most predictable:** the most consistent actions across repeats, and the fewest policy-order violations (a refund before the eligibility check).
2. **The graph workflow is the weakest on unexpected input:** on two-intent tickets it misses the second intent more often than the single loop.
3. **A plan needs validation:** the planner proposes invalid plans (a refund before the check, a tool that doesn't exist) at a measurable rate, and validation plus replanning recovers most of them.
4. **Multi-agent costs the most** (LLM calls, tokens, latency) and brings its own failures: lost handoff context and duplicate replies.
5. **On ordinary refund tickets, all four reach the policy outcome** within 5 points of each other. The designs differ in cost and failure modes, not basic correctness.

## The setup

- **Backend:** the Kestrel Outfitters mock API from the [Agentic AI book series](https://iambishal.com/resources/books) (`kestrel-outfitters` 0.3.0, seed 2026, 10% scale), started in-process.
- **Tickets** ([`scenarios.py`](src/four_shapes/scenarios.py)): 16 tickets over real orders, each with its correct outcome computed from the data.

  | Kind | n | Correct outcome |
  |---|---|---|
  | Refund policy | 8 | refund (3), escalate to Finance (2), decline (3, one a trade customer's 14-day window) |
  | Two intents | 4 | the policy outcome **and** `route_to_engineering` for a website bug in the same ticket |
  | Where is my order? | 2 | look the order up, reply; no money moves (one delivered, one in transit) |
  | No order id | 2 | find the order with `customer_orders`, then the policy outcome |

- **Server** ([`servers/support.py`](src/four_shapes/servers/support.py)): one MCP tool server for every design. Every call, reads included, goes to an in-memory **ledger** in order.
- **Conditions:** `normal`, and `faults`, where the first `check_return_eligibility` call fails with a retryable "service unavailable".
- **The four designs** ([`architectures/`](src/four_shapes/architectures)):

  | | Who owns the next step | Built as |
  |---|---|---|
  | [`loop`](src/four_shapes/architectures/loop.py) | the model | one tool-calling loop with every tool |
  | [`graph`](src/four_shapes/architectures/graph.py) | code | classify (one intent, with a fallback edge to a human) → resolve order → policy decision in code → act → LLM-drafted reply |
  | [`planner`](src/four_shapes/architectures/planner.py) | model plans, code runs | a JSON plan validated in code (registered tools only, eligibility before money); an executor fills each step; errors go back for a replan (at most 2) |
  | [`multi`](src/four_shapes/architectures/multi.py) | supervisor + specialists | a read-only triage agent hands **structured fields checked in code** to a billing agent (the only one that can refund) and a technical agent |

  All four get the same policy text ([`episode.py`](src/four_shapes/episode.py) `POLICY`) and run under the same hard limits: 30 LLM calls per ticket and a circuit breaker on repeated failures.
- **Grading** ([`grading.py`](src/four_shapes/grading.py)) is code only, from the ledger. It records success, money moved before the eligibility check, a missed second intent, a handoff to a human, duplicate replies, and the action sequence, which gives predictability across repeats. No LLM judge.
- **Models:** Claude Haiku 5.5 and Claude Sonnet 5.5, adaptive thinking, effort `medium`, prompt caching on. Structured outputs (the graph's classifier, the planner, the triage handoff) use a JSON schema.
- **Matrix:** 16 tickets × 4 designs × 2 conditions × 2 models × 3 repeats = 768 episodes.

A note on fairness: the graph's classifier picks **one** intent, as the article's design does. On a two-intent ticket it can only take one path, so its success there is capped by design. What those tickets measure for the graph is *how* it fails (the wrong path silently, or the human fallback), and whether the other three designs do better.

## Run it

```bash
uv sync
uv run pytest                          # tickets, server, grader, plan validation, every design end to end; offline
uv run four-shapes --dry-run           # 256 scripted episodes: an "oracle" and a "sloppy" fake on every design; $0
uv run four-shapes --estimate          # what a live run should cost, before you spend anything

cp .env.example .env                   # add ANTHROPIC_API_KEY
uv run --env-file .env four-shapes --repeats 3 --max-usd 15
uv run python -m four_shapes.report    # summary.csv, by_condition.csv, by_kind.csv, charts/
```

`--max-usd` is a hard stop, and re-running the same command resumes from `results/episodes.jsonl`. Subsets: `--scenarios multi,p1`, `--archs loop,graph`, `--conditions faults`, `--models haiku`.

The dry run proves the grading before any money is spent:
- the oracle reaches every ground truth on every design, faults included;
- the sloppy fake's refund-before-check and dropped bug report are flagged;
- its invented-tool plan is rejected and then recovered by replanning;
- the graph can't be made to move money before checking.

### A fresh live run

The runner skips any episode already in `results/episodes.jsonl`; that's how resume works. Once published episodes are committed there, move them aside first:

```bash
mv results results-published
uv run --env-file .env four-shapes --models haiku --repeats 1 --scenarios p1,m1,s1,n1 --max-usd 0.5   # a few cents
uv run --env-file .env four-shapes --repeats 3 --max-usd 15                                         # the full matrix
uv run python -m four_shapes.report
```

Compare your `results/summary.csv` with `results-published/summary.csv`. Expect small differences: models don't give identical answers on every run.

## Learning the code

Read `src/four_shapes/` in this order; each file builds on the one before:

1. **[`scenarios.py`](src/four_shapes/scenarios.py)**: the 16 tickets and how each correct outcome is computed from the data.
2. **[`servers/support.py`](src/four_shapes/servers/support.py)** and **[`ledger.py`](src/four_shapes/ledger.py)**: the tools every design shares, and the ordered record of every call.
3. **[`episode.py`](src/four_shapes/episode.py)**: the shared `Context` (`llm`, `json`, `call`, `tool_loop`) that counts every call the same way for every design.
4. **[`architectures/loop.py`](src/four_shapes/architectures/loop.py)**: the whole single loop is one line on top of `tool_loop`.
5. **[`architectures/graph.py`](src/four_shapes/architectures/graph.py)**: the same job with code deciding every transition. Compare its policy decision with the loop's prompt.
6. **[`architectures/planner.py`](src/four_shapes/architectures/planner.py)**: `validate()` is the article's rule set in a dozen lines.
7. **[`architectures/multi.py`](src/four_shapes/architectures/multi.py)**: the structured handoff, and the code check that drops an order id that isn't the customer's.
8. **[`grading.py`](src/four_shapes/grading.py) → [`run.py`](src/four_shapes/run.py) → [`report.py`](src/four_shapes/report.py)**: what counts as success and failure, the matrix runner, then the tables and charts.

Then change something. Each of these takes a few lines, and `--dry-run` or `pytest` shows the effect in seconds:

- **Give the graph two intents:** let the classifier return a list, and add an engineering branch. Re-run the two-intent tickets (`--scenarios multi --archs graph`) and see what one design change buys.
- **Turn off plan validation:** make `validate()` return `[]` and run the planner live on a few tickets. Count the violations it now lets through.
- **Leak the ticket:** pass the raw ticket text into the billing agent's handoff in `multi.py`, and think about what an injected instruction could now reach (see the [Poisoned Ticket](../prompt-injection-defence-benchmark) experiment).

## Results

See [RESULTS.md](RESULTS.md) once the live run is in. The raw episodes, tables and charts will be in [`results/`](results).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
