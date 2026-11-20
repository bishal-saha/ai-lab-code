# Tool Shape Benchmark: Endpoint Wrappers vs Task-Shaped MCP Tools

> Code and raw results for **[Tool Shape Benchmark](https://iambishal.com/ai-lab/projects/mcp-tool-shape-benchmark)** in the AI Lab on iambishal.com.
> It puts numbers on the argument in **[Stop Wrapping Endpoints as Tools](https://iambishal.com/resources/articles/agent-tool-design-task-shaped-tools-not-endpoints)**.

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.11 · mcp 2.3 · kestrel-outfitters 0.3.0 |
| Run | `uv run tool-shape --dry-run` (offline) · `uv run tool-shape` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

Most MCP servers start life as a wrapper around an existing REST API: one tool per endpoint, raw JSON out. The alternative is a few tools shaped around the jobs the agent actually does, which join records, apply business rules and answer in plain language. Does the shape of the tools change how well, how cheaply and how reliably an agent does the work?

**Hypothesis, written before the first live run:** task-shaped tools give a higher task success rate with fewer tool calls and fewer tokens, and the gap is wider on a smaller model.

## The setup

- **Backend:** the Kestrel Outfitters mock API from the [Agentic AI book series](https://iambishal.com/resources/books) (`kestrel-outfitters` 0.3.0, seed 2026, 10% scale: 2,000 orders, 1,800 shipments, 300 tickets, 40 products), started in-process for every run.
- **Server A, `endpoints`** ([`servers/endpoints.py`](src/tool_shape/servers/endpoints.py)): 8 tools, one per API route, returning the API's JSON. Descriptions are accurate and well written; this is a good endpoint wrapper, not a strawman.
- **Server B, `tasks`** ([`servers/tasks.py`](src/tool_shape/servers/tasks.py)): 5 tools (`order_status`, `check_return_eligibility`, `ticket_context`, `stock_check`, `read_policy`) over **the same API client**, so neither design can see data the other can't.
- **Tasks** ([`tasks.py`](src/tool_shape/tasks.py)): 30 customer-support questions in five families: delivery status (6), return eligibility and refund approval (8), ticket triage (8), stock (4) and policy facts (4). Some carry a trap: a trade customer's 14-day window, or the outdated shipping policy that still sits in the policy list.
- **Ground truth** is computed straight from the dataset and the written policies, never through either server. A test checks that every task-shaped tool agrees with it, so a buggy tool can't win or lose the benchmark.
- **Grading** is code only: the agent submits typed fields through a strict `submit_answer` tool and [`grading.py`](src/tool_shape/grading.py) compares them exactly. No LLM judge.
- **Agent** ([`agent.py`](src/tool_shape/agent.py)): one loop for both designs: the same system prompt, at most 12 turns, one reminder if the model stops without answering, and a circuit breaker (the same failing call twice, or three errors in a row).
- **Models:** Claude Haiku 5.5 and Claude Sonnet 5.5, adaptive thinking, effort `medium`, prompt caching on.
- **Matrix:** 30 tasks × 2 designs × 2 models × 3 repeats = 360 episodes.

## Run it

```bash
uv sync
uv run pytest                       # servers, ground truth, grader, agent loop; offline
uv run tool-shape --dry-run         # the whole matrix with a scripted fake model; $0
uv run tool-shape --estimate        # what a live run should cost, before you spend anything

cp .env.example .env                # add ANTHROPIC_API_KEY
uv run --env-file .env tool-shape --models haiku,sonnet --repeats 3 --max-usd 20
uv run python -m tool_shape.report  # summary.csv, by_family.csv and charts/ from the episodes
```

`--max-usd` is a hard stop. Episodes are appended to `results/episodes.jsonl` as they finish, so re-running the same command after a stop picks up where it left off. Use `--tasks returns,ticket-3` to run a subset.

### A fresh live run

The runner skips any episode already in `results/episodes.jsonl`; that's how resume works. The published episodes are committed there, so the live command above does nothing until you move them aside:

```bash
mv results results-published
uv run --env-file .env tool-shape --models haiku --repeats 1 --tasks delivery --max-usd 0.5   # a few cents: one family, Haiku
uv run --env-file .env tool-shape --models haiku,sonnet --repeats 3 --max-usd 20             # the published matrix, about $1
uv run python -m tool_shape.report
```

Compare your `results/summary.csv` with `results-published/summary.csv`. Expect small differences: models don't give identical answers on every run.

## Learning the code

Read `src/tool_shape/` in this order; each file builds on the one before:

1. **[`tasks.py`](src/tool_shape/tasks.py)**: the 30 tasks and their ground truth, computed from the dataset, never through a server. `_returns` holds the returns-policy logic.
2. **[`servers/endpoints.py`](src/tool_shape/servers/endpoints.py)**: design A, one tool per API route.
3. **[`servers/tasks.py`](src/tool_shape/servers/tasks.py)**: design B, five tools that join records and apply the rules. Find the in-transit branch of `order_status`: the one fact it leaves out explains every task-shaped miss in [RESULTS.md](RESULTS.md).
4. **[`agent.py`](src/tool_shape/agent.py)**: the loop that both designs share: call Claude, run the tools it asks for, nudge once, stop on `submit_answer` or the circuit breaker.
5. **[`grading.py`](src/tool_shape/grading.py)**: exact comparison of the submitted fields.
6. **[`run.py`](src/tool_shape/run.py) → [`report.py`](src/tool_shape/report.py)**: the matrix runner (dry run, estimate, cost cap, resume), then the tables and charts.

Then change something. Each of these takes a few lines, and `--dry-run` or `pytest` shows the effect in seconds:

- **Fix the miss:** make `order_status` say "not late yet" for parcels in transit, then re-run the delivery family live (`--tasks delivery`, a few cents) and see whether Sonnet stops answering *late*.
- **Add a task:** write one in `tasks.py` with its ground truth; the test that checks every task-shaped tool against the ground truth tells you if your tool disagrees.
- **Make the endpoint wrapper worse:** shorten the endpoint descriptions to one line each and measure what good descriptions were worth.

## Results

**In short:** task-shaped tools halved tool calls and input tokens and cut the cost per correct answer by 1.5–2.6×, but they did *not* raise accuracy: endpoint wrappers scored 100% on both models, task-shaped tools 99% (Haiku) and 93% (Sonnet), every miss traced to one fact the task-shaped tool left out. Full write-up in [RESULTS.md](RESULTS.md); raw episodes, tables and charts in [`results/`](results).

## What's here

| Path | What it is |
|---|---|
| `src/tool_shape/servers/` | the two MCP servers |
| `src/tool_shape/tasks.py`, `grading.py` | the task set, ground truth and grader |
| `src/tool_shape/agent.py`, `models.py` | the agent loop, the Claude client and the offline fake |
| `src/tool_shape/run.py`, `report.py`, `pricing.py` | the matrix runner, the report and per-model prices |
| `tests/` | everything above, offline |

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
