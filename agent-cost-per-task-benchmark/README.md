# The Bill for One Ticket: Agent Cost per Task, Measured

> Code and raw results for **[The Bill for One Ticket](https://iambishal.com/ai-lab/projects/agent-cost-per-task-benchmark)** in the AI Lab on iambishal.com.
> It measures **[AI Agent Cost per Task](https://iambishal.com/resources/articles/ai-agent-cost-per-task)**, whose cost model is filled with numbers marked *"illustrative throughout"*, and which ends: *"Fill the model with ten real tickets before you commit."*

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.12 · kestrel-outfitters 0.3.0 |
| Run | `uv run bill-for-one-ticket --dry-run` (offline) · `uv run bill-for-one-ticket` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

The article argues that an agent's cost per task depends on its structure more than on its token price. It names the levers:
- steps, and the history re-sent at each one;
- prompt caching, and what silently breaks it;
- a capped history with a running summary;
- routing steps to a cheaper model;
- retries and handoffs, priced as human time.

It sets these out in a planning model, `ticket_cost.py`, and runs a sensitivity analysis on invented inputs.

This experiment runs the real support agent on 24 tickets under each lever, one at a time. It then fills the article's model with the measured numbers, as the article tells you to, and checks what the model gets right.

**Hypothesis, written before any live run** (from the article):

1. **History makes input grow faster than linearly:** one step's input × steps underestimates a real attempt's input.
2. **Prompt caching is the biggest token lever:** turning it off raises attempt cost more than any other structural change.
3. **A cache break is silent and expensive:** a timestamp at the top of the system prompt costs close to no caching at all, with no error, visible only in the cache fields. A prompt below the cache minimum caches nothing, with no error.
4. **Routing pays only when it keeps the cache:** phase routing is cheaper than the baseline; switching models every step costs more.
5. **A history cap loses on a short run:** it costs more per attempt than plain caching.
6. **A cheaper model is a weaker lever than structure:** per resolved ticket, all in, Haiku saves less than caching does. Haiku costs 1/20 of Sonnet here, against the article's illustrative 1/5, so this is a hard test.
7. **The planning model, filled with pilot numbers, predicts each condition's mean attempt cost within ±25%, and ranks the levers correctly.**

## The setup

- **Tickets** ([`scenarios.py`](src/bill/scenarios.py)): 24 support tickets over real Kestrel Outfitters orders (seed 2026):
  - 12 return requests (refund, escalate to Finance, or decline);
  - 4 returns that also report a website bug;
  - 4 delivery questions;
  - 4 returns without an order id.

  Every order is still `delivered`, or `shipped` for an in-transit delivery question. Experiment 7's builder let three tickets land on orders already returned.
- **One attempt** ([`agent.py`](src/bill/agent.py)): one ticket in a fresh window, at most 12 model calls. Lookups return the full record an untrimmed API would, about 1,060 tokens each ([`tools.py`](src/bill/tools.py)).
  - A failed attempt is retried once, as in the article's `max_attempts = 2`.
  - A handoff to a person is final.
  - A ticket that fails twice is abandoned.
- **Conditions** ([`conditions.py`](src/bill/conditions.py)), one lever changed at a time from a cached Sonnet baseline:

  | id | What changes |
  |---|---|
  | `cache` | Sonnet 5.5, append-only automatic prompt caching (the baseline) |
  | `no-cache` | caching off |
  | `cache-break` | a timestamp at the top of the system prompt, new on every call |
  | `cap` | keep the last 2 tool exchanges word for word, plus a running summary written by Haiku |
  | `haiku` | Haiku 5.5 for every call |
  | `phase` | the first 2 calls on Haiku, the rest on Sonnet |
  | `switch` | Haiku and Sonnet alternate on every call |

  In the routing conditions each model sees its own earlier thinking and not the other model's, so each model's history only ever grows by appending.
- **Grading** ([`grading.py`](src/bill/grading.py)): from the ledger of tool calls only. Each attempt is resolved, handed off, or failed.
- **Cost:** every model call is priced from the usage the API reports (input, cache writes, cache reads, output). The cap's summaries are counted.
  - **All-in cost per ticket** follows the article: every attempt plus human time, divided by tickets.
  - Human time uses the article's illustrative $6 per handoff and $9 per abandoned ticket. These are placeholders, labelled as such, and a tokens-only figure is always reported beside them.
- **The planning model** ([`planning_model.py`](src/bill/planning_model.py)): the article's `ticket_cost.py`, copied verbatim. A test checks it reproduces the article's ten printed lines.
  - `fit()` fills it from the cached baseline's first attempts: steps, prefix, tokens added per step, output per step, real prices, and a tool fee of 0, since these tools are internal.
  - `predict()` maps each condition onto the model's own levers.
- **Matrix:** 24 tickets × 7 conditions × 2 repeats = 336 tickets. Three were planned; the pilot cost twice the estimate. Effort `medium` on both models.

## Run it

```bash
uv sync
uv run pytest                                 # the vendored model, tickets, conditions, accounting; offline
uv run bill-for-one-ticket --dry-run          # a fake that bills like the API; $0
uv run bill-for-one-ticket --estimate         # a rough projection (it undercounts: the real run cost $9.09)

cp .env.example .env                          # add ANTHROPIC_API_KEY
uv run --env-file .env bill-for-one-ticket --cache-min-check      # a few cents
uv run --env-file .env bill-for-one-ticket --pilot --max-usd 2
uv run --env-file .env bill-for-one-ticket --max-usd 12           # resumes from results/tickets.jsonl
uv run python -m bill.report                                       # tables and charts/
```

The dry run proves the accounting. Its fake model reports usage from a small cache simulator, and every recorded cost must recompute exactly from that usage. The simulator must also bill the conditions in the expected order: no caching and the cache break above the baseline, Haiku and phase routing below.

The cap is checked before every model call and shared by all concurrent tickets. A ticket stopped by the cap isn't recorded, so the next run redoes it.

### A fresh live run

The committed results mean a live run has nothing left to do. Move them aside first:

```bash
mv results results-published
uv run --env-file .env bill-for-one-ticket --max-usd 12
uv run python -m bill.report
```

## Learning the code

1. **[`planning_model.py`](src/bill/planning_model.py):** the article's model, then `fit()` and `predict()`. Read `attempt_cost()` with the article open beside it.
2. **[`conditions.py`](src/bill/conditions.py):** seven levers in about 40 lines.
3. **[`agent.py`](src/bill/agent.py):** `build_window()` decides what each call sends, and `attempt()` is the loop.
4. **[`run.py`](src/bill/run.py) → [`report.py`](src/bill/report.py):** retries, outcomes, and the comparison with the model.

Then change something:

- **Price your own model:** edit `SONNET` and `HAIKU` in `planning_model.py` and `PRICES` in `pricing.py`, then rerun the report (free).
- **Your human costs:** the $6 and $9 are the article's placeholders. Put in your own and see how much of the per-ticket figure is labour.
- **Break the cache yourself:** move the timestamp to the end of the system prompt in `agent.py` and rerun `cache-break`. Where the change sits decides what is lost.

## Results

**A cheaper model was the biggest lever here, caching the next, and the article's model predicted the rest once given real step counts.** Haiku cut cost per ticket by 95% and resolved every ticket, as Sonnet did, so the failure costs the article's "weaker lever" argument rests on never arose. Turning caching off raised cost by 86%; a timestamp at the top of the system prompt raised it by 125%, silently. A history cap cost 73% more on these short tickets; phase routing saved 24%, and switching models every call saved 17% instead of costing more. Real input was 2.1x the one-step-times-steps estimate. Filled from an 8-ticket pilot, the article's planning model was within ±25% for 4 of 7 conditions, and for all 7 once averaged over the real step counts. A flaw found in the pilot (conditions sharing a cached prompt for the same ticket) is disclosed and fixed. Each claim is judged in [RESULTS.md](RESULTS.md).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
