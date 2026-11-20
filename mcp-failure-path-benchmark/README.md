# The Unhappy Path: A Production MCP Server's Failure Path, Measured

> Code and raw results for **[The Unhappy Path](https://iambishal.com/ai-lab/projects/mcp-failure-path-benchmark)** in the AI Lab on iambishal.com.
> It measures the failure-path advice in **[Build a Production MCP Server](https://iambishal.com/resources/articles/build-production-mcp-server)**, whose snippets *"are illustrative and haven't been run"*.

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.12 · mcp 2.x · kestrel-outfitters 0.3.0 |
| Run | `uv run unhappy-path --dry-run` (offline) · `uv run unhappy-path` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

A quickstart MCP server passes the inspector. The article argues that the trouble starts on the failure path, where a production server has to do more:
- return errors the model can act on, naming the field and the next step;
- keep schemas strict, because free strings "invite invented field names, and some guesses succeed at the wrong thing";
- say how many results there are, because "otherwise the model will reasonably treat the list as complete";
- make a resend after a dropped write safe, because "the model may simply call again, whether or not the comment was written".

Experiment 1 measured the article's tool-shape advice. This one builds its running example, an MCP server over a ticketing API, as a quickstart and as a production server, injects each failure on purpose, and measures what the agent does.

**Hypothesis, written before any live run** (from the article):

1. **The production server beats the quickstart** on task success across the six failure modes.
2. **Actionable errors drive recovery:** without them, success falls on the not-found, transient and permission tasks, and failed calls get repeated.
3. **Strict schemas stop silent wrong answers:** with free strings, the model invents enum values and reports the empty result as fact.
4. **Without a total, the model treats a truncated list as complete,** and undercounts.
5. **Without idempotency, a dropped write produces duplicates;** with server-derived keys it never does.
6. **A permission error isn't retried when the message says so;** an opaque one is.

## The setup

- **The ticketing API** ([`ticketing.py`](src/unhappy/ticketing.py)): Kestrel Outfitters' 300 support tickets in memory, with search, get and add-comment.
  - Every variant gets the same page-size cap of 20.
  - Comments honour an optional idempotency key.
  - A per-task `Faults` plan injects one failure deterministically: a read timeout, a rate limit, a write that lands and then times out, or a ticket locked against comments.
- **Six real MCP servers** ([`servers.py`](src/unhappy/servers.py), Python SDK v2, driven through the in-process client). All have the same task-shaped tools (`search_tickets`, `get_ticket`, `add_ticket_comment`, plus `submit_answer`) with the same descriptions. Four practices are switched on or off:

  | id | strict schema | actionable errors | totals and cursors | idempotency keys |
  |---|---|---|---|---|
  | `quickstart` | no | no ("Request failed.") | no | no |
  | `production` | yes | yes | yes | yes |
  | `prod-no-errors` | yes | **no** | yes | yes |
  | `prod-no-schema` | **no** | yes | yes | yes |
  | `prod-no-pages` | yes | yes | **no** | yes |
  | `prod-no-keys` | yes | yes | yes | **no** |

  A strict schema puts enums and bounds in `tools/list`. Python SDK v2 rejects a bad value as an `isError` result that names the field, before the handler runs. With free strings, a guessed value is valid input that matches nothing.
- **24 tasks** ([`tasks.py`](src/unhappy/tasks.py)), four for each failure mode in the article's table:

  | Mode | The task | The right outcome |
  |---|---|---|
  | not found | the user quotes a ticket id that doesn't exist | find it by customer, add the note there |
  | enum | "waiting on the customer", "closed", "critical" | the API's `pending`, `solved`, `urgent`; answers of 4–12 |
  | large | count 21–57 tickets | more than one page |
  | dropped write | add a note; the first write lands, then times out | exactly one note |
  | transient | a lookup or count whose first call fails | retry and answer |
  | permission | add a note to a locked ticket | say so, without retrying in a loop |
- **Grading** ([`grading.py`](src/unhappy/grading.py)): from the API's comment store, the agent's call log and its `submit_answer` call. It records success, duplicate notes, silent wrong answers (done, but wrong), giving up, and repeats of a failed call.
- **Models:** Claude Haiku 5.5 and Sonnet 5.5, effort `medium`, automatic prompt caching, at most 15 turns.
- **Matrix:** 24 tasks × 6 variants × 2 models × 3 repeats = 864 episodes.

## Run it

```bash
uv sync
uv run pytest                          # the API, every variant's failure path, tasks, grader; offline
uv run unhappy-path --dry-run          # a literal-minded fake against every variant; $0
uv run unhappy-path --estimate         # what a live run should cost

cp .env.example .env                   # add ANTHROPIC_API_KEY
uv run --env-file .env unhappy-path --pilot --max-usd 1
uv run --env-file .env unhappy-path --max-usd 8      # resumes from results/episodes.jsonl
uv run python -m unhappy.report                      # tables and charts/
```

The dry run proves the graders can see each failure. Its fake follows the server's messages literally: it searches when told to use `search_tickets`, retries when told it's safe, and stops when told retrying won't help. When a message tells it nothing, it does the naive thing.
- On `production` it finishes all 24 tasks; on `quickstart`, none.
- Each ablation breaks exactly the mode it removes, and nothing else.

The spending cap is checked before every model call and shared by all concurrent episodes.

### A fresh live run

The committed results mean a live run has nothing left to do. Move them aside first:

```bash
mv results results-published
uv run --env-file .env unhappy-path --max-usd 8
uv run python -m unhappy.report
```

## Learning the code

1. **[`servers.py`](src/unhappy/servers.py):** read `add_ticket_comment` first. The four practices are a few lines each: the key, and the three error texts.
2. **[`ticketing.py`](src/unhappy/ticketing.py):** the API and its `Faults`. Note that the dropped write *lands* before it times out.
3. **[`tasks.py`](src/unhappy/tasks.py) and [`grading.py`](src/unhappy/grading.py):** one right outcome per task, checked in code.
4. **[`agent.py`](src/unhappy/agent.py):** a plain loop over the MCP client.

Then change something:

- **Make an opaque error a little better:** change `OPAQUE` to "Request failed (404)." and see whether a status code alone helps.
- **List the values in the description** of a free-string `status`, and measure how much of the strict schema's effect that recovers.
- **Lower the page cap** to 10 in `ticketing.py`, and watch the large-result tasks without totals.

## Results

**The production server finished every task on both models; the quickstart finished 43% on Haiku and 89% on Sonnet.** Without totals Haiku reported the 20-ticket page as the whole answer; with free strings both models invented values like "closed" and "critical", and Haiku reported the empty result as zero; told a refusal was final, both models stopped after one attempt, but after "Request failed." they retried. The surprise: without idempotency keys there were no duplicates, because an honest "may or may not have been posted" and a ticket that shows its notes made both models check first; after an opaque error Haiku duplicated the note 12 times out of 12. Two flaws found in the pilot are disclosed and fixed. Each claim is judged in [RESULTS.md](RESULTS.md).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
