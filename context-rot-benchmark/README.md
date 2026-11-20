# Ticket Thirty: Context Rot Inside a Tool Loop

> Code and raw results for **[Ticket Thirty](https://iambishal.com/ai-lab/projects/context-rot-benchmark)** in the AI Lab on iambishal.com.
> It measures the remedies in **[Context Engineering for AI Agents](https://iambishal.com/resources/articles/context-engineering-ai-agents)**, which says of context rot: *"I'm not aware of either effect being measured inside a tool loop."*

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.12 · kestrel-outfitters 0.3.0 |
| Run | `uv run ticket-thirty --dry-run` (offline) · `uv run ticket-thirty` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

*Lost in the Middle* and Chroma's *Context Rot* found that models use long inputs unevenly. Neither measured an agent. The article proposes remedies for a long-running agent's window:
- trim tool output at the tool boundary;
- clear stale tool results, "nearly lossless";
- summarise older turns, but pin constraints verbatim, because a summary paraphrases them away;
- start fresh windows for sub-tasks;
- keep the tool count down, since selection degrades beyond 30–50 tools.

This experiment gives one agent a **queue of 24 support tickets in one conversation** and measures each remedy as a context policy.

**Hypothesis, written before any live run** (from the article):

1. **Context rot shows up inside a tool loop:** under `raw`, accuracy or adherence to the standing instructions declines from the first third of the queue to the last.
2. **Trimming at the tool boundary removes most of the input and most of the decline:** `trim` uses a fraction of `raw`'s tokens with no worse accuracy.
3. **Clearing stale results is nearly lossless:** `clear` matches `raw`'s accuracy with far fewer tokens.
4. **An unpinned summary loses constraints:** `summary` breaks the freeze or drops the audit flag more often than `summary+pin`, and its summaries paraphrase or omit the instructions.
5. **A fresh window per ticket is the accuracy ceiling,** and the cheapest per ticket.
6. **Tool selection degrades with 40 similar tools:** more wrong-tool calls and lower success than with 10.

## The setup

- **Tickets** ([`scenarios.py`](src/ticket_thirty/scenarios.py)): 24 tickets over real Kestrel Outfitters orders (seed 2026), no order used twice:
  - 12 return requests;
  - 4 returns that also report a website bug;
  - 4 delivery questions;
  - 4 returns without an order id (find it by email).

  The returns policy decides each outcome: refund, escalate to Finance above USD 500, or decline outside the window.
- **Two standing instructions,** given once in the opening user message ([`queue.py`](src/ticket_thirty/queue.py)). They're in the conversation, not the system prompt, because that's where mid-task instructions live and what a compaction rewrites:
  1. *Finance has frozen escalations until Monday:* above USD 500, don't escalate; reply that the refund is delayed. This changes the right outcome of 5 tickets.
  2. *Flag every ticket you finish with `flag_for_review`, reason `Q4-AUDIT`.*
- **Tools** ([`tools.py`](src/ticket_thirty/tools.py)): ten support tools, called in-process.
  - Order lookups return the full record an untrimmed API would: line items, customer, order history, contact history and tracking events, **about 1,060 tokens**.
  - The policy decision (window open or closed, who approves) is computed in code and stated in every mode, so no policy makes the reasoning harder.
- **Context policies** ([`policies.py`](src/ticket_thirty/policies.py)):

  | id | What the window holds at each request |
  |---|---|
  | `raw` | everything, in order of arrival |
  | `trim` | the same, but tools return only the lines the task needs (about 60 tokens) |
  | `clear` | raw tools; once a ticket closes, its tool results become a one-line stub with a reference |
  | `summary` | a running summary (written by Claude Haiku after each ticket) plus the current ticket; the instructions survive only if the summariser keeps them |
  | `summary+pin` | the same summary, with both instructions pinned word for word |
  | `fresh` | a new window per ticket, seeded with the opening message |

  Thinking blocks from closed tickets are dropped under every policy.
- **Tool-count study:** each ticket alone in a fresh window, with the 10 real tools or with 30 look-alikes added (`lookup_order`, `refund_payment`, `issue_credit` …). Each look-alike refuses and names the right tool.
- **Models:** Claude Haiku 5.5 and Sonnet 5.5 as the agent, effort `medium`, automatic prompt caching on the growing transcript. The summariser is Haiku at effort `low`.
- **Matrix:**
  - queue: 6 policies × 2 models × 3 queue orders (the same tickets, shuffled) = 36 queues, 864 tickets;
  - tools: 24 tickets × 2 tool sets × 2 models × 2 repeats = 192 episodes.
- **Grading** ([`grading.py`](src/ticket_thirty/grading.py)), from the ledger of tool calls only, never from what the agent says:
  - outcome under the freeze, and success (the outcome plus a bug report when one was due);
  - freeze violations and audit flags;
  - money before the eligibility check;
  - wrong-tool calls.

  Each is broken down by queue position. For the summary policies, every summary is checked for each instruction.

## Run it

```bash
uv sync
uv run pytest                          # tickets, tools, policies, grader, the dry run; offline
uv run ticket-thirty --dry-run         # an oracle and a forgetful fake; $0
uv run ticket-thirty --estimate        # what a live run should cost (about $11)

cp .env.example .env                   # add ANTHROPIC_API_KEY
uv run --env-file .env ticket-thirty --pilot --max-usd 2
uv run --env-file .env ticket-thirty --max-usd 15     # resumes from results/runs.jsonl
uv run python -m ticket_thirty.report                 # tables and charts/
```

The dry run proves the grader can see what it measures. The **forgetful** fake keeps an instruction only while its text is in the window it was sent. Under every policy but `summary` it handles all 24 tickets correctly. Under `summary`, whose fake summariser leaves the instructions out, it escalates all 5 frozen refunds and stops flagging after the first ticket.

The cap is checked before every model call and shared by all concurrent runs. A run stopped by the cap isn't recorded, so the next run redoes it.

### A fresh live run

The committed results mean a live run has nothing left to do. Move them aside first:

```bash
mv results results-published
uv run --env-file .env ticket-thirty --max-usd 15
uv run python -m ticket_thirty.report
```

## Learning the code

1. **[`queue.py`](src/ticket_thirty/queue.py):** the system prompt, the opening message with the two instructions, and the queue orders.
2. **[`tools.py`](src/ticket_thirty/tools.py):** compare `_full_record()` with `_summary()`. That gap is the whole `trim` policy.
3. **[`policies.py`](src/ticket_thirty/policies.py):** `build_messages()` is the experiment: six ways to turn one history into one window.
4. **[`agent.py`](src/ticket_thirty/agent.py):** the loop, about 60 lines.
5. **[`grading.py`](src/ticket_thirty/grading.py) → [`report.py`](src/ticket_thirty/report.py):** from ledger to verdicts.

Then change something:

- **Pin only one instruction:** drop the audit line from the pinned block in `build_messages()` and watch which instruction the summary keeps.
- **A smarter summariser:** add "keep every standing instruction verbatim" to `SUMMARISER_SYSTEM`, which is the article's advice moved into the prompt, and compare with pinning.
- **A longer queue:** add rows to `POLICY` in `scenarios.py` (each new row is a new ticket over a fresh order) to push `raw` past 60k tokens.

## Results

Not run yet: the hypothesis above was written first. The results, with a verdict on each claim, will be in `RESULTS.md`.

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
