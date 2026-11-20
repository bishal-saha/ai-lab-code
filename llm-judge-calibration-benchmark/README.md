# Judging the Judge: LLM-as-a-Judge, Calibrated

> Code and raw results for **[Judging the Judge](https://iambishal.com/ai-lab/projects/llm-judge-calibration-benchmark)** in the AI Lab on iambishal.com.
> It measures the judge-design advice in **[LLM Evaluation in Production](https://iambishal.com/resources/articles/llm-evaluation-in-production)**, whose calibration example is marked *"every number illustrative"*.

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.12 · kestrel-outfitters 0.3.0 |
| Run | `uv run judge-the-judge --dry-run` (offline) · `uv run judge-the-judge` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

An LLM judge is only as useful as its calibration: how many real failures it catches, and how many good answers it lets through. The article argues for specific design choices:
- one criterion, pass or fail, not a 1–10 score;
- both orders for pairwise comparisons;
- length-neutral wording;
- separate TPR and TNR instead of raw agreement.

It illustrates the calibration step with invented numbers. This experiment runs six judge designs over 200 replies whose correct verdict is **known by construction**, so the calibration needs no human labels.

**Hypothesis, written before any live run** (from the article):

1. **Raw agreement flatters the judge:** with 20% failures, raw agreement sits well above the share of failures caught (TPR).
2. **A 1–10 score ranks better than it gates:** the score's rank correlation is high, but its κ at a ≥ 7 gate is clearly below the pass/fail judge's κ.
3. **One criterion beats three:** a judge checking three criteria at once catches fewer refund-policy failures than the one-criterion judge.
4. **Judges are verbose-biased, and length-neutral wording reduces it:** long wrong replies pass more often than short ones, and "a short response can pass" narrows the gap.
5. **Pairwise verdicts carry a position bias:** some verdicts flip when A and B swap, and between two good replies the first position wins more than half the time.
6. **A measured change is the real change scaled by (TPR + TNR − 1).**
7. **Critique first doesn't hurt, and may help:** it matches or beats plain pass/fail on TPR, at a higher cost.

**Not testable here:** self-enhancement bias needs a judge from a different family from the generator. These replies are templated, not model-written.

## The setup

- **Cases** ([`cases.py`](src/judge_the_judge/cases.py), written to [`cases/`](cases)): 200 support replies to refund tickets over real Kestrel Outfitters orders (seed 2026).
  - Each ticket's policy outcome (refund, escalate to Finance, decline) is computed from the order, as in the earlier experiments.
  - Each reply is built as a **PASS** (it states that outcome and promises nothing more) or one of five **FAIL** types, 8 each:
    - wrong amount;
    - wrong outcome (refunds a declined return);
    - wrong window (a trade customer "within our 30-day window");
    - extra promise (a USD 40 credit);
    - missing fact (an escalation reply that never says what happens).
  - 160 pass, 40 fail. Every reply exists **short** and **long** (the same decision, padded with friendly filler).
  - **Pairs:** 100. 50 pair a right reply with a wrong one to the same ticket; 50 pair a short and a long reply that are both right.
- **Judge designs** ([`judges.py`](src/judge_the_judge/judges.py)). All judge one criterion: *"the reply states the outcome the returns policy requires for this order, and promises nothing the policy doesn't allow"*, given the ticket, the order facts and the policy:

  | id | Design |
  |---|---|
  | `pf` | pass/fail (the article's design) |
  | `pf-crit` | pass/fail, critique first |
  | `pf-neutral` | pass/fail, "a short response can pass" |
  | `score` | 1–10; the gate is ≥ 7 |
  | `multi` | one pass/fail over policy, tone and completeness together |
  | `pair` | A or B, each pair judged in both orders |

- **Judges:** Claude Haiku 5.5 and Sonnet 5.5, effort `low`, structured output, policy and instructions in a cached system block. **3 repeats**, with a majority verdict per case.
- **Metrics** ([`metrics.py`](src/judge_the_judge/metrics.py)), with FAIL as the positive class:
  - TPR (failures caught) and TNR (good replies passed), with Wilson 95% intervals;
  - raw agreement, Cohen's κ, and Spearman ρ for the score;
  - pass rates by length;
  - pairwise swap consistency and first-position win rate;
  - self-consistency across repeats;
  - a gate simulation that draws baseline and candidate sets with known failure rates and compares the judge's measured change with the true one.

  The tests check the arithmetic on the article's own illustrative matrix (12 / 4 / 4 / 60).

## Run it

```bash
uv sync
uv run pytest                          # cases, metrics (incl. the article's matrix), the dry run; offline
uv run judge-the-judge --dry-run       # a perfect and a biased fake judge; $0
uv run judge-the-judge --estimate      # what a live run should cost

cp .env.example .env                   # add ANTHROPIC_API_KEY
uv run --env-file .env judge-the-judge --pilot --max-usd 1
uv run --env-file .env judge-the-judge --max-usd 10    # resumes from results/judgements.jsonl
uv run python -m judge_the_judge.report                # calibration.csv, pairwise.csv, charts/
```

The dry run proves the metrics can see what they measure. The biased fake passes most long wrong replies and always prefers position A, and the report shows both effects: about 92% raw agreement while catching only 57% of failures.

### A fresh live run

The committed judgements mean a live run has nothing left to do. Move them aside first:

```bash
mv results results-published
uv run --env-file .env judge-the-judge --max-usd 10
uv run python -m judge_the_judge.report
```

## Learning the code

1. **[`cases.py`](src/judge_the_judge/cases.py):** how a reply is built to be right or wrong. `core()` holds every decision sentence. Open [`cases/cases.jsonl`](cases) next to it.
2. **[`judges.py`](src/judge_the_judge/judges.py):** the six designs are six short instructions and schemas in `INSTRUCTIONS`.
3. **[`metrics.py`](src/judge_the_judge/metrics.py):** TPR, TNR, κ, Wilson intervals and the gate simulation, each a few lines.
4. **[`run.py`](src/judge_the_judge/run.py) → [`report.py`](src/judge_the_judge/report.py):** the runs, and the reduction to one verdict per case.

Then change something:

- **Tighten the criterion:** add "an exact amount must match the order total" to `CRITERION` and see whether the wrong-amount catch rate moves.
- **Move the gate:** set the score gate to ≥ 8 or ≥ 6 in `verdict()` and rerun the report (free): watch TPR and TNR trade off.
- **Fewer repeats:** report with only `repeat == 0` and see how much majority voting was buying.

## Results

**Every judge caught every failure; the errors all went the other way.** All five designs on both models caught 100% of the 40 wrong replies. Sonnet passed 97–100% of good replies; Haiku passed 82–94%. Critique-first made Haiku fail 65% of correct escalation replies, by reading the policy more strictly than it's written. Haiku's 1–10 score tilted towards long replies (98% vs 70% passed); length-neutral wording removed the tilt. Pairwise verdicts never flipped with a wrong reply in the pair. The (TPR + TNR − 1) scaling predicted measured changes to within 0.12 points. Each claim, and one design flaw (correct replies' outcome and length are confounded), are in [RESULTS.md](RESULTS.md).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
