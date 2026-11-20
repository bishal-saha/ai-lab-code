# Right Shape, Wrong Number: Structured Outputs, Measured

> Code and raw results for **[Right Shape, Wrong Number](https://iambishal.com/ai-lab/projects/structured-output-benchmark)** in the AI Lab on iambishal.com.
> It tests the claims in **[Structured Outputs from LLMs: JSON Schema, Constrained Decoding and Validation](https://iambishal.com/resources/articles/llm-structured-output-json-schema-constrained-decoding-validation)**, above all this one: *"Constrained decoding solves syntax, not semantics."*

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · anthropic 2.12 · jsonschema 4.26 · kestrel-outfitters 0.3.0 |
| Run | `uv run right-shape --dry-run` (offline) · `uv run right-shape` (live) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for a live run: see `.env.example` |

## The question

Structured outputs can now guarantee that a model's JSON parses and matches your schema. Does that make the data right? This experiment extracts 120 invoices with known true values in seven ways: four mechanisms for asking, and three schema designs on the strongest one. It grades every field, not just the shape.

**Hypothesis, written before any live run** (from the article):

1. **Structured outputs and strict tool use give 100% schema-valid records;** instructions-only and non-strict tool use fall short.
2. **Constrained decoding doesn't fix values:** structured outputs still return valid-but-wrong records at a rate well above zero, concentrated in the traps.
3. **A closed enum forces a wrong answer:** with USD/GBP/EUR only, JPY and CAD invoices come back as one of the three. Adding `OTHER` lets the model say "none of these".
4. **Required-but-nullable beats optional:** with no due date printed, the designed schema returns `null` more reliably than the base schema leaves the field out, instead of inventing a date.
5. **An explanation-first `notes` field improves accuracy on the ambiguous traps,** the gain the article explicitly calls untested.
6. **Business-rule validation catches most of the wrong records the schema can't, and one retry repairs a meaningful share,** at the cost of some false alarms and extra calls.

## The setup

- **Invoices** ([`invoices.py`](src/right_shape/invoices.py), written to [`invoices/`](invoices)): 120 documents rendered from real Kestrel Outfitters orders (line items, quantities, prices, USD/GBP/EUR, customers, dates; seed 2026), so every field has a known answer. Four layouts (table, forwarded email, noisy OCR scan, prose), and eight trap types, 15 invoices each:

  | Trap | What's tempting | Right answer |
  |---|---|---|
  | subtotal | "ORDER VALUE" printed big at the top | the amount due at the bottom |
  | discount | a pre-discount total printed first | the discounted amount due |
  | currency | a partner invoice in JPY or CAD | `OTHER` (or the real code); the base enum can't say it |
  | no due date | none printed | none: `null` or left out |
  | credit note | a credit note | document type `credit_note`, no due date |
  | date | `09/10/2025` on a UK invoice | 9 October |
  | words | the amount due only in words | the number |
  | none | a control invoice | – |

- **Conditions** ([`run.py`](src/right_shape/run.py) `CONDITIONS`, [`models.py`](src/right_shape/models.py), [`schemas.py`](src/right_shape/schemas.py)):

  | id | Mechanism | Schema |
  |---|---|---|
  | `prompt` | instructions only, a JSON example in the prompt, parsed defensively | base |
  | `tool` | a `record_invoice` tool, not strict | base |
  | `strict` | the same tool with `strict: true` | base |
  | `so` | structured outputs (`output_config.format`) | base |
  | `so+d` | structured outputs | designed |
  | `so+d-n` | structured outputs | designed without `notes` |
  | `so+d+v` | `so+d`, then business-rule validation and one retry with the errors | designed |

  - **Base schema:** fields in natural order, no descriptions, a closed USD/GBP/EUR enum, an optional due date.
  - **Designed schema:** the article's recommendations. It adds a short `notes` field first, descriptions on the ambiguous fields, `OTHER` in the enum, and a required-but-nullable due date.
- **Rules** ([`rules.py`](src/right_shape/rules.py)): the invoice's numbers must agree with each other. Line items must sum to the subtotal, subtotal − discount + tax + shipping must equal the total, and the due date can't fall before the issue date.
- **Grading** ([`grading.py`](src/right_shape/grading.py)) is code only: eight fields per record against the true values, schema validity with `jsonschema`, and **valid-but-wrong** (schema-valid, a field wrong). No LLM judge.
- **Models:** Claude Haiku 5.5 and Sonnet 5.5, adaptive thinking, effort `medium`. **Matrix:** 120 invoices × 7 conditions × 2 models = 1,680 records, plus retries.

## Run it

```bash
uv sync
uv run pytest                     # invoices, schemas, rules, grader, the retry path; offline
uv run right-shape --dry-run      # an oracle and a mistake-maker through every condition; $0
uv run right-shape --estimate     # what a live run should cost

cp .env.example .env              # add ANTHROPIC_API_KEY
uv run --env-file .env right-shape --pilot --max-usd 1    # 16 invoices, every condition
uv run --env-file .env right-shape --max-usd 10           # the full matrix; resumes
uv run python -m right_shape.report                       # summary.csv, by_trap.csv, charts/
```

`--max-usd` is a hard stop, and rows are appended to `results/extractions.jsonl`, so a stopped run resumes. Use `--conditions so,so+d` and `--models haiku` for subsets.

### A fresh live run

The committed results mean a live run has nothing left to do. Move them aside first:

```bash
mv results results-published
uv run --env-file .env right-shape --max-usd 10
uv run python -m right_shape.report
```

Expect small differences from the published numbers; the models don't answer identically on every run.

## Learning the code

Read `src/right_shape/` in this order:

1. **[`invoices.py`](src/right_shape/invoices.py):** how an order becomes an invoice, and where each trap is planted. Open a file in [`invoices/`](invoices) next to it.
2. **[`schemas.py`](src/right_shape/schemas.py):** the base and designed schemas side by side. `_designed()` is the article's advice in about 15 lines.
3. **[`models.py`](src/right_shape/models.py):** four ways to ask for the same record. The only differences are a prompt, a tool definition, one `strict` flag, and one `output_config.format`.
4. **[`rules.py`](src/right_shape/rules.py):** what a schema can't say, in plain functions.
5. **[`grading.py`](src/right_shape/grading.py) → [`run.py`](src/right_shape/run.py) → [`report.py`](src/right_shape/report.py):** what counts as right, the matrix, then the charts.

Then change something:

- **Add a rule:** for example, tax = rate × (subtotal − discount). See which wrong records it catches, and whether it raises false alarms.
- **Remove a description:** delete the `total` description in `DESCRIPTIONS` and rerun `--conditions so+d` on the subtotal and discount traps.
- **Close the enum again:** remove `OTHER` and watch what the currency invoices turn into.

## Results

**The models read every value trap correctly; every error came from the schema.** Across 1,620 parsed records there was not one wrong total, subtotal, date or invoice number. A closed currency enum made structured outputs record USD on all 30 JPY/CAD records (schema-valid, wrong). The tool conditions didn't call the tool at all on those invoices, and an optional due date was dropped 4 times. The designed schema (descriptions, `OTHER`, a required-but-nullable due date) took structured outputs from 88% to 100% fully right on both models. Each claim is judged in [RESULTS.md](RESULTS.md); raw records, tables and charts are in [`results/`](results).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
