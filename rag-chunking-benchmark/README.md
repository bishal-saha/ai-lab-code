# Cut Here: RAG Chunking Strategies, Measured

> Code and raw results for **[Cut Here](https://iambishal.com/ai-lab/projects/rag-chunking-benchmark)** in the AI Lab on iambishal.com.
> It tests the claims in **[RAG Chunking Strategies: How to Choose, and How to Measure the Choice](https://iambishal.com/resources/articles/rag-chunking-strategies-choose-and-measure)**, using the method that article ends on: *"measuring recall@k on queries you labelled yourself."*

| | |
|---|---|
| Type | AI Lab experiment |
| Tested with | Python 3.12 · fastembed 0.9 (BAAI/bge-small-en-v1.5) · rank-bm25 0.2 · anthropic 2.11 |
| Run | `uv run cut-here --retrieval` (offline, free) · `--contextualize` / `--answer` (paid, Claude) |
| Tests | `uv run pytest` (no key needed) |
| Keys | only for the two paid steps: see `.env.example` |

## The question

When a RAG system answers with the wrong plan's rate limit, the passage usually never reached the model, and chunking is the first suspect. Most advice is a number ("512 tokens, 15% overlap"). The article argues instead that chunking is a retrieval-design decision. This experiment indexes one structured corpus eight ways, asks 204 questions with known answers, and measures which chunks actually come back usable.

**Hypothesis, written before any paid run** (from the article):

1. **Structure-aware chunks with the heading path (S+H) beat fixed-size and recursive splitting on recall@5**, and the gap comes mostly from table-row and heading-dependent questions.
2. **Without the heading path, small chunks fail on tables:** S+H beats plain structure-aware (S) by more on heading-dependent questions than on prose.
3. **Overlap raises fixed-size recall but roughly doubles the chunk count;** once boundaries follow structure, overlap has little left to fix.
4. **Semantic chunking is no better than S+H on documentation,** at a higher index-time cost.
5. **Contextual retrieval reduces retrieval failures on top of S+H, but by less than Anthropic's reported 35%,** because the heading path already restores much of the missing context.
6. **Recall@5 predicts end-to-end answer accuracy:** the strategies rank the same on both.

## The setup

- **Corpus** ([`corpus.py`](src/cut_here/corpus.py), written to [`corpus/`](corpus)): "Kestrel Docs", 47 Markdown documents built by code from the Kestrel Outfitters dataset (seed 2026), about 11,400 words.
  - 40 product pages with a specifications table.
  - A shipping guide: a section per carrier, a sub-section per region, a rate table in each.
  - 2 warehouse pages with 40-row stock tables.
  - The 4 real Kestrel policy documents, the outdated shipping policy included.

  The spec tables all share their column names, and so do the rate tables: a row means nothing without its heading. The dataset also has "(edition 2)" near-duplicates of some products, which makes retrieval harder, as real catalogues do.
- **Questions** ([`questions.py`](src/cut_here/questions.py)): 204 questions, each with its answer's exact span in the corpus and the **context** a chunk must also carry for the answer to be usable:
  - 120 table rows (spec rows need the product name; stock rows need the warehouse);
  - 32 heading-dependent shipping rows (need the carrier *and* the region);
  - 52 prose facts that carry their own context.
- **Strategies** ([`chunkers/`](src/cut_here/chunkers)):

  | id | Strategy |
  |---|---|
  | F256 / F256o | fixed 256 tokens, no overlap / 50% overlap |
  | F512o | fixed 510 tokens, 15% overlap (the common default; 510 = the embedder's limit) |
  | R512 | recursive splitting at 510 tokens: paragraphs, then lines, sentences, words |
  | S | structure-aware: split on headings, tables kept whole or split by rows with the header repeated (400-token limit) |
  | S+H | S plus the heading path (`Ridgeline 40L pack > Specifications`) prefixed to the embedded text |
  | SEM | semantic: sentences and table rows embedded, split where neighbour similarity drops into the doc's lowest quartile |
  | S+H+C | S+H plus contextual retrieval: Claude Haiku writes a short context for each chunk from the whole document (Anthropic's published prompt) |

  Token counts use the embedding model's own tokenizer.
- **Retrievers** ([`index.py`](src/cut_here/index.py)): **dense** (BAAI/bge-small-en-v1.5 through fastembed, run locally: free, no GPU), **BM25**, and **hybrid** (reciprocal-rank fusion).
- **Grading** ([`grading.py`](src/cut_here/grading.py)), code only:
  - **Recall@k:** a top-k chunk covers the answer span **and** its embedded text carries the context. The right row with no product name is not an answer a reader can use.
  - Also reported: span-only recall (the passage came back, context or not), MRR, and why each miss happened ("no span" or "no context").
- **End to end** ([`run.py`](src/cut_here/run.py) `--answer`): Claude Haiku 5.5 and Sonnet 5.5 answer each question from the top-5 hybrid chunks, for F512o, R512, S+H and S+H+C. The answer is exact-match graded in code: the expected value must appear as written.

## Run it

```bash
uv sync
uv run pytest                     # corpus, questions, chunkers, grader; offline
uv run cut-here --dry-run         # every step with a fake model, in a temp folder; $0
uv run cut-here --retrieval       # 7 strategies x 3 retrievers x 204 questions, offline; $0
uv run cut-here --estimate        # what the two paid steps should cost

cp .env.example .env              # add ANTHROPIC_API_KEY
uv run --env-file .env cut-here --contextualize --max-usd 1   # S+H+C contexts, cached in cache/
uv run cut-here --retrieval                                   # again, now with S+H+C
uv run --env-file .env cut-here --answer --max-usd 5          # end-to-end answers
uv run python -m cut_here.report  # results/*_summary.csv and charts/
```

Retrieval is deterministic and rewritten on each run. Contexts are cached in `cache/contexts.jsonl` and answers are appended to `results/answers.jsonl`, so a paid step stopped by `--max-usd` resumes where it left off.

### A fresh live run

The committed contexts and answers mean the paid steps have nothing left to do. Move them aside first:

```bash
mv cache cache-published && mv results results-published
uv run cut-here --retrieval                                   # free
uv run --env-file .env cut-here --contextualize --max-usd 1
uv run cut-here --retrieval
uv run --env-file .env cut-here --answer --max-usd 5
uv run python -m cut_here.report
```

Retrieval should match exactly. Contexts and answers come from a model, so expect small differences there.

## Learning the code

Read `src/cut_here/` in this order:

1. **[`corpus.py`](src/cut_here/corpus.py):** how the documents are built. Open a page in [`corpus/`](corpus) next to it.
2. **[`questions.py`](src/cut_here/questions.py):** each question's answer span and its required context.
3. **[`chunkers/fixed.py`](src/cut_here/chunkers/fixed.py)** → **[`recursive.py`](src/cut_here/chunkers/recursive.py)** → **[`structure.py`](src/cut_here/chunkers/structure.py):** from blind to structure-aware. In `structure.py`, `_table_pieces` is the header-repeat rule and `heading_path` is the one-line prefix the article recommends.
4. **[`chunkers/semantic.py`](src/cut_here/chunkers/semantic.py)** and **[`contextual.py`](src/cut_here/chunkers/contextual.py):** the two strategies that spend compute at index time.
5. **[`index.py`](src/cut_here/index.py)** and **[`grading.py`](src/cut_here/grading.py):** three retrievers in about 60 lines, and the strict hit rule.
6. **[`run.py`](src/cut_here/run.py)** → **[`report.py`](src/cut_here/report.py):** the steps and the charts.

Then change something. Retrieval is free, so `uv run cut-here --retrieval` shows every effect in about a minute:

- **Give S a heading path on tables only:** prefix the path just to table pieces, and see how much of S+H's gain that keeps.
- **Change the overlap:** try F256 at 25% and 75%, and watch chunk count against recall.
- **Swap the embedder:** set `EMBED_MODEL` in `chunkers/base.py` to another fastembed model and rerun retrieval.

## Results

See [RESULTS.md](RESULTS.md) once the paid steps are in. The raw retrieval rows, tables and charts are in [`results/`](results).

## Licence

[MIT](../LICENSE). Kestrel Outfitters is a fictional company; its data is synthetic.
