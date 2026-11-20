# Filter Before You Rank: Multi-Tenant RAG Isolation, Measured

> Code and raw results for **[Filter Before You Rank](https://iambishal.com/ai-lab/projects/multi-tenant-rag-isolation-benchmark)** in the AI Lab on iambishal.com.
> It measures the claims in the tutorial **[Build a Secure RAG API](https://iambishal.com/resources/tutorials/build-a-secure-rag-api)**, whose [companion code](https://github.com/bishal-saha/tutorials-code/tree/main/build-a-secure-rag-api) runs on a fake embedder and a fake model over two small tenants.

| | |
|---|---|
| Type | AI Lab experiment (the first on a tutorial) |
| Tested with | Python 3.12 · PostgreSQL 18 + pgvector 0.8.1 · voyage-4-lite · anthropic 2.12 |
| Run | `uv run filter-before-rank --dry-run` (needs Postgres; no keys) · `uv run filter-before-rank` (live) |
| Tests | `uv run pytest` (database tests run when `DATABASE_URL_ADMIN` is set) |
| Keys | only for a live run: see `.env.example` |

## The question

The tutorial makes three claims a multi-tenant RAG API depends on:
- **Filter before you rank:** "tenant and role filtering must happen inside the retrieval query, before the top k results are chosen", keeping k results with pgvector's iterative index scans.
- **Row-level security as a backstop:** RLS backs up the query filter. The app role must not own the tables or bypass RLS, the principal travels in transaction-local settings so a pooled connection doesn't carry it on, and a missing setting becomes null rather than an error.
- **Refuse before you guess:** a similarity threshold refuses before any model call when nothing relevant was found, and the citation check is "a floor rather than proof".

The companion code demonstrates each one on a toy corpus. This experiment measures them on a shared index of 30 tenants, with real embeddings, on real PostgreSQL.

**Hypothesis, written before any live run** (from the tutorial):

1. **Post-filtering starves small tenants:** filtering after a global top-k returns fewer than k results, worse as a tenant's share of the index shrinks. Oversampling softens it without fixing it.
2. **Filtering inside the HNSW query without iterative scans still under-returns for small tenants;** iterative scans restore k results at close to exact recall.
3. **RLS alone behaves like a post-filter on an HNSW index,** so it under-returns too. It is a backstop, not the retrieval filter.
4. **Each layer alone misses a bug the pair catches:**
   - the query filter misses its own omission;
   - RLS misses an owner connection;
   - a session-level `SET` leaks the principal across pooled requests;
   - a cast without `NULLIF` errors.
5. **Without a threshold the model sometimes answers what its chunks don't support;** a threshold turns those into refusals, at a cost in wrongly refused questions.
6. **The citation check is a floor:** it catches invalid citations, but some answers cite valid chunks that don't contain what they claim.

## The setup

- **Corpus** ([`corpus.py`](src/fbr/corpus.py)): 10,974 synthetic SaaS support chunks from **30 tenants in one index**.
  - One tenant holds 32.8% of the index, four are medium, and 25 are small, down to 0.23%.
  - Help articles are for agents and end users; runbooks only for agents.
  - Every tenant writes about the same 32 subtopics in its own products and numbers. A question therefore lands close to every tenant's answer, which is the hard case for a filter.
  - Facts (limits, durations, menu paths) are fixed per tenant and subtopic, so a grader can tell a right answer from another tenant's.
- **Embeddings** ([`embed.py`](src/fbr/embed.py)): `voyage-4-lite` (1,024 dimensions), cached in `cache/` after the first run. The cache is 45 MB, so it isn't committed: a rerun embeds the corpus once with a Voyage key, about 0.54M tokens, a few cents.
- **Database** ([`db.py`](src/fbr/db.py)): the tutorial's schema on a throwaway database:
  - chunks carry their document's tenant and ACL, under a three-column foreign key;
  - an HNSW index (cosine);
  - three roles: owner, app (SELECT only) and ingest (BYPASSRLS), all prefixed `fbr_`;
  - the RLS policy with `NULLIF` and `FORCE`.

  The harness creates all of it and drops it at the end.
- **Study A, where the filter goes** ([`study_a.py`](src/fbr/study_a.py)): 287 tenant-scoped questions × k = 5 and 10. Each is answered by:
  - a global top-k (or top-4k) filtered in the app;
  - the filter in the vector query, with and without `hnsw.iterative_scan = relaxed_order`;
  - RLS alone.

  Recall is measured against an exact scan, which a test checks against numpy. Every mode runs twice: as the planner chooses, and with HNSW forced. Postgres may answer a selective filter from the plain tenant index, which is exact.
- **Study B, which layer catches which bug** ([`study_b.py`](src/fbr/study_b.py)): the tutorial's cross-tenant probe, with negative controls (tenant and role) and a positive one. It runs under 6 bug states × 3 defences (query filter, RLS, both). Each cell ends as leaked, error, blind (safe, but the caller can't see its own data) or safe.
- **Study C, refusals and citations** ([`study_c.py`](src/fbr/study_c.py)): 100 questions from 20 tenants:
  - 40 answerable from the asker's own documents;
  - 40 answerable only from *another* tenant's;
  - 20 answerable nowhere.

  Each is retrieved with the in-query filter and iterative scans, k = 5, then answered by Claude Haiku 5.5 and Sonnet 5.5 with lesson 3's rules (only the chunks, cite `[chunk:ID]`, else `NO_ANSWER`), 2 repeats. The threshold runs before the model, so the threshold sweep is computed offline from the same runs.

## Run it

```bash
uv sync
export DATABASE_URL_ADMIN=postgresql://postgres@localhost:5432/postgres   # a superuser; pgvector installed
uv run pytest
uv run filter-before-rank --dry-run        # fake embedder, fake model, every study; $0
uv run filter-before-rank --estimate

cp .env.example .env                       # ANTHROPIC_API_KEY, VOYAGE_API_KEY (the corpus is embedded once, a few cents)
uv run --env-file .env filter-before-rank --max-usd 3
uv run python -m fbr.report                # tables and charts/
```

The dry run proves the measurements can see what they measure:
- exact retrieval matches numpy;
- post-filtering starves;
- iterative scans return k results;
- every Study B cell matches the tutorial's reasoning;
- the answer graders work.

### A fresh live run

Studies A and B are recomputed on every run (no model needed), and Study C resumes from `results/study_c.jsonl`. Move the results aside for a fresh Study C:

```bash
mv results results-published
uv run --env-file .env filter-before-rank --max-usd 3
uv run python -m fbr.report
```

## Learning the code

1. **[`db.py`](src/fbr/db.py):** `policy_sql()` is the tutorial's RLS policy, and `set_rls()` switches on each bug.
2. **[`study_a.py`](src/fbr/study_a.py):** the same question asked five ways. Compare `post-filter` and `in-query-iter`.
3. **[`study_b.py`](src/fbr/study_b.py):** `cell()` is one probe under one bug and one defence.
4. **[`study_c.py`](src/fbr/study_c.py):** lesson 3's prompt and the graders.

Then change something:

- **Raise `hnsw.ef_search`** in `study_a.py` and watch the in-query filter recover without iterative scans, more slowly.
- **Shrink the large tenant** (`SIZES` in `corpus.py`) and see post-filtering recover as tenants even out.
- **Remove `FORCE`** in `db.setup()` and rerun Study B.

## Results

**Where the filter goes decides whether a tenant gets its answers.** Filtering a global top-10 in the app found 4% of the right chunks (under 2% for small tenants); the filter inside a plain HNSW query was no better than post-filtering a top-40, because HNSW hands back only ef_search candidates; pgvector's iterative scans, the tutorial's setting, restored 98.5-100%. RLS alone behaved like the in-query filter, and worse for the largest tenant. The isolation matrix matched the tutorial cell for cell: each layer caught the other's bugs, and no bug leaked through both. Neither model ever answered an unanswerable question (0 of 240), so the similarity threshold could only refuse good answers; Sonnet over-refused 12 answerable questions. Each claim is judged in [RESULTS.md](RESULTS.md).

## Licence

[MIT](../LICENSE). The tenants and documents are synthetic.
