"""Run the three studies on a throwaway database.

    uv run filter-before-rank --dry-run                          # fake embedder and fake model; $0
    uv run filter-before-rank --estimate
    uv run --env-file .env filter-before-rank --pilot --max-usd 0.5
    uv run --env-file .env filter-before-rank --max-usd 5        # studies A, B, C; resumes Study C

Needs DATABASE_URL_ADMIN (a Postgres superuser URL with pgvector). The harness creates its own
database and three fbr_ roles, and drops them when it finishes (unless --keep-db).
Then `uv run python -m fbr.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anthropic
import numpy as np

from fbr import db, study_a, study_b, study_c
from fbr.corpus import build, questions, retrieval_queries
from fbr.embed import MODEL as EMBED_MODEL
from fbr.embed import Embedder
from fbr.pricing import cost_usd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
REPEATS = 2
KS = (5, 10)

# What Study B must show (the tutorial's reasoning, cell by cell).
EXPECTED_B = {
    ("filter", "none"): "safe",
    ("filter", "no-tenant"): "leaked",
    ("filter", "no-role"): "leaked",
    ("filter", "owner"): "safe",
    ("filter", "session-set"): "safe",
    ("filter", "no-nullif"): "safe",
    ("rls", "none"): "safe",
    ("rls", "no-tenant"): "safe",
    ("rls", "no-role"): "safe",
    ("rls", "owner"): "leaked",
    ("rls", "session-set"): "leaked",
    ("rls", "no-nullif"): "error",
    ("both", "none"): "safe",
    ("both", "no-tenant"): "safe",
    ("both", "no-role"): "safe",
    ("both", "owner"): "safe",
    ("both", "session-set"): "blind",
    ("both", "no-nullif"): "error",
}


def prepare(fake: bool) -> tuple[Embedder, list, list, np.ndarray]:
    tenants, chunks = build()
    emb = Embedder(fake=fake)
    vectors = emb.embed([c.text for c in chunks], "document")
    db.setup()
    db.load(tenants, chunks, vectors)
    return emb, tenants, chunks, vectors


def run_study_a(emb: Embedder, tenants, chunks, vectors, out: Path | None) -> list[dict]:
    qs = retrieval_queries(tenants)
    qv = emb.embed([q for _, _, q in qs], "query")
    sizes = {t.name: t.size for t in tenants}
    total = len(chunks)
    rows, mismatches = [], 0
    meta = [(c.id, c.tenant, c.allowed_roles) for c in chunks]
    for present, force in ((True, False), (False, True)):  # planner chooses; HNSW forced
        study_a.tenant_index(present)
        for (tenant, role, q), v in zip(qs, qv, strict=True):
            for k in KS:
                res = study_a.run_query(tenant, role, v, k, force_hnsw=force)
                if not np.allclose(
                    res["exact"]["dist"],
                    study_a.brute_force_distances(vectors, meta, v, tenant, role, k),
                    atol=1e-4,
                ):
                    mismatches += 1
                rows.append(
                    {
                        "pass": "auto" if present else "hnsw",
                        "tenant": tenant,
                        "share": sizes[tenant] / total,
                        "role": role,
                        "question": q,
                        "k": k,
                        **study_a.score(res, k),
                    }
                )
    study_a.tenant_index(True)
    if out:
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"  study A: {len(rows)} query runs; exact mode differs from numpy on {mismatches}")
    return rows


def run_study_b(out: Path | None) -> list[dict]:
    rows = study_b.run()
    if out:
        out.write_text(json.dumps(rows, indent=1) + "\n")
    bad = [r for r in rows if EXPECTED_B[(r["defence"], r["bug"])] != r["result"]]
    print(f"  study B: {len(rows)} cells; {len(bad)} differ from the tutorial's reasoning")
    return rows


async def run_study_c(
    emb: Embedder,
    tenants,
    models: list[str],
    repeats: int,
    max_usd: float,
    out: Path | None,
    fake: bool,
    limit: int | None = None,
) -> tuple[list[dict], float]:
    qs = questions(tenants)[:limit] if limit else questions(tenants)
    qv = emb.embed([q.text for q in qs], "query")
    retrieved = [study_c.retrieve(q.tenant, q.role, v) for q, v in zip(qs, qv, strict=True)]
    done = (
        {
            (json.loads(x)["question"], json.loads(x)["model"], json.loads(x)["repeat"])
            for x in out.read_text().splitlines()
        }
        if out and out.exists()
        else set()
    )
    client = None if fake else anthropic.AsyncAnthropic(max_retries=6)
    spent, rows, gate = 0.0, [], asyncio.Semaphore(8)

    async def one(q, chunks, model, r):
        nonlocal spent
        if (q.id, model, r) in done or spent >= max_usd:
            return
        async with gate:
            if spent >= max_usd:
                return
            if fake:
                reply, usage, cost = study_c.fake_reply(q, chunks), {}, 0.0
            else:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=1500,
                    system=study_c.SYSTEM,
                    messages=[{"role": "user", "content": study_c.user_prompt(q.text, chunks)}],
                    output_config={"effort": "low"},
                )
                reply = "".join(b.text for b in resp.content if b.type == "text")
                u = resp.usage
                usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
                cost = cost_usd(model, u.input_tokens, u.output_tokens)
            spent += cost
            row = study_c.record(q, chunks, reply, usage, cost, model, r)
            rows.append(row)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    await asyncio.gather(
        *(
            one(q, ch, m, r)
            for r in range(repeats)
            for m in models
            for q, ch in zip(qs, retrieved, strict=True)
        )
    )
    print(f"  study C: {len(rows)} answers this session, ${spent:.3f}")
    return rows, spent


def dry_run() -> int:
    problems = []
    try:
        emb, tenants, chunks, vectors = prepare(fake=True)
        a = run_study_a(emb, tenants, chunks, vectors, None)
        post = np.mean([r["post-filter"]["recall"] for r in a])
        exact_iter = np.mean(
            [r["in-query-iter"]["returned"] == r["k"] for r in a if r["pass"] == "hnsw"]
        )
        if post > 0.5:
            problems.append(
                f"post-filter recall {post:.2f}: the starvation the study measures doesn't show"
            )
        if exact_iter < 0.99:
            problems.append("iterative scans didn't return k results")
        b = run_study_b(None)
        problems += [
            f"study B {r['defence']}/{r['bug']}: {r['result']}"
            for r in b
            if EXPECTED_B[(r["defence"], r["bug"])] != r["result"]
        ]
        c, _ = asyncio.run(run_study_c(emb, tenants, ["fake"], 1, 1e9, None, fake=True))
        if not any(r["correct"] for r in c) or any(r["invalid_citation"] for r in c):
            problems.append(
                "study C graders: the fake should answer some questions correctly with valid citations"
            )
    finally:
        db.teardown()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "\nDry run OK: exact retrieval matches numpy, post-filtering starves, iterative scans return k, the bug matrix\n"
        "matches the tutorial's reasoning cell by cell, and the answer graders work. The database is dropped."
    )
    return 0


def estimate() -> None:
    tenants, chunks = build()
    words = sum(len(c.text.split()) for c in chunks) + sum(
        len(q.split()) for _, _, q in retrieval_queries(tenants)
    )
    print(
        f"  embeddings ({EMBED_MODEL}): about {words * 1.35 / 1e6:.2f}M tokens, a few cents (cached after the first run)"
    )
    n = len(questions(tenants))
    total = 0.0
    for m in MODELS:
        c = n * REPEATS * cost_usd(m, 700, 120)
        total += c
        print(f"  study C {m:18} ${c:6.2f}  ({n} questions x {REPEATS} repeats)")
    print(f"  {'total':26} ${total:6.2f}  (studies A and B need no model)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument(
        "--pilot", action="store_true", help="embed everything; Study C on 20 questions with Haiku"
    )
    parser.add_argument("--max-usd", type=float, default=3.0)
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.estimate:
        estimate()
        return 0
    RESULTS.mkdir(exist_ok=True)
    try:
        emb, tenants, chunks, vectors = prepare(fake=False)
        print(f"  embedded with {EMBED_MODEL}: {emb.tokens} new tokens")
        if args.pilot:
            asyncio.run(
                run_study_c(
                    emb,
                    tenants,
                    ["claude-haiku-5-5"],
                    1,
                    args.max_usd,
                    RESULTS / "study_c.jsonl",
                    False,
                    limit=20,
                )
            )
        else:
            run_study_a(emb, tenants, chunks, vectors, RESULTS / "study_a.jsonl")
            run_study_b(RESULTS / "study_b.json")
            asyncio.run(
                run_study_c(
                    emb, tenants, MODELS, REPEATS, args.max_usd, RESULTS / "study_c.jsonl", False
                )
            )
    finally:
        if not args.keep_db:
            db.teardown()
    (RESULTS / "run.json").write_text(
        json.dumps(
            {
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "embedding_model": EMBED_MODEL,
                "models": MODELS,
                "repeats": REPEATS,
                "k": list(KS),
                "chunks": len(chunks),
                "tenants": len(tenants),
                "packages": {p: version(p) for p in ("anthropic", "psycopg", "pgvector", "numpy")},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
