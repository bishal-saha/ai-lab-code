"""Study A: where the tenant filter goes, and what it does to the top k (retrieval only, no LLM).

For each tenant-scoped query, each mode returns up to k chunk ids; recall@k is measured against the
exact answer (the filter in the query, sequential scan):

    post-filter       global HNSW top-k (and top-4k, oversampled), then the app keeps the caller's rows
    in-query          WHERE tenant and role in the vector query; iterative scans off (pgvector's default)
    in-query-iter     the same with hnsw.iterative_scan = relaxed_order (the tutorial's setting)
    rls-only          no WHERE: the row-level security policy filters, as the app role
    exact             WHERE in the query, index scans off: the ground truth

Postgres may answer a selective filter from the plain tenant index instead of HNSW, which is exact.
That is realistic, so each mode runs as the planner chooses ("auto") and with the HNSW index forced
("hnsw"), and the plan actually used is recorded.
"""

from __future__ import annotations

import time

import numpy as np

from fbr import db

MODES = ["post-filter", "post-filter-4x", "in-query", "in-query-iter", "rls-only"]


def _principal(c, tenant: str, role: str) -> None:
    c.execute(
        "SELECT set_config('app.tenant_id', %s, true), set_config('app.roles', %s, true)",
        (str(db.tenant_uuid(tenant)), role),
    )


def _plan_uses(c, sql: str, args) -> str:
    plan = c.execute("EXPLAIN (FORMAT JSON) " + sql, args).fetchone()[0]
    text = str(plan)
    return (
        "hnsw"
        if "chunks_embedding_hnsw" in text
        else "tenant-index"
        if "chunks_tenant" in text
        else "seq"
    )


def run_query(tenant: str, role: str, qvec: np.ndarray, k: int, force_hnsw: bool) -> dict:
    tid = db.tenant_uuid(tenant)
    where = "tenant_id = %s AND allowed_roles && ARRAY[%s]::text[]"
    out: dict = {}
    # forcing HNSW: no sequential or bitmap scans (the caller also drops the tenant index for that pass)
    force = (
        ["SET LOCAL enable_seqscan = off", "SET LOCAL enable_bitmapscan = off"]
        if force_hnsw
        else []
    )

    with db.connect("fbr_ingest") as c:  # BYPASSRLS: only the WHERE clause filters
        with c.transaction():
            c.execute("SET LOCAL enable_indexscan = off")
            c.execute("SET LOCAL enable_bitmapscan = off")
            rows = c.execute(
                f"SELECT id, embedding <=> %s FROM chunks WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
                (qvec, tid, role, qvec, k),
            ).fetchall()
            out["exact"] = {
                "ids": [r[0] for r in rows],
                "dist": [float(r[1]) for r in rows],
                "ms": 0.0,
                "plan": "seq",
            }
        for mode, mult in (("post-filter", 1), ("post-filter-4x", 4)):
            with c.transaction():
                for s in force:
                    c.execute(s)
                t0 = time.perf_counter()
                rows = c.execute(
                    "SELECT id, tenant_id, allowed_roles FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                    (qvec, k * mult),
                ).fetchall()
                ids = [r[0] for r in rows if r[1] == tid and role in r[2]][:k]
                out[mode] = {
                    "ids": ids,
                    "ms": (time.perf_counter() - t0) * 1000,
                    "plan": "hnsw (global)",
                }
        for mode, iterative in (("in-query", False), ("in-query-iter", True)):
            with c.transaction():
                for s in force:
                    c.execute(s)
                if iterative:
                    c.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                sql, args = (
                    f"SELECT id FROM chunks WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
                    (tid, role, qvec, k),
                )
                plan = _plan_uses(c, sql, args)
                t0 = time.perf_counter()
                rows = c.execute(sql, args).fetchall()
                out[mode] = {
                    "ids": [r[0] for r in rows],
                    "ms": (time.perf_counter() - t0) * 1000,
                    "plan": plan,
                }
    with (
        db.connect("fbr_app") as c,
        c.transaction(),
    ):  # the RLS policy filters; the query has no WHERE
        for s in force:
            c.execute(s)
        _principal(c, tenant, role)
        sql, args = "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s", (qvec, k)
        plan = _plan_uses(c, sql, args)
        t0 = time.perf_counter()
        rows = c.execute(sql, args).fetchall()
        out["rls-only"] = {
            "ids": [r[0] for r in rows],
            "ms": (time.perf_counter() - t0) * 1000,
            "plan": plan,
        }
    return out


def score(result: dict, k: int) -> dict:
    truth = set(result["exact"]["ids"])
    denom = max(1, min(k, len(truth)))
    return {
        m: {
            "returned": len(r["ids"]),
            "recall": len(set(r["ids"]) & truth) / denom,
            "ms": round(r["ms"], 2),
            "plan": r["plan"],
        }
        for m, r in result.items()
        if m != "exact"
    } | {"exact_returned": len(truth)}


def tenant_index(present: bool) -> None:
    """Drop or recreate the plain tenant index: without it, a filtered vector query has to use HNSW."""
    import psycopg

    with psycopg.connect(db.admin_url(db.DB), autocommit=True) as c:
        c.execute(
            "CREATE INDEX IF NOT EXISTS chunks_tenant ON chunks (tenant_id)"
            if present
            else "DROP INDEX IF EXISTS chunks_tenant"
        )
        c.execute("ANALYZE chunks")


def brute_force_distances(
    chunk_vecs: np.ndarray,
    chunk_rows: list[tuple[int, str, tuple]],
    qvec: np.ndarray,
    tenant: str,
    role: str,
    k: int,
) -> list[float]:
    """The k smallest cosine distances computed in numpy, to check the database's exact mode
    (distances, not ids: tied chunks can swap at the k-th place)."""
    mask = np.array([t == tenant and role in roles for _, t, roles in chunk_rows])
    norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(qvec)
    dist = 1 - (chunk_vecs @ qvec) / np.maximum(norms, 1e-12)
    return sorted(dist[mask].tolist())[:k]
