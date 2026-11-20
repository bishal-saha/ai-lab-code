"""Offline tests. The database tests need DATABASE_URL_ADMIN (a Postgres superuser with pgvector) and are
skipped without it; they build the throwaway database once with the fake embedder and drop it after."""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pytest
from fbr.corpus import build, questions, retrieval_queries
from fbr.db import policy_sql
from fbr.embed import DIM, fake_embed
from fbr.study_c import grade

needs_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL_ADMIN"), reason="no DATABASE_URL_ADMIN"
)


def test_corpus_shape():
    tenants, chunks = build()
    assert len(tenants) == 30 and len({c.text for c in chunks}) == len(chunks)
    assert all(sum(c.tenant == t.name for c in chunks) == t.size for t in tenants)
    shares = [t.size / len(chunks) for t in tenants]
    assert shares[0] > 0.3 and shares[-1] < 0.003
    for c in chunks:  # runbooks are agent-only, help articles are for both roles
        assert c.allowed_roles == (("agent",) if c.kind == "runbook" else ("agent", "end_user"))


def test_facts_are_per_tenant_and_subtopic():
    tenants, chunks = build()
    t = tenants[7]
    for key in t.subtopics:
        fact = t.facts[key]["key"]
        assert all(fact in c.text for c in chunks if c.tenant == t.name and c.subtopic == key)


def test_questions():
    tenants, _ = build()
    qs = questions(tenants)
    assert Counter(q.case for q in qs) == {"answerable": 40, "other-tenant": 40, "nowhere": 20}
    by = {t.name: t for t in tenants}
    for q in qs:
        if q.case == "answerable":
            assert q.subtopic in by[q.tenant].subtopics and q.fact
        elif q.case == "other-tenant":
            assert q.subtopic not in by[q.tenant].subtopics
    assert len(retrieval_queries(tenants)) > 250


def test_fake_embedder_is_normalised_and_similar_for_similar_text():
    v = fake_embed(
        [
            "rotate an API key without downtime",
            "rotate the API key with no downtime",
            "invoice VAT number",
        ]
    )
    assert v.shape == (3, DIM) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_policy_has_nullif():
    assert "NULLIF(current_setting('app.tenant_id', true), '')::uuid" in policy_sql()
    assert "NULLIF(current_setting('app.tenant_id'" not in policy_sql(nullif=False)


def test_answer_grader():
    tenants, _ = build()
    qs = questions(tenants)
    a = next(q for q in qs if q.case == "answerable")
    chunks = [(11, f"... {a.fact} ...", 0.1), (12, "unrelated", 0.4)]
    g = grade(a, f"It is {a.fact}. [chunk:11]", chunks)
    assert g.correct and g.supported and not g.invalid_citation
    g = grade(a, f"It is {a.fact}. [chunk:12]", chunks)
    assert g.correct and not g.supported  # right, but cites a chunk without the fact
    assert grade(a, "It is something. [chunk:99]", chunks).invalid_citation
    o = next(q for q in qs if q.case == "other-tenant")
    assert (
        grade(o, "NO_ANSWER", chunks).refused
        and grade(o, "Do this. [chunk:11]", chunks).confabulated
    )


@pytest.fixture(scope="module")
def database():
    from fbr import db

    tenants, chunks = build()
    vectors = fake_embed([c.text for c in chunks])
    db.setup()
    db.load(tenants, chunks, vectors)
    yield db, tenants, chunks, vectors
    db.teardown()


@needs_db
def test_study_b_matches_the_tutorial(database):
    from fbr import study_b
    from fbr.run import EXPECTED_B

    rows = study_b.run()
    assert {(r["defence"], r["bug"]): r["result"] for r in rows} == EXPECTED_B


@needs_db
def test_principal_lasts_one_transaction(database):
    db = database[0]
    with db.connect("fbr_app") as c:
        with c.transaction():
            c.execute(
                "SELECT set_config('app.tenant_id', %s, true), set_config('app.roles', 'agent', true)",
                (str(db.tenant_uuid("Globex")),),
            )
            assert c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1200
        assert (
            c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        )  # '' becomes NULL: no rows


@needs_db
def test_exact_mode_matches_numpy_and_iterative_returns_k(database):
    from fbr import study_a

    db, tenants, chunks, vectors = database
    meta = [(c.id, c.tenant, c.allowed_roles) for c in chunks]
    tenant, role, q = retrieval_queries(tenants)[-3]
    qv = fake_embed([q])[0]
    study_a.tenant_index(False)
    try:
        res = study_a.run_query(tenant, role, qv, 10, force_hnsw=True)
    finally:
        study_a.tenant_index(True)
    assert np.allclose(
        res["exact"]["dist"],
        study_a.brute_force_distances(vectors, meta, qv, tenant, role, 10),
        atol=1e-5,
    )
    assert len(res["in-query-iter"]["ids"]) == 10


def test_fact_match_ignores_word_order():
    from fbr.study_c import has_fact

    assert has_fact("on the Enterprise plan only", "It is available only on the Enterprise plan.")
    assert not has_fact("on the Enterprise plan only", "It is available on the Team plan.")
