"""The throwaway database: the tutorial's schema, roles and row-level security, with every bug as a switch.

Everything lives in one database the harness creates and drops (`ai_lab_rag_bench`), plus three
roles prefixed `fbr_` (roles are cluster-wide in Postgres, so the prefix keeps them apart from
anything else on the server). `teardown()` drops all of it.

From lessons 2 and 4 of "Build a Secure RAG API":
- chunks carry their document's tenant and ACL, kept in step by a three-column foreign key with
  ON UPDATE CASCADE, so the filter sits on the row being ranked;
- an HNSW index on the embedding (cosine), as in the tutorial;
- fbr_owner owns the tables, fbr_app serves requests (SELECT only), fbr_ingest loads (BYPASSRLS);
- the RLS policy reads the principal from two transaction-local settings, with NULLIF so a
  connection without a principal sees nothing instead of raising, and FORCE so the owner obeys it.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from fbr.corpus import Chunk, Tenant
from fbr.embed import DIM

DB = "ai_lab_rag_bench"
ROLES = ("fbr_owner", "fbr_app", "fbr_ingest")
PASSWORD = "fbr-local-only"  # local throwaway roles; never reuse


def admin_url(db: str = "postgres") -> str:
    u = urlparse(os.environ["DATABASE_URL_ADMIN"])
    return urlunparse(u._replace(path=f"/{db}"))


def role_url(role: str) -> str:
    u = urlparse(os.environ["DATABASE_URL_ADMIN"])
    netloc = f"{role}:{PASSWORD}@{u.hostname}:{u.port or 5432}"
    return urlunparse(u._replace(netloc=netloc, path=f"/{DB}"))


def tenant_uuid(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"fbr-tenant:{name}")


def policy_sql(nullif: bool = True) -> str:
    tid = (
        "NULLIF(current_setting('app.tenant_id', true), '')::uuid"
        if nullif
        else "current_setting('app.tenant_id', true)::uuid"
    )
    roles = "string_to_array(NULLIF(current_setting('app.roles', true), ''), ',')"
    return f"tenant_id = {tid} AND allowed_roles && {roles}"


def setup() -> None:
    """Create the database, roles, schema and policies (idempotent: drops any previous run first)."""
    teardown()
    with psycopg.connect(admin_url(), autocommit=True) as c:
        c.execute(f"CREATE DATABASE {DB}")
        for r in ROLES:
            extra = " BYPASSRLS" if r == "fbr_ingest" else ""
            c.execute(f"CREATE ROLE {r} LOGIN PASSWORD '{PASSWORD}'{extra}")
    with psycopg.connect(admin_url(DB), autocommit=True) as c:
        c.execute("CREATE EXTENSION vector")
        c.execute(f"""
            CREATE TABLE tenants (id uuid PRIMARY KEY, name text UNIQUE NOT NULL);
            CREATE TABLE documents (
                id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants(id), title text NOT NULL,
                allowed_roles text[] NOT NULL, UNIQUE (id, tenant_id, allowed_roles));
            CREATE TABLE chunks (
                id bigint PRIMARY KEY, document_id uuid NOT NULL, tenant_id uuid NOT NULL, allowed_roles text[] NOT NULL,
                subtopic text NOT NULL, kind text NOT NULL, content text NOT NULL, embedding vector({DIM}) NOT NULL,
                FOREIGN KEY (document_id, tenant_id, allowed_roles) REFERENCES documents (id, tenant_id, allowed_roles) ON UPDATE CASCADE);
            CREATE INDEX chunks_tenant ON chunks (tenant_id);
            ALTER TABLE tenants OWNER TO fbr_owner; ALTER TABLE documents OWNER TO fbr_owner; ALTER TABLE chunks OWNER TO fbr_owner;
            GRANT CONNECT ON DATABASE {DB} TO fbr_app, fbr_ingest, fbr_owner;
            GRANT USAGE ON SCHEMA public TO fbr_app, fbr_ingest, fbr_owner;
            GRANT SELECT ON tenants, documents, chunks TO fbr_app;
            GRANT SELECT, INSERT, UPDATE, DELETE ON tenants, documents, chunks TO fbr_ingest;
        """)
        set_rls(True, force=True, nullif=True, conn=c)


def set_rls(enabled: bool, *, force: bool = True, nullif: bool = True, conn=None) -> None:
    """Switch row-level security on or off, with or without FORCE and NULLIF (the bug switches)."""

    def apply(c):
        for t in ("documents", "chunks"):
            c.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {t}")
            if enabled:
                c.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
                c.execute(f"ALTER TABLE {t} {'FORCE' if force else 'NO FORCE'} ROW LEVEL SECURITY")
                c.execute(
                    f"CREATE POLICY tenant_isolation ON {t} FOR SELECT TO fbr_app USING ({policy_sql(nullif)})"
                )
            else:
                c.execute(f"ALTER TABLE {t} DISABLE ROW LEVEL SECURITY")

    if conn is not None:
        apply(conn)
    else:
        with psycopg.connect(admin_url(DB), autocommit=True) as c:
            apply(c)


def load(tenants: list[Tenant], chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Ingest as fbr_ingest (BYPASSRLS), then build the HNSW index."""
    docs: dict[str, tuple] = {}
    for ch in chunks:
        docs.setdefault(
            ch.document + "|" + ch.kind,
            (
                uuid.uuid5(uuid.NAMESPACE_URL, ch.document + "|" + ch.kind),
                tenant_uuid(ch.tenant),
                ch.document,
                list(ch.allowed_roles),
            ),
        )
    with psycopg.connect(role_url("fbr_ingest")) as c:
        register_vector(c)
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO tenants VALUES (%s, %s)",
                [(tenant_uuid(t.name), t.name) for t in tenants],
            )
            cur.executemany("INSERT INTO documents VALUES (%s, %s, %s, %s)", list(docs.values()))
            with cur.copy(
                "COPY chunks (id, document_id, tenant_id, allowed_roles, subtopic, kind, content, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as cp:
                cp.set_types(["int8", "uuid", "uuid", "text[]", "text", "text", "text", "vector"])
                for ch, v in zip(chunks, vectors, strict=True):
                    d = docs[ch.document + "|" + ch.kind]
                    cp.write_row(
                        (
                            ch.id,
                            d[0],
                            d[1],
                            list(ch.allowed_roles),
                            ch.subtopic,
                            ch.kind,
                            ch.text,
                            v,
                        )
                    )
        c.commit()
    with psycopg.connect(admin_url(DB), autocommit=True) as c:
        c.execute(
            "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        c.execute("ANALYZE chunks")


def teardown() -> None:
    with psycopg.connect(admin_url(), autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
        for r in ROLES:
            c.execute(f"DROP ROLE IF EXISTS {r}")


@contextmanager
def connect(role: str):
    # autocommit, so `with conn.transaction()` is a real transaction. Without it the first statement
    # opens one implicitly, a later transaction() block becomes a savepoint inside it, and SET LOCAL
    # then lasts until the outer transaction ends: the principal outlives the request.
    with psycopg.connect(role_url(role), autocommit=True) as c:
        register_vector(c)
        yield c
