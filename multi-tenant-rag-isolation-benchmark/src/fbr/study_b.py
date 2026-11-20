"""Study B: which isolation layer catches which bug (deterministic, no LLM).

The tutorial's cross-tenant probe, with controls, is run under every (defence, bug) pair:

    defences  filter   the query filters by tenant and role; RLS off
              rls      the query has no filter; the RLS policy filters
              both     the query filter and RLS
    bugs      none           the setup as the tutorial builds it
              no-tenant      the query forgets the tenant filter
              no-role        the query forgets the role filter
              owner          the app connects as the table owner, and RLS isn't FORCEd
              session-set    an earlier request set the principal with session-level SET on this
                             pooled connection, and this request's principal step was skipped
              no-nullif      the policy casts the setting without NULLIF, on a connection where an
                             earlier transaction set the principal (so the setting is now '')

The probe: an Acme end user searches with the vectors of a Globex help article, a Globex runbook
and an Acme runbook (each one's nearest chunk is itself).
    negative controls  no Globex row comes back (tenant), no Acme runbook comes back (role)
    positive control   the same caller, probing with an Acme help chunk, gets Acme help back

Each cell ends as `leaked` (a negative control failed), `error` (the query raised), `blind` (safe,
but the positive control failed: the caller can't see its own data) or `safe`.
"""

from __future__ import annotations

import psycopg

from fbr import db

DEFENCES = ["filter", "rls", "both"]
BUGS = ["none", "no-tenant", "no-role", "owner", "session-set", "no-nullif"]
ASKER, ROLE, OTHER = "Acme", "end_user", "Globex"


def _vector(c, chunk_id: int):
    return c.execute("SELECT embedding FROM chunks WHERE id = %s", (chunk_id,)).fetchone()[0]


def probes() -> dict:
    """Pick the probe chunks: a Globex runbook, an Acme runbook and an Acme help article."""
    with db.connect("fbr_ingest") as c:
        pick = lambda tenant, kind: c.execute(  # noqa: E731
            "SELECT id FROM chunks WHERE tenant_id = %s AND kind = %s ORDER BY id LIMIT 1",
            (db.tenant_uuid(tenant), kind),
        ).fetchone()[0]
        # a tenant leak must be probed with a chunk the asker's role may see (a help article):
        # probing only with a runbook lets the role filter hide a missing tenant filter
        ids = {
            "other_help": pick(OTHER, "help"),
            "other_runbook": pick(OTHER, "runbook"),
            "own_runbook": pick(ASKER, "runbook"),
            "own_help": pick(ASKER, "help"),
        }
        return {k: (v, _vector(c, v)) for k, v in ids.items()}


def _search(c, qvec, *, tenant_filter: bool, role_filter: bool) -> list[tuple]:
    conds, args = [], []
    if tenant_filter:
        conds.append("tenant_id = %s")
        args.append(db.tenant_uuid(ASKER))
    if role_filter:
        conds.append("allowed_roles && ARRAY[%s]::text[]")
        args.append(ROLE)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return c.execute(
        f"SELECT id, tenant_id, kind FROM chunks {where} ORDER BY embedding <=> %s LIMIT 10",
        (*args, qvec),
    ).fetchall()


def cell(defence: str, bug: str, probe: dict) -> dict:
    rls = defence in ("rls", "both")
    query_filter = defence in ("filter", "both")
    db.set_rls(rls, force=bug != "owner", nullif=bug != "no-nullif")
    role = "fbr_owner" if bug == "owner" else "fbr_app"
    if defence == "filter" and role == "fbr_app":
        pass  # RLS is off: fbr_app reads whatever the query asks for
    outcome = {"defence": defence, "bug": bug}
    try:
        with db.connect(role) as c:
            if (
                bug == "session-set"
            ):  # an earlier request on this pooled connection, using session SET
                c.execute(
                    "SELECT set_config('app.tenant_id', %s, false), set_config('app.roles', 'agent', false)",
                    (str(db.tenant_uuid(OTHER)),),
                )
            if (
                bug == "no-nullif"
            ):  # an earlier transaction on this connection set the principal locally
                with c.transaction():
                    c.execute(
                        "SELECT set_config('app.tenant_id', %s, true), set_config('app.roles', %s, true)",
                        (str(db.tenant_uuid(ASKER)), ROLE),
                    )
            runs = {}
            for name, (_cid, qvec) in probe.items():
                with c.transaction():
                    if bug not in (
                        "session-set",
                        "no-nullif",
                    ):  # the normal request path sets the principal
                        c.execute(
                            "SELECT set_config('app.tenant_id', %s, true), set_config('app.roles', %s, true)",
                            (str(db.tenant_uuid(ASKER)), ROLE),
                        )
                    runs[name] = _search(
                        c,
                        qvec,
                        tenant_filter=query_filter and bug != "no-tenant",
                        role_filter=query_filter and bug != "no-role",
                    )
    except psycopg.Error as err:
        return outcome | {
            "result": "error",
            "detail": type(err).__name__ + ": " + str(err).splitlines()[0],
        }
    other_tid = db.tenant_uuid(OTHER)
    leaked_tenant = any(r[1] == other_tid for rows in runs.values() for r in rows)
    leaked_role = any(
        r[1] == db.tenant_uuid(ASKER) and r[2] == "runbook" for rows in runs.values() for r in rows
    )
    own_help_seen = any(r[0] == probe["own_help"][0] for r in runs["own_help"])
    result = "leaked" if (leaked_tenant or leaked_role) else "safe" if own_help_seen else "blind"
    return outcome | {
        "result": result,
        "leaked_tenant": leaked_tenant,
        "leaked_role": leaked_role,
        "positive_control": own_help_seen,
        "rows_returned": {k: len(v) for k, v in runs.items()},
    }


def run() -> list[dict]:
    probe = probes()
    try:
        return [cell(d, b, probe) for d in DEFENCES for b in BUGS]
    finally:
        db.set_rls(True, force=True, nullif=True)  # leave the tutorial's setup in place
