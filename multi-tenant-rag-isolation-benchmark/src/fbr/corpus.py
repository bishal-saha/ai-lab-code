"""A synthetic multi-tenant SaaS support corpus in the tutorial's shape, big enough to show the effects.

The tutorial's companion code has two tenants (Acme and Globex) with three or four documents each.
Filtering after the top k only starves a tenant when it shares an index with much larger
neighbours, so this corpus has 30 tenants of very different sizes in one index:

    one large tenant (about 30% of all chunks), four medium ones, and 25 small ones down to about
    0.2% of the index

Every tenant writes about the same 32 subtopics (exports, API keys, SSO, billing, webhooks ...)
in its own product names and with its own facts, so a question lands close to every tenant's
answer: the realistic hard case for a tenant filter. A fact (a limit, a duration, a menu path)
depends only on the tenant and the subtopic, so all of a tenant's chunks on a subtopic agree and
other tenants' values differ: a grader can tell a right answer from another tenant's.

Help articles are visible to agents and end users; runbooks only to agents. Each chunk carries its
document's tenant and ACL, as in the tutorial's schema. Everything is seeded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SEED = 2026
PLANS = ("Starter", "Team", "Enterprise")
REGIONS = ("US", "EU", "APAC")
WORDS = (
    "Ledger",
    "Beacon",
    "Harbor",
    "Atlas",
    "Pulse",
    "Quarry",
    "Vertex",
    "Cinder",
    "Meadow",
    "Summit",
    "Lumen",
    "Orbit",
    "Tundra",
    "Willow",
    "Fathom",
    "Granite",
    "Juniper",
    "Kestrel",
    "Marlin",
    "Nimbus",
)
COMPANIES = (
    "Acme",
    "Globex",
    "Initech",
    "Umbrella",
    "Hooli",
    "Vandelay",
    "Stark",
    "Wayne",
    "Tyrell",
    "Cyberdyne",
    "Soylent",
    "Wonka",
    "Gringotts",
    "Monarch",
    "Oscorp",
    "Pied Piper",
    "Aperture",
    "Black Mesa",
    "Massive Dynamic",
    "Nakatomi",
    "Prestige",
    "Sterling",
    "Virtucon",
    "Duff",
    "Krusty",
    "Gekko",
    "Bluth",
    "Dunder",
    "Weyland",
    "Zorg",
)
SIZES = [3600, 1200, 900, 700, 500] + [
    round(500 * (25 / 500) ** (i / 24)) for i in range(25)
]  # 500 ... 25


def _pick(rng: random.Random, options):
    return options[rng.randrange(len(options))]


# (key, kind, topic, question, chunk template, fact generator). The fact named "key" is what a
# grader looks for in an answer; templates use {p} (product), {plan}, {region} and the facts.
SUBTOPICS = [
    (
        "export-schedule",
        "help",
        "Data exports",
        "How often can I schedule a full data export?",
        "Full data exports in {p} can be scheduled every {key} on the {plan} plan; each export is kept for {keep} days in the {region} region before it is deleted.",
        lambda r: {
            "key": _pick(r, ["6 hours", "12 hours", "24 hours", "7 days"]),
            "keep": _pick(r, ["7", "14", "30"]),
        },
    ),
    (
        "export-format",
        "help",
        "Data exports",
        "Which file formats can a data export use?",
        "{p} exports data as {key}. On the {plan} plan in the {region} region, files over {cap} GB are split into numbered parts.",
        lambda r: {
            "key": _pick(r, ["CSV and JSON", "Parquet and CSV", "JSON Lines only", "CSV only"]),
            "cap": _pick(r, ["2", "5", "10"]),
        },
    ),
    (
        "export-stuck",
        "runbook",
        "Data exports",
        "How do I recover a data export that is stuck?",
        "Runbook: if a {p} export has been running for more than {key}, cancel it from the job console, clear the export lock with `exports unlock --tenant`, and re-run it. Escalate to the {region} on-call after {tries} failed retries.",
        lambda r: {
            "key": _pick(r, ["45 minutes", "2 hours", "90 minutes", "4 hours"]),
            "tries": _pick(r, ["2", "3", "5"]),
        },
    ),
    (
        "export-region",
        "runbook",
        "Data exports",
        "Where are export files stored?",
        "Runbook: {p} writes export files to the {key} bucket for {region} customers on the {plan} plan; access is logged and keys rotate every {rot} days.",
        lambda r: {
            "key": _pick(r, ["exports-primary", "cold-archive", "tenant-dumps", "export-vault"]),
            "rot": _pick(r, ["30", "60", "90"]),
        },
    ),
    (
        "apikey-rotate",
        "help",
        "API keys",
        "How do I rotate an API key without downtime?",
        "To rotate a {p} API key without downtime, create a second key, deploy it, then revoke the old key within {key}. The {plan} plan allows {n} active keys per workspace in {region}.",
        lambda r: {
            "key": _pick(r, ["24 hours", "48 hours", "7 days", "72 hours"]),
            "n": _pick(r, ["2", "3", "5"]),
        },
    ),
    (
        "apikey-limit",
        "help",
        "API keys",
        "How many API keys can a workspace have?",
        "Each {p} workspace on the {plan} plan can hold up to {key}; keys unused for {idle} days are flagged for review in the {region} console.",
        lambda r: {
            "key": _pick(r, ["3 active keys", "5 active keys", "10 active keys", "25 active keys"]),
            "idle": _pick(r, ["30", "60", "90"]),
        },
    ),
    (
        "apikey-leak",
        "runbook",
        "API keys",
        "What do I do if an API key is leaked?",
        "Runbook: for a leaked {p} key, revoke it immediately, then search the audit log for calls from unknown IPs over the last {key}. Notify the {region} security channel and the customer within {notify} hours.",
        lambda r: {
            "key": _pick(r, ["30 days", "14 days", "72 hours", "7 days"]),
            "notify": _pick(r, ["4", "12", "24"]),
        },
    ),
    (
        "apikey-scope",
        "help",
        "API keys",
        "Can I limit what an API key is allowed to do?",
        "{p} API keys can be scoped to {key}. Scopes are set when the key is created on the {plan} plan and cannot be widened later in {region}.",
        lambda r: {
            "key": _pick(
                r,
                ["read-only or read-write", "single projects", "named endpoints", "IP allowlists"],
            )
        },
    ),
    (
        "sso-setup",
        "help",
        "Single sign-on",
        "How do I set up single sign-on?",
        "To set up SSO in {p}, open {key}, upload your identity provider's metadata and test with one user before enforcing it. SSO needs the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "Settings › Security › SSO",
                    "Admin › Identity",
                    "Organisation › Login",
                    "Workspace › Access › SAML",
                ],
            )
        },
    ),
    (
        "sso-lockout",
        "runbook",
        "Single sign-on",
        "How do I let an admin in when SSO is broken?",
        "Runbook: when SSO locks every admin out of {p}, issue a break-glass login with `access grant --breakglass`, valid for {key}, after the account owner confirms by phone. Log it in the {region} incident tracker.",
        lambda r: {"key": _pick(r, ["15 minutes", "1 hour", "30 minutes", "4 hours"])},
    ),
    (
        "sso-scim",
        "help",
        "Single sign-on",
        "Does single sign-on provision users automatically?",
        "{p} provisions users through SCIM {key}. Deprovisioned users lose access within {lag} minutes on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "on the Enterprise plan only",
                    "on every paid plan",
                    "after a support request",
                    "with Okta and Azure AD",
                ],
            ),
            "lag": _pick(r, ["5", "15", "60"]),
        },
    ),
    (
        "sso-cert",
        "runbook",
        "Single sign-on",
        "What happens when the SSO certificate expires?",
        "Runbook: {p} warns admins {key} before an SSO signing certificate expires; if it lapses, logins fail closed. Rotate it from the identity console and re-test in {region}.",
        lambda r: {"key": _pick(r, ["30 days", "14 days", "7 days", "60 days"])},
    ),
    (
        "billing-invoice",
        "help",
        "Billing",
        "When are invoices sent?",
        "{p} sends invoices on the {key}. On the {plan} plan in {region}, payment is due within {due} days.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "1st of each month",
                    "last day of each month",
                    "anniversary of signup",
                    "15th of each month",
                ],
            ),
            "due": _pick(r, ["14", "30", "45"]),
        },
    ),
    (
        "billing-refund",
        "help",
        "Billing",
        "Can I get a refund for unused seats?",
        "{p} refunds unused seats {key}. Refunds on the {plan} plan in {region} go back to the original payment method.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "within 14 days of purchase",
                    "as account credit only",
                    "within 30 days of purchase",
                    "only on annual plans",
                ],
            )
        },
    ),
    (
        "billing-failed",
        "runbook",
        "Billing",
        "What happens when a card payment fails?",
        "Runbook: after a failed {p} payment the account enters a {key} grace period, then becomes read-only. Retry the charge from the billing console; {region} finance approves extensions.",
        lambda r: {"key": _pick(r, ["7-day", "14-day", "3-day", "21-day"])},
    ),
    (
        "billing-tax",
        "help",
        "Billing",
        "How do I add a VAT number?",
        "Add a VAT number in {p} under {key}; it applies from the next invoice on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "Billing › Tax details",
                    "Account › Company profile",
                    "Settings › Invoicing",
                    "Admin › Billing entity",
                ],
            )
        },
    ),
    (
        "webhook-retry",
        "help",
        "Webhooks",
        "How many times is a failed webhook retried?",
        "{p} retries a failed webhook {key} with exponential backoff, then disables the endpoint. The {plan} plan in {region} keeps delivery logs for {logs} days.",
        lambda r: {
            "key": _pick(r, ["5 times", "8 times", "3 times", "12 times"]),
            "logs": _pick(r, ["7", "14", "30"]),
        },
    ),
    (
        "webhook-sign",
        "help",
        "Webhooks",
        "How do I verify a webhook signature?",
        "Each {p} webhook carries an {key} header; compute an HMAC-SHA256 of the body with your signing secret and compare. Secrets rotate per endpoint on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r, ["X-Signature", "Webhook-Signature", "X-Hub-Signature-256", "Svix-Signature"]
            )
        },
    ),
    (
        "webhook-replay",
        "runbook",
        "Webhooks",
        "How do I replay webhooks after an outage?",
        "Runbook: replay {p} webhooks with `hooks replay --since`, at most {key} per run, after the customer's endpoint is healthy. Announce it in the {region} status channel first.",
        lambda r: {
            "key": _pick(r, ["10,000 events", "50,000 events", "1,000 events", "5,000 events"])
        },
    ),
    (
        "webhook-ip",
        "help",
        "Webhooks",
        "Which IP addresses do webhooks come from?",
        "{p} sends webhooks from {key} in the {region} region; allowlist them on the {plan} plan to block other senders.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "a fixed /28 range",
                    "three static IPs",
                    "the published egress list",
                    "region-specific NAT gateways",
                ],
            )
        },
    ),
    (
        "user-invite",
        "help",
        "Users",
        "How do I invite a teammate?",
        "Invite teammates to {p} from {key}; invitations expire after {exp} days on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r, ["People › Invite", "Settings › Members", "Team › Add member", "Admin › Users"]
            ),
            "exp": _pick(r, ["3", "7", "14"]),
        },
    ),
    (
        "user-delete",
        "runbook",
        "Users",
        "How do I permanently delete a user's data?",
        "Runbook: permanent deletion in {p} runs `users purge` after a {key} soft-delete window; it cannot be undone and is logged for {region} compliance.",
        lambda r: {"key": _pick(r, ["30-day", "14-day", "7-day", "90-day"])},
    ),
    (
        "user-roles",
        "help",
        "Users",
        "What roles can a user have?",
        "{p} has {key} roles on the {plan} plan in {region}; custom roles need the Enterprise plan.",
        lambda r: {
            "key": _pick(r, ["three built-in", "four built-in", "two built-in", "five built-in"])
        },
    ),
    (
        "user-2fa",
        "help",
        "Users",
        "Can I require two-factor authentication?",
        "Admins can require 2FA in {p} under {key}; users get {grace} days to enrol on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "Security › Two-factor",
                    "Settings › Login security",
                    "Admin › Policies",
                    "Organisation › MFA",
                ],
            ),
            "grace": _pick(r, ["3", "7", "14"]),
        },
    ),
    (
        "limit-rate",
        "help",
        "Limits",
        "What is the API rate limit?",
        "The {p} API allows {key} on the {plan} plan in {region}; bursts above it get HTTP 429 with a Retry-After header.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "100 requests per minute",
                    "600 requests per minute",
                    "50 requests per second",
                    "1,000 requests per minute",
                ],
            )
        },
    ),
    (
        "limit-storage",
        "help",
        "Limits",
        "How much storage is included?",
        "Each {p} workspace on the {plan} plan includes {key} of storage in {region}; extra storage is billed per GB.",
        lambda r: {"key": _pick(r, ["10 GB", "50 GB", "100 GB", "1 TB"])},
    ),
    (
        "limit-raise",
        "runbook",
        "Limits",
        "How do I raise a customer's rate limit?",
        "Runbook: raise a {p} rate limit with `limits set` for at most {key}, with a ticket reference; permanent raises need {region} engineering approval.",
        lambda r: {"key": _pick(r, ["72 hours", "7 days", "24 hours", "30 days"])},
    ),
    (
        "limit-files",
        "help",
        "Limits",
        "What is the largest file I can upload?",
        "{p} accepts uploads up to {key} on the {plan} plan in {region}; larger files must use the multipart API.",
        lambda r: {"key": _pick(r, ["100 MB", "250 MB", "2 GB", "5 GB"])},
    ),
    (
        "outage-status",
        "help",
        "Incidents",
        "Where can I see if there is an outage?",
        "{p} posts incidents on {key} and emails admins on the {plan} plan in {region}.",
        lambda r: {
            "key": _pick(
                r, ["status page", "in-app banner", "the trust centre", "the community forum"]
            )
        },
    ),
    (
        "outage-sev",
        "runbook",
        "Incidents",
        "When is an incident a SEV1?",
        "Runbook: a {p} incident is SEV1 when {key}; page the {region} incident commander within {page} minutes.",
        lambda r: {
            "key": _pick(
                r,
                [
                    "logins fail for every tenant",
                    "data loss is suspected",
                    "the API error rate passes 5%",
                    "exports fail for over an hour",
                ],
            ),
            "page": _pick(r, ["5", "10", "15"]),
        },
    ),
    (
        "outage-credit",
        "help",
        "Incidents",
        "Do I get credit for downtime?",
        "{p} credits {key} of the monthly fee per hour below the {plan} plan's uptime target in {region}.",
        lambda r: {"key": _pick(r, ["5%", "10%", "2%", "25%"])},
    ),
    (
        "outage-postmortem",
        "runbook",
        "Incidents",
        "When is a postmortem due after an incident?",
        "Runbook: {p} postmortems are due within {key} of a SEV1 or SEV2 and are shared with affected {region} customers.",
        lambda r: {"key": _pick(r, ["5 business days", "72 hours", "10 days", "48 hours"])},
    ),
]
# Questions no tenant's documents answer: every answer to them is unsupported.
NOWHERE = [
    "How do I connect a hardware security key to the mobile app?",
    "What is the data retention policy for voice call recordings?",
    "How do I migrate dashboards from a competitor's product?",
    "Can I host the product on my own servers?",
    "What does the loyalty points programme include?",
    "How do I change the language of email notifications?",
]


@dataclass(frozen=True)
class Chunk:
    id: int
    tenant: str
    document: str  # tenant/product/topic
    subtopic: str
    kind: str  # help | runbook
    allowed_roles: tuple[str, ...]
    text: str


@dataclass
class Tenant:
    name: str
    products: list[str]
    size: int
    subtopics: set[str] = field(default_factory=set)
    facts: dict[str, dict] = field(default_factory=dict)


def build() -> tuple[list[Tenant], list[Chunk]]:
    rng = random.Random(SEED)
    tenants, chunks = [], []
    for i, size in enumerate(SIZES):
        name = COMPANIES[i]
        # small tenants cover only some subtopics, so some questions are unanswerable for them
        n_sub = len(SUBTOPICS) if i < 5 else max(4, min(len(SUBTOPICS) - 8, size // 9))
        covered = set(rng.sample([s[0] for s in SUBTOPICS], n_sub))
        # enough products that (subtopic x product x plan x region) reaches the tenant's size
        n_products = -(-size // (n_sub * len(PLANS) * len(REGIONS)))
        products = [f"{name} {w}" for w in rng.sample(WORDS, n_products)]
        t = Tenant(name, products, size)
        for key, *_rest, factgen in SUBTOPICS:
            t.facts[key] = factgen(random.Random(f"{SEED}:{name}:{key}"))
        combos = [
            (s, p, plan, reg)
            for s in SUBTOPICS
            if s[0] in covered
            for p in products
            for plan in PLANS
            for reg in REGIONS
        ]
        rng.shuffle(combos)
        for (key, kind, topic, _q, template, _f), p, plan, reg in combos[:size]:
            facts = t.facts[key]
            where = (
                "help centre" if kind == "help" else "internal runbook"
            )  # every chunk names its plan and region
            text = f"{p} {where}, {topic} ({plan} plan, {reg}): " + template.format(
                p=p, plan=plan, region=reg, **facts
            )
            chunks.append(
                Chunk(
                    len(chunks) + 1,
                    name,
                    f"{name}/{p}/{topic}",
                    key,
                    kind,
                    ("agent", "end_user") if kind == "help" else ("agent",),
                    text,
                )
            )
            t.subtopics.add(key)
        tenants.append(t)
    return tenants, chunks


@dataclass(frozen=True)
class Question:
    id: str
    tenant: str
    role: str
    text: str
    subtopic: str | None
    case: str  # answerable | other-tenant | nowhere
    fact: str | None  # the fact a right answer contains (answerable only)


def questions(tenants: list[Tenant]) -> list[Question]:
    """Study C's questions. Small and medium tenants (1-4 products, so one value per fact),
    agents asking, so runbooks are in scope."""
    rng = random.Random(SEED + 1)
    by_key = {s[0]: s for s in SUBTOPICS}
    out: list[Question] = []
    for t in tenants[5:25]:
        have = sorted(t.subtopics)
        lack = sorted(
            k
            for k in by_key
            if k not in t.subtopics and any(k in o.subtopics for o in tenants if o is not t)
        )
        for k in rng.sample(have, min(2, len(have))):
            out.append(
                Question(
                    f"{t.name}:a:{k}",
                    t.name,
                    "agent",
                    by_key[k][3],
                    k,
                    "answerable",
                    t.facts[k]["key"],
                )
            )
        for k in rng.sample(lack, min(2, len(lack))):
            out.append(
                Question(f"{t.name}:o:{k}", t.name, "agent", by_key[k][3], k, "other-tenant", None)
            )
        q = NOWHERE[rng.randrange(len(NOWHERE))]
        out.append(
            Question(f"{t.name}:n:{NOWHERE.index(q)}", t.name, "agent", q, None, "nowhere", None)
        )
    return out


def retrieval_queries(tenants: list[Tenant], per_tenant: int = 12) -> list[tuple[str, str, str]]:
    """Study A's queries: (tenant, role, question) for subtopics each tenant covers, both roles."""
    rng = random.Random(SEED + 2)
    by_key = {s[0]: s for s in SUBTOPICS}
    out = []
    for t in tenants:
        for k in rng.sample(sorted(t.subtopics), min(per_tenant, len(t.subtopics))):
            out.append((t.name, rng.choice(["agent", "end_user"]), by_key[k][3]))
    return out
