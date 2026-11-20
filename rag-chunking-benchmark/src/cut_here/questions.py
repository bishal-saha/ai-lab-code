"""About 200 questions, each with its answer's exact place in the corpus.

Every question records the answer span (document, start, end) and the **context**
a retrieved chunk must also carry for the answer to be usable: a spec-table row
is only an answer if the same chunk says which product it belongs to. Four types:

    table     a product spec row (needs the product name) or a stock row (needs the warehouse)
    heading   a shipping-rate row: needs the carrier *and* the region, both only in headings
    prose     a sentence that carries its own context (product prose, policy facts)

Grading (grading.py) counts a hit only when one chunk holds the span and its context.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cut_here.corpus import CARRIERS, REGIONS, SEED, Doc, build, dataset

# (question, document, the exact text holding the answer, the answer as a reader states it)
POLICY_FACTS = [
    (
        "How long do trade customers have to return an item?",
        "policies/returns-policy.md",
        "trade customers within 14 days",
        "14 days",
    ),
    (
        "How many days do consumers have to return unused items?",
        "policies/returns-policy.md",
        "within 30 days of delivery",
        "30 days",
    ),
    (
        "How quickly are damaged or wrong items refunded once received?",
        "policies/returns-policy.md",
        "within 5 working days of receiving the item",
        "5 working days",
    ),
    (
        "Up to what amount can Support approve a refund?",
        "policies/returns-policy.md",
        "approve refunds up to USD 500",
        "USD 500",
    ),
    (
        "What is the order cut-off time for same-day shipping under the current shipping policy?",
        "policies/shipping-policy.md",
        "Orders placed before 14:00",
        "14:00",
    ),
    (
        "What is the current standard delivery promise?",
        "policies/shipping-policy.md",
        "Standard delivery promise: 3 working days",
        "3 working days",
    ),
    (
        "What is the delivery promise for bulky items over 20 kg?",
        "policies/shipping-policy.md",
        "Bulky (over 20 kg): 5 working days",
        "5 working days",
    ),
    (
        "Which warehouse ships bulky items?",
        "policies/shipping-policy.md",
        "bulky items ship from North (Eastgate)",
        "North (Eastgate)",
    ),
    (
        "How often is carrier performance reviewed?",
        "policies/shipping-policy.md",
        "reviewed every quarter",
        "every quarter",
    ),
    (
        "Within how many days must expense claims be submitted?",
        "policies/expense-policy.md",
        "submitted within 60 days",
        "60 days",
    ),
    (
        "What is the meals expense limit?",
        "policies/expense-policy.md",
        "| Meals | USD 90 per day |",
        "USD 90 per day",
    ),
    (
        "Who approves equipment expense claims?",
        "policies/expense-policy.md",
        "| Always | Finance (Priya Nair) |",
        "Finance (Priya Nair)",
    ),
]
SPEC_FIELDS = {
    "warranty": ("Warranty", "What is the warranty on the {name}?"),
    "material": ("Material", "What is the {name} made of?"),
    "waterproof": ("Waterproof rating", "What is the waterproof rating of the {name}?"),
    "weight": ("Weight", "How much does the {name} weigh?"),
}


@dataclass(frozen=True)
class Question:
    id: str
    qtype: str  # table | heading | prose
    text: str
    answer: str
    doc: str
    start: int
    end: int
    context: tuple[str, ...]  # strings the same chunk must also contain

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


def _locate(doc: Doc, needle: str, after: int = 0) -> tuple[int, int]:
    i = doc.text.find(needle, after)
    if i < 0:
        raise ValueError(f"{needle!r} not found in {doc.id}")
    return i, i + len(needle)


def build_questions(docs: list[Doc] | None = None) -> list[Question]:
    if docs is None:
        docs, specs, table = build(dataset())
    else:
        _, specs, table = build(dataset())
    by_id = {d.id: d for d in docs}
    rng = random.Random(SEED + 6)
    out: list[Question] = []

    def add(qtype, text, answer, doc_id, needle, context=(), after=0):
        start, end = _locate(by_id[doc_id], needle, after)
        out.append(
            Question(
                f"q{len(out) + 1:03d}", qtype, text, answer, doc_id, start, end, tuple(context)
            )
        )

    # table: two spec rows per product (the product name lives only in the H1 and the prose)
    for s in specs:
        for field in rng.sample(sorted(SPEC_FIELDS), 2):
            label, template = SPEC_FIELDS[field]
            value = getattr(s, field)
            add(
                "table",
                template.format(name=s.name),
                value,
                f"products/{s.sku.lower()}.md",
                f"| {label} | {value} |",
                [s.name],
            )

    # table: stock rows, 20 products per warehouse (the warehouse lives only in the H1)
    for doc in (d for d in docs if d.id.startswith("warehouses/")):
        warehouse = doc.title.removesuffix(" warehouse")
        for s in rng.sample(specs, 20):
            row_start = doc.text.index(f"| {s.sku} | {s.name} |")
            row = doc.text[row_start : doc.text.index("\n", row_start)]
            units = row.rstrip(" |").rsplit("|", 1)[1].strip()
            add(
                "table",
                f"How many units of the {s.name} are on hand at the {warehouse} warehouse?",
                units,
                doc.id,
                row,
                [warehouse],
            )

    # heading: shipping rows; carrier (H2) and region (H3) appear only in headings
    guide = by_id["shipping-rates.md"].text
    for carrier in CARRIERS:
        for region in REGIONS:
            # Rows repeat across tables, so search from this carrier's region heading.
            section = guide.index(f"### {region}\n", guide.index(f"## {carrier}\n"))
            for svc, days, price in rng.sample(table[(carrier, region)], 2):
                if rng.random() < 0.5:
                    text, answer = (
                        f"How long does {svc} delivery to {region} take with {carrier}?",
                        days,
                    )
                else:
                    text, answer = (
                        f"What does {svc} shipping to {region} cost with {carrier}?",
                        price,
                    )
                add(
                    "heading",
                    text,
                    answer,
                    "shipping-rates.md",
                    f"| {svc} | {days} | {price} |",
                    [carrier, region],
                    after=section,
                )

    # prose: the assembly sentence names its product; policy facts carry their own context
    for s in specs:
        sentence = f"The {s.name} takes about {s.assembly} minutes to assemble"
        add(
            "prose",
            f"How long does the {s.name} take to assemble?",
            f"{s.assembly} minutes",
            f"products/{s.sku.lower()}.md",
            sentence,
        )
    for text, doc_id, needle, answer in POLICY_FACTS:
        add("prose", text, answer, doc_id, needle)
    return out
