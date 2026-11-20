"""200 support replies whose correct verdict is known by construction, and 100 reply pairs.

Each case is a refund ticket over a real Kestrel order (seed 2026). The policy
outcome is computed from the order, as in the earlier experiments:

    inside the window (30 days consumer, 14 trade) and total <= USD 500  -> refund the total
    inside the window and total > USD 500                                -> escalate to Finance
    outside the window                                                   -> decline

The reply is then written from templates as a PASS (it states that outcome and
promises nothing more) or one of five FAIL types:

    wrong_amount    refunds, but the wrong amount
    wrong_outcome   refunds a return the policy declines
    wrong_window    a trade customer outside 14 days, refunded "within our 30-day window"
    extra_promise   the right outcome plus a USD 40 credit the policy doesn't allow
    missing_fact    an escalation reply that never says what happens

Every reply exists short (two sentences) and long (the same content, padded with
friendly filler), so correctness and length are crossed for the verbosity test.
160 pass, 40 fail: most answers are fine, as in production.

    uv run python -m judge_the_judge.cases --write   # refresh cases/ (committed, so they can be read)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

from kestrel.data import generate
from kestrel.services import KestrelData

SEED = 2026
SCALE = 0.1
ROOT = Path(__file__).resolve().parents[2]
LIMIT = 500
WINDOW = {"consumer": 30, "trade": 14}
FAIL_TYPES = ["wrong_amount", "wrong_outcome", "wrong_window", "extra_promise", "missing_fact"]
POLICY = """Kestrel Outfitters returns policy:
- Consumers may return unused items within 30 days of delivery for a full refund of the order total; trade customers within 14 days.
- Support may refund up to USD 500. Above USD 500, Support escalates the refund to Finance and tells the customer so.
- Outside the window, the return is declined and the customer is told why.
- Support may not offer credits, discounts or goodwill payments."""
# Severity for the ordinal label used with the 1-10 score: 0 = the outcome itself is wrong, 1 = right outcome with a defect, 2 = pass.
SEVERITY = {
    "wrong_amount": 0,
    "wrong_outcome": 0,
    "wrong_window": 0,
    "extra_promise": 1,
    "missing_fact": 1,
    None: 2,
}

OPEN = ["Hi {first},", "Hello {first},", "Dear {first},"]
PAD = [
    "Thank you so much for getting in touch with us, and I'm sorry to hear the item didn't work out for you.",
    "We know how important it is that your outdoor gear fits the way you live, and we really appreciate you shopping with Kestrel Outfitters.",
    "I've looked carefully through your order history and the details of your request so that I could give you a complete answer today.",
    "If anything else comes up, whether it's about this order or anything in our range, our team is always happy to help.",
    "Thanks again for your patience while we looked into this, and have a wonderful week ahead.",
]
CLOSE = [
    "Best wishes, Kestrel Support",
    "Kind regards, the Kestrel Support team",
    "All the best, Kestrel Outfitters Support",
]


@dataclass(frozen=True)
class Case:
    id: str
    label: str  # PASS | FAIL
    fail_type: str | None
    length: str  # short | long
    outcome: str  # refund | escalate | decline (what the policy requires)
    customer: str
    customer_type: str
    order_id: str
    total: float
    delivered: str
    today: str
    ticket: str
    reply: str

    @property
    def severity(self) -> int:
        return SEVERITY[self.fail_type]

    def facts(self) -> str:
        return (
            f"Order {self.order_id}: {self.customer_type} customer {self.customer}, order total USD {self.total:.2f}, "
            f"delivered {self.delivered}. Today is {self.today}."
        )


@dataclass(frozen=True)
class Pair:
    id: str
    kind: str  # correct_vs_wrong | short_vs_long (both correct)
    a: Case  # the "better" reply for correct_vs_wrong; the short one for short_vs_long
    b: Case


def core(outcome: str, kind: str, c: dict, fail: str | None) -> str:
    """The sentences that carry the decision: everything a judge needs to grade."""
    total, window = c["total"], WINDOW[c["customer_type"]]
    if fail == "wrong_amount":
        wrong = round(total - 7.95, 2) if total > 20 else round(total + 10, 2)
        return f"Your return is approved, and we've refunded USD {wrong:.2f} to your original payment method."
    if fail == "wrong_outcome":
        return f"Your return is approved, and we've refunded the full USD {total:.2f} to your original payment method."
    if fail == "wrong_window":
        return f"Your order is within our 30-day return window, so we've refunded the full USD {total:.2f} to your original payment method."
    if fail == "missing_fact":
        return "Thanks for sending this through. We've received your request about this order and it's with the right people now."
    base = {
        "refund": f"Your return is approved, and we've refunded the full USD {total:.2f} to your original payment method.",
        "escalate": f"Because your order total of USD {total:.2f} is above USD {LIMIT}, I've passed your refund to our Finance team, who will review it and contact you.",
        "decline": f"Your order was delivered on {c['delivered']}, which is outside our {window}-day return window, so I'm sorry but we can't accept this return.",
    }[outcome]
    if fail == "extra_promise":
        base += " As a thank-you, we've also added a USD 40 credit to your account."
    return base


def write_reply(outcome: str, c: dict, fail: str | None, length: str, rng: random.Random) -> str:
    first = c["customer"].split()[0]
    body = core(outcome, c["customer_type"], c, fail)
    if length == "short":
        return f"{rng.choice(OPEN).format(first=first)} {body} {rng.choice(CLOSE)}"
    pad = rng.sample(PAD, 3)
    return "\n\n".join(
        [
            rng.choice(OPEN).format(first=first),
            pad[0] + " " + pad[1],
            body,
            pad[2],
            rng.choice(CLOSE),
        ]
    )


def _pick(
    d: KestrelData,
    rng: random.Random,
    used: set[str],
    want: str,
    customer_type: str | None = None,
    days: int | None = None,
) -> dict:
    """An order and a 'today' that make the policy outcome `want`."""
    orders = sorted(
        (
            o
            for o in d.dataset.orders
            if o.currency == "USD"
            and o.id not in used
            and o.id in d.shipments_by_order
            and d.shipments_by_order[o.id].delivered_at
        ),
        key=lambda o: o.id,
    )
    while True:
        o = rng.choice(orders)
        cust = d.customers[o.customer_id]
        ctype = cust.type.value
        if customer_type and ctype != customer_type:
            continue
        total = o.total_cents / 100
        window = WINDOW[ctype]
        if want == "refund" and total > LIMIT:
            continue
        if want == "escalate" and total <= LIMIT:
            continue
        gap = (
            days
            if days is not None
            else (
                rng.randint(3, window)
                if want in ("refund", "escalate")
                else rng.randint(window + 5, window + 40)
            )
        )
        used.add(o.id)
        delivered = d.shipments_by_order[o.id].delivered_at.date()
        return {
            "order_id": o.id,
            "customer": cust.name,
            "customer_type": ctype,
            "total": total,
            "delivered": delivered.isoformat(),
            "today": (delivered + timedelta(days=gap)).isoformat(),
        }


def _case(
    n: int,
    c: dict,
    outcome: str,
    fail: str | None,
    length: str,
    rng: random.Random,
    prefix: str = "c",
) -> Case:
    ticket = f"Hi, I'd like to return my order {c['order_id']} for a refund. The item is unused. Thanks, {c['customer'].split()[0]}"
    return Case(
        f"{prefix}{n:03d}",
        "FAIL" if fail else "PASS",
        fail,
        length,
        outcome,
        c["customer"],
        c["customer_type"],
        c["order_id"],
        c["total"],
        c["delivered"],
        c["today"],
        ticket,
        write_reply(outcome, c, fail, length, rng),
    )


def build(d: KestrelData | None = None) -> tuple[list[Case], list[Pair]]:
    d = d or KestrelData(generate(SEED, SCALE))
    rng = random.Random(SEED + 8)
    used: set[str] = set()
    cases: list[Case] = []
    # 160 passes: outcomes mixed, half short, half long
    for i in range(160):
        outcome = ["refund", "refund", "escalate", "decline"][i % 4]
        cases.append(
            _case(
                len(cases) + 1,
                _pick(d, rng, used, outcome),
                outcome,
                None,
                "short" if i % 2 == 0 else "long",
                rng,
            )
        )
    # 40 fails: 8 per type, 4 short and 4 long
    for fail in FAIL_TYPES:
        for k in range(8):
            if fail == "wrong_outcome":
                c, outcome = _pick(d, rng, used, "decline"), "decline"
            elif fail == "wrong_window":
                c, outcome = (
                    _pick(d, rng, used, "decline", customer_type="trade", days=rng.randint(16, 29)),
                    "decline",
                )
            elif fail == "missing_fact":
                c, outcome = _pick(d, rng, used, "escalate"), "escalate"
            elif fail == "extra_promise":
                outcome = ["refund", "escalate", "decline"][k % 3]
                c = _pick(d, rng, used, outcome)
            else:  # wrong_amount
                c, outcome = _pick(d, rng, used, "refund"), "refund"
            cases.append(
                _case(len(cases) + 1, c, outcome, fail, "short" if k % 2 == 0 else "long", rng)
            )
    rng.shuffle(cases)
    pairs: list[Pair] = []
    for k in range(100):
        if k < 50:
            fail = FAIL_TYPES[k % len(FAIL_TYPES)]
            want = {
                "wrong_outcome": "decline",
                "wrong_window": "decline",
                "missing_fact": "escalate",
                "wrong_amount": "refund",
                "extra_promise": "refund",
            }[fail]
            c = _pick(
                d,
                rng,
                used,
                want,
                customer_type="trade" if fail == "wrong_window" else None,
                days=rng.randint(16, 29) if fail == "wrong_window" else None,
            )
            length = rng.choice(["short", "long"])
            good, bad = (
                _case(k, c, want, None, length, rng, "pg"),
                _case(k, c, want, fail, length, rng, "pb"),
            )
            pairs.append(Pair(f"p{k + 1:03d}", "correct_vs_wrong", good, bad))
        else:
            want = rng.choice(["refund", "escalate", "decline"])
            c = _pick(d, rng, used, want)
            short, long = (
                _case(k, c, want, None, "short", rng, "ps"),
                _case(k, c, want, None, "long", rng, "pl"),
            )
            pairs.append(Pair(f"p{k + 1:03d}", "short_vs_long", short, long))
    return cases, pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    cases, pairs = build()
    print(f"{len(cases)} cases, {len(pairs)} pairs")
    if args.write:
        folder = ROOT / "cases"
        folder.mkdir(exist_ok=True)
        (folder / "cases.jsonl").write_text(
            "".join(json.dumps({**asdict(c), "facts": c.facts()}) + "\n" for c in cases)
        )
        (folder / "pairs.jsonl").write_text(
            "".join(
                json.dumps({"id": p.id, "kind": p.kind, "a": asdict(p.a), "b": asdict(p.b)}) + "\n"
                for p in pairs
            )
        )
        (folder / "policy.txt").write_text(POLICY + "\n")
        print(f"written to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
