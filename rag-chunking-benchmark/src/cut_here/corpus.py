"""Kestrel Docs: the structured corpus every chunking strategy indexes.

Built by code from the Kestrel Outfitters dataset (seed 2026), so every fact a
question asks about sits at a known place in a known document:

    products/<sku>.md     40 product pages: overview, a specifications table, care, warranty, FAQ
    shipping-rates.md     one guide: a section per carrier, a sub-section per region, a rate table in each
    warehouses/<n>.md     2 warehouse pages: a dispatch table and a 40-row stock table
    policies/<n>.md       the 4 real Kestrel policy documents, the outdated shipping policy included

The product spec tables all share their column names and the shipping tables all
share theirs: a row means nothing without its heading. That is the article's
"Pro vs Enterprise rate limit" problem, 56 times over.

    uv run python -m cut_here.corpus --write    # refresh corpus/ (committed, so it can be read)
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

from kestrel.data import generate
from kestrel.services import WAREHOUSES, KestrelData

SEED = 2026
SCALE = 0.1
ROOT = Path(__file__).resolve().parents[2]

MATERIALS = {
    "Camping": [
        "Recycled ripstop nylon",
        "Polyester canvas",
        "Aluminium alloy",
        "Down-filled nylon",
        "Stainless steel",
    ],
    "Garden": [
        "Powder-coated steel",
        "Carbon steel",
        "Recycled polypropylene",
        "FSC oak",
        "Galvanised steel",
    ],
    "Home storage": ["Recycled polypropylene", "Bamboo", "Powder-coated steel", "Seagrass", "Pine"],
    "Outdoor furniture": [
        "FSC acacia",
        "Powder-coated aluminium",
        "Teak",
        "Recycled HDPE",
        "Rattan-effect resin",
    ],
    "Small appliances": [
        "Brushed stainless steel",
        "BPA-free plastic",
        "Borosilicate glass",
        "Cast aluminium",
        "Ceramic-coated steel",
    ],
}
WATERPROOF = ["Not waterproof", "IPX2", "IPX4", "IPX5", "IPX6", "IPX7"]
WARRANTY = [1, 2, 3, 5, 10]
COLOURS = ["Moss, Slate", "Charcoal", "Sand, Ember", "Natural", "Forest, Stone", "White, Graphite"]
ASSEMBLY = [5, 10, 15, 20, 30, 45, 60]
REGIONS = ["Domestic", "Canada", "Europe", "Asia-Pacific"]
CARRIERS = ["Northline Freight", "Swiftway Parcel", "Harbour Express", "Tallpine Logistics"]
CARE = {
    "Camping": "Brush off dirt and let it dry fully before packing it away. Store it loose, out of direct sunlight.",
    "Garden": "Rinse after use and dry before storing. Oil moving parts at the start of each season.",
    "Home storage": "Wipe clean with a damp cloth. Keep away from radiators and direct heat.",
    "Outdoor furniture": "Wash with mild soapy water and cover or store indoors over winter.",
    "Small appliances": "Unplug and let it cool before cleaning. Wipe the outside with a damp cloth; never immerse the base.",
}


@dataclass(frozen=True)
class Doc:
    id: str  # path under corpus/, e.g. "products/sku-00001.md"
    title: str
    text: str


@dataclass(frozen=True)
class Spec:
    sku: str
    name: str
    category: str
    weight: str
    material: str
    waterproof: str
    colours: str
    warranty: str
    assembly: int
    max_load: int


def dataset() -> KestrelData:
    return KestrelData(generate(SEED, SCALE))


def specs(d: KestrelData) -> list[Spec]:
    rng = random.Random(SEED + 4)
    out = []
    for p in sorted(d.dataset.products, key=lambda p: p.sku):
        out.append(
            Spec(
                sku=p.sku,
                name=p.name,
                category=p.category,
                weight=f"{p.weight_kg:.1f} kg",
                material=rng.choice(MATERIALS[p.category]),
                waterproof=rng.choice(WATERPROOF),
                colours=rng.choice(COLOURS),
                warranty=f"{rng.choice(WARRANTY)} years".replace("1 years", "1 year"),
                assembly=rng.choice(ASSEMBLY),
                max_load=rng.choice([20, 50, 100, 120, 150, 200]),
            )
        )
    return out


def product_page(s: Spec) -> str:
    return f"""# {s.name}

The {s.name} is part of the Kestrel Outfitters {s.category.lower()} range, designed for everyday use and built to last through years of it. Every unit is inspected at our warehouse before it ships.

Most customers set it up on the day it arrives. The {s.name} takes about {s.assembly} minutes to assemble, and no tools beyond those in the box are needed.

## Specifications

| Specification | Value |
|---|---|
| SKU | {s.sku} |
| Category | {s.category} |
| Weight | {s.weight} |
| Material | {s.material} |
| Waterproof rating | {s.waterproof} |
| Colours | {s.colours} |
| Maximum load | {s.max_load} kg |
| Warranty | {s.warranty} |

## Care

{CARE[s.category]} Following these steps keeps the warranty valid.

## Warranty

The warranty covers manufacturing defects from the date of delivery. It does not cover wear and tear, accidental damage or misuse. To make a claim, contact Support with your order number and a photo of the fault; we repair, replace or refund at our discretion.

## FAQ

**Can I return it if it doesn't suit me?** Yes. Unused items can be returned under our returns policy.

**Do you sell spare parts?** Common spare parts are available through Support while the product is in our catalogue.
"""


def rates(rng: random.Random) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    out = {}
    for carrier in CARRIERS:
        for i, region in enumerate(REGIONS):
            base = [3, 6, 8, 10][i] + rng.choice([0, 1, 2])
            std = (base, base + 2)
            exp = (max(1, base - 3 + rng.choice([0, 1])), max(2, base - 1))
            price = [6.95, 14.95, 19.95, 24.95][i] + rng.choice([0, 2, 4])
            rows = [
                ("Standard", f"{std[0]}–{std[1]} business days", f"USD {price:.2f}"),
                ("Express", f"{exp[0]}–{exp[1]} business days", f"USD {price * 2 + 4:.2f}"),
            ]
            if region == "Domestic":
                rows.append(("Overnight", "1 business day", f"USD {price * 4 + 3:.2f}"))
            out[(carrier, region)] = rows
    return out


def shipping_guide(table: dict) -> str:
    parts = [
        "# Shipping rates and delivery times\n",
        "Kestrel Outfitters ships with four carriers. The tables below give the delivery time and price for each "
        "service, by carrier and destination region. Delivery times count from dispatch, not from the order date. "
        "Prices are per parcel up to 20 kg; bulky items are quoted at checkout.\n",
    ]
    for carrier in CARRIERS:
        parts.append(f"## {carrier}\n")
        parts.append(f"Rates and delivery times when your order ships with {carrier}.\n")
        for region in REGIONS:
            parts.append(f"### {region}\n")
            parts.append("| Service | Delivery time | Price |\n|---|---|---|")
            parts.append(
                "\n".join(
                    f"| {svc} | {days} | {price} |" for svc, days, price in table[(carrier, region)]
                )
                + "\n"
            )
    return "\n".join(parts)


def warehouse_page(name: str, d: KestrelData, all_specs: list[Spec], cutoff: str) -> str:
    rows = []
    for s in all_specs:
        on_hand = next(w["on_hand"] for w in d.stock(s.sku)["warehouses"] if w["warehouse"] == name)
        rows.append(f"| {s.sku} | {s.name} | {on_hand} |")
    return (
        f"""# {name} warehouse

The {name} warehouse handles picking, packing and dispatch for its share of Kestrel Outfitters orders. Stock levels below are counted every night.

## Dispatch

| Order placed by | Ships |
|---|---|
| {cutoff} on a working day | Same day |
| After {cutoff} | Next working day |

## Stock levels

Units on hand at the last nightly count, by product.

| SKU | Product | Units on hand |
|---|---|---|
"""
        + "\n".join(rows)
        + "\n"
    )


def build(d: KestrelData | None = None) -> tuple[list[Doc], list[Spec], dict]:
    d = d or dataset()
    all_specs = specs(d)
    rng = random.Random(SEED + 5)
    table = rates(rng)
    docs = [Doc(f"products/{s.sku.lower()}.md", s.name, product_page(s)) for s in all_specs]
    docs.append(
        Doc("shipping-rates.md", "Shipping rates and delivery times", shipping_guide(table))
    )
    for name, cutoff in zip(WAREHOUSES, ["14:00", "12:00"], strict=True):
        slug = re.sub(r"[^a-z]+", "-", name.lower()).strip("-")
        docs.append(
            Doc(
                f"warehouses/{slug}.md",
                f"{name} warehouse",
                warehouse_page(name, d, all_specs, cutoff),
            )
        )
    for name, body in sorted(d.policies.items()):
        docs.append(
            Doc(
                f"policies/{name}.md",
                body.splitlines()[0].lstrip("# "),
                body if body.endswith("\n") else body + "\n",
            )
        )
    return docs, all_specs, table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the documents to corpus/")
    args = parser.parse_args(argv)
    docs, _, _ = build()
    words = sum(len(doc.text.split()) for doc in docs)
    print(f"{len(docs)} documents, {words:,} words")
    if args.write:
        for doc in docs:
            path = ROOT / "corpus" / doc.id
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(doc.text)
        print(f"written to {ROOT / 'corpus'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
