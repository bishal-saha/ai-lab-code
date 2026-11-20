"""Run the benchmark: free retrieval first, then the two paid steps.

    uv run cut-here --retrieval                     # 7 strategies x 3 retrievers x 204 questions, offline, $0
    uv run cut-here --estimate                      # what the paid steps should cost
    uv run --env-file .env cut-here --contextualize --max-usd 2    # S+H+C contexts (Claude Haiku), cached
    uv run cut-here --retrieval                     # again: now includes S+H+C
    uv run --env-file .env cut-here --answer --max-usd 8           # end-to-end answers (Haiku, Sonnet)
    uv run cut-here --dry-run                       # every step with a fake model, in a temp folder

Retrieval rows go to results/retrieval.jsonl (rewritten each run, deterministic);
answers to results/answers.jsonl (appended; a rerun skips what is there).
Then `uv run python -m cut_here.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from cut_here.chunkers import CONTEXTUAL, STRATEGIES
from cut_here.chunkers import contextual as ctx
from cut_here.chunkers.base import EMBED_MODEL, Chunk, count
from cut_here.corpus import Doc, build, dataset
from cut_here.grading import answer_correct, first_rank, miss_reason
from cut_here.index import RETRIEVERS, Index
from cut_here.models import ClaudeModel, FakeModel, Usage
from cut_here.pricing import cost_usd, resolve
from cut_here.questions import Question, build_questions

ROOT = Path(__file__).resolve().parents[2]
ANSWER_STRATEGIES = ["F512o", "R512", "S+H", CONTEXTUAL]
ANSWER_RETRIEVER = "hybrid"
ANSWER_K = 5


def all_chunks(docs: list[Doc], cache: Path) -> dict[str, list[Chunk]]:
    out = {name: fn(docs) for name, fn in STRATEGIES.items()}
    with contextlib.suppress(LookupError):  # contexts not written yet
        out[CONTEXTUAL] = ctx.apply(out["S+H"], cache)
    return out


def retrieval(
    docs: list[Doc], questions: list[Question], cache: Path, out: Path
) -> dict[str, dict]:
    chunk_sets = all_chunks(docs, cache)
    stats, rows = {}, []
    texts = [q.text for q in questions]
    for name, chunks in chunk_sets.items():
        started = time.perf_counter()
        index = Index(chunks)
        index_s = time.perf_counter() - started
        stats[name] = {
            "chunks": len(chunks),
            "mean_tokens": round(sum(count(c.embed_text) for c in chunks) / len(chunks), 1),
            "embedded_tokens": sum(count(c.embed_text) for c in chunks),
            "index_seconds": round(index_s, 2),
        }
        for retriever in RETRIEVERS:
            for q, ranked in zip(questions, index.search(texts, retriever, 10), strict=True):
                top = [chunks[i] for i in ranked]
                rank, span_rank = first_rank(top, q), first_rank(top, q, need_context=False)
                rows.append(
                    {
                        "strategy": name,
                        "retriever": retriever,
                        "qid": q.id,
                        "qtype": q.qtype,
                        "rank": rank,
                        "span_rank": span_rank,
                        "miss5": miss_reason(rank, span_rank, 5),
                        "top": [c.id for c in top],
                    }
                )
        print(f"  {name:6} {len(chunks):4} chunks  indexed in {index_s:5.1f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "retrieval.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (out / "chunks.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _summary(rows_path: Path) -> None:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    strategies = list(dict.fromkeys(r["strategy"] for r in rows))
    print(f"\nrecall@5 {'':6} " + " ".join(f"{r:>7}" for r in RETRIEVERS))
    for s in strategies:
        cells = []
        for r in RETRIEVERS:
            sel = [x for x in rows if x["strategy"] == s and x["retriever"] == r]
            cells.append(
                f"{sum(x['rank'] is not None and x['rank'] <= 5 for x in sel) / len(sel):7.0%}"
            )
        print(f"  {s:12} " + " ".join(cells))


async def answers(
    docs, questions, cache: Path, out: Path, models: list, max_usd: float, concurrency: int
) -> float:
    chunk_sets = all_chunks(docs, cache)
    missing = [s for s in ANSWER_STRATEGIES if s not in chunk_sets]
    if missing:
        raise SystemExit(f"no chunks for {missing}: run --contextualize first")
    path = out / "answers.jsonl"
    done = set()
    if path.exists():
        done = {
            (r["strategy"], r["model"], r["qid"])
            for r in map(json.loads, path.read_text().splitlines())
        }
    texts = [q.text for q in questions]
    usage, spent, gate = Usage(), 0.0, asyncio.Semaphore(concurrency)
    path.parent.mkdir(parents=True, exist_ok=True)

    async def one(strategy: str, chunks: list[Chunk], top: list[int], q: Question, model) -> None:
        nonlocal spent
        async with gate:
            if spent >= max_usd:
                return
            passages = [chunks[i].embed_text for i in top]
            given, u = await model.answer(q.text, passages)
            spent += usage.add(model.name, u)
            row = {
                "strategy": strategy,
                "model": model.name,
                "qid": q.id,
                "qtype": q.qtype,
                "expected": q.answer,
                "given": given,
                "correct": answer_correct(q.answer, given),
                "hit5": first_rank([chunks[i] for i in top], q) is not None,
                "cost_usd": cost_usd(
                    model.name,
                    u["input_tokens"],
                    u["output_tokens"],
                    u.get("cache_read", 0),
                    u.get("cache_write", 0),
                ),
            }
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    jobs = []
    for strategy in ANSWER_STRATEGIES:
        chunks = chunk_sets[strategy]
        ranked = Index(chunks).search(texts, ANSWER_RETRIEVER, ANSWER_K)
        for model in models:
            jobs += [
                one(strategy, chunks, top, q, model)
                for q, top in zip(questions, ranked, strict=True)
                if (strategy, model.name, q.id) not in done
            ]
    await asyncio.gather(*jobs)
    return spent


def estimate(docs: list[Doc], questions: list[Question], cache: Path) -> None:
    """Characters/4 on the real prompts; output 60 tokens a context, 30 an answer. No caching counted."""
    sh = STRATEGIES["S+H"](docs)
    by_doc = {d.id: d.text for d in docs}
    have = ctx.load(cache)
    todo = [c for c in sh if ctx.key(c) not in have]
    ctx_in = sum((len(by_doc[c.doc]) + len(c.text) + 400) // 4 for c in todo)
    ctx_cost = cost_usd("haiku", ctx_in, 60 * len(todo))
    print(f"contextualize: {len(todo)} chunks to write -> Haiku ${ctx_cost:.2f}")
    chunk_sets = {s: STRATEGIES[s](docs) for s in ANSWER_STRATEGIES if s in STRATEGIES}
    chunk_sets[CONTEXTUAL] = [
        Chunk(c.id, c.doc, c.spans, c.text, "x" * 240 + c.embed_text) for c in sh
    ]  # ~60-token contexts
    per_q = {
        s: sum(len(c.embed_text) for c in ch) / len(ch) * ANSWER_K / 4 + 150
        for s, ch in chunk_sets.items()
    }
    total = ctx_cost
    for m in ("haiku", "sonnet"):
        c = sum(
            cost_usd(m, int(per_q[s]) * len(questions), 30 * len(questions))
            for s in ANSWER_STRATEGIES
        )
        total += c
        print(
            f"answers, {resolve(m)}: {len(ANSWER_STRATEGIES)} strategies x {len(questions)} questions -> ${c:.2f}"
        )
    print(f"total: ${total:.2f}")


def dry_run(docs: list[Doc], questions: list[Question]) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sh = STRATEGIES["S+H"](docs)
        asyncio.run(ctx.contextualize(docs, sh, FakeModel(), tmp / "contexts.jsonl", max_usd=1e9))
        retrieval(docs, questions, tmp / "contexts.jsonl", tmp)
        fake = FakeModel({q.text: q.answer for q in questions})
        asyncio.run(answers(docs, questions, tmp / "contexts.jsonl", tmp, [fake], 1e9, 16))
        rows = [json.loads(line) for line in (tmp / "answers.jsonl").read_text().splitlines()]
        retrieved = {
            json.loads(line)["strategy"]
            for line in (tmp / "retrieval.jsonl").read_text().splitlines()
        }
    problems = []
    if retrieved != set(STRATEGIES) | {CONTEXTUAL}:
        problems.append(f"retrieval covered {sorted(retrieved)}")
    if len(rows) != len(ANSWER_STRATEGIES) * len(questions):
        problems.append(
            f"{len(rows)} answer rows, expected {len(ANSWER_STRATEGIES) * len(questions)}"
        )
    # The fake answers right only when the answer text is in a passage, so a hit must mean correct.
    wrong_on_hit = [r for r in rows if r["hit5"] and not r["correct"]]
    if wrong_on_hit:
        problems.append(f"{len(wrong_on_hit)} hits graded wrong, e.g. {wrong_on_hit[0]}")
    print(f"Dry run: retrieval for {len(retrieved)} strategies, {len(rows)} fake answers.")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "Dry run OK: contexts, retrieval, answering and grading all run end to end with a fake model."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag, text in [
        ("--retrieval", "free, offline"),
        ("--contextualize", "paid: S+H+C contexts"),
        ("--answer", "paid: end-to-end answers"),
        ("--estimate", "projected cost"),
        ("--dry-run", "fake model, temp folder"),
    ]:
        mode.add_argument(flag, action="store_true", help=text)
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    docs, _, _ = build(dataset())
    questions = build_questions(docs)
    cache, out = ROOT / "cache" / "contexts.jsonl", ROOT / "results"
    if args.dry_run:
        return dry_run(docs, questions)
    if args.estimate:
        estimate(docs, questions, cache)
        return 0
    if args.retrieval:
        stats = retrieval(docs, questions, cache, out)
        _summary(out / "retrieval.jsonl")
        meta = {
            "embed_model": EMBED_MODEL,
            "questions": len(questions),
            "docs": len(docs),
            "strategies": list(stats),
        }
    elif args.contextualize:
        spent, n = asyncio.run(
            ctx.contextualize(
                docs,
                STRATEGIES["S+H"](docs),
                ClaudeModel("haiku"),
                cache,
                max_usd=args.max_usd,
                concurrency=args.concurrency,
            )
        )
        print(f"contextualized {n} chunks for ${spent:.3f}")
        meta = {"contextualize_usd": round(spent, 4)}
    else:
        models = [ClaudeModel(m) for m in args.models.split(",") if m]
        spent = asyncio.run(
            answers(docs, questions, cache, out, models, args.max_usd, args.concurrency)
        )
        print(f"answers: ${spent:.3f}")
        meta = {"answer_models": [m.name for m in models], "answer_usd_this_run": round(spent, 4)}
    run_json = out / "run.json"
    previous = json.loads(run_json.read_text()) if run_json.exists() else {}
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(
        json.dumps(
            {
                **previous,
                **meta,
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "packages": {
                    p: version(p)
                    for p in ("anthropic", "fastembed", "rank-bm25", "kestrel-outfitters")
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
