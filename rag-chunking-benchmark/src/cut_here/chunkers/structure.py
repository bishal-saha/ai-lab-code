"""Structure-aware chunking: let the document draw the boundaries.

Split on Markdown headings; keep each section whole when it fits. A section that
doesn't fit is packed block by block (paragraphs, lists, tables); a table that
doesn't fit is split by rows with its header row repeated in every piece, because
`| POST /v1/exports | 120 |` means nothing without its column names.

With `heading_path=True` (strategy S+H) the path of headings above the section
(`Ridgeline 40L pack > Specifications`) is prefixed to the embedded text, so a
short chunk still says what it is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cut_here.chunkers.base import Chunk, count, make
from cut_here.chunkers.recursive import _split
from cut_here.corpus import Doc

HEADING = re.compile(r"^(#{1,6}) (.+)$")


@dataclass
class Section:
    path: list[str]
    heading: tuple[int, int] | None  # the heading line's span
    blocks: list[tuple[str, int, int]] = field(
        default_factory=list
    )  # (kind, start, end): para | table


def sections(text: str) -> list[Section]:
    out: list[Section] = [Section([], None)]
    stack: list[tuple[int, str]] = []
    block: tuple[str, int, int] | None = None
    pos = 0
    for line in text.splitlines(keepends=True):
        start, end, body = pos, pos + len(line.rstrip("\n")), line.strip()
        pos += len(line)
        m = HEADING.match(line.rstrip("\n"))
        kind = "table" if body.startswith("|") else "para"
        if m or not body or (block and block[0] != kind):
            if block:
                out[-1].blocks.append(block)
            block = None
        if m:
            level = len(m.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2).strip())]
            out.append(Section([t for _, t in stack], (start, end)))
        elif body:
            block = (kind, block[1], end) if block else (kind, start, end)
    if block:
        out[-1].blocks.append(block)
    return [s for s in out if s.heading or s.blocks]


def _table_pieces(text: str, start: int, end: int, limit: int) -> list[list[tuple[int, int]]]:
    lines, at = [], start
    for line in text[start:end].split("\n"):
        lines.append((at, at + len(line)))
        at += len(line) + 1
    header, rows = lines[:2], lines[2:]
    pieces, current = [], []
    for row in rows:
        trial = header + current + [row]
        if current and count("\n".join(text[a:b] for a, b in trial)) > limit:
            pieces.append(header + current)
            current = []
        current.append(row)
    if current:
        pieces.append(header + current)
    return pieces


def structure(docs: list[Doc], limit: int, name: str, heading_path: bool) -> list[Chunk]:
    out = []
    for doc in docs:
        t, n = doc.text, 0
        for sec in sections(t):
            prefix = (" > ".join(sec.path) + "\n\n") if heading_path and sec.path else ""
            whole = [sec.heading] if sec.heading else []
            whole += [(a, b) for _, a, b in sec.blocks]
            pieces: list[list[tuple[int, int]]]
            if count("\n".join(t[a:b] for a, b in whole)) <= limit:
                pieces = [whole]
            else:
                pieces, current = [], [sec.heading] if sec.heading else []
                for kind, a, b in sec.blocks:
                    parts = (
                        _table_pieces(t, a, b, limit)
                        if kind == "table" and count(t[a:b]) > limit
                        else [[span] for span in _split(t, a, b, limit, ["\n\n", "\n", ". ", " "])]
                    )
                    for part in parts:
                        if current and count("\n".join(t[x:y] for x, y in current + part)) > limit:
                            pieces.append(current)
                            current = []
                        current = current + part
                if current:
                    pieces.append(current)
            for spans in pieces:
                out.append(make(name, doc.id, n, spans, t, prefix))
                n += 1
    return out
