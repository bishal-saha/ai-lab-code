"""The seven conditions: one cost lever changed at a time from a cached Sonnet baseline.

cache        Sonnet on every call, append-only automatic prompt caching (the baseline)
no-cache     caching off
cache-break  a timestamp at the top of the system prompt, new on every call: the silent break
             the article warns about. Tool definitions come before the system prompt, so they
             can still be read from the cache; the system prompt and history cannot
cap          keep the last 2 tool exchanges word for word; older ones are folded into a running
             summary (written by Haiku, its cost counted)
haiku        Haiku on every call
phase        the first 2 calls (read the ticket, first lookup) on Haiku, the rest on Sonnet
switch       Haiku and Sonnet alternate on every call
"""

from __future__ import annotations

from dataclasses import dataclass

CONDITIONS = ["cache", "no-cache", "cache-break", "cap", "haiku", "phase", "switch"]

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"


@dataclass(frozen=True)
class Condition:
    id: str
    caching: bool = True
    timestamp: bool = False
    history_cap: int | None = None  # tool exchanges kept word for word
    cheap_first: int = 0  # phase routing: the first N calls on Haiku
    alternate: bool = False  # per-step switching
    all_haiku: bool = False

    def model_for(self, step: int) -> str:
        """The model for the 0-based call `step` of an attempt."""
        if self.all_haiku or step < self.cheap_first or (self.alternate and step % 2 == 0):
            return HAIKU
        return SONNET


def condition(cid: str) -> Condition:
    return {
        "cache": Condition("cache"),
        "no-cache": Condition("no-cache", caching=False),
        "cache-break": Condition("cache-break", timestamp=True),
        "cap": Condition("cap", history_cap=2),
        "haiku": Condition("haiku", all_haiku=True),
        "phase": Condition("phase", cheap_first=2),
        "switch": Condition("switch", alternate=True),
    }[cid]
