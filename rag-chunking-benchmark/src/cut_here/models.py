"""Claude for the two paid steps, and an offline fake for the dry run.

contextualize  Claude Haiku writes a short context for each S+H chunk from its whole document
               (Anthropic's published contextual-retrieval prompt), document block prompt-cached
answer         Claude Haiku or Sonnet answers a question from the top-5 chunks, as structured output
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from cut_here.pricing import cost_usd, resolve

CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, quoting the value exactly as the documents give it, or 'not found'",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    by_model: dict[str, list[int]] = field(default_factory=dict)

    def add(self, model: str, u: dict[str, int]) -> float:
        self.calls += 1
        self.input_tokens += u["input_tokens"]
        self.output_tokens += u["output_tokens"]
        self.cache_read += u.get("cache_read", 0)
        self.cache_write += u.get("cache_write", 0)
        return cost_usd(
            model,
            u["input_tokens"],
            u["output_tokens"],
            u.get("cache_read", 0),
            u.get("cache_write", 0),
        )


class ClaudeModel:
    def __init__(
        self, model: str, *, effort: str = "low", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic()

    async def _create(self, **kwargs: Any) -> tuple[str, dict[str, int]]:
        r = await self.client.messages.create(model=self.name, max_tokens=2000, **kwargs)
        text = "".join(getattr(b, "text", "") for b in r.content if b.type == "text")
        u = r.usage
        return text, {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
        }

    async def context(self, document: str, chunk: str) -> tuple[str, dict[str, int]]:
        return await self._create(
            output_config={"effort": self.effort},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{document}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": CONTEXT_PROMPT.format(chunk=chunk)},
                    ],
                }
            ],
        )

    async def answer(self, question: str, passages: list[str]) -> tuple[str, dict[str, int]]:
        docs = "\n\n".join(
            f"<passage {i}>\n{p}\n</passage {i}>" for i, p in enumerate(passages, start=1)
        )
        text, usage = await self._create(
            system="Answer the question using only the passages from the Kestrel Outfitters documentation. "
            "Quote the value exactly as the passages give it. If the passages don't contain the answer, say 'not found'.",
            messages=[{"role": "user", "content": f"{docs}\n\nQuestion: {question}"}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
        )
        return json.loads(text)["answer"], usage


class FakeModel:
    """Offline stand-in: a context naming the document; the right answer only when it is in the passages."""

    name = "fake"

    def __init__(self, answers: dict[str, str] | None = None):
        self.answers = answers or {}

    async def context(self, document: str, chunk: str) -> tuple[str, dict[str, int]]:
        title = document.splitlines()[0].lstrip("# ")
        return f"This chunk is from the {title} page.", {
            "input_tokens": (len(document) + len(chunk)) // 4,
            "output_tokens": 20,
        }

    async def answer(self, question: str, passages: list[str]) -> tuple[str, dict[str, int]]:
        expected = self.answers.get(question, "")
        found = any(expected and expected in p for p in passages)
        return (expected if found else "not found"), {
            "input_tokens": sum(map(len, passages)) // 4,
            "output_tokens": 10,
        }
