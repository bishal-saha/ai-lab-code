"""Planner-executor: the model writes the plan; code validates it and runs it step by step.

    read ticket (code) -> plan (LLM) -> validate (code) --invalid--> replan
      for each step: executor (LLM, may only fill this step's tool, or skip a conditional step)
                     -> run the call (code) --error--> replan with the results so far
      at most MAX_REPLANS plans after the first

Validation is the article's rule set: only registered tools, and a return-eligibility
check before any money action. A plan is a model output like any other, so it can be
wrong; how often, and whether replanning recovers, is what this design measures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from four_shapes.episode import POLICY, Context, Stop

MAX_REPLANS = 2
MAX_STEPS = 10
MONEY = ("issue_refund", "escalate_to_finance")
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "The tool this step calls"},
                    "purpose": {
                        "type": "string",
                        "description": "What the step is for, and any condition on running it",
                    },
                },
                "required": ["tool", "purpose"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}
EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call", "skip"]},
        "args_json": {
            "type": "string",
            "description": "The tool's arguments as a JSON object; {} when skipping",
        },
        "note": {"type": "string"},
    },
    "required": ["action", "args_json", "note"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Step:
    tool: str
    purpose: str


def validate(steps: list[Step], registered: set[str], checked_already: bool = False) -> list[str]:
    """Problems with a plan, in words the planner can act on. Empty means valid.

    `checked_already`: an earlier plan's eligibility check already succeeded, so a replan
    covering only the remaining steps may start with the money action.
    """
    problems = []
    if not steps:
        problems.append("the plan has no steps")
    if len(steps) > MAX_STEPS:
        problems.append(f"the plan has {len(steps)} steps; use at most {MAX_STEPS}")
    for st in steps:
        if st.tool not in registered:
            problems.append(
                f"step '{st.tool}' is not a registered tool; use only: {', '.join(sorted(registered))}"
            )
    first_check = next(
        (i for i, st in enumerate(steps) if st.tool == "check_return_eligibility"), None
    )
    for i, st in enumerate(steps):
        if st.tool in MONEY and not checked_already and (first_check is None or first_check > i):
            problems.append(
                f"step {i + 1} ({st.tool}) moves money before check_return_eligibility; check first"
            )
            break
    return problems


def _tool_lines(ctx: Context) -> str:
    return "\n".join(
        f"- {t['name']}: {t['description'].splitlines()[0]}"
        for t in ctx.tools()
        if t["name"] != "get_ticket"
    )


def _history(done: list[tuple[str, dict, str, bool]]) -> str:
    if not done:
        return "(nothing yet)"
    return "\n".join(
        f"{i + 1}. {tool}({json.dumps(args)}) -> {'ERROR: ' if err else ''}{text}"
        for i, (tool, args, text, err) in enumerate(done)
    )


async def run(ctx: Context) -> None:
    s, ep = ctx.s, ctx.ep
    ticket, err = await ctx.call("get_ticket", {"ticket_id": s.ticket_id})
    if err:
        raise Stop("breaker", ticket)
    registered = {t["name"] for t in ctx.tools()} - {"get_ticket"}
    schemas = {t["name"]: t["input_schema"] for t in ctx.tools()}
    planner_system = f"""You are the planner for a Kestrel Outfitters support desk. You don't call tools yourself:
you write an ordered plan of tool steps that an executor carries out one at a time.

{POLICY}

Tools:
{_tool_lines(ctx)}

Each step names one tool and says what it is for. A step may be conditional on earlier results (say so in its
purpose, for example "only if the window is open and the total is USD 500 or less"); the executor fills in the
arguments and skips a step whose condition isn't met. The ticket has already been read."""

    done: list[tuple[str, dict, str, bool]] = []
    feedback = ""
    attempt = 0
    while True:
        user = f"Today is {s.today}.\n\nTicket:\n{ticket}\n\nSteps already run:\n{_history(done)}"
        if feedback:
            user += f"\n\nYour previous plan could not be used: {feedback}\nWrite the plan for what remains."
        raw = await ctx.json(
            "plan",
            planner_system,
            user,
            PLAN_SCHEMA,
            meta={"attempt": attempt, "done": [d[0] for d in done if not d[3]]},
        )
        steps = [Step(st["tool"], st["purpose"]) for st in raw.get("steps", [])]
        ep.notes.append("plan: " + " > ".join(st.tool for st in steps))
        checked = any(
            tool == "check_return_eligibility" and not failed for tool, _, _, failed in done
        )
        problems = validate(steps, registered, checked)
        if problems:
            ep.invalid_plans += 1
            feedback = "; ".join(problems)
        else:
            feedback = ""
            for st in steps:
                decision = await ctx.json(
                    "execute",
                    f"You are the executor. Carry out one plan step for a Kestrel Outfitters support ticket by giving the "
                    f"arguments for its tool, or skip it if its condition isn't met.\n\n{POLICY}",
                    f"Today is {s.today}.\n\nTicket:\n{ticket}\n\nSteps already run:\n{_history(done)}\n\n"
                    f"This step: {st.tool} - {st.purpose}\nIts arguments (JSON schema): {json.dumps(schemas[st.tool])}",
                    EXECUTE_SCHEMA,
                    meta={"tool": st.tool},
                )
                if decision["action"] == "skip":
                    continue
                try:
                    args = json.loads(decision["args_json"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as bad:
                    done.append((st.tool, {}, f"invalid arguments: {bad}", True))
                    feedback = f"step {st.tool} had invalid arguments"
                    break
                text, err = await ctx.call(st.tool, args)
                done.append((st.tool, args, text, err))
                if err:
                    feedback = f"step {st.tool} failed: {text}"
                    break
            if not feedback:
                return
        if attempt >= MAX_REPLANS:
            raise Stop("breaker", f"no usable plan after {attempt + 1} attempts: {feedback}")
        attempt += 1
        ep.replans += 1
