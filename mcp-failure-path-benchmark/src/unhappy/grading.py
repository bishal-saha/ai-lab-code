"""Grading one episode, in code, from the API's comment store, the agent's call log and its answer.

success         not_found, dropped_write: exactly one comment, on the right ticket, nothing elsewhere,
                outcome done
                enum, large, transient counts: the right count
                transient lookup: the answer names the ticket's status and priority
                permission: no comment, outcome could_not_complete, at most 2 attempts to post
duplicate       more than one comment on the target ticket
silent_wrong    outcome done, yet the task failed (an empty or truncated result taken as the truth,
                or a note reported as added that isn't there)
gave_up         outcome could_not_complete on a task that could be completed
repeats         calls identical to an earlier call that returned an error
post_attempts   calls to add_ticket_comment
"""

from __future__ import annotations

import json
from typing import Any

from unhappy.tasks import Task
from unhappy.ticketing import Comment


def grade(
    task: Task, comments: list[Comment], calls: list[dict[str, Any]], answer: dict | None
) -> dict[str, Any]:
    on_target = sum(c.ticket_id == task.comment_on for c in comments) if task.comment_on else 0
    elsewhere = len(comments) - on_target
    outcome = (answer or {}).get("outcome")
    if task.mode in ("not_found", "dropped_write"):
        ok = on_target == 1 and elsewhere == 0 and outcome == "done"
    elif task.mode == "permission":
        posts = sum(c["tool"] == "add_ticket_comment" for c in calls)
        ok = not comments and outcome == "could_not_complete" and posts <= 2
    elif task.count is not None:
        ok = outcome == "done" and (answer or {}).get("count") == task.count
    else:
        text = str((answer or {}).get("answer", "")).lower()
        ok = outcome == "done" and all(w in text for w in task.must_mention)
    errored: set[str] = set()
    repeats = 0
    for c in calls:
        k = c["tool"] + json.dumps(c["args"], sort_keys=True)
        if k in errored:
            repeats += 1
        if c["is_error"]:
            errored.add(k)
    return {
        "success": ok,
        "answered": answer is not None,
        "duplicate": on_target > 1,
        "comments_on_target": on_target,
        "comments_elsewhere": elsewhere,
        "silent_wrong": outcome == "done" and not ok,
        "gave_up": outcome == "could_not_complete" and not task.must_refuse,
        "repeats": repeats,
        "post_attempts": sum(c["tool"] == "add_ticket_comment" for c in calls),
        "answer_count": (answer or {}).get("count"),
    }
