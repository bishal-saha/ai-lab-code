"""The agent loop against the offline fake: answering, wrong answers, silence, the breaker."""

import pytest
from tool_shape.agent import run_episode
from tool_shape.models import FakeModel
from tool_shape.pricing import cost_usd


@pytest.mark.parametrize("server", ["endpoints", "tasks"])
async def test_every_task_completes_with_the_fake(servers, task_list, server):
    for task in task_list:
        ep = await run_episode(task, servers[server], server, FakeModel(task, server))
        assert ep.success, (task.id, ep.stop, ep.error, ep.wrong_fields)
        assert ep.stop == "answered" and ep.tool_errors == 0, task.id


async def test_wrong_answer_is_graded_wrong(servers, task_list):
    task = task_list[0]
    ep = await run_episode(task, servers["tasks"], "tasks", FakeModel(task, "tasks", "wrong"))
    assert ep.stop == "answered" and not ep.success
    assert ep.wrong_fields == [next(iter(task.fields))]


async def test_silent_model_gets_one_nudge_then_fails(servers, task_list):
    task = task_list[0]
    ep = await run_episode(task, servers["tasks"], "tasks", FakeModel(task, "tasks", "silent"))
    assert ep.nudged and ep.stop == "no_answer" and ep.turns == 2 and not ep.success


async def test_repeated_failing_call_trips_the_breaker(servers, task_list):
    task = task_list[0]
    ep = await run_episode(
        task, servers["endpoints"], "endpoints", FakeModel(task, "endpoints", "broken")
    )
    assert ep.stop == "breaker" and ep.tool_errors == 2 and ep.turns == 2


def test_cost_includes_cache_pricing():
    assert cost_usd("haiku", 1_000_000, 0) == pytest.approx(0.10)
    assert cost_usd("claude-sonnet-5-5", 0, 1_000_000) == pytest.approx(10.0)
    assert cost_usd("sonnet", 0, 0, cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(
        0.2 + 2.5
    )
