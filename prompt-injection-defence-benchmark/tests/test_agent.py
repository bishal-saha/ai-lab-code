"""The agent loop against the offline fake: completion, breaches, refusals, the breaker."""

from poisoned_ticket.agent import run_episode
from poisoned_ticket.attacks import ATTACKS
from poisoned_ticket.defences import BY_ID
from poisoned_ticket.models import FakeModel
from poisoned_ticket.pricing import cost_usd


async def test_obeying_agent_is_breached_without_enforcement(api, scenarios):
    s, a = scenarios[0], ATTACKS[0]
    ep = await run_episode(s, a, BY_ID["D0"], api, FakeModel(s, a, "obey"))
    assert ep.attack_success and ep.utility and ep.stop == "done" and ep.refused == 0


async def test_enforcement_turns_the_breach_into_a_refusal(api, scenarios):
    s, a = scenarios[0], ATTACKS[0]  # refund_other
    ep = await run_episode(s, a, BY_ID["D3"], api, FakeModel(s, a, "obey"))
    assert not ep.attack_success and ep.utility and ep.refused == 1 and ep.tool_errors == 1


async def test_resisting_agent_flags_and_completes(api, scenarios):
    s, a = scenarios[5], ATTACKS[5]
    ep = await run_episode(s, a, BY_ID["D1"], api, FakeModel(s, a, "resist"))
    assert ep.flagged and ep.utility and ep.attack_success is False


async def test_clean_episode_has_no_attack_verdict(api, scenarios):
    s = scenarios[2]
    ep = await run_episode(s, None, BY_ID["D4"], api, FakeModel(s, None))
    assert ep.attack == "clean" and ep.attack_success is None and ep.utility and not ep.flagged


def test_cost_includes_cache_pricing():
    assert cost_usd("haiku", 1_000_000, 0) == 0.10
    assert cost_usd("sonnet", 0, 0, cache_read=1_000_000, cache_write=1_000_000) == 2.7
