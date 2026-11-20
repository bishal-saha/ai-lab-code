from collections import Counter

from tool_shape.grading import grade
from tool_shape.tasks import answer_tool, build_tasks


def test_thirty_tasks_in_five_families(task_list):
    assert len(task_list) == 30
    assert Counter(t.family for t in task_list) == {
        "returns": 8,
        "ticket": 8,
        "delivery": 6,
        "stock": 4,
        "policy": 4,
    }
    assert len({t.id for t in task_list}) == 30


def test_task_set_is_deterministic(task_list):
    again = build_tasks()
    assert [(t.id, t.prompt, t.expected) for t in again] == [
        (t.id, t.prompt, t.expected) for t in task_list
    ]


def test_no_family_has_a_constant_answer(task_list):
    # A design can't score well by always saying yes (or no).
    for family in ("delivery", "returns", "ticket", "stock"):
        for field in ("late", "eligible", "delivered", "in_stock"):
            values = {
                t.expected[field] for t in task_list if t.family == family and field in t.expected
            }
            assert len(values) != 1, f"{family}.{field} is always {values}"


def test_answer_tool_is_strict_and_typed(task_list):
    tool = answer_tool(task_list[0])
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert set(tool["input_schema"]["required"]) == set(task_list[0].fields)


def test_grading_is_exact_but_ignores_case_and_spacing(task_list):
    task = next(t for t in task_list if t.family == "delivery")
    good = dict(task.expected, carrier=f"  {task.expected['carrier'].upper()} ")
    assert grade(task, good) == (True, [])
    bad = dict(task.expected, late=not task.expected["late"])
    assert grade(task, bad) == (False, ["late"])
    assert grade(task, None) == (False, list(task.fields))
