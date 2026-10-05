"""
Định nghĩa các kiểm thử cho Evaluation của ba mẫu Agent
"""

from evaluation.evaluate import (
    PATTERNS,
    run_evaluation,
    summarize_runs,
)
from evaluation.scenarios import SCENARIOS


def find_run(
    runs,
    scenario: str,
    pattern: str,
):
    for run in runs:
        if run.scenario == scenario and run.pattern == pattern:
            return run

    raise AssertionError("The requested evaluation run was not found.")


def test_evaluation_runs_every_pattern_on_every_scenario():
    runs = run_evaluation()

    assert len(runs) == (len(SCENARIOS) * len(PATTERNS))

    pairs = {
        (
            run.scenario,
            run.pattern,
        )
        for run in runs
    }

    assert len(pairs) == len(runs)


def test_happy_path_completes_for_all_patterns():
    runs = run_evaluation()

    for pattern in PATTERNS:
        run = find_run(
            runs,
            scenario="happy_path",
            pattern=pattern,
        )

        assert run.success is True
        assert run.status == "complete"
        assert run.trace_clarity is True


def test_recovery_scenario_distinguishes_pattern_adaptability():
    runs = run_evaluation()

    react = find_run(
        runs,
        scenario=("recoverable_constraint_violation"),
        pattern="react",
    )

    plan = find_run(
        runs,
        scenario=("recoverable_constraint_violation"),
        pattern="plan_then_execute",
    )

    hybrid = find_run(
        runs,
        scenario=("recoverable_constraint_violation"),
        pattern="hybrid",
    )

    assert react.success is True
    assert react.error_recovery is True
    assert react.adaptability is True

    assert plan.success is False
    assert plan.error_recovery is False
    assert plan.adaptability is False

    assert hybrid.success is True
    assert hybrid.error_recovery is True
    assert hybrid.adaptability is True


def test_payment_denied_handoffs_for_all_patterns():
    runs = run_evaluation()

    for pattern in PATTERNS:
        run = find_run(
            runs,
            scenario="payment_denied",
            pattern=pattern,
        )

        assert run.success is False
        assert run.human_handoff_count == 1

        assert run.first_failure is not None
        assert run.debuggability is True


def test_plan_and_hybrid_count_planning_search_as_tool_call():
    runs = run_evaluation()

    plan = find_run(
        runs,
        scenario="happy_path",
        pattern="plan_then_execute",
    )

    hybrid = find_run(
        runs,
        scenario="happy_path",
        pattern="hybrid",
    )

    assert plan.tool_calls == plan.total_steps + 1

    assert hybrid.tool_calls == hybrid.total_steps + 1


def test_summary_is_derived_from_run_results():
    runs = run_evaluation()
    summary = summarize_runs(runs)

    assert set(summary) == set(PATTERNS)

    for pattern in PATTERNS:
        assert summary[pattern]["scenario_count"] == len(SCENARIOS)

        assert 0 <= summary[pattern]["success_rate"] <= 1

        assert summary[pattern]["avg_model_calls"] > 0
