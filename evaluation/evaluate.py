"""
Đánh giá cả ba mẫu Agent trên cùng một tập kịch bản
"""

import json
from dataclasses import asdict, dataclass

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evaluation.scenarios import (
    SCENARIOS,
    EvaluationScenario,
)
from flight_agent.harness import FlightAgentHarness
from flight_agent.hybrid_agent import HybridFlightAgent
from flight_agent.plan_execute_agent import (
    FlightPlan,
    PlanStep,
    PlanThenExecuteFlightAgent,
)
from flight_agent.react_agent import ReActFlightAgent
from flight_agent.tools import reset_booking_store

PATTERNS = (
    "react",
    "plan_then_execute",
    "hybrid",
)


class CountingReActModel:
    """Model giả lập cho ReAct và đếm số lần được gọi."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = responses
        self.index = 0
        self.invoke_count = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, _):
        if self.index >= len(self.responses):
            raise RuntimeError("No scripted ReAct response is available.")

        response = self.responses[self.index]

        self.index += 1
        self.invoke_count += 1

        return response


class CountingPlannerModel:
    """Model giả lập cho Planner và đếm số lần được gọi."""

    def __init__(self, plans: list[FlightPlan]):
        self.plans = plans
        self.index = 0
        self.invoke_count = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return RunnableLambda(self._invoke)

    def _invoke(self, _):
        if self.index >= len(self.plans):
            raise RuntimeError("No scripted flight plan is available.")

        plan = self.plans[self.index]

        self.index += 1
        self.invoke_count += 1

        return plan


"""Các hàm hỗ trợ (Helper Function)"""


def tool_call_message(
    name: str,
    args: dict,
    call_id: str,
) -> AIMessage:
    """Tạo phản hồi Model chứa một yêu cầu gọi công cụ."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def flight_plan_for(
    flight_number: str,
) -> FlightPlan:
    """Tạo kế hoạch đặt vé hoàn chỉnh cho một chuyến bay."""
    return FlightPlan(
        steps=[
            PlanStep(
                tool="book_seat",
                args={
                    "flight_number": flight_number,
                },
            ),
            PlanStep(
                tool="pay",
                args={
                    "booking_code": "$booking_code",
                },
            ),
            PlanStep(
                tool="get_booking",
                args={
                    "booking_code": "$booking_code",
                },
            ),
        ]
    )


"""Tạo chuỗi phản hồi của từng Agent cho một kịch bản"""


def build_react_responses(
    scenario: EvaluationScenario,
) -> list[AIMessage]:
    """Tạo chuỗi phản hồi ReAct cho một kịch bản."""
    responses = [
        tool_call_message(
            name="search_flights",
            args={
                "origin": scenario.constraints.origin,
                "destination": scenario.constraints.destination,
                "date": scenario.constraints.date,
            },
            call_id="search-1",
        ),
        tool_call_message(
            name="book_seat",
            args={
                "flight_number": scenario.initial_flight_number,
            },
            call_id="book-1",
        ),
    ]

    final_flight_number = scenario.initial_flight_number

    if scenario.recovery_flight_number is not None:
        final_flight_number = scenario.recovery_flight_number

        responses.append(
            tool_call_message(
                name="book_seat",
                args={
                    "flight_number": final_flight_number,
                },
                call_id="book-2",
            )
        )

    responses.append(
        tool_call_message(
            name="pay",
            args={
                "booking_code": f"{final_flight_number}-1",
            },
            call_id="pay-1",
        )
    )

    return responses


def build_plan_sequence(
    scenario: EvaluationScenario,
    include_recovery: bool,
) -> list[FlightPlan]:
    """Tạo chuỗi kế hoạch từ cùng một kịch bản đánh giá."""
    plans = [flight_plan_for(scenario.initial_flight_number)]

    if include_recovery and scenario.recovery_flight_number is not None:
        plans.append(flight_plan_for(scenario.recovery_flight_number))

    return plans


"""Định nghĩa các Metrics"""


@dataclass(frozen=True)
class EvaluationRun:
    """Kết quả đánh giá một Pattern trên một Scenario."""

    scenario: str
    pattern: str
    status: str
    success: bool

    model_calls: int
    tool_calls: int
    total_steps: int

    constraint_violations: int
    human_handoff_count: int

    error_recovery: bool | None
    adaptability: bool | None

    trace_clarity: bool
    debuggability: bool | None

    first_failure: str | None


def first_failure_location(
    trace: list[dict],
) -> str | None:
    """Tìm vị trí Observation thất bại lần đầu tiên."""
    for entry in trace:
        observation = entry.get(
            "observation",
            {},
        )

        if observation.get("status") == "success":
            continue

        if "plan_number" in entry:
            return f"plan {entry['plan_number']}, " f"step {entry['step']}"

        return f"step {entry['step']}"

    return None


def count_constraint_violations(
    trace: list[dict],
) -> int:
    """Đếm tổng số Observation vi phạm ràng buộc."""
    count = 0

    for entry in trace:
        observation = entry.get(
            "observation",
            {},
        )

        if observation.get("status") == "constraint_violation":
            count += 1

    return count


def has_clear_trace(
    pattern: str,
    trace: list[dict],
) -> bool:
    """Kiểm tra Trace có đủ dữ liệu cần thiết để quan sát và chuẩn đoán hay không."""
    if not trace:
        return False

    required_keys = {
        "step",
        "action",
        "args",
        "observation",
    }

    for entry in trace:
        if not required_keys.issubset(entry):
            return False

        observation = entry["observation"]

        if not isinstance(observation, dict):
            return False

        if "status" not in observation:
            return False

        if pattern == "hybrid" and "plan_number" not in entry:
            return False

    return True


def adapted_after_failure(
    scenario: EvaluationScenario,
    trace: list[dict],
) -> bool | None:
    """Đánh giá khả năng thích nghi của Agent."""
    if not scenario.expects_recovery:
        return None

    for index, entry in enumerate(trace):
        observation = entry.get(
            "observation",
            {},
        )

        if (
            entry.get("action") != "book_seat"
            or observation.get("status") != "constraint_violation"
        ):
            continue

        failed_args = entry.get(
            "args",
            {},
        )

        for later_entry in trace[index + 1 :]:
            if (
                later_entry.get("action") == "book_seat"
                and later_entry.get("args") != failed_args
            ):
                return True

        return False

    return False


def count_tool_calls(
    pattern: str,
    result: dict,
) -> int:
    """Đếm số lần gọi Tool Calls của Agent."""
    trace_calls = len(result.get("trace", []))

    if pattern == "react":
        return trace_calls

    if pattern == "plan_then_execute":
        return 1 + trace_calls

    replan_count = result.get(
        "replan_count",
        0,
    )

    return 1 + replan_count + trace_calls


"""Chạy từng Pattern trên từng Scenario và chuẩn hóa kết quả để phục vụ cho Evaluation"""


def build_harness(
    scenario: EvaluationScenario,
) -> FlightAgentHarness:
    """Tạo Harness từ Constraints và Permissions của Scenario."""
    return FlightAgentHarness(
        constraints=scenario.constraints,
        permissions=scenario.permissions,
    )


def build_evaluation_run(
    scenario: EvaluationScenario,
    pattern: str,
    result: dict,
    model_calls: int,
) -> EvaluationRun:
    """Chuẩn hóa kết quả chạy của Agent và các Metrics thành một bản ghi Evaluation."""
    trace = result.get(
        "trace",
        [],
    )

    constraint_violations = count_constraint_violations(trace)

    failure_location = first_failure_location(trace)

    if scenario.expects_recovery:
        error_recovery = (
            result.get("status") == "complete" and constraint_violations > 0
        )

    else:
        error_recovery = None

    adaptability = adapted_after_failure(
        scenario=scenario,
        trace=trace,
    )

    if failure_location is None:
        debuggability = None
    else:
        debuggability = True

    return EvaluationRun(
        scenario=scenario.name,
        pattern=pattern,
        status=result.get(
            "status",
            "unknown",
        ),
        success=(result.get("status") == "complete"),
        model_calls=model_calls,
        tool_calls=count_tool_calls(
            pattern=pattern,
            result=result,
        ),
        total_steps=len(trace),
        constraint_violations=(constraint_violations),
        human_handoff_count=(1 if result.get("handoff") is not None else 0),
        error_recovery=error_recovery,
        adaptability=adaptability,
        trace_clarity=has_clear_trace(
            pattern=pattern,
            trace=trace,
        ),
        debuggability=debuggability,
        first_failure=failure_location,
    )


def evaluate_pattern(
    pattern: str,
    scenario: EvaluationScenario,
) -> EvaluationRun:
    """Thực hiện kích hoạt chạy 1 Pattern trên 1 Scenario."""
    reset_booking_store()

    try:
        harness = build_harness(scenario)

        if pattern == "react":
            model = CountingReActModel(build_react_responses(scenario))

            agent = ReActFlightAgent(
                model=model,
                harness=harness,
            )

        elif pattern == "plan_then_execute":
            model = CountingPlannerModel(
                build_plan_sequence(
                    scenario=scenario,
                    include_recovery=False,
                )
            )

            agent = PlanThenExecuteFlightAgent(
                model=model,
                harness=harness,
            )

        elif pattern == "hybrid":
            model = CountingPlannerModel(
                build_plan_sequence(
                    scenario=scenario,
                    include_recovery=True,
                )
            )

            agent = HybridFlightAgent(
                model=model,
                harness=harness,
            )

        else:
            raise ValueError(f"Unknown evaluation pattern: {pattern}")

        result = agent.run()

        return build_evaluation_run(
            scenario=scenario,
            pattern=pattern,
            result=result,
            model_calls=model.invoke_count,
        )

    finally:
        reset_booking_store()


"""Tổng hợp kết quả đánh giá hoàn chỉnh"""


def run_evaluation() -> list[EvaluationRun]:
    """Chạy toàn bộ Pattern trên toàn bộ Scenario."""
    runs = []

    for scenario in SCENARIOS:
        for pattern in PATTERNS:
            runs.append(
                evaluate_pattern(
                    pattern=pattern,
                    scenario=scenario,
                )
            )

    return runs


def average(
    values: list[int],
) -> float:
    """Tính trung bình số học."""
    return round(
        sum(values) / len(values),
        2,
    )


def optional_boolean_rate(
    values: list[bool | None],
) -> float | None:
    """Tính tỷ lệ True và bỏ qua các giá trị không áp dụng."""
    applicable = [value for value in values if value is not None]

    if not applicable:
        return None

    true_count = sum(1 for value in applicable if value)

    return round(
        true_count / len(applicable),
        2,
    )


def summarize_runs(
    runs: list[EvaluationRun],
) -> dict:
    """Tạo hợp toàn bộ Metrics trên từng Pattern."""
    summary = {}

    for pattern in PATTERNS:
        pattern_runs = [run for run in runs if run.pattern == pattern]

        success_count = sum(1 for run in pattern_runs if run.success)

        summary[pattern] = {
            "scenario_count": len(pattern_runs),
            "success_rate": round(
                success_count / len(pattern_runs),
                2,
            ),
            "avg_model_calls": average([run.model_calls for run in pattern_runs]),
            "avg_tool_calls": average([run.tool_calls for run in pattern_runs]),
            "avg_total_steps": average([run.total_steps for run in pattern_runs]),
            "constraint_violations": sum(
                run.constraint_violations for run in pattern_runs
            ),
            "human_handoffs": sum(run.human_handoff_count for run in pattern_runs),
            "error_recovery_rate": optional_boolean_rate(
                [run.error_recovery for run in pattern_runs]
            ),
            "adaptability_rate": optional_boolean_rate(
                [run.adaptability for run in pattern_runs]
            ),
            "trace_clarity_rate": optional_boolean_rate(
                [run.trace_clarity for run in pattern_runs]
            ),
            "debuggability_rate": optional_boolean_rate(
                [run.debuggability for run in pattern_runs]
            ),
        }

    return summary


def main() -> None:
    """Chạy Evaluation và in ra bảng tổng kết so sánh hoàn chỉnh."""
    runs = run_evaluation()

    output = {
        "runs": [asdict(run) for run in runs],
        "summary": summarize_runs(runs),
    }

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
