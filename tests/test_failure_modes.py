"""
Định nghĩa các kiểm thử cho Failure Mode của Flight Agent
"""

import pytest

from langchain_core.messages import AIMessage

import flight_agent.react_agent as react_agent_module

from flight_agent.failure_modes import (
    LoopDetector,
    validate_tool_result,
)
from flight_agent.harness import (
    AgentPermissions,
    FlightAgentHarness,
)
from flight_agent.models import FlightConstraints
from flight_agent.react_agent import ReActFlightAgent
from flight_agent.tools import (
    get_booking,
    reset_booking_store,
)


class ScriptedFailureModel:
    """Khởi tạo Model giả lập để phục vụ cho việc kiểm thử."""

    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        response = self.responses[self.index]
        self.index += 1
        return response


def tool_call_message(
    name: str,
    args: dict,
    call_id: str,
) -> AIMessage:
    """Tạo phản hồi Model chứa yêu cầu gọi công cụ."""
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


@pytest.fixture(autouse=True)
def reset_store():
    """Đặt lại kho lưu đặt chỗ trước và sau mỗi kiểm thử."""
    reset_booking_store()
    yield
    reset_booking_store()


@pytest.fixture
def constraints():
    """Tạo bộ ràng buộc dùng chung cho kiểm thử Failure Mode."""
    return FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )


def build_harness(
    constraints,
) -> FlightAgentHarness:
    """Tạo Harness có đầy đủ permission để phục vụ cho việc kiểm thử."""
    return FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
    )


def test_loop_detector_flags_repeated_call():
    detector = LoopDetector(
        repeat_limit=3,
    )

    args = {
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
    }

    assert (
        detector.check(
            "search_flights",
            args,
        )
        is False
    )

    assert (
        detector.check(
            "search_flights",
            args,
        )
        is False
    )

    assert (
        detector.check(
            "search_flights",
            args,
        )
        is True
    )


def test_validate_tool_result_rejects_missing_status():
    result = validate_tool_result(
        tool_name="search_flights",
        result={},
    )

    assert result["status"] == "tool_error"
    assert result["error_type"] == "invalid_tool_result"


def test_react_agent_detects_infinite_loop(
    constraints,
):
    repeated_args = {
        "origin": "SGN",
        "destination": "DAD",
        "date": "2026-10-07",
    }

    model = ScriptedFailureModel(
        responses=[
            tool_call_message(
                name="search_flights",
                args=repeated_args,
                call_id="call-1",
            ),
            tool_call_message(
                name="search_flights",
                args=repeated_args,
                call_id="call-2",
            ),
            tool_call_message(
                name="search_flights",
                args=repeated_args,
                call_id="call-3",
            ),
        ]
    )

    agent = ReActFlightAgent(
        model=model,
        harness=build_harness(constraints),
        max_steps=7,
        loop_repeat_limit=3,
    )

    result = agent.run()

    assert result["status"] == "loop_detected"
    assert result["failure_step"] == 3

    assert result["trace"][-1]["observation"]["status"] == "loop_detected"

    assert len(result["trace"]) == 3


def test_react_agent_rejects_unknown_tool(
    constraints,
):
    model = ScriptedFailureModel(
        responses=[
            tool_call_message(
                name="get_flight_discount",
                args={
                    "flight_number": "VN122",
                },
                call_id="call-1",
            ),
            AIMessage(content=("I could not complete the booking.")),
        ]
    )

    agent = ReActFlightAgent(
        model=model,
        harness=build_harness(constraints),
    )

    result = agent.run()

    assert result["status"] == "incomplete"

    assert result["trace"][0]["observation"]["status"] == "tool_error"

    assert result["trace"][0]["observation"]["tool"] == "get_flight_discount"


def test_react_agent_does_not_trust_hallucinated_completion(
    constraints,
):
    model = ScriptedFailureModel(
        responses=[
            tool_call_message(
                name="search_flights",
                args={
                    "origin": "SGN",
                    "destination": "DAD",
                    "date": "2026-10-07",
                },
                call_id="call-1",
            ),
            AIMessage(
                content=("Done! Booking VN999-1 was created " "and paid successfully.")
            ),
        ]
    )

    agent = ReActFlightAgent(
        model=model,
        harness=build_harness(constraints),
    )

    result = agent.run()

    assert result["status"] == "incomplete"

    assert get_booking("VN999-1")["status"] == "not_found"

    assert len(result["trace"]) == 1


def test_harness_blocks_goal_drift(
    constraints,
):
    model = ScriptedFailureModel(
        responses=[
            tool_call_message(
                name="search_flights",
                args={
                    "origin": "SGN",
                    "destination": "DAD",
                    "date": "2026-10-07",
                },
                call_id="call-1",
            ),
            tool_call_message(
                name="book_seat",
                args={
                    "flight_number": "QH118",
                },
                call_id="call-2",
            ),
            AIMessage(content=("QH118 is the cheapest option.")),
        ]
    )

    agent = ReActFlightAgent(
        model=model,
        harness=build_harness(constraints),
    )

    result = agent.run()

    assert result["status"] == "incomplete"

    assert result["trace"][1]["observation"]["status"] == "constraint_violation"

    assert get_booking("QH118-1")["status"] == "not_found"


def test_invalid_tool_result_is_not_treated_as_empty_state(
    constraints,
    monkeypatch,
):
    def broken_search_flights(
        origin: str,
        destination: str,
        date: str,
    ) -> dict:
        return {}

    monkeypatch.setattr(
        react_agent_module,
        "search_flights",
        broken_search_flights,
    )

    model = ScriptedFailureModel(
        responses=[
            tool_call_message(
                name="search_flights",
                args={
                    "origin": "SGN",
                    "destination": "DAD",
                    "date": "2026-10-07",
                },
                call_id="call-1",
            ),
            AIMessage(content=("There are no flights available.")),
        ]
    )

    agent = ReActFlightAgent(
        model=model,
        harness=build_harness(constraints),
    )

    result = agent.run()

    observation = result["trace"][0]["observation"]

    assert result["status"] == "incomplete"

    assert observation["status"] == "tool_error"

    assert observation["error_type"] == "invalid_tool_result"
