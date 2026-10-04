"""
Định nghĩa các Unit Test (Kiểm thử đơn vị) cho các Agent (Tác tử) của hệ thống Flight Agent
"""

import pytest

from langchain_core.messages import AIMessage

from flight_agent.harness import AgentPermissions, FlightAgentHarness
from flight_agent.models import FlightConstraints
from flight_agent.react_agent import ReActFlightAgent
from flight_agent.tools import reset_booking_store


class ScriptedModel:
    """Khởi tạo Model giả lập bằng danh sách các phản hồi đã được xác định trước."""

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
    """Tạo Model Response chứa một yêu cầu gọi công cụ."""
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
    """Đặt lại kho lưu đặt chỗ trước và sau mỗi trường hợp kiểm thử."""
    reset_booking_store()
    yield
    reset_booking_store()


@pytest.fixture
def constraints():
    """Tạo bộ ràng buộc dùng chung cho các trường hợp kiểm thử."""
    return FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )


def test_react_agent_completes_booking(constraints):
    model = ScriptedModel(
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
                    "flight_number": "VN122",
                },
                call_id="call-2",
            ),
            tool_call_message(
                name="pay",
                args={
                    "booking_code": "VN122-1",
                },
                call_id="call-3",
            ),
        ]
    )

    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
    )

    agent = ReActFlightAgent(
        model=model,
        harness=harness,
    )

    result = agent.run()

    assert result["status"] == "complete"
    assert result["booking"]["code"] == "VN122-1"
    assert result["booking"]["paid"] is True
    assert len(result["trace"]) == 3

    assert [item["action"] for item in result["trace"]] == [
        "search_flights",
        "book_seat",
        "pay",
    ]


def test_react_agent_uses_observation_to_retry(constraints):
    model = ScriptedModel(
        responses=[
            tool_call_message(
                name="book_seat",
                args={
                    "flight_number": "VJ604",
                },
                call_id="call-1",
            ),
            tool_call_message(
                name="book_seat",
                args={
                    "flight_number": "VN122",
                },
                call_id="call-2",
            ),
            tool_call_message(
                name="pay",
                args={
                    "booking_code": "VN122-1",
                },
                call_id="call-3",
            ),
        ]
    )

    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
    )

    agent = ReActFlightAgent(
        model=model,
        harness=harness,
    )

    result = agent.run()

    assert result["status"] == "complete"

    assert result["trace"][0]["observation"]["status"] == "constraint_violation"

    assert result["booking"]["flight"]["flight_number"] == "VN122"
    assert result["booking"]["paid"] is True


def test_react_agent_handoffs_when_payment_is_denied(constraints):
    model = ScriptedModel(
        responses=[
            tool_call_message(
                name="book_seat",
                args={
                    "flight_number": "VN122",
                },
                call_id="call-1",
            ),
            tool_call_message(
                name="pay",
                args={
                    "booking_code": "VN122-1",
                },
                call_id="call-2",
            ),
        ]
    )

    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=False,
        ),
    )

    agent = ReActFlightAgent(
        model=model,
        harness=harness,
    )

    result = agent.run()

    assert result["status"] == "handoff"
    assert result["handoff"]["booking"]["code"] == "VN122-1"
    assert result["handoff"]["booking"]["paid"] is False
    assert "Permission is required" in result["handoff"]["reason"]
    assert len(result["trace"]) == 2


def test_react_agent_stops_at_max_steps(constraints):
    model = ScriptedModel(
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
                name="search_flights",
                args={
                    "origin": "SGN",
                    "destination": "DAD",
                    "date": "2026-10-07",
                },
                call_id="call-2",
            ),
        ]
    )

    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
    )

    agent = ReActFlightAgent(
        model=model,
        harness=harness,
        max_steps=2,
    )

    result = agent.run()

    assert result["status"] == "max_steps_exceeded"
    assert len(result["trace"]) == 2

    assert (
        result["handoff"]["reason"] == "The maximum number of ReAct steps was reached."
    )
