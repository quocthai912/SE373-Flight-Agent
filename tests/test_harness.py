# Định nghĩa các Unit Tests (kiểm thử đơn vị) cho Harness của Flight Agent

import pytest

from flight_agent.harness import AgentPermissions, FlightAgentHarness
from flight_agent.models import FlightConstraints
from flight_agent.tools import book_seat, get_booking, reset_booking_store


@pytest.fixture(autouse=True)
def reset_store():
    """Đặt lại kho lưu đặt chỗ trước và sau mỗi trường hợp kiểm thử."""
    reset_booking_store()
    yield
    reset_booking_store()


@pytest.fixture
def constraints():
    """Tạo bộ ràng buộc dữ liệu dùng chung cho các trường hợp kiểm thử."""
    return FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )


@pytest.fixture
def harness(constraints):
    """Tạo Harness có đầy đủ quyền đặt chỗ và thanh toán."""
    permissions = AgentPermissions(
        allow_booking=True,
        allow_payment=True,
    )

    return FlightAgentHarness(
        constraints=constraints,
        permissions=permissions,
    )


def test_check_constraints_accepts_valid_flight(harness):
    result = harness.check_constraints("VN122")

    assert result["status"] == "satisfied"
    assert result["message"] == "The selected flight satisfies all user constraints."
    assert result["flight"]["flight_number"] == "VN122"


def test_check_constraints_rejects_invalid_flight(harness):
    result = harness.check_constraints("VJ604")

    assert result["status"] == "constraint_violation"
    assert result["message"] == "The selected flight violates the user constraints."
    assert result["flight"]["flight_number"] == "VJ604"


def test_check_constraints_returns_not_found(harness):
    result = harness.check_constraints("VN999")

    assert result["status"] == "not_found"
    assert result["message"] == "The requested flight was not found."
    assert result["flight_number"] == "VN999"


def test_execute_book_seat_requires_permission(constraints):
    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=False,
            allow_payment=True,
        ),
    )

    result = harness.execute_book_seat("VN122")

    assert result["status"] == "denied"
    assert result["message"] == "The requested action is not permitted."
    assert result["action"] == "book_seat"

    stored = get_booking("VN122-1")

    assert stored["status"] == "not_found"


def test_execute_book_seat_books_valid_flight(harness):
    result = harness.execute_book_seat("VN122")

    assert result["status"] == "success"
    assert result["message"] == "Seat booking completed successfully."
    assert result["booking"]["flight"]["flight_number"] == "VN122"
    assert result["booking"]["paid"] is False


def test_execute_book_seat_blocks_constraint_violation(harness):
    result = harness.execute_book_seat("VJ604")

    assert result["status"] == "constraint_violation"

    stored = get_booking("VJ604-1")

    assert stored["status"] == "not_found"


def test_execute_pay_requires_permission(constraints):
    booked = book_seat("VN122")
    booking_code = booked["booking"]["code"]

    harness = FlightAgentHarness(
        constraints=constraints,
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=False,
        ),
    )

    result = harness.execute_pay(booking_code)

    assert result["status"] == "denied"
    assert result["message"] == "The requested action is not permitted."
    assert result["action"] == "pay"

    stored = get_booking(booking_code)

    assert stored["booking"]["paid"] is False


def test_execute_pay_blocks_invalid_booking(harness):
    booked = book_seat("VJ604")
    booking_code = booked["booking"]["code"]

    result = harness.execute_pay(booking_code)

    assert result["status"] == "constraint_violation"

    stored = get_booking(booking_code)

    assert stored["booking"]["paid"] is False


def test_execute_pay_updates_valid_booking(harness):
    booked = harness.execute_book_seat("VN122")
    booking_code = booked["booking"]["code"]

    result = harness.execute_pay(booking_code)

    assert result["status"] == "success"
    assert result["message"] == "Payment completed successfully."
    assert result["booking"]["paid"] is True

    stored = get_booking(booking_code)

    assert stored["booking"]["paid"] is True


def test_check_completion_returns_incomplete_before_payment(harness):
    booked = harness.execute_book_seat("VN122")
    booking_code = booked["booking"]["code"]

    result = harness.check_completion(booking_code)

    assert result["status"] == "incomplete"
    assert "booking has not been paid" in result["message"]
    assert result["booking"]["paid"] is False


def test_check_completion_returns_complete_after_payment(harness):
    booked = harness.execute_book_seat("VN122")
    booking_code = booked["booking"]["code"]

    harness.execute_pay(booking_code)

    result = harness.check_completion(booking_code)

    assert result["status"] == "complete"
    assert (
        result["message"]
        == "The booking task is complete and verified from system state"
    )
    assert result["booking"]["paid"] is True


def test_check_completion_returns_incomplete_when_booking_not_found(harness):
    result = harness.check_completion("UNKNOWN-001")

    assert result["status"] == "incomplete"
    assert (
        result["message"] == "The task is incomplete because the booking was not found"
    )
    assert result["code"] == "UNKNOWN-001"


def test_create_handoff_contains_required_context(harness):
    booked = harness.execute_book_seat("VN122")
    booking_code = booked["booking"]["code"]

    result = harness.create_handoff(
        reason="Payment permission is missing.",
        attempted_actions=[
            "Selected VN122.",
            "Booked seat 1A.",
        ],
        side_effects=[
            "Created booking VN122-1.",
        ],
        question="Should payment be approved?",
        booking_code=booking_code,
    )

    assert result["status"] == "handoff"
    assert result["message"] == "Human handoff required."
    assert result["reason"] == "Payment permission is missing."
    assert result["question"] == "Should payment be approved?"
    assert result["attempted_actions"] == [
        "Selected VN122.",
        "Booked seat 1A.",
    ]
    assert result["side_effects"] == [
        "Created booking VN122-1.",
    ]
    assert result["constraints"]["origin"] == "SGN"
    assert result["booking"]["code"] == booking_code


def test_format_handoff_returns_readable_text(harness):
    handoff = harness.create_handoff(
        reason="Payment permission is missing.",
        attempted_actions=[
            "Selected VN122.",
        ],
        side_effects=[
            "Created booking VN122-1.",
        ],
        question="Should payment be approved?",
    )

    result = harness.format_handoff(handoff)

    assert "Flight Agent requires user assistance." in result
    assert "Reason: Payment permission is missing." in result
    assert "Actions already performed:" in result
    assert "- Selected VN122." in result
    assert "Changes already made:" in result
    assert "- Created booking VN122-1." in result
    assert "User decision required: Should payment be approved?" in result
