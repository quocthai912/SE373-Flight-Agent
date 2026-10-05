"""
Định nghĩa các kịch bản dùng chung để đánh giá ba mẫu Agent
"""

from dataclasses import dataclass

from flight_agent.harness import AgentPermissions
from flight_agent.models import FlightConstraints


@dataclass(frozen=True)
class EvaluationScenario:
    """Mô tả một kịch bản đánh giá dùng chung cho 3 mẫu Agent."""

    name: str
    description: str
    constraints: FlightConstraints
    permissions: AgentPermissions
    initial_flight_number: str
    recovery_flight_number: str | None = None
    expects_recovery: bool = False


def default_constraints() -> FlightConstraints:
    """Tạo bộ ràng buộc dữ liệu chuẩn dùng cho đánh giá."""
    return FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )


SCENARIOS: tuple[EvaluationScenario, ...] = (
    EvaluationScenario(
        name="happy_path",
        description=(
            "A valid flight can be booked and paid " "without an intermediate failure."
        ),
        constraints=default_constraints(),
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
        initial_flight_number="VN122",
    ),
    EvaluationScenario(
        name="recoverable_constraint_violation",
        description=(
            "The first selected flight violates the "
            "price constraint, but a valid alternative exists."
        ),
        constraints=default_constraints(),
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=True,
        ),
        initial_flight_number="VJ604",
        recovery_flight_number="VN122",
        expects_recovery=True,
    ),
    EvaluationScenario(
        name="payment_denied",
        description=(
            "Booking is allowed, but payment permission "
            "is denied and requires human handoff."
        ),
        constraints=default_constraints(),
        permissions=AgentPermissions(
            allow_booking=True,
            allow_payment=False,
        ),
        initial_flight_number="VN122",
    ),
)
