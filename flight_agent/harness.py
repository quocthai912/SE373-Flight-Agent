"""
Định nghĩa lớp Harness kiểm soát các hành động của Flight Agent
"""

from dataclasses import asdict, dataclass

from flight_agent.models import Flight, FlightConstraints
from flight_agent.tools import FLIGHTS, book_seat, get_booking, pay


@dataclass(frozen=True)
class AgentPermissions:
    """Kiểm soát quyền của Flight Agent."""

    allow_booking: bool = False
    allow_payment: bool = False


class FlightAgentHarness:
    """Kiểm soát các hành động của Flight Agent."""

    def __init__(
        self,
        constraints: FlightConstraints,
        permissions: AgentPermissions,
    ):
        self.constraints = constraints
        self.permissions = permissions

    def _find_flight(self, flight_number: str) -> Flight | None:
        """Tra cứu thông tin chuyến bay mà Flight Agent đề xuất."""
        for flight in FLIGHTS:
            if flight.flight_number == flight_number:
                return flight

        return None

    def check_constraints(self, flight_number: str) -> dict:
        """Kiểm tra thông tin chuyến bay có thỏa mãn toàn bộ ràng buộc của người dùng không."""
        flight = self._find_flight(flight_number)

        if flight is None:
            return {
                "status": "not_found",
                "message": "The requested flight was not found.",
                "flight_number": flight_number,
            }

        if not self.constraints.is_satisfied_by(flight):
            return {
                "status": "constraint_violation",
                "message": "The selected flight violates the user constraints.",
                "flight": asdict(flight),
            }

        return {
            "status": "satisfied",
            "message": "The selected flight satisfies all user constraints.",
            "flight": asdict(flight),
        }

    def check_permission(self, action: str) -> dict:
        """Kiểm tra Agent có được phép thực hiện hành động hay không."""
        allowed = False

        if action == "book_seat":
            allowed = self.permissions.allow_booking

        elif action == "pay":
            allowed = self.permissions.allow_payment

        if allowed:
            return {
                "status": "allowed",
                "message": "The requested action is permitted.",
                "action": action,
            }

        return {
            "status": "denied",
            "message": "The requested action is not permitted.",
            "action": action,
        }

    def execute_book_seat(self, flight_number: str) -> dict:
        """Harness kiểm tra quyền và ràng buộc dữ liệu trước khi thực thi công cụ đặt vé chuyến bay."""
        permission = self.check_permission("book_seat")

        if permission["status"] != "allowed":
            return permission

        constraint_check = self.check_constraints(flight_number)

        if constraint_check["status"] != "satisfied":
            return constraint_check

        return book_seat(flight_number)

    def execute_pay(self, booking_code: str) -> dict:
        """Harness kiểm tra quyền và ràng buộc dữ liệu trước khi cho phép thanh toán."""
        permission = self.check_permission("pay")

        if permission["status"] != "allowed":
            return permission

        booking_result = get_booking(booking_code)

        if booking_result["status"] != "success":
            return booking_result

        booking = booking_result["booking"]
        flight = booking["flight"]

        constraint_check = self.check_constraints(flight["flight_number"])

        if constraint_check["status"] != "satisfied":
            return constraint_check

        return pay(booking_code)

    def check_completion(self, booking_code: str) -> dict:
        """Tiêu chí hoàn thành được kiểm bằng mã chương trình"""
        booking_result = get_booking(booking_code)

        if booking_result["status"] != "success":
            return {
                "status": "incomplete",
                "message": "The task is incomplete because the booking was not found",
                "code": booking_code,
            }

        booking = booking_result["booking"]
        flight = booking["flight"]

        reasons = []

        constraint_check = self.check_constraints(flight["flight_number"])
        if constraint_check["status"] != "satisfied":
            reasons.append("flight constraints are not satisfied")

        if booking["paid"] == False:
            reasons.append("booking has not been paid")

        if reasons:
            return {
                "status": "incomplete",
                "message": "The task is not complete because: "
                + "; ".join(reasons)
                + ".",
                "booking": booking,
            }

        return {
            "status": "complete",
            "message": "The booking task is complete and verified from system state",
            "booking": booking,
        }

    def create_handoff(
        self,
        reason: str,
        attempted_actions: list[str],
        side_effects: list[str],
        question: str,
        booking_code: str | None = None,
    ) -> dict:
        """Tạo thông tin bàn giao cho con người."""
        booking = None

        if booking_code is not None:
            booking_result = get_booking(booking_code)

            if booking_result["status"] == "success":
                booking = booking_result["booking"]

        return {
            "status": "handoff",
            "message": "Human handoff required.",
            "reason": reason,
            "question": question,
            "attempted_actions": attempted_actions,
            "side_effects": side_effects,
            "constraints": asdict(self.constraints),
            "booking": booking,
        }

    def format_handoff(self, handoff: dict) -> str:
        """Chuyển thông tin bàn giao có cấu trúc thành nội dung dễ đọc cho con người."""
        attempted_actions = "\n".join(
            f"- {action}" for action in handoff["attempted_actions"]
        )

        side_effects = "\n".join(f"- {effect}" for effect in handoff["side_effects"])

        return (
            f"Flight Agent requires user assistance.\n\n"
            f"Reason: {handoff['reason']}\n\n"
            f"Actions already performed:\n{attempted_actions}\n\n"
            f"Changes already made:\n{side_effects}\n\n"
            f"User decision required: {handoff['question']}"
        )
