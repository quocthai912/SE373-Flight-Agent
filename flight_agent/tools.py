"""
Định nghĩa kho dữ liệu mô phỏng và các công cụ dùng chung cho Flight Agent
"""

from dataclasses import asdict

from flight_agent.models import Booking, Flight

# Tập dữ liệu các chuyến bay mô phỏng
FLIGHTS: tuple[Flight, ...] = (
    Flight(
        flight_number="VN122",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T08:10",
        price=1_850_000,
    ),
    Flight(
        flight_number="QH118",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T15:40",
        price=1_640_000,
    ),
    Flight(
        flight_number="VJ604",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T08:10",
        price=2_480_000,
    ),
    Flight(
        flight_number="VN210",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-08T09:00",
        price=1_900_000,
    ),
    Flight(
        flight_number="VN250",
        origin="SGN",
        destination="HAN",
        depart_at="2026-10-07T07:30",
        price=1_700_000,
    ),
)


# Kho lưu trạng thái đặt chỗ
BOOKINGS: dict[str, Booking] = {}


# Các hàm hỗ trợ nội bộ
def _find_flight(flight_number: str) -> Flight | None:
    """Tìm chuyến bay theo mã chuyến bay."""
    for flight in FLIGHTS:
        if flight.flight_number == flight_number:
            return flight
    return None


def _booking_to_dict(booking: Booking) -> dict:
    """Chuyển thông tin đặt chỗ thành đầu ra có cấu trúc."""
    return asdict(booking)


def _booking_count_for_flight(flight_number: str) -> int:
    """Đếm số lượt đặt chỗ hiện có của một chuyến bay."""
    count = 0
    for booking in BOOKINGS.values():
        if booking.flight.flight_number == flight_number:
            count += 1
    return count


def _next_booking_code(flight_number: str) -> str:
    """Tạo mã đặt chỗ."""
    booking_number = _booking_count_for_flight(flight_number) + 1
    return f"{flight_number}-{booking_number}"


def _next_seat(flight_number: str) -> str:
    """Tạo số ghế."""
    seat_number = _booking_count_for_flight(flight_number) + 1
    return f"{seat_number}A"


def reset_booking_store() -> None:
    """Xóa toàn bộ trạng thái đặt chỗ."""
    BOOKINGS.clear()


# Các Tools được Flight Agent sử dụng
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Tìm các chuyến bay theo điểm đi, điểm đến và ngày khởi hành."""
    flights = [
        asdict(flight)
        for flight in FLIGHTS
        if flight.origin == origin
        and flight.destination == destination
        and flight.depart_at[:10] == date
    ]

    return {
        "status": "success",
        "message": (
            "Flight search completed successfully."
            if flights
            else "Flight search completed successfully, but no matching flights were found."
        ),
        "count": len(flights),
        "flights": flights,
    }


def book_seat(flight_number: str) -> dict:
    """Tạo thông tin đặt chỗ trên chuyến bay đã tìm được."""
    flight = _find_flight(flight_number)

    if flight is None:
        return {
            "status": "not_found",
            "message": "The requested flight was not found.",
            "flight_number": flight_number,
        }

    booking = Booking(
        code=_next_booking_code(flight_number),
        flight=flight,
        seat=_next_seat(flight_number),
        paid=False,
    )

    BOOKINGS[booking.code] = booking

    return {
        "status": "success",
        "message": "Seat booking completed successfully.",
        "booking": _booking_to_dict(booking),
    }


def pay(booking_code: str) -> dict:
    """Thanh toán cho chỗ ngồi đã được đặt."""
    booking = BOOKINGS.get(booking_code)

    if booking is None:
        return {
            "status": "not_found",
            "message": "The requested booking was not found.",
            "code": booking_code,
        }

    booking.paid = True

    return {
        "status": "success",
        "message": "Payment completed successfully.",
        "booking": _booking_to_dict(booking),
    }


def get_booking(booking_code: str) -> dict:
    """Đọc lại thông tin đặt chỗ từ kho lưu trạng thái."""
    booking = BOOKINGS.get(booking_code)

    if booking is None:
        return {
            "status": "not_found",
            "message": "The requested booking was not found.",
            "code": booking_code,
        }

    return {
        "status": "success",
        "message": "Booking information retrieved successfully.",
        "booking": _booking_to_dict(booking),
    }
