# Định nghĩa các Unit Test (Kiểm thử đơn vị) cho các công cụ mô phỏng (Tools) của Flight Agent

import pytest

from flight_agent.tools import (
    book_seat,
    get_booking,
    pay,
    reset_booking_store,
    search_flights,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Đặt lại kho lưu đặt chỗ trước và sau mỗi trường hợp kiểm thử."""
    reset_booking_store()
    yield
    reset_booking_store()


# TEST 1 (Tool search_flights): Truy vấn hợp lệ phải trả đúng các chuyến bay cùng tuyến và cùng ngày
def test_search_flights_returns_matching_flights():
    result = search_flights(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
    )

    assert result["status"] == "success"
    assert result["message"] == "Flight search completed successfully."
    assert result["count"] == 3

    assert {flight["flight_number"] for flight in result["flights"]} == {
        "VN122",
        "QH118",
        "VJ604",
    }


# TEST 2 (Tool search_flights): Truy vấn không có chuyến bay phù hợp thì phải trả danh sách rỗng
def test_search_flights_returns_empty_list_when_not_found():
    result = search_flights(
        origin="DAD",
        destination="SGN",
        date="2026-10-07",
    )

    assert result["status"] == "success"
    assert (
        result["message"]
        == "Flight search completed successfully, but no matching flights were found."
    )
    assert result["count"] == 0
    assert result["flights"] == []


# TEST 3 (Tool book_seat): Chuyến bay tồn tại phải tạo được đặt chỗ ở trạng thái chưa thanh toán
def test_book_seat_creates_unpaid_booking():
    result = book_seat("VN122")

    assert result["status"] == "success"
    assert result["message"] == "Seat booking completed successfully."
    assert result["booking"]["flight"]["flight_number"] == "VN122"
    assert result["booking"]["paid"] is False

    stored = get_booking(result["booking"]["code"])

    assert stored["status"] == "success"


# TEST 4 (Tool book_seat): Chuyến bay không tồn tại phải bị từ chối đặt chỗ
def test_book_seat_rejects_unknown_flight():
    result = book_seat("VN999")

    assert result["status"] == "not_found"
    assert result["message"] == "The requested flight was not found."
    assert result["flight_number"] == "VN999"


# TEST 5 (Tool pay): Đặt chỗ ở chuyến bay hợp lệ phải được cập nhật sang trạng thái đã thanh toán
def test_pay_updates_booking_to_paid():
    booked = book_seat("VN122")
    booking_code = booked["booking"]["code"]

    result = pay(booking_code)

    assert result["status"] == "success"
    assert result["message"] == "Payment completed successfully."
    assert result["booking"]["code"] == booking_code
    assert result["booking"]["paid"] is True

    stored = get_booking(booking_code)

    assert stored["status"] == "success"
    assert stored["booking"]["paid"] is True


# TEST 6 (Tool pay): Mã đặt chỗ không hợp lệ phải bị từ chối thanh toán
def test_pay_rejects_unknown_booking():
    result = pay("UNKNOWN-001")

    assert result["status"] == "not_found"
    assert result["message"] == "The requested booking was not found."
    assert result["code"] == "UNKNOWN-001"


# TEST 7 (Tool get_booking): Phải truy xuất đúng thông tin của trạng thái đặt chỗ sau khi đặt
def test_get_booking_returns_existing_booking():
    booked = book_seat("VN122")
    booking_code = booked["booking"]["code"]

    result = get_booking(booking_code)

    assert result["status"] == "success"
    assert result["message"] == "Booking information retrieved successfully."
    assert result["booking"]["code"] == booking_code
    assert result["booking"]["flight"]["flight_number"] == "VN122"
    assert result["booking"]["seat"] == "1A"
    assert result["booking"]["paid"] is False


# TEST 8 (Tool get_booking): Mã đặt chỗ không hợp lệ phải trả trạng thái không tìm thấy
def test_get_booking_returns_not_found():
    result = get_booking("UNKNOWN-001")

    assert result["status"] == "not_found"
    assert result["message"] == "The requested booking was not found."
    assert result["code"] == "UNKNOWN-001"
