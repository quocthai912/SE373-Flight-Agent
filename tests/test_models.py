"""
Định nghĩa các Unit Test (Kiểm thử đơn vị) cho Domain Model (Mô hình miền nghiệp vụ) và
các logic kiểm tra ràng buộc của Flight Agent
"""

from flight_agent.models import Flight, FlightConstraints


# TEST 1: Chuyến bay hợp lệ phải thỏa mãn toàn bộ ràng buộc
def test_flight_satisfies_all_constraints():
    constraints = FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )

    flight = Flight(
        flight_number="VN122",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T08:10",
        price=1_850_000,
    )

    assert constraints.is_satisfied_by(flight)


# TEST 2: Chuyến bay vượt quá giá tối đa phải bị từ chối
def test_flight_exceeds_max_price():
    constraints = FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )

    flight = Flight(
        flight_number="VJ604",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T08:10",
        price=2_480_000,
    )

    assert not constraints.is_satisfied_by(flight)


# TEST 3: Chuyến bay khởi hành sau thời gian cho phép phải bị từ chối
def test_flight_departs_too_late():
    constraints = FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )

    flight = Flight(
        flight_number="QH118",
        origin="SGN",
        destination="DAD",
        depart_at="2026-10-07T15:40",
        price=1_640_000,
    )

    assert not constraints.is_satisfied_by(flight)


# TEST 4: Chuyến bay sai tuyến yêu cầu phải bị từ chối
def test_flight_has_wrong_route():
    constraints = FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )

    flight = Flight(
        flight_number="VN999",
        origin="SGN",
        destination="HAN",
        depart_at="2026-10-07T08:10",
        price=1_500_000,
    )

    assert not constraints.is_satisfied_by(flight)
