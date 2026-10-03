"""
Định nghĩa các Domain Model (Mô hình miền nghiệp vụ) và các thực thể dữ liệu cốt lõi của hệ thống Flight Agent
"""

from dataclasses import dataclass
from datetime import date, datetime, time


@dataclass(frozen=True)
class Flight:
    """Thông tin một chuyến bay."""

    flight_number: str  # Số hiệu chuyến bay
    origin: str  # Điểm khởi hành
    destination: str  # Điểm đến
    depart_at: str  # Ngày và giờ khởi hành theo định dạng ISO
    price: int  # Giá vé, tính bằng tiền VNĐ


@dataclass
class Booking:
    """Thông tin đặt chỗ của một chuyến bay."""

    code: str  # Mã đặt chỗ
    flight: Flight  # Chuyến bay được đặt
    seat: str  # Số ghế được giữ
    paid: bool = False  # Trạng thái thanh toán


@dataclass(frozen=True)
class FlightConstraints:
    """Các ràng buộc mà chuyến bay phải thỏa mãn."""

    origin: str  # Điểm khởi hành yêu cầu
    destination: str  # Điểm đến yêu cầu
    date: str  # Ngày khởi hành yêu cầu theo định dạng YYYY-MM-DD
    depart_before: str  # Chuyến bay được đặt phải khởi hành trước thời điểm này
    max_price: int  # Giá vé tối đa được phép

    def to_prompt(self) -> str:
        """Chuyển các ràng buộc thành nội dung cung cấp cho mô hình."""
        return (
            f"Book one ticket from {self.origin} to {self.destination} "
            f"on {self.date}, departing before {self.depart_before}, "
            f"with a maximum price of {self.max_price:,} VND."
        )

    def is_satisfied_by(self, flight: Flight) -> bool:
        """Kiểm tra chuyến bay có thỏa mãn tất cả ràng buộc hay không."""
        try:
            # Chuyển chuỗi thời gian thành kiểu dữ liệu thời gian để so sánh
            depart_at = datetime.fromisoformat(flight.depart_at)
            required_date = date.fromisoformat(self.date)
            required_time = time.fromisoformat(self.depart_before)
        except ValueError:
            return False

        return (
            flight.origin == self.origin
            and flight.destination == self.destination
            and depart_at.date() == required_date
            and depart_at.time() < required_time
            and flight.price <= self.max_price
        )
