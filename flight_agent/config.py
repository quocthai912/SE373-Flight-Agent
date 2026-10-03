"""
Định nghĩa cấu hình dùng chung cho API Key và Model của Flight Agent
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình dùng chung của ứng dụng."""

    api_key: str | None
    model_name: str | None


def load_config() -> AppConfig:
    """Đọc cấu hình từ biến môi trường."""
    load_dotenv()

    return AppConfig(
        api_key=os.getenv("API_KEY"),
        model_name=os.getenv("FLIGHT_AGENT_MODEL"),
    )
