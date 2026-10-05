"""
Định nghĩa cấu hình dùng chung cho API Key và Model của Flight Agent
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình dùng chung của ứng dụng."""

    api_key: str
    model_name: str


def load_config() -> AppConfig:
    """Đọc cấu hình từ biến môi trường."""
    load_dotenv()

    api_key = os.getenv("API_KEY")
    model_name = os.getenv("FLIGHT_AGENT_MODEL")

    missing = []

    if not api_key:
        missing.append("API_KEY")

    if not model_name:
        missing.append("FLIGHT_AGENT_MODEL")

    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return AppConfig(
        api_key=api_key,
        model_name=model_name,
    )
