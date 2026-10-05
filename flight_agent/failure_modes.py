"""
Định nghĩa các cơ chế phát hiện lỗi dùng chung cho Flight Agent
"""

import json


class LoopDetector:
    """Phát hiện một lời gọi công cụ bị lặp quá nhiều lần."""

    def __init__(
        self,
        repeat_limit: int = 3,
        window: int = 6,
    ):
        if repeat_limit < 2:
            raise ValueError("repeat_limit must be at least two.")

        if window < repeat_limit:
            raise ValueError("window must be greater than or equal to repeat_limit.")

        self.repeat_limit = repeat_limit
        self.window = window
        self.recent_calls = []

    def check(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> bool:
        """Kiểm tra lời gọi hiện tại có tạo thành vòng lặp hay không."""
        fingerprint = (
            tool_name,
            json.dumps(
                tool_args,
                sort_keys=True,
                default=str,
            ),
        )

        self.recent_calls.append(fingerprint)

        if len(self.recent_calls) > self.window:
            self.recent_calls.pop(0)

        return self.recent_calls.count(fingerprint) >= self.repeat_limit


def validate_tool_result(
    tool_name: str,
    result: object,
) -> dict:
    """Kiểm tra kết quả trả về từ công cụ có trạng thái rõ ràng hay không."""
    if not isinstance(result, dict):
        return {
            "status": "tool_error",
            "message": "The tool returned an invalid structured result.",
            "tool": tool_name,
            "error_type": "invalid_tool_result",
        }

    if "status" not in result:
        return {
            "status": "tool_error",
            "message": "The tool returned an invalid structured result.",
            "tool": tool_name,
            "error_type": "invalid_tool_result",
        }

    return result
