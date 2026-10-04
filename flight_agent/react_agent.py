"""
Định nghĩa vòng lặp ReAct cho Flight Agent
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from flight_agent.harness import FlightAgentHarness
from flight_agent.tools import get_booking, search_flights


class ReActFlightAgent:
    """Điều phối vòng lặp ReAct của Flight Agent."""

    def __init__(
        self,
        model: BaseChatModel,
        harness: FlightAgentHarness,
        max_steps: int = 7,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        self.harness = harness
        self.max_steps = max_steps

        self.tools = self._build_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.model = model.bind_tools(self.tools)

    def _build_tools(self) -> list[StructuredTool]:
        """Tạo các công cụ mà Model được phép đề xuất sử dụng."""

        def search_flights_tool(
            origin: str,
            destination: str,
            date: str,
        ) -> dict:
            return search_flights(
                origin=origin,
                destination=destination,
                date=date,
            )

        def book_seat_tool(flight_number: str) -> dict:
            return self.harness.execute_book_seat(flight_number)

        def pay_tool(booking_code: str) -> dict:
            return self.harness.execute_pay(booking_code)

        def get_booking_tool(booking_code: str) -> dict:
            return get_booking(booking_code)

        return [
            StructuredTool.from_function(
                func=search_flights_tool,
                name="search_flights",
                description=(
                    "Search available flights by origin, destination, and date."
                ),
            ),
            StructuredTool.from_function(
                func=book_seat_tool,
                name="book_seat",
                description="Book a seat on a selected flight.",
            ),
            StructuredTool.from_function(
                func=pay_tool,
                name="pay",
                description="Pay for an existing booking.",
            ),
            StructuredTool.from_function(
                func=get_booking_tool,
                name="get_booking",
                description="Retrieve the current state of a booking.",
            ),
        ]

    def _build_messages(self) -> list:
        """Tạo ngữ cảnh ban đầu cho vòng lặp ReAct."""
        system_prompt = (
            "You are a flight booking agent. "
            "Use the available tools to satisfy the user's goal. "
            "Choose one tool action at a time. "
            "Use each tool observation to decide the next action. "
            "Do not invent flight or booking data. "
            "Do not declare the task complete yourself because "
            "completion is verified by system code."
        )

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.harness.constraints.to_prompt()),
        ]

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        """Thực thi công cụ đã được Model đề xuất."""
        tool = self.tools_by_name.get(tool_name)

        if tool is None:
            return {
                "status": "tool_error",
                "message": "The requested tool does not exist.",
                "tool": tool_name,
            }

        try:
            return tool.invoke(tool_args)

        except Exception as exc:
            return {
                "status": "tool_error",
                "message": "The requested tool failed during execution.",
                "tool": tool_name,
                "error": str(exc),
            }

    def run(self) -> dict:
        """Chạy vòng lặp ReAct cho tới khi hoàn thành hoặc đạt điều kiện dừng của Agent Loop."""
        messages = self._build_messages()

        trace = []
        attempted_actions = []
        side_effects = []

        booking_code = None

        for step in range(1, self.max_steps + 1):
            response_model = self.model.invoke(messages)
            messages.append(response_model)

            tool_calls = response_model.tool_calls

            if not tool_calls:
                handoff = self.harness.create_handoff(
                    reason=("The model stopped before the task was complete."),
                    attempted_actions=attempted_actions,
                    side_effects=side_effects,
                    question=("Would you like to continue the booking task?"),
                    booking_code=booking_code,
                )

                return {
                    "status": "incomplete",
                    "message": self.harness.format_handoff(handoff),
                    "handoff": handoff,
                    "trace": trace,
                }

            if len(tool_calls) != 1:
                return {
                    "status": "invalid_model_output",
                    "message": "The model must request exactly one tool per ReAct step.",
                    "trace": trace,
                }

            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            observation = self._execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            messages.append(
                ToolMessage(
                    content=json.dumps(observation),
                    tool_call_id=tool_call["id"],
                )
            )

            trace.append(
                {
                    "step": step,
                    "action": tool_name,
                    "args": tool_args,
                    "observation": observation,
                }
            )

            attempted_actions.append(
                f"Called {tool_name} with {json.dumps(tool_args, sort_keys=True)}."
            )

            if tool_name == "book_seat" and observation["status"] == "success":
                booking_code = observation["booking"]["code"]
                side_effects.append(f"Created booking {booking_code}.")

            if tool_name == "pay" and observation["status"] == "success":
                side_effects.append(f"Paid booking {observation['booking']['code']}.")

            if observation["status"] == "denied":
                handoff = self.harness.create_handoff(
                    reason=(f"Permission is required to execute {tool_name}."),
                    attempted_actions=attempted_actions,
                    side_effects=side_effects,
                    question=(f"Should the {tool_name} action be approved?"),
                    booking_code=booking_code,
                )

                return {
                    "status": "handoff",
                    "message": self.harness.format_handoff(handoff),
                    "handoff": handoff,
                    "trace": trace,
                }

            if booking_code is not None:
                completion = self.harness.check_completion(booking_code)

                if completion["status"] == "complete":
                    return {
                        "status": "complete",
                        "message": ("The booking task was completed successfully."),
                        "booking": completion["booking"],
                        "trace": trace,
                    }

        handoff = self.harness.create_handoff(
            reason="The maximum number of ReAct steps was reached.",
            attempted_actions=attempted_actions,
            side_effects=side_effects,
            question="Would you like to continue the booking task?",
            booking_code=booking_code,
        )

        return {
            "status": "max_steps_exceeded",
            "message": self.harness.format_handoff(handoff),
            "handoff": handoff,
            "trace": trace,
        }
