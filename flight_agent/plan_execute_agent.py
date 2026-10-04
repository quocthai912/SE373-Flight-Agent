"""
Định nghĩa Plan-then-Execute Agent
"""

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from flight_agent.harness import FlightAgentHarness
from flight_agent.tools import get_booking, search_flights


class PlanStep(BaseModel):
    """Định nghĩa của một bước trong Plan."""

    tool: Literal["book_seat", "pay", "get_booking"]
    args: dict


class FlightPlan(BaseModel):
    """Định nghĩa toàn bộ Plan đặt vé."""

    steps: list[PlanStep] = Field(
        default_factory=list,
        max_length=5,
    )


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You plan a flight booking task. "
                "Create the whole plan before any action is executed. "
                "Available tools are book_seat, pay, and get_booking. "
                "Use only flights provided in the available flight data. "
                "The booking code does not exist yet, so use "
                '"$booking_code" for pay and get_booking. '
                "If no flight can satisfy the user's goal, "
                "return an empty list of steps."
            ),
        ),
        (
            "human",
            "Goal: {goal}\nAvailable flights: {flights}",
        ),
    ]
)


class PlanThenExecuteFlightAgent:
    """Điều phối mẫu Plan-then-Execute của Flight Agent."""

    def __init__(
        self,
        model: BaseChatModel,
        harness: FlightAgentHarness,
    ):
        self.harness = harness
        self.planner = PLANNER_PROMPT | model.with_structured_output(FlightPlan)

    def create_plan(self) -> tuple[FlightPlan, dict]:
        """Tạo toàn bộ Plan trước khi thực thi."""
        constraints = self.harness.constraints

        search_result = search_flights(
            origin=constraints.origin,
            destination=constraints.destination,
            date=constraints.date,
        )

        plan = self.planner.invoke(
            {
                "goal": constraints.to_prompt(),
                "flights": json.dumps(
                    search_result["flights"],
                    sort_keys=True,
                ),
            }
        )

        return plan, search_result

    def _validate_plan(
        self,
        plan: FlightPlan,
        search_result: dict,
    ) -> dict:
        """Kiểm tra Plan do Model tạo ra trước khi thực thi."""
        if not plan.steps:
            return {
                "status": "empty_plan",
                "message": "The model did not produce an executable plan.",
            }

        expected_tools = [
            "book_seat",
            "pay",
            "get_booking",
        ]

        actual_tools = [step.tool for step in plan.steps]

        if actual_tools != expected_tools:
            return {
                "status": "invalid_plan",
                "message": "The plan must execute book_seat, pay and get_booking in that order.",
            }

        flight_number = plan.steps[0].args.get("flight_number")

        available_flight_numbers = {
            flight["flight_number"] for flight in search_result["flights"]
        }

        if flight_number not in available_flight_numbers:
            return {
                "status": "invalid_plan",
                "message": "The planned flight was not present in the search result.",
            }

        for step in plan.steps[1:]:
            if step.args.get("booking_code") != "$booking_code":
                return {
                    "status": "invalid_plan",
                    "message": "The plan must use $booking_code until a real booking code exists.",
                }

        return {
            "status": "valid",
            "message": "The plan is valid.",
        }

    def _resolve_args(
        self,
        args: dict,
        booking_code: str | None,
    ) -> dict:
        """Điền dữ liệu thực tế vào các Placeholder."""
        resolved_args = {}

        for key, value in args.items():
            if value == "$booking_code":
                resolved_args[key] = booking_code
            else:
                resolved_args[key] = value

        return resolved_args

    def _execute_step(
        self,
        step: PlanStep,
        booking_code: str | None,
    ) -> tuple[dict, dict]:
        """Thực thi một bước trong Plan."""
        resolved_args = self._resolve_args(step.args, booking_code)

        if step.tool == "book_seat":
            result = self.harness.execute_book_seat(resolved_args["flight_number"])

        elif step.tool == "pay":
            if resolved_args["booking_code"] is None:
                result = {
                    "status": "plan_error",
                    "message": "A booking code is required before payment.",
                }
            else:
                result = self.harness.execute_pay(resolved_args["booking_code"])

        elif step.tool == "get_booking":
            if resolved_args["booking_code"] is None:
                result = {
                    "status": "plan_error",
                    "message": "A booking code is required before retrieving a booking.",
                }
            else:
                result = get_booking(resolved_args["booking_code"])

        return resolved_args, result

    def execute_plan(
        self,
        plan: FlightPlan,
        search_result: dict,
    ) -> dict:
        """Thực thi tuần tự Plan mà Model đã tạo."""
        validation = self._validate_plan(plan, search_result)

        plan_data = plan.model_dump()

        if validation["status"] != "valid":
            handoff = self.harness.create_handoff(
                reason=validation["message"],
                attempted_actions=[],
                side_effects=[],
                question="Would you like to review or regenerate the plan?",
            )

            return {
                "status": validation["status"],
                "message": self.harness.format_handoff(handoff),
                "plan": plan_data,
                "handoff": handoff,
                "trace": [],
            }

        trace = []
        attempted_actions = []
        side_effects = []

        booking_code = None

        for step_number, step in enumerate(plan.steps, start=1):
            args, observation = self._execute_step(step, booking_code)

            trace.append(
                {
                    "step": step_number,
                    "action": step.tool,
                    "args": args,
                    "observation": observation,
                }
            )

            attempted_actions.append(
                f"Executed planned step {step_number}: {step.tool} with {json.dumps(args, sort_keys=True)}."
            )

            if step.tool == "book_seat" and observation["status"] == "success":
                booking_code = observation["booking"]["code"]
                side_effects.append(f"Created booking {booking_code}.")

            elif step.tool == "pay" and observation["status"] == "success":
                side_effects.append(f"Paid booking {observation["booking"]["code"]}")

            if observation["status"] != "success":
                if observation["status"] == "denied":
                    question = f"Should the {step.tool} action be approved?"

                else:
                    question = "Would you like to review the failed plan?"

                handoff = self.harness.create_handoff(
                    reason=f"Planned step {step_number} failed with status {observation["status"]}",
                    attempted_actions=attempted_actions,
                    side_effects=side_effects,
                    question=question,
                    booking_code=booking_code,
                )

                return {
                    "status": "handoff",
                    "message": self.harness.format_handoff(handoff),
                    "plan": plan_data,
                    "handoff": handoff,
                    "trace": trace,
                }

        if booking_code is not None:
            completion = self.harness.check_completion(booking_code)

            if completion["status"] == "complete":
                return {
                    "status": "complete",
                    "message": "The booking task was completed successfully.",
                    "plan": plan_data,
                    "booking": completion["booking"],
                    "trace": trace,
                }

        handoff = self.harness.create_handoff(
            reason="The plan finished without satisfying the completion criteria.",
            attempted_actions=attempted_actions,
            side_effects=side_effects,
            question="Would you like to review or regenerate the plan?",
            booking_code=booking_code,
        )

        return {
            "status": "incomplete",
            "message": self.harness.format_handoff(handoff),
            "plan": plan_data,
            "handoff": handoff,
            "trace": trace,
        }

    def run(self) -> dict:
        """Thực thi Plan-Then-Execute-Flight-Agent."""
        try:
            plan, search_result = self.create_plan()

        except Exception as exc:
            handoff = self.harness.create_handoff(
                reason="The planner failed to create a valid plan.",
                attempted_actions=[],
                side_effects=[],
                question="Would you like to retry the planning step?",
            )

            return {
                "status": "planning_error",
                "message": self.harness.format_handoff(handoff),
                "error": str(exc),
                "handoff": handoff,
                "trace": [],
            }

        return self.execute_plan(plan, search_result)
