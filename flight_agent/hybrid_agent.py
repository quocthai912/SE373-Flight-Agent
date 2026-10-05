"""
Định nghĩa Hybrid Agent
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from flight_agent.harness import FlightAgentHarness
from flight_agent.plan_execute_agent import (
    FlightPlan,
    PlanStep,
    PlanThenExecuteFlightAgent,
)
from flight_agent.tools import search_flights

REPLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You replan a flight booking task after "
                "a previous plan failed. "
                "Create a new complete plan before more "
                "actions are executed. "
                "Available tools are book_seat, pay, "
                "and get_booking. "
                "Use only flights provided in the "
                "available flight data. "
                "The booking code does not exist yet, "
                'so use "$booking_code" for pay '
                "and get_booking. "
                "Use the previous plan and the last "
                "failure to avoid repeating the same "
                "failed choice. "
                "Replanning is only allowed before a "
                "booking has been created. "
                "If no viable plan remains, return an "
                "empty list of steps."
            ),
        ),
        (
            "human",
            (
                "Goal: {goal}\n"
                "Available flights: {flights}\n"
                "Previous plan: {previous_plan}\n"
                "Last failure: {failure}"
            ),
        ),
    ]
)


class HybridFlightAgent(PlanThenExecuteFlightAgent):
    """Điều phối mẫu Hybrid của Flight Agent."""

    def __init__(
        self,
        model: BaseChatModel,
        harness: FlightAgentHarness,
        max_replans: int = 2,
    ):
        if max_replans < 0:
            raise ValueError("max_replans must be zero or greater.")

        super().__init__(
            model=model,
            harness=harness,
        )

        self.max_replans = max_replans
        self.replanner = REPLANNER_PROMPT | model.with_structured_output(FlightPlan)

    def _create_replan(
        self, previous_plan: FlightPlan, failure: dict
    ) -> tuple[FlightPlan, dict]:
        """Thực hiện tạo lại Plan mới (Replan) sau khi Plan cũ thất bại."""
        constraints = self.harness.constraints

        search_result = search_flights(
            constraints.origin, constraints.destination, constraints.date
        )

        new_plan = self.replanner.invoke(
            {
                "goal": constraints.to_prompt(),
                "flights": json.dumps(
                    search_result["flights"],
                    sort_keys=True,
                ),
                "previous_plan": json.dumps(
                    previous_plan.model_dump(),
                    sort_keys=True,
                ),
                "failure": json.dumps(
                    failure,
                    sort_keys=True,
                ),
            }
        )

        return new_plan, search_result

    def _should_replan(
        self, step: PlanStep, observation: dict, booking_code: str | None
    ) -> bool:
        """Xác định khi nào thì cần phải lập lại kế hoạch (Replan)."""
        recoverable_statuses = {"constraint_violation", "not_found"}

        return (
            booking_code is None
            and step.tool == "book_seat"
            and observation["status"] in recoverable_statuses
        )

    def run(self) -> dict:
        """Chạy Hybrid Agent với khả năng Replan có giới hạn."""
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
                "plan_history": [],
                "replan_count": 0,
            }

        trace = []
        attempted_actions = []
        side_effects = []
        plan_history = []

        booking_code = None
        replan_count = 0

        while True:
            plan_number = replan_count + 1

            plan_history.append(
                {
                    "plan_number": plan_number,
                    "plan": plan.model_dump(),
                }
            )

            validation = self._validate_plan(plan, search_result)

            if validation["status"] != "valid":
                handoff = self.harness.create_handoff(
                    reason=validation["message"],
                    attempted_actions=attempted_actions,
                    side_effects=side_effects,
                    question="Would you like to review or regenerate the plan?",
                    booking_code=booking_code,
                )

                return {
                    "status": validation["status"],
                    "message": self.harness.format_handoff(handoff),
                    "handoff": handoff,
                    "trace": trace,
                    "plan_history": plan_history,
                    "replan_count": replan_count,
                }

            replan_requested = False

            for step_number, step in enumerate(plan.steps, start=1):
                args, observation = self._execute_step(step, booking_code)

                trace.append(
                    {
                        "plan_number": plan_number,
                        "step": step_number,
                        "action": step.tool,
                        "args": args,
                        "observation": observation,
                    }
                )

                attempted_actions.append(
                    f"Executed plan {plan_number}, step {step_number}: {step.tool} with {json.dumps(args, sort_keys=True)}."
                )

                if step.tool == "book_seat" and observation["status"] == "success":
                    booking_code = observation["booking"]["code"]
                    side_effects.append(f"Created booking {booking_code}.")

                elif step.tool == "pay" and observation["status"] == "success":
                    side_effects.append(
                        f"Paid booking {observation["booking"]["code"]}."
                    )

                if observation["status"] == "success":
                    continue

                if observation["status"] == "denied":
                    handoff = self.harness.create_handoff(
                        reason=f"Permission is required to execute {step.tool}.",
                        attempted_actions=attempted_actions,
                        side_effects=side_effects,
                        question=f"Should the {step.tool} action be approved?",
                        booking_code=booking_code,
                    )

                    return {
                        "status": "handoff",
                        "message": self.harness.format_handoff(handoff),
                        "handoff": handoff,
                        "trace": trace,
                        "plan_history": plan_history,
                        "replan_count": replan_count,
                    }

                if self._should_replan(step, observation, booking_code):
                    if replan_count >= self.max_replans:
                        handoff = self.harness.create_handoff(
                            reason="The maximum number of replans was reached.",
                            attempted_actions=attempted_actions,
                            side_effects=side_effects,
                            question="Would you like to review the failed plans?",
                            booking_code=booking_code,
                        )

                        return {
                            "status": "max_replans_exceeded",
                            "message": self.harness.format_handoff(handoff),
                            "handoff": handoff,
                            "trace": trace,
                            "plan_history": plan_history,
                            "replan_count": replan_count,
                        }

                    try:
                        plan, search_result = self._create_replan(
                            previous_plan=plan, failure=observation
                        )

                    except Exception as exc:
                        handoff = self.harness.create_handoff(
                            reason="The replanner failed to create a valid plan.",
                            attempted_actions=attempted_actions,
                            side_effects=side_effects,
                            question="Would you like to review the failed planning state?",
                            booking_code=booking_code,
                        )

                        return {
                            "status": "planning_error",
                            "message": self.harness.format_handoff(handoff),
                            "error": str(exc),
                            "handoff": handoff,
                            "trace": trace,
                            "plan_history": plan_history,
                            "replan_count": replan_count,
                        }

                    replan_count += 1
                    replan_requested = True
                    break

                handoff = self.harness.create_handoff(
                    reason=(
                        f"Planned step {step_number} failed with status {observation["status"]}."
                    ),
                    attempted_actions=attempted_actions,
                    side_effects=side_effects,
                    question="Would you like to review the failed plan?",
                    booking_code=booking_code,
                )

                return {
                    "status": "handoff",
                    "message": self.harness.format_handoff(handoff),
                    "handoff": handoff,
                    "trace": trace,
                    "plan_history": plan_history,
                    "replan_count": replan_count,
                }

            if replan_requested:
                continue

            if booking_code is not None:
                completion = self.harness.check_completion(booking_code)

                if completion["status"] == "complete":
                    return {
                        "status": "complete",
                        "message": "The booking task was completed successfully.",
                        "booking": completion["booking"],
                        "trace": trace,
                        "plan_history": plan_history,
                        "replan_count": replan_count,
                    }

            handoff = self.harness.create_handoff(
                reason="The current plan finished without satisfying the completion criteria.",
                attempted_actions=attempted_actions,
                side_effects=side_effects,
                question="Would you like to review the current state?",
                booking_code=booking_code,
            )

            return {
                "status": "incomplete",
                "message": self.harness.format_handoff(handoff),
                "handoff": handoff,
                "trace": trace,
                "plan_history": plan_history,
                "replan_count": replan_count,
            }
