import argparse
from pprint import pprint

from langchain_google_genai import ChatGoogleGenerativeAI

from flight_agent.config import AppConfig, load_config
from flight_agent.harness import AgentPermissions, FlightAgentHarness
from flight_agent.models import FlightConstraints
from flight_agent.hybrid_agent import HybridFlightAgent
from flight_agent.plan_execute_agent import PlanThenExecuteFlightAgent
from flight_agent.react_agent import ReActFlightAgent
from flight_agent.tools import reset_booking_store


def create_model(
    config: AppConfig,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.model_name,
        api_key=config.api_key,
    )


def create_harness() -> FlightAgentHarness:
    constraints = FlightConstraints(
        origin="SGN",
        destination="DAD",
        date="2026-10-07",
        depart_before="12:00",
        max_price=2_000_000,
    )

    permissions = AgentPermissions(
        allow_booking=True,
        allow_payment=True,
    )

    return FlightAgentHarness(
        constraints=constraints,
        permissions=permissions,
    )


def create_agent(
    pattern: str,
    model,
    harness: FlightAgentHarness,
):
    if pattern == "react":
        return ReActFlightAgent(
            model=model,
            harness=harness,
        )

    if pattern == "plan_then_execute":
        return PlanThenExecuteFlightAgent(
            model=model,
            harness=harness,
        )

    if pattern == "hybrid":
        return HybridFlightAgent(
            model=model,
            harness=harness,
        )

    raise ValueError(f"Unknown agent pattern: {pattern}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SE373 Flight Agent.")

    parser.add_argument(
        "pattern",
        choices=[
            "react",
            "plan_then_execute",
            "hybrid",
        ],
        help="Agent pattern to run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config = load_config()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    reset_booking_store()

    model = create_model(config)
    harness = create_harness()
    agent = create_agent(
        pattern=args.pattern,
        model=model,
        harness=harness,
    )

    result = agent.run()

    pprint(result, sort_dicts=False)


if __name__ == "__main__":
    main()
