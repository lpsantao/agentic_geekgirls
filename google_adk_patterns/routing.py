"""
Routing Pattern — Google ADK
Use case: A coordinator agent that routes travel requests to specialist sub-agents.

Key concept: sub_agents enables LLM-driven delegation (Auto-Flow).
The coordinator reads the user's intent and forwards the request to the right agent.
No hard-coded routing logic — the LLM decides based on descriptions alone.
"""
import asyncio
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
APP_NAME = "travel_router"

# USER_ID identifies the person interacting with the agent.
USER_ID = "medtiles"


# --- Step 1: Define Tool Functions ---
# These functions simulate actions that specialist agents will perform.
# In a real system these would call actual booking/info APIs.

def booking_handler(request: str) -> str:
    """
    Handles booking-related requests.
    Args:
        request: The user's booking request.
    Returns:
        A message indicating the booking action was simulated.
    """
    print("-------------------------- Booking Handler Called ----------------------------")
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's information request.
    Returns:
        A message indicating the information retrieval was simulated.
    """
    print("-------------------------- Info Handler Called ----------------------------")
    return f"Information request for '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """
    Handles requests that cannot be clearly categorised.
    Args:
        request: The user's unclear request.
    Returns:
        A message indicating that the request could not be delegated.
    """
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


# --- Step 2: Wrap Functions as ADK Tools ---
# FunctionTool makes a Python function callable by an LLM agent.
# ADK automatically generates the tool schema from the function signature and docstring.
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)


# --- Step 3: Define Specialist Agents ---
# Each specialist has a clear description — this is how the coordinator knows
# which agent to route to. Descriptions are part of the routing signal.
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="A specialized agent that handles booking-related requests by calling the booking tool.",
    tools=[booking_tool],
)

info_agent = Agent(
    name="Info",
    model="gemini-2.0-flash",
    description="A specialized agent that handles general information requests by calling the info tool.",
    tools=[info_tool],
)


# --- Step 4: Define the Coordinator ---
# The coordinator uses sub_agents to enable LLM-driven delegation (Auto-Flow).
# It reads the user's message and forwards it to the best-matching sub-agent
# based on their descriptions — no if/else routing needed.
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyse incoming user requests "
        "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
        "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
        "- For all other general information questions, delegate to the 'Info' agent."
    ),
    description="A coordinator that routes user requests to the correct specialist agent.",
    # sub_agents activates Auto-Flow: the LLM picks which sub-agent to invoke.
    sub_agents=[booking_agent, info_agent],
)


# --- Step 5: Execution Logic ---

async def run_coordinator(runner: Runner, request: str):
    """
    Runs the coordinator with a user request and prints the final response.

    Each call creates its own session so requests are fully isolated.
    The runner is shared across calls (it's stateless).
    """
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""

    try:
        # A fresh session per request keeps conversations isolated.
        session_id = str(uuid.uuid4())

        # runner.session_service and runner.app_name give us the service and name
        # that were set when the runner was created — avoids duplicating constants here.
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            session_id=session_id,
        )

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):
            if event.is_final_response() and event.content:
                # Prefer event.content.text (direct) over iterating parts to avoid SDK warnings.
                if hasattr(event.content, "text") and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                break

        print(f"\nCoordinator Final Response: {final_result}")
        return final_result

    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


# --- Step 6: Main Execution ---

def main():
    print("--- Google ADK Routing Example ---")

    # Create the session service and runner once; reuse across multiple requests.
    # Runner is stateless between calls — all state lives in the session.
    session_service = InMemorySessionService()
    runner = Runner(agent=coordinator, app_name=APP_NAME, session_service=session_service)

    result_a = asyncio.run(run_coordinator(runner, "Book me a hotel in Paris."))
    print(f"Final Output A: {result_a}")

    result_b = asyncio.run(run_coordinator(runner, "What is the highest mountain in the world?"))
    print(f"Final Output B: {result_b}")

    result_c = asyncio.run(run_coordinator(runner, "Tell me a random fact."))
    print(f"Final Output C: {result_c}")

    result_d = asyncio.run(run_coordinator(runner, "Find flights to Tokyo next month."))
    print(f"Final Output D: {result_d}")


if __name__ == "__main__":
    main()
