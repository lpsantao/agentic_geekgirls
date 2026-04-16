import asyncio
import uuid

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# Step 1: Define Tool Functions
# These functions simulate the actions that the specialist agents will perform when they receive a request.
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
    Handles requests that cannot be clearly categorized.
    Args:
        request: The user's unclear request.
    Returns:
        A message indicating that the request could not be delegated.
    """
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# Step 2: Create Tools from Functions
# These tools will be used by the specialist agents to perform their respective tasks.
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Step 3: Define Specialist Agents
# Define the specialist agents that will handle specific types of requests using the tools defined above
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="A specialized agent that handles booking-related requests by calling the booking tool.",
    tools=[booking_tool]
)

info_agent = Agent(
    name="Info",
    model="gemini-2.0-flash",
    description="A specialized agent that handles general information requests by calling the info tool.",
    tools=[info_tool]
)

# Step 4: Define the main coordinator
# This agent that will route requests to the appropriate sub-agent based on the content of the request
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyze incoming user requests "
        "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
        "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
        "- For all other general information questions, delegate to the 'Info' agent."
    ),
    description="A coordinator that routes user requests to the correct specialist agent.",
    # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
    sub_agents=[booking_agent, info_agent]
)

# Step 5: Execution Logic 

async def run_coordinator(runner: InMemoryRunner, request: str):
    """
    Runs the coordinator agent with a given user request and captures the final response from the appropriate sub-agent.
    Args:
        runner: An instance of InMemoryRunner initialized with the coordinator agent.
        request: The user's request to be processed by the coordinator and its sub-agents.
    Returns:
        The final response from the sub-agent that handled the request.
    """
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "medtiles"
        session_id = str(uuid.uuid4())

        await runner.session_service.create_session(
            app_name=runner.app_name, 
            user_id=user_id, 
            session_id=session_id
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                # Try to get text directly from event.content to avoid iterating parts
                if hasattr(event.content, 'text') and event.content.text:
                     final_result = event.content.text
                elif event.content.parts:
                    # Fallback: Iterate through parts and extract text (might trigger warning)
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                # Assuming the loop should break after the final response
                break

        print(f"\nCoordinator Final Response: {final_result}")
        return final_result
    
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


# ---------------------------------------------------------
# Step 6: Main Execution
# ---------------------------------------------------------
def main():
    """Main function to run the ADK example."""
    print("--- Google ADK Routing Example ---")

    runner = InMemoryRunner(coordinator)
    # Example Usage
    result_a = asyncio.run(run_coordinator(runner, "Book me a hotel in Paris."))
    print(f"Final Output A: {result_a}")
    result_b = asyncio.run(run_coordinator(runner, "What is the highest mountain in the world?"))
    print(f"Final Output B: {result_b}")
    result_c = asyncio.run(run_coordinator(runner, "Tell me a random fact.")) # Should go to Info
    print(f"Final Output C: {result_c}")
    result_d = asyncio.run(run_coordinator(runner, "Find flights to Tokyo next month.")) # Should go to Booker
    print(f"Final Output D: {result_d}")


if __name__ == "__main__":
    main()
     
