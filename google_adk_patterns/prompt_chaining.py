"""
Prompt Chaining Pattern — Google ADK
Use case: Extract raw laptop specs from unstructured text, then transform them into a typed JSON object.

Key concept: SequentialAgent chains LlmAgents so each agent's output_key feeds the
next agent's instruction via {state_key} interpolation. No manual data passing needed.

Pipeline: extract_specs_agent → transform_specs_agent
"""
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
import uuid
import asyncio

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
APP_NAME = "specs_pipeline"

# USER_ID identifies the person interacting with the agent.
USER_ID = "medtiles"

# Model to use for all agents in this pipeline.
MODEL = "gemini-2.0-flash"


# --- Pydantic schema for the final structured output ---
# output_schema forces the transform agent to return valid JSON matching this shape.
# ADK validates the response and raises an error if the schema isn't satisfied.
class SpecsJson(BaseModel):
    cpu: str = Field(description="CPU details")
    memory: str = Field(description="RAM size")
    storage: str = Field(description="Storage details")


# --- Step 1: Extract agent ---
# Reads the raw user message and pulls out the specs as bullet points.
# output_key="specifications" saves the result into session state.
extract_agent = LlmAgent(
    name="extract_specs_agent",
    model=MODEL,
    instruction="Extract the technical specifications from the user's message. Return only the extracted specs as concise bullet points.",
    output_key="specifications",  # Written to session state; the next agent reads it.
)


# --- Step 2: Transform agent ---
# Reads the extracted specs from session state and converts them to structured JSON.
# Using a callable instruction lets us inject the current state value at runtime.
def build_transform_instruction(ctx):
    # ctx.state holds the current session state — read the previous agent's output here.
    specs = ctx.state.get("specifications", "")
    return f"Transform to JSON with cpu, memory, storage keys:\n\n{specs}"

transform_agent = LlmAgent(
    name="transform_specs_agent",
    model=MODEL,
    instruction=build_transform_instruction,  # Called fresh each invocation with current state.
    output_schema=SpecsJson,  # Forces the LLM to return valid JSON matching SpecsJson.
    output_key="result",
)


# --- Step 3: Build the pipeline ---
# SequentialAgent runs sub_agents left-to-right.
# Each agent's output_key is automatically available in state for the next agent.
pipeline = SequentialAgent(
    name="specs_pipeline",
    sub_agents=[extract_agent, transform_agent],
)


async def run_pipeline(runner: Runner, request: str):
    """Runs the pipeline with a user request and returns the final structured output."""
    print(f"\n--- Running Pipeline with request: '{request}' ---")
    final_result = ""

    try:
        # Each call gets its own session so requests are fully isolated.
        session_id = str(uuid.uuid4())

        # runner.session_service and runner.app_name are set when the runner is created.
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            session_id=session_id,
        )

        # Identify the last agent so we only capture its final response.
        last_agent_name = pipeline.sub_agents[-1].name

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):
            # Only capture the final response from the last agent in the chain.
            if (
                event.author == last_agent_name
                and hasattr(event, "is_final_response")
                and event.is_final_response()
                and event.content
            ):
                if hasattr(event.content, "text") and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)

        print(f"\nAnswer: {final_result}")
        return final_result

    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


# --- Step 4: Main Execution ---

def main():
    print("--- Prompt Chaining Pipeline Example ---")

    # Create the session service and runner once; reuse across multiple requests.
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline,
        session_service=session_service,
        app_name=APP_NAME,
    )

    asyncio.run(run_pipeline(
        runner,
        "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD.",
    ))


if __name__ == "__main__":
    main()
