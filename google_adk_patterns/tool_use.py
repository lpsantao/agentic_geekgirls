"""
Tool Use Pattern — Google ADK
Use case: An agent that uses Google Search to answer user questions.

Key concept: Tools extend what an agent can do beyond its training data.
google_search is a pre-built ADK tool — the agent decides when to call it.
"""
import asyncio
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
# Think of it as the "namespace" for your agent system.
APP_NAME = "google_search_agent"

# USER_ID identifies who is interacting with the agent.
# In production this would come from your auth system (e.g. a real user ID).
USER_ID = "medtiles"

# --- Define the agent ---
# tools=[google_search] tells ADK to make the Google Search tool available.
# The LLM decides autonomously when to call it based on the user's question.
root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[google_search],
)


async def call_agent_async(query: str):
    # A session is a container for one conversation: it holds the message history
    # and a key-value state store that agents can read/write.
    # InMemorySessionService keeps sessions in RAM — perfect for demos and workshops.
    session_service = InMemorySessionService()

    # Generate a unique session ID so different conversations don't collide.
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    # Runner is the orchestrator: it connects the agent, the session service,
    # and the app context, and manages the execution loop.
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    # ADK uses a structured Content/Part format for messages (same as the Gemini API).
    # role="user" marks this as the human's turn.
    content = types.Content(role="user", parts=[types.Part(text=query)])

    # run_async is a streaming interface — it yields Event objects as the agent works.
    # Events include tool calls, intermediate thoughts, and the final answer.
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        # is_final_response() is True only for the agent's last, complete answer.
        # Ignore intermediate events (tool calls, partial responses) unless you need them.
        if event.is_final_response():
            print("Agent Response: ", event.content.parts[0].text)


def main():
    asyncio.run(call_agent_async("what's the latest ai news?"))


if __name__ == "__main__":
    main()
