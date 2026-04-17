"""
Human-in-the-Loop (HITL) Pattern — Google ADK
Use case: AI email drafter that pauses for human review before sending.
The agent drafts, you approve or request edits, it revises — loops until you're happy.
"""
import asyncio
import uuid
from typing import AsyncGenerator

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

APP_NAME = "email_hitl"


# --- Step 1: Initial drafter — reads brief from state via {brief} interpolation ---
drafter = LlmAgent(
    name="EmailDrafter",
    model="gemini-2.0-flash",
    description="Drafts a professional email from a brief.",
    instruction=(
        "You are an expert professional email writer.\n"
        "Write a professional email for the following brief: {brief}\n"
        "Return only the email text, no explanation."
    ),
    output_key="draft",
)

# --- Step 2b: Reviser — runs only when human requested changes ---
reviser = LlmAgent(
    name="EmailReviser",
    model="gemini-2.0-flash",
    description="Revises an email draft based on human feedback.",
    instruction=(
        "You are an expert professional email writer.\n"
        "Revise this email draft based on the feedback provided.\n\n"
        "Current draft:\n{draft}\n\n"
        "Feedback: {feedback}\n\n"
        "Return only the revised email, no explanation."
    ),
    output_key="draft",
)


# --- Step 2a: Human review — pauses, shows draft, gets approval or feedback ---
class HumanReviewAgent(BaseAgent):
    """Interrupts the loop to show the current draft and collect human input."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        draft = ctx.session.state.get("draft", "")
        revision = ctx.session.state.get("revision", 0)

        print(f"\n Email draft (revision #{revision}):")
        print("-" * 50)
        print(draft)
        print("-" * 50)
        print("\nType 'send' to approve, or describe the changes you want:")

        feedback = input("Your input: ").strip()

        actions = EventActions()
        if feedback.lower() == "send":
            actions.escalate = True
            actions.state_delta = {"approved": True}
        else:
            actions.state_delta = {"feedback": feedback, "revision": revision + 1}

        yield Event(author=self.name, actions=actions)


# --- Loop: HumanReviewer → (escalate if approved) → Reviser → repeat ---
approval_loop = LoopAgent(
    name="ApprovalLoop",
    sub_agents=[HumanReviewAgent(name="HumanReviewer"), reviser],
    max_iterations=10,
)

# --- Full pipeline: Draft once, then review/revise until approved ---
pipeline = SequentialAgent(
    name="EmailHITLPipeline",
    sub_agents=[drafter, approval_loop],
)


async def run(brief: str):
    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME,
        user_id="user",
        session_id=session_id,
        state={"brief": brief, "draft": "", "feedback": "", "revision": 0},
    )

    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    print(f"\n Brief: {brief}\n")

    async for _ in runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=brief)]),
    ):
        pass  # HumanReviewAgent handles all I/O inline

    # Pipeline is done — check final state
    state = (await session_service.get_session(
        app_name=APP_NAME, user_id="user", session_id=session_id
    )).state

    if state.get("approved"):
        print("\n Email approved and sent!")
        print("\nFinal email:")
        print("=" * 50)
        print(state.get("draft", ""))
        print("=" * 50)
        print(f"\n(Total revisions: {state.get('revision', 0)})")


if __name__ == "__main__":
    asyncio.run(run(
        "Write an email declining a job offer from Acme Corp, keeping the door open for future opportunities."
    ))
