"""
Reflection Pattern — Google ADK
Use case: Self-correcting content generation with a fact-check loop.

Key concept: A LoopAgent keeps iterating sub-agents until a termination condition
is met. Here a reviewer checks the draft and either approves (escalate=True, stops
the loop) or asks a rewriter to fix it (loop continues).

Pipeline: DraftWriter → [LoopAgent: FactChecker → Rewriter] (max MAX_ITERATIONS)
"""
from google.adk.agents import SequentialAgent, LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import uuid
import asyncio

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
APP_NAME = "reflection_pipeline"

# USER_ID identifies the person interacting with the agent.
USER_ID = "medtiles"

# Maximum number of review → rewrite cycles before the loop stops unconditionally.
MAX_ITERATIONS = 2


# --- Step 1: Generator ---
# Runs once at the start of the pipeline to produce the initial draft.
# output_key="draft_text" saves the result into session state under that key —
# downstream agents read it from state using {draft_text} interpolation.
generator = LlmAgent(
    name="DraftWriter",
    description="Generates initial draft content on a given subject.",
    instruction="Write a short, informative paragraph about the user's subject.",
    output_key="draft_text",
)


# --- Callback: early-exit when the draft is already accurate ---
# after_agent_callback runs after the agent's turn completes.
# Setting actions.escalate = True tells the LoopAgent to stop immediately
# instead of running the next sub-agent (the rewriter).
def check_accuracy(callback_context: CallbackContext):
    review = callback_context.state.get("review_output", "")
    if "INACCURATE" not in review and "ACCURATE" in review:
        # The draft is accurate — no rewrite needed; exit the loop early.
        callback_context.actions.escalate = True


# --- Step 2a: Reviewer ---
# Reads the current draft from state and classifies it as ACCURATE or INACCURATE.
# The after_agent_callback fires immediately after this agent writes its output —
# if the draft is good, it escalates (stops the loop) before the rewriter runs.
reviewer = LlmAgent(
    name="FactChecker",
    description="Reviews a draft for factual accuracy and signals when it is correct.",
    instruction="""
    You are a meticulous fact-checker.
    1. Read the text in state key 'draft_text'.
    2. Carefully verify the factual accuracy of every claim.
    3. Respond with:
       - First line: exactly "ACCURATE" if all facts are correct, or "INACCURATE" if any are wrong.
       - Remaining lines: a clear explanation. If INACCURATE, cite each specific issue.
    """,
    output_key="review_output",
    after_agent_callback=check_accuracy,
)


# --- Step 2b: Rewriter ---
# Only runs when the reviewer found issues (i.e. did NOT escalate).
# Overwrites draft_text so the next reviewer pass sees the corrected version.
rewriter = LlmAgent(
    name="Rewriter",
    description="Rewrites a draft to fix factual issues identified by the reviewer.",
    instruction="""
    You are a careful writer tasked with correcting a draft.
    1. Read the original draft in state key 'draft_text'.
    2. Read the critique in state key 'review_output'.
    3. Rewrite the paragraph to fix every identified factual issue.
    4. Output only the corrected paragraph — no preamble, no explanation.
    """,
    output_key="draft_text",  # Overwrites the draft so the next reviewer pass sees the fix.
)


# --- Step 3: Reflection loop ---
# LoopAgent runs its sub_agents in sequence, then repeats from the first sub-agent.
# The loop ends when: escalate=True is set (accurate draft) OR max_iterations is reached.
reflection_loop = LoopAgent(
    name="ReflectionLoop",
    sub_agents=[reviewer, rewriter],
    max_iterations=MAX_ITERATIONS,
)


# --- Step 4: Full pipeline ---
# SequentialAgent runs sub_agents one after another in order.
# generate first, then enter the reflect-until-accurate loop.
reflection_pipeline = SequentialAgent(
    name="ReflectionPipeline",
    sub_agents=[generator, reflection_loop],
)


async def run_pipeline(runner: Runner, request: str):
    """Runs the reflection pipeline and prints state updates as the loop progresses."""
    print(f"\n--- Running Reflection Pipeline ---")
    print(f"Input: '{request}'\n")

    session_id = str(uuid.uuid4())

    # runner.session_service and runner.app_name are set when the runner is created —
    # accessing them here avoids duplicating APP_NAME across functions.
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
        # state_delta contains the session state keys that changed in this event.
        # We print them to observe the reflection loop in action.
        if event.actions.state_delta:
            delta = event.actions.state_delta
            if "draft_text" in delta:
                print(f"[{event.author}] Draft updated:\n{delta['draft_text']}\n")
            if "review_output" in delta:
                print(f"[{event.author}] Review:\n{delta['review_output']}\n")


async def main():
    # Create the session service and runner once; pass the runner to run_pipeline.
    session_service = InMemorySessionService()
    runner = Runner(
        agent=reflection_pipeline,
        session_service=session_service,
        app_name=APP_NAME,
    )

    # Deliberately wrong facts so the reflection loop has something to correct.
    await run_pipeline(runner, "The Eiffel Tower is located in Berlin.")


if __name__ == "__main__":
    asyncio.run(main())
