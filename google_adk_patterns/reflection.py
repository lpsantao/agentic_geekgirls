from google.adk.agents import SequentialAgent, LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import uuid
import asyncio

from dotenv import load_dotenv
load_dotenv()

MAX_ITERATIONS = 2

# Step 1: Generate the initial draft (runs once before the loop).
generator = LlmAgent(
    name="DraftWriter",
    description="Generates initial draft content on a given subject.",
    instruction="Write a short, informative paragraph about the user's subject.",
    output_key="draft_text"
)

# Callback on the reviewer: exit the loop as soon as the draft is accurate,
# so the rewriter doesn't run on an already-correct draft.
def check_accuracy(callback_context: CallbackContext):
    review = callback_context.state.get("review_output", "")
    if "INACCURATE" not in review and "ACCURATE" in review:
        callback_context.actions.escalate = True

# Step 2a: Critique the current draft.
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
    after_agent_callback=check_accuracy
)

# Step 2b: Revise the draft based on the critique.
# Only runs when the reviewer found issues (i.e. did NOT escalate).
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
    output_key="draft_text"  # Overwrites the draft for the next reviewer pass.
)

# The reflection loop: reviewer → (escalate if ACCURATE) → rewriter → repeat.
reflection_loop = LoopAgent(
    name="ReflectionLoop",
    sub_agents=[reviewer, rewriter],
    max_iterations=MAX_ITERATIONS
)

# Full pipeline: generate once, then reflect until accurate.
reflection_pipeline = SequentialAgent(
    name="ReflectionPipeline",
    sub_agents=[generator, reflection_loop]
)


async def run_pipeline(runner: Runner, request: str):
    print(f"\n--- Running Reflection Pipeline ---")
    print(f"Input: '{request}'\n")

    user_id = "user"
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
        # Print intermediate state updates so we can follow the loop.
        if event.actions.state_delta:
            delta = event.actions.state_delta
            if "draft_text" in delta:
                print(f"[{event.author}] Draft updated:\n{delta['draft_text']}\n")
            if "review_output" in delta:
                print(f"[{event.author}] Review:\n{delta['review_output']}\n")


async def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=reflection_pipeline,
        session_service=session_service,
        app_name="ReflectionApp"
    )

    # Deliberately wrong facts to exercise the reflection loop.
    await run_pipeline(runner, "The Eiffel Tower is located in Berlin.")


if __name__ == "__main__":
    asyncio.run(main())
