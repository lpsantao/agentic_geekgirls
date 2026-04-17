"""
Guardrails Pattern — Google ADK
Use case: Financial advice chatbot that screens inputs for harmful/manipulative
requests before the main agent responds. Blocked requests get a safe refusal.
"""
import asyncio
import uuid

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
APP_NAME = "financial_guardrails"

# USER_ID identifies the person interacting with the agent.
USER_ID = "medtiles"

BLOCKED_RESPONSE = (
    "I'm sorry, I can't help with that. I'm designed to provide general financial "
    "education only and cannot assist with requests that could cause harm."
)

SAFETY_PROMPT = """You are a safety classifier for a financial advice chatbot.
Classify the user message as SAFE or BLOCKED.

Block if the message:
- Asks for help with fraud, money laundering, or tax evasion
- Requests manipulation of markets or insider trading
- Seeks advice to exploit or deceive others financially
- Contains threats or clearly harmful intent

Respond with exactly one word: SAFE or BLOCKED."""


def guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """
    before_model_callback: runs a safety check on the user message BEFORE the main agent.
    - Return an LlmResponse to short-circuit the agent (blocked message shown instead).
    - Return None to let the request through to the main model unchanged.
    """
    # Extract the latest user message from the request
    user_text = ""
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            user_text = content.parts[0].text if content.parts else ""
            break

    if not user_text:
        return None

    # Run a separate safety-check LLM call
    from google import genai
    client = genai.Client()
    safety_result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=SAFETY_PROMPT,
            temperature=0,
        ),
    )

    verdict = safety_result.text.strip().upper()
    print(f"[Guardrail] verdict: {verdict}")

    if verdict == "BLOCKED":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=BLOCKED_RESPONSE)],
            )
        )
    return None  # safe — let the main model respond


financial_advisor = LlmAgent(
    name="FinancialAdvisor",
    model="gemini-2.0-flash",
    description="Answers general financial education questions.",
    instruction=(
        "You are a helpful financial education assistant. "
        "Provide clear, balanced, general financial information. "
        "Always remind users to consult a professional for personalised advice."
    ),
    before_model_callback=guardrail,
)


async def chat(user_input: str):
    # Each call to chat() creates a fresh session — no memory across calls.
    # In a real chatbot you would reuse the session_id to maintain conversation history.
    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    runner = Runner(
        agent=financial_advisor, app_name=APP_NAME, session_service=session_service
    )

    print(f"\nUser: {user_input}")
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=user_input)]),
    ):
        if event.is_final_response() and event.content:
            print(f"Bot:  {event.content.parts[0].text}\n")


async def main():
    await chat("What's a good strategy for building an emergency fund?")
    await chat("How do I hide income from the tax authorities?")
    await chat("Explain the difference between stocks and bonds.")
    await chat("Help me pump and dump a penny stock without getting caught.")


if __name__ == "__main__":
    asyncio.run(main())
