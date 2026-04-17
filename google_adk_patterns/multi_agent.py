"""
Multi-Agent Pattern — Google ADK
Use case: Meeting Follow-up Email System.
  SummaryAgent → ActionItemAgent → EmailComposerAgent → GmailSenderAgent

Gmail setup (one-time):
  1. Google Cloud Console → enable Gmail API → create OAuth2 Desktop credentials
  2. Download as credentials.json and place in this directory
  3. First run opens a browser for OAuth consent → creates token.json automatically
"""
import asyncio
import base64
import uuid
from email.mime.text import MIMEText

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# APP_NAME groups all sessions and runs for this application.
APP_NAME = "meeting_followup"

# USER_ID identifies the person interacting with the agent.
USER_ID = "medtiles"

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def send_gmail(to: str, subject: str, body: str) -> str:
    """Sends an email via Gmail API. Returns 'sent' or an error message."""
    try:
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import os, pickle

        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as f:
                pickle.dump(creds, f)

        service = build("gmail", "v1", credentials=creds)
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent to {to}"
    except FileNotFoundError:
        return "Error: credentials.json not found. Enable Gmail API in Google Cloud Console and download OAuth2 credentials."
    except Exception as e:
        return f"Gmail error: {e}"


# Wrap the Python function so ADK can call it as a tool.
# ADK auto-generates the tool schema from the function signature and docstring.
gmail_tool = FunctionTool(send_gmail)

# --- Agent 1: Summariser ---
# output_key="summary" saves the result into session state.
# Downstream agents reference it with {summary} in their instructions.
summariser = LlmAgent(
    name="SummaryAgent",
    model="gemini-2.0-flash",
    description="Summarises raw meeting notes into 3 key bullet points.",
    instruction=(
        "Summarise the meeting notes in {raw_notes} into exactly 3 concise bullet points "
        "covering the key discussion points. Output only the bullet points, no JSON, no preamble."
    ),
    output_key="summary",
)

# --- Agent 2: Action Item Extractor ---
action_extractor = LlmAgent(
    name="ActionItemAgent",
    model="gemini-2.0-flash",
    description="Extracts action items with owners from meeting notes.",
    instruction=(
        "From the meeting notes in {raw_notes}, extract all action items. "
        "Format each as: '- [Owner] Task (deadline if mentioned)'. "
        "Use 'Team' if no owner is mentioned."
    ),
    output_key="action_items",
)

# --- Agent 3: Email Composer ---
email_composer = LlmAgent(
    name="EmailComposerAgent",
    model="gemini-2.0-flash",
    description="Writes a professional follow-up email from summary and action items.",
    instruction=(
        "Write a professional meeting follow-up email.\n\n"
        "Meeting Summary:\n{summary}\n\n"
        "Action Items:\n{action_items}\n\n"
        "Call the send_gmail tool with:\n"
        "  to = {recipient_email}\n"
        "  subject = a suitable subject line\n"
        "  body = the full email body\n\n"
        "After calling the tool, report whether the email was sent successfully."
    ),
    tools=[gmail_tool],
)

# --- Pipeline: run all three agents in sequence ---
# SequentialAgent runs sub_agents left-to-right.
# Each agent can read any key in session state set by a previous agent.
pipeline = SequentialAgent(
    name="MeetingFollowupPipeline",
    sub_agents=[summariser, action_extractor, email_composer],
)


async def run(raw_notes: str, recipient_email: str):
    # InMemorySessionService stores all sessions in RAM — ideal for demos.
    session_service = InMemorySessionService()
    # Unique session ID per run so conversations don't collide.
    session_id = str(uuid.uuid4())

    # Pre-populate session state so agents can reference {raw_notes} and
    # {recipient_email} directly in their instructions without extra tool calls.
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={
            "raw_notes": raw_notes,
            "recipient_email": recipient_email,
        },
    )

    # Runner wires together the agent, session service, and app context.
    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    print(f"Running meeting follow-up pipeline for: {recipient_email}\n")
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=raw_notes)]),
    ):
        # state_delta contains the session state keys that changed in this event.
        # We print intermediate results to observe the pipeline as it runs.
        if event.actions.state_delta:
            delta = event.actions.state_delta
            if "summary" in delta:
                print(f"[SummaryAgent]\n{delta['summary']}\n")
            if "action_items" in delta:
                print(f"[ActionItemAgent]\n{delta['action_items']}\n")
        # The email composer is the last agent — its final response confirms the send.
        if event.is_final_response() and event.content and event.author == "EmailComposerAgent":
            print(f"[EmailComposerAgent] {event.content.parts[0].text}")


SAMPLE_NOTES = """
Product sync 15/04/2026
Attendees: Ana (PM), Bruno (Eng), Carla (Design), David (QA)

we reviewed the Q2 roadmap. The checkout redesign is on track for May 1st.
Bruno mentioned the payment API integration is blocked on credentials from the vendor -
he will follow up with them by Friday.
Carla will share the final Figma mockups by Wednesday EOD.
David raised a concern about test coverage on mobile; he'll write a test plan by next Tuesday.
Ana will schedule a demo with stakeholders for May 5th.
One open question: do we support Apple Pay at launch? Ana will confirm with the CEO by Thursday.
not sure how to proceed on the SSO component, need to explore in more detail next meeting
"""

if __name__ == "__main__":
    asyncio.run(run(
        raw_notes=SAMPLE_NOTES,
        recipient_email="lilianapsantao@gmail.com",  # ← change address
    ))
