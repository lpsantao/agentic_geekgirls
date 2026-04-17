"""
Multi-Agent Pattern — LangGraph
Use case: Meeting Follow-up Email System.
  SummaryAgent → ActionItemAgent → EmailComposerAgent → GmailSenderAgent

Gmail setup (one-time):
  1. Google Cloud Console → enable Gmail API → create OAuth2 Desktop credentials
  2. Download as credentials.json and place in this directory
  3. First run opens a browser for OAuth consent → creates token.json automatically
"""
import json
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_community.gmail.send_message import GmailSendMessage
from langchain_google_community.gmail.utils import build_resource_service

from dotenv import load_dotenv
load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


class MeetingState(TypedDict):
    raw_notes: str
    recipient_email: str
    summary: str
    action_items: str
    email_subject: str
    email_body: str
    sent: bool


# --- Agent 1: Summariser ---
def summarise(state: MeetingState) -> MeetingState:
    print("\n[SummaryAgent] Summarising meeting notes...")
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert at summarising meeting notes. "
            "Produce a concise 3-bullet summary of the key discussion points."
        )),
        HumanMessage(content=state["raw_notes"]),
    ])
    print(f"Summary:\n{response.content}")
    return {**state, "summary": response.content}


# --- Agent 2: Action Item Extractor ---
def extract_actions(state: MeetingState) -> MeetingState:
    print("\n[ActionItemAgent] Extracting action items...")
    response = llm.invoke([
        SystemMessage(content=(
            "Extract all action items from the meeting notes. "
            "Format each as: '- [Owner] Task (deadline if mentioned)'. "
            "If no owner is mentioned, use 'Team'."
        )),
        HumanMessage(content=state["raw_notes"]),
    ])
    print(f"Action items:\n{response.content}")
    return {**state, "action_items": response.content}


# --- Agent 3: Email Composer ---
def compose_email(state: MeetingState) -> MeetingState:
    print("\n[EmailComposerAgent] Drafting follow-up email...")
    prompt = (
        f"Write a professional meeting follow-up email using the content below.\n\n"
        f"Meeting Summary:\n{state['summary']}\n\n"
        f"Action Items:\n{state['action_items']}\n\n"
        f"Return a JSON object with exactly two keys: 'subject' and 'body'."
    )
    response = llm.invoke([
        SystemMessage(content="You are a professional business communication expert."),
        HumanMessage(content=prompt),
    ])

    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    parsed = json.loads(match.group()) if match else {
        "subject": "Meeting Follow-up",
        "body": response.content,
    }
    print(f"Subject: {parsed['subject']}")
    return {**state, "email_subject": parsed["subject"], "email_body": parsed["body"]}


# --- Agent 4: Gmail Sender ---
def send_email(state: MeetingState) -> MeetingState:
    print(f"\n[GmailSenderAgent] Sending email to {state['recipient_email']}...")
    try:
        service = build_resource_service()
        tool = GmailSendMessage(api_resource=service)
        tool.run({
            "to": [state["recipient_email"]],
            "subject": state["email_subject"],
            "message": state["email_body"],
        })
        print("Email sent!")
        return {**state, "sent": True}
    except Exception as e:
        print(f"Gmail error: {e}")
        print("(Ensure credentials.json is present and Gmail API is enabled in Google Cloud Console)")
        return {**state, "sent": False}


def build_graph():
    graph = StateGraph(MeetingState)

    graph.add_node("summarise", summarise)
    graph.add_node("extract_actions", extract_actions)
    graph.add_node("compose_email", compose_email)
    graph.add_node("send_email", send_email)

    graph.set_entry_point("summarise")
    graph.add_edge("summarise", "extract_actions")
    graph.add_edge("extract_actions", "compose_email")
    graph.add_edge("compose_email", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()


SAMPLE_NOTES = """
Product sync — 15 April 2026
Attendees: Ana (PM), Bruno (Eng), Carla (Design), David (QA)

We reviewed the Q2 roadmap. The checkout redesign is on track for May 1st.
Bruno mentioned the payment API integration is blocked on credentials from the vendor —
he will follow up with them by Friday.
Carla will share the final Figma mockups by Wednesday EOD.
David raised a concern about test coverage on mobile; he'll write a test plan by next Tuesday.
Ana will schedule a demo with stakeholders for May 5th.
One open question: do we support Apple Pay at launch? Ana will confirm with the CEO by Thursday.
"""

if __name__ == "__main__":
    if not llm:
        print("Skipping — LLM not available.")
    else:
        build_graph().invoke({
            "raw_notes": SAMPLE_NOTES,
            "recipient_email": "lilianapsantao@gmail.com",  # ← change address
            "summary": "",
            "action_items": "",
            "email_subject": "",
            "email_body": "",
            "sent": False,
        })
