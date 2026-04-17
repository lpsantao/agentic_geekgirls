"""
Human-in-the-Loop (HITL) Pattern
Use case: AI email drafter that pauses for human review before sending.
The agent drafts, you approve or request edits, it revises — loops until you're happy.
"""
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


class EmailState(TypedDict):
    brief: str          # what the email should achieve
    draft: str          # current draft
    feedback: str       # human feedback (empty = approved)
    revision: int       # revision count


def draft_email(state: EmailState) -> EmailState:
    """LLM drafts (or revises) the email based on brief + optional feedback."""
    brief = state["brief"]
    feedback = state.get("feedback", "")
    revision = state.get("revision", 0)

    if feedback:
        prompt = (
            f"Revise this email draft based on feedback.\n\n"
            f"Original brief: {brief}\n\n"
            f"Current draft:\n{state['draft']}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Return only the revised email, no explanation."
        )
    else:
        prompt = (
            f"Write a professional email for the following brief:\n{brief}\n\n"
            f"Return only the email text, no explanation."
        )

    response = llm.invoke([
        SystemMessage(content="You are an expert professional email writer."),
        HumanMessage(content=prompt),
    ])

    action = "Revised" if revision > 0 else "Drafted"
    print(f"\n {action} email (revision #{revision}):\n")
    print("-" * 50)
    print(response.content)
    print("-" * 50)

    return {**state, "draft": response.content, "revision": revision + 1}


def human_review(state: EmailState) -> Command:
    """Pause and ask the human to approve or request changes."""
    feedback = interrupt({
        "message": "\n📬 Email is ready for your review.\nType 'send' to approve, or describe changes you want:",
        "draft": state["draft"],
    })

    if feedback.strip().lower() == "send":
        return Command(goto="send_email", update={**state, "feedback": ""})
    else:
        return Command(goto="draft_email", update={**state, "feedback": feedback})


def send_email(state: EmailState) -> EmailState:
    """Simulate sending the approved email."""
    print("\n🚀 Email approved and sent!")
    print("\nFinal email:")
    print("=" * 50)
    print(state["draft"])
    print("=" * 50)
    print(f"\n(Total revisions: {state['revision'] - 1})")
    return state


def build_graph():
    graph = StateGraph(EmailState)

    graph.add_node("draft_email", draft_email)
    graph.add_node("human_review", human_review)
    graph.add_node("send_email", send_email)

    graph.set_entry_point("draft_email")
    graph.add_edge("draft_email", "human_review")
    graph.add_edge("send_email", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run(brief: str):
    """Run the HITL email drafter interactively."""
    app = build_graph()
    config = {"configurable": {"thread_id": "email-session-1"}}

    initial_state: EmailState = {
        "brief": brief,
        "draft": "",
        "feedback": "",
        "revision": 0,
    }

    print(f"\n Brief: {brief}\n")

    state = app.invoke(initial_state, config=config)

    # Loop: resume with human input until approved (graph reaches END)
    while app.get_state(config).next:
        interrupt_value = app.get_state(config).tasks[0].interrupts[0].value
        print(interrupt_value["message"])

        feedback = input("\nYour input: ").strip()
        state = app.invoke(Command(resume=feedback), config=config)


if __name__ == "__main__":
    run("Write an email declining a job offer from Acme Corp, keeping the door open for future opportunities.")
