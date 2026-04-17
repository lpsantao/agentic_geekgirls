"""
Guardrails Pattern — LangGraph
Use case: Financial advice chatbot that screens inputs for harmful/manipulative
requests before the main agent responds. Blocked requests get a safe refusal.
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

BLOCKED_RESPONSE = (
    "I'm sorry, I can't help with that. I'm designed to provide general financial "
    "education only and cannot assist with requests that could cause harm."
)


class ChatState(TypedDict):
    user_input: str
    verdict: Literal["safe", "blocked"]
    response: str


def safety_check(state: ChatState) -> ChatState:
    """LLM-based guardrail: classifies input as safe or blocked."""
    result = llm.invoke([
        SystemMessage(content=(
            "You are a safety classifier for a financial advice chatbot.\n"
            "Classify the user message as SAFE or BLOCKED.\n\n"
            "Block if the message:\n"
            "- Asks for help with fraud, money laundering, or tax evasion\n"
            "- Requests manipulation of markets or insider trading\n"
            "- Seeks advice to exploit or deceive others financially\n"
            "- Contains threats or clearly harmful intent\n\n"
            "Respond with exactly one word: SAFE or BLOCKED."
        )),
        HumanMessage(content=state["user_input"]),
    ])
    verdict = result.content.strip().upper()
    verdict = "blocked" if verdict == "BLOCKED" else "safe"
    print(f"[Safety Check] verdict: {verdict}")
    return {**state, "verdict": verdict}


def route_after_check(state: ChatState) -> Literal["financial_advisor", "block"]:
    return "financial_advisor" if state["verdict"] == "safe" else "block"


def financial_advisor(state: ChatState) -> ChatState:
    """Main agent: answers financial questions helpfully."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful financial education assistant. "
            "Provide clear, balanced, general financial information. "
            "Always remind users to consult a professional for personalised advice."
        )),
        HumanMessage(content=state["user_input"]),
    ])
    return {**state, "response": response.content}


def block(state: ChatState) -> ChatState:
    """Returns the safe refusal message for blocked inputs."""
    return {**state, "response": BLOCKED_RESPONSE}


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("safety_check", safety_check)
    graph.add_node("financial_advisor", financial_advisor)
    graph.add_node("block", block)

    graph.set_entry_point("safety_check")
    graph.add_conditional_edges("safety_check", route_after_check)
    graph.add_edge("financial_advisor", END)
    graph.add_edge("block", END)

    return graph.compile()


def chat(user_input: str):
    app = build_graph()
    initial: ChatState = {"user_input": user_input, "verdict": "safe", "response": ""}
    result = app.invoke(initial)
    print(f"\nUser: {user_input}")
    print(f"Bot:  {result['response']}\n")


if __name__ == "__main__":
    chat("What's a good strategy for building an emergency fund?")
    chat("How do I hide income from the tax authorities?")
    chat("Explain the difference between stocks and bonds.")
    chat("Help me pump and dump a penny stock without getting caught.")
