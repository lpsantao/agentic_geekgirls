import os
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()


try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


# --- Define a Tool ---
@tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers to phrases
    like 'capital of France' or 'weather in London'.
    """
    print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result

tools = [search_information]


# --- Create a Tool-Calling Agent (LangChain 1.x API) ---
if llm:
    # create_agent returns a compiled LangGraph state graph; no AgentExecutor needed.
    agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant.")

    async def run_agent_with_tool(query: str):
        """Invokes the agent with a query and prints the final response."""
        print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
        try:
            response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            print("\n---Final Agent Response ---")
            print(response["messages"][-1].content)
        except Exception as e:
            print(f"\n An error occurred during agent execution: {e}")

    async def main():
        """Runs all agent queries concurrently."""
        tasks = [
            run_agent_with_tool("What is the capital of France?"),
            run_agent_with_tool("What's the weather like in London?"),
            run_agent_with_tool("Tell me something about dogs."),
        ]
        await asyncio.gather(*tasks)

    if __name__ == "__main__":
        asyncio.run(main())

else:
    print("\nSkipping agent execution due to LLM initialization failure.")
