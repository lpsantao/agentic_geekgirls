from google.adk.agents import LlmAgent, SequentialAgent, Agent
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
import uuid
import asyncio

MODEL = "openai/gpt-4o-mini"
from dotenv import load_dotenv
load_dotenv()

class SpecsJson(BaseModel):
    cpu: str = Field(description="CPU details")
    memory: str = Field(description="RAM size")
    storage: str = Field(description="Storage details")

# Step 1: Extract agent
extract_agent = LlmAgent(
    name="extract_specs_agent",
    model=MODEL,
    instruction="Extract the technical specifications from the user's message. Return only the extracted specs as concise bullet points:}",
    output_key="specifications",
)

# Step 2: Transform agent
def build_transform_instruction(ctx):
    specs = ctx.state.get("specifications", "")
    return f"Transform to JSON with cpu, memory, storage keys:\n\n{specs}"

transform_agent = LlmAgent(
    name="transform_specs_agent",
    model=MODEL,
    instruction=build_transform_instruction,
    output_schema=SpecsJson,
    output_key="result",
)

# Step 3: Build pipeline
pipeline = SequentialAgent(
    name="specs_pipeline",
    sub_agents=[extract_agent, transform_agent],
)

async def run_pipeline(runner: Runner, request: str):
    print(f"\n--- Running Pipeline with request: '{request}' ---")
    final_result = ""
    
    try:
        user_id = "medtiles"
        session_id = str(uuid.uuid4())
        
        await runner.session_service.create_session(
            app_name=runner.app_name, 
            user_id=user_id, 
            session_id=session_id
        )

        # The last agent in the pipeline is the one whose response we want
        last_agent_name = pipeline.sub_agents[-1].name
        # print(f"Listening for final response from agent: '{last_agent_name}'...\n")
        # print(f"Request: {request}")

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            # print(f"Event from [{event.author}]: {event}")
            
            ## Only capture the final response from the last agent
            if (
                event.author == last_agent_name
                and hasattr(event, 'is_final_response')
                and event.is_final_response()
                and event.content
            ):
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                # break

        print(f"\nAnswer: {final_result}")
        return final_result
        
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"

# ---------------------------------------------------------
# Step 4. Main Execution
# ---------------------------------------------------------
def main():
    print("--- Custom LLM Sequential Pipeline Example ---")
    
    ## Initialize the runner with your SequentialAgent instead of the coordinator
    # runner = InMemoryRunner(agent=writing_pipeline)
    
    ## With embedded InMemorySessionService
    session_service = InMemorySessionService()
    runner = Runner(
        agent=pipeline, 
        session_service=session_service, 
        app_name="myFirstADKApp"
    )
    ## Run the user input through the custom pipeline)
    result = asyncio.run(run_pipeline(runner, "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."))
    # result = asyncio.run(run_pipeline("Topic: AI Agents"))
    # print(result)

if __name__ == "__main__":
    main()