from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import dotenv
import asyncio
import os

dotenv.load_dotenv()

class ResearchOutput(BaseModel):
    is_research: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about research.",
    output_type=ResearchOutput,
)

cs_research_agent = Agent(
    name="CS Researcher",
    handoff_description="Specialist agent for Computer Science Research",
    instructions="You provide help with CS research. Explain your reasoning at each step and include examples",
)

bio_research_agent = Agent(
    name="Biology Researcher",
    handoff_description="Specialist agent for Biological Research",
    instructions="You assist with biological research. Explain important events and context clearly.",
)


async def research_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ResearchOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_research,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's research question",
    handoffs=[cs_research_agent, bio_research_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=research_guardrail),
    ],
)

async def main():
    result = await Runner.run(triage_agent, "What is Agentic AI?")
    print(result.final_output)

if __name__ == "__main__":
  asyncio.run(main())