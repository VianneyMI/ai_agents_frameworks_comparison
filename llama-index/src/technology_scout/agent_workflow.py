from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

DEFAULT_MODEL_NAME = "gpt-4o"


def get_llm(model_name: str = DEFAULT_MODEL_NAME) -> OpenAI:
    """Returns an OpenAI LLM."""

    return OpenAI(model=model_name)


def _create_planner(model: OpenAI = get_llm()) -> FunctionAgent:
    """Creates a planner agent."""
    return FunctionAgent(
        name="Planner",
        description="Break the user goal into an ordered list of tool calls",
        system_prompt=(
            "You create a JSON plan: [{name: tool_name, args:{...}}, …]. "
            "Only emit the JSON – no dialogue."
        ),
        llm=model,
        tools=[],
        can_handoff_to=["Executor"],
    )


def _create_executor(
    model: OpenAI = get_llm(), tools: list[FunctionTool] | None = None
) -> FunctionAgent:
    """Creates an executor agent."""
    return FunctionAgent(
        name="Executor",
        description="Execute each plan item sequentially, call tools, store results.",
        system_prompt=(
            "Receive `plan` (a list) + `state.results` (dict). "
            "Pick the next unfinished step, call its tool, append the result. "
            "Hand back to yourself until every step done, then hand-off to Reviewer."
        ),
        llm=model,
        tools=tools,
        can_handoff_to=["Planner", "Reviewer"],
    )


def _create_reviewer(model: OpenAI = get_llm()) -> FunctionAgent:
    """Creates a reviewer agent."""
    return FunctionAgent(
        name="Reviewer",
        description="Combine all results into the final answer for the user.",
        system_prompt=(
            "Read state.results and craft a concise, cited summary. "
            "If something is missing, hand-off to Planner again."
        ),
        llm=model,
        tools=[],
        can_handoff_to=["Planner"],
    )


def create_agent_workflow(
    model: OpenAI = get_llm(), tools: list[FunctionTool] | None = None
) -> AgentWorkflow:
    """Creates a technology scout agent workflow."""

    planner = _create_planner(model)
    executor = _create_executor(model, tools)
    reviewer = _create_reviewer(model)

    multi_agent = AgentWorkflow(
        agents=[planner, executor, reviewer],
        root_agent=planner.name,
        initial_state={"plan": None, "results": {}},
    )

    return multi_agent


async def run_agent_workflow(agent: AgentWorkflow, task: str) -> str:
    handler = agent.run(user_msg=task)

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'=' * 50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

    # Await the final result after streaming all events
    result = await handler

    # Extract the final response content
    if hasattr(result, "response") and hasattr(result.response, "content"):
        return result.response.content
    elif isinstance(result, str):
        return result
    else:
        return str(result)
