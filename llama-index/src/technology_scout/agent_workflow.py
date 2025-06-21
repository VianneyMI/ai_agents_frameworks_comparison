from typing import Any, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

DEFAULT_MODEL_NAME = "gpt-4o"
MAX_STEPS = 15  # Maximum number of reasoning steps to prevent infinite loops - increased for complex reasoning tasks


def get_llm(model_name: str = DEFAULT_MODEL_NAME) -> OpenAI:
    """Returns an OpenAI LLM."""
    return OpenAI(model=model_name)


# Event definitions for the workflow
class PrepEvent(Event):
    """Event to prepare for the next reasoning step."""

    pass


class InputEvent(Event):
    """Event containing the formatted input for the LLM."""

    input: List[ChatMessage]


class ToolCallEvent(Event):
    """Event containing tool calls to execute."""

    tool_calls: List[ToolSelection]


class EvaluationEvent(Event):
    """Event to evaluate if we need to continue reasoning or stop."""

    reasoning_complete: bool
    final_response: str = ""


class ReasoningAgent(Workflow):
    """
    A ReAct-based reasoning agent that can loop through:
    1. Reasoning about the task
    2. Planning actions
    3. Acting using tools
    4. Evaluating results
    5. Repeating until task is complete or max steps reached
    """

    def __init__(
        self,
        *args: Any,
        llm: OpenAI | None = None,
        tools: List[FunctionTool] | None = None,
        max_steps: int = MAX_STEPS,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or get_llm()
        self.max_steps = max_steps
        # Enhanced context for better reasoning
        enhanced_context = (
            (extra_context or "")
            + """

You are an advanced reasoning agent that can work through complex problems step by step.

Key reasoning principles:
1. BREAK DOWN complex problems into smaller, manageable steps
2. TRY MULTIPLE APPROACHES when the first attempt fails
3. LEARN FROM ERRORS and adapt your strategy
4. BE PERSISTENT - don't give up after one failure

When working with APIs and tools:
- Author IDs are often formatted as lowercase-with-hyphens (e.g., "yann-lecun" for "Yann LeCun")
- If a tool call fails, try alternative parameter formats
- When searching for specific items in lists, iterate through all results
- Pay attention to error messages and use them to inform your next actions

Output formatting:
- When asked to return specific information (like a title), return ONLY that information
- Do not add extra explanatory text unless specifically requested
- Match the format requested in the task description

Remember: Each step should build toward the final goal. If one approach doesn't work, explain why and try a different approach."""
        )

        self.formatter = ReActChatFormatter.from_defaults(context=enhanced_context)
        self.output_parser = ReActOutputParser()

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        """Initialize the reasoning process with a new user message."""
        # Clear previous reasoning state
        await ctx.set("sources", [])
        await ctx.set("step_count", 0)

        # Initialize memory if needed
        memory = await ctx.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # Get user input and add to memory
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # Clear current reasoning for new task
        await ctx.set("current_reasoning", [])
        await ctx.set("memory", memory)

        return PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        """Prepare the chat history with ReAct formatting."""
        # Get chat history and current reasoning
        memory = await ctx.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])

        # Format with ReAct instructions
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def reason_and_act(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | EvaluationEvent:
        """Core reasoning step - decide whether to use tools or provide final answer."""
        chat_history = ev.input
        current_reasoning = await ctx.get("current_reasoning", default=[])
        step_count = await ctx.get("step_count", default=0)

        # Check if we've exceeded max steps
        if step_count >= self.max_steps:
            return EvaluationEvent(
                reasoning_complete=True,
                final_response="I've reached the maximum number of reasoning steps. Let me provide you with the best answer I can based on my current understanding.",
            )

        # Increment step counter
        await ctx.set("step_count", step_count + 1)

        try:
            # Get LLM response
            response = await self.llm.achat(chat_history)

            # Parse the reasoning step
            reasoning_step = self.output_parser.parse(response.message.content)
            current_reasoning.append(reasoning_step)
            await ctx.set("current_reasoning", current_reasoning)

            # Check if reasoning is complete
            if reasoning_step.is_done:
                memory = await ctx.get("memory")
                memory.put(
                    ChatMessage(role="assistant", content=reasoning_step.response)
                )
                await ctx.set("memory", memory)

                return EvaluationEvent(
                    reasoning_complete=True, final_response=reasoning_step.response
                )

            # If it's an action step, prepare tool calls
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id=f"call_{step_count}",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )

        except Exception as e:
            # Handle parsing errors by adding observation and continuing
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}. Let me try a different approach."
                )
            )
            await ctx.set("current_reasoning", current_reasoning)

        # If no tool calls or final response, continue reasoning
        return PrepEvent()

    @step
    async def execute_tools(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        """Execute tool calls and observe results."""
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = await ctx.get("current_reasoning", default=[])
        sources = await ctx.get("sources", default=[])

        # Execute tools safely
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist. Available tools: {list(tools_by_name.keys())}"
                    )
                )
                continue

            try:
                # Execute the tool
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                current_reasoning.append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                error_msg = str(e)
                # Provide specific guidance for common API errors
                if "404" in error_msg and "authors" in error_msg:
                    guidance = " This suggests the author ID format is incorrect. Try converting the author name to lowercase with hyphens (e.g., 'Yann LeCun' becomes 'yann-lecun')."
                elif "404" in error_msg:
                    guidance = " This suggests the resource wasn't found. Check if the ID format is correct."
                else:
                    guidance = " Let me try a different approach."

                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}.{guidance}"
                    )
                )

        # Save updated state
        await ctx.set("sources", sources)
        await ctx.set("current_reasoning", current_reasoning)

        # Continue to next reasoning step
        return PrepEvent()

    @step
    async def finalize_response(self, ctx: Context, ev: EvaluationEvent) -> StopEvent:
        """Finalize and return the response."""
        if ev.reasoning_complete:
            sources = await ctx.get("sources", default=[])
            current_reasoning = await ctx.get("current_reasoning", default=[])

            return StopEvent(
                result={
                    "response": ev.final_response,
                    "sources": sources,
                    "reasoning": current_reasoning,
                }
            )

        # This shouldn't happen, but just in case
        return StopEvent(
            result={"response": "Unable to complete the reasoning process."}
        )


def create_agent_workflow(
    model: OpenAI = get_llm(),
    tools: List[FunctionTool] | None = None,
    max_steps: int = MAX_STEPS,
) -> ReasoningAgent:
    """Creates a reasoning agent workflow."""
    return ReasoningAgent(
        llm=model,
        tools=tools or [],
        max_steps=max_steps,
        timeout=120,  # 2 minute timeout for safety
        verbose=True,
    )


async def run_agent_workflow(agent: ReasoningAgent, task: str) -> str:
    """Run the reasoning agent workflow."""
    handler = agent.run(input=task)

    # Stream events for debugging
    async for event in handler.stream_events():
        # We could add custom event handling here if needed
        pass

    # Get the final result
    result = await handler

    # Extract the response
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    elif hasattr(result, "response"):
        return result.response
    else:
        return str(result)
