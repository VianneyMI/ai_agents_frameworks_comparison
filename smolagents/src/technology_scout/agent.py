"""SmolAgents Technology Scout Agent."""

import litellm
from smolagents import CodeAgent, LiteLLMModel, Tool

from technology_scout.observer import setup_langfuse_tracer


setup_langfuse_tracer()

# DEFAULT_MODEL_ID = "gemini/gemini-2.5-flash-preview-04-17"
DEFAULT_MODEL_ID = "openai/gpt-4o"


litellm._turn_on_debug()


def get_model(model_id: str = DEFAULT_MODEL_ID) -> LiteLLMModel:
    """Get a model."""

    return LiteLLMModel(
        model_id=model_id,
    )


def create_agent(
    model: LiteLLMModel = get_model(),
    tools: list[Tool] | None = None,
) -> CodeAgent:
    """Create a technology scout agent."""

    setup_langfuse_tracer()

    if tools is None:
        tools = []

    agent = CodeAgent(
        model=model,
        tools=tools,
        add_base_tools=True,
    )
    return agent
