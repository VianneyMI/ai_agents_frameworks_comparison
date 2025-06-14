"""SmolAgents Technology Scout Agent."""

import litellm
from smolagents import CodeAgent, LiteLLMModel, Tool

DEFAULT_MODEL_ID = "gemini/gemini-2.5-flash-preview-04-17"


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

    if tools is None:
        tools = []

    agent = CodeAgent(
        model=model,
        tools=tools,
        add_base_tools=True,
    )
    return agent
