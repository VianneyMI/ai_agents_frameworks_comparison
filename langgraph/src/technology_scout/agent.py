"""LangGraph Technology Scout Agent."""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

# DEFAULT_MODEL_ID = "gpt-4o"
DEFAULT_MODEL_ID = "gpt-4o-mini"


def get_model(model_id: str = DEFAULT_MODEL_ID) -> ChatOpenAI:
    """Get a model."""
    return ChatOpenAI(model=model_id, temperature=0)


def create_agent(
    model: ChatOpenAI | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str = "You are a helpful technology scout assistant that can help users find and analyze technology trends, AI personalities, and research papers.",
) -> object:
    """Create a technology scout agent using LangGraph's create_react_agent.

    Args:
        model: The language model to use. If None, uses the default model.
        tools: List of tools for the agent to use. If None, uses empty list.
        system_prompt: System prompt for the agent.

    Returns:
        The compiled LangGraph agent.
    """
    if model is None:
        model = get_model()

    if tools is None:
        tools = []

    # Create the agent using LangGraph's prebuilt create_react_agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
    )

    return agent
