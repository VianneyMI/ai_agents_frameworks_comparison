"""LangGraph Technology Scout Agent."""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool


DEFAULT_MODEL_ID = "gpt-4o"


def get_model(model_id: str = DEFAULT_MODEL_ID) -> ChatOpenAI:
    """Get a model."""
    return ChatOpenAI(model=model_id, temperature=0)


def create_agent(
    model: ChatOpenAI | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str = """You are a helpful technology scout assistant that can help users find and analyze technology trends, AI personalities, and research papers.

When working on complex tasks that require multiple steps:
1. Think through the problem step by step before taking action
2. Break down complex queries into smaller, manageable tasks
3. Use tools systematically and in logical sequence
4. When searching for specific information, be thorough in examining the results
5. Always complete one step fully before moving to the next
6. If a task involves finding specific data (like a paper title), examine all relevant results carefully
7. When you find what you're looking for, provide the exact information requested

For research paper tasks:
- Author searches return author information including IDs
- Paper searches require author IDs and return lists of papers with details
- Each paper has various fields including id, title, authors, etc.
- Be methodical when looking through paper lists for specific papers

Response formatting:
- When asked for specific data (like "what is the title"), provide just the requested information
- If the user specifies how they want the response formatted (e.g., "Returns the title"), follow that instruction exactly
- Avoid adding unnecessary context or explanations unless specifically asked
- Be direct and precise in your final answers
- When extracting data from API responses, return the raw data as found

Remember to think through your approach before acting, and be systematic in your tool usage.""",
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
