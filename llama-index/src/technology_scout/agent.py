from llama_index.core.agent.react import ReActAgent
from llama_index.core.memory import Memory
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import FunctionTool


DEFAULT_MODEL_NAME = "gpt-4o"


def get_llm(model_name: str = DEFAULT_MODEL_NAME) -> OpenAI:
    """Returns an OpenAI LLM."""

    return OpenAI(model=model_name)


def get_memory() -> Memory:
    """Returns a ChatMemoryBuffer."""
    return Memory(token_limit=1_000_000)


def create_agent(
    llm: OpenAI = get_llm(), tools: list[FunctionTool] | None = None
) -> ReActAgent:
    """Creates a technology scout agent."""

    if tools is None:
        tools = []

    return ReActAgent(
        tools,
        llm=llm,
        memory=get_memory(),
    )
