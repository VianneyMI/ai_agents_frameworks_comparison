from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import FunctionTool


DEFAULT_MODEL_NAME = "gpt-4o"


def get_llm(model_name: str = DEFAULT_MODEL_NAME) -> OpenAI:
    """Returns an OpenAI LLM."""

    return OpenAI(model=model_name)


def create_agent(llm: OpenAI, tools: list[FunctionTool] | None = None) -> AgentWorkflow:
    """Creates a technology scout agent."""

    if tools is None:
        tools = []

    return AgentWorkflow.from_tools_or_functions(
        tools,
        llm=llm,
    )
