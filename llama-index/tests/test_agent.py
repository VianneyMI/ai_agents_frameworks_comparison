import pytest
from llama_index.core.agent.react import ReActAgent
from technology_scout.agent import create_agent


class TestCreateAgent:
    def test_with_default_model(self):
        """Test that the agent is created with the default model."""

        agent = create_agent()
        assert isinstance(agent, ReActAgent)


class TestAgent:
    @pytest.mark.asyncio
    async def test_on_simple_task(self) -> None:
        """Test that the agent can solve a simple task."""

        agent = create_agent()
        result = await agent.aquery("What is the capital of France?")
        assert "paris" in result.response.lower()
