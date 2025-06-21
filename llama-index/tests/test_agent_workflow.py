import asyncio
import json
from technology_scout.agent_workflow import (
    create_agent_workflow,
    run_agent_workflow,
    ReasoningAgent,
)
import pytest
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class TestCreateAgent:
    def test_with_default_model(self):
        """Test that the agent is created with the default model."""

        agent = create_agent_workflow()
        assert isinstance(agent, ReasoningAgent)


class TestAgent:
    @pytest.mark.asyncio
    async def test_on_simple_task(self) -> None:
        """Test that the agent can solve a simple task."""

        agent = create_agent_workflow()
        result = await run_agent_workflow(agent, "What is the capital of France?")
        assert "paris" in str(result).lower()


async def main() -> None:
    """Allows to run the tests without using pytest."""

    test_agent = TestAgent()
    await test_agent.test_on_simple_task()


if __name__ == "__main__":
    asyncio.run(main())
