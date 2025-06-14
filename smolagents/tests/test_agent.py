import pytest
from smolagents import CodeAgent
from technology_scout.agent import create_agent


class TestCreateAgent:
    def test_with_default_model(self):
        """Test that the agent is created with the default model."""

        agent = create_agent()
        assert isinstance(agent, CodeAgent)


class TestAgent:
    def test_on_simple_task(self) -> None:
        """Test that the agent can solve a simple task."""

        agent = create_agent()
        result = agent.run("What is the capital of France?")
        assert "paris" in result.lower()
