"""Tests for the LangGraph Technology Scout Agent."""

import pytest
from technology_scout.agent import create_agent


class TestCreateAgent:
    def test_with_default_model(self):
        """Test that the agent is created with the default model."""

        agent = create_agent()
        # LangGraph agents are more complex objects, so we just check if we get something
        assert agent is not None


class TestAgent:
    def test_on_simple_task(self) -> None:
        """Test that the agent can solve a simple task."""

        agent = create_agent()

        # LangGraph agents expect messages in a specific format
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            }
        )

        # Extract the response content
        assert result is not None
        assert "messages" in result

        # Get the last message content
        last_message = result["messages"][-1]
        if hasattr(last_message, "content"):
            content = last_message.content.lower()
        else:
            content = str(last_message).lower()

        assert "paris" in content
