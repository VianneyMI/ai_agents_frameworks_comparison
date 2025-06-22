"""Tests for the select_from_db tool."""

import pandas as pd
from technology_scout.agent import create_agent
from technology_scout.tools.select_from_db import (
    select_from_db,
    select_from_db_tool,
    influences_table_name,
)


class TestSelectFromDb:
    def test_on_simple_query(self) -> None:
        """Tests that the tool returns the correct data with a simple query."""

        demo_query = f"""
        SELECT name, twitter_username, nb_twitter_followers
        FROM {influences_table_name}
        ORDER BY nb_twitter_followers DESC
        LIMIT 10
        """

        result = select_from_db(demo_query)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 3), f"Result: {result}"


class TestSelectFromDbToolUsageByAgent:
    def test_on_simple_task(self) -> None:
        """Tests that the tool is used correctly by the agent."""

        agent = create_agent(
            tools=[select_from_db_tool],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Who is the most prominent AI personality?",
                    }
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

        assert "lex fridman" in content, f"Result: {result}"


def main() -> None:
    """Main function."""

    test_select_from_db = TestSelectFromDb()
    test_select_from_db.test_on_simple_query()

    test_select_from_db_tool_usage_by_agent = TestSelectFromDbToolUsageByAgent()
    test_select_from_db_tool_usage_by_agent.test_on_simple_task()


if __name__ == "__main__":
    main()
