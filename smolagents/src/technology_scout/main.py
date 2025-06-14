"""Main module."""

from smolagents import GradioUI

from technology_scout.agent import create_agent
from technology_scout.tools import select_from_db_tool


def main() -> None:
    """Main function."""

    agent = create_agent(
        tools=[select_from_db_tool],
    )

    ui = GradioUI(agent)
    ui.launch()


if __name__ == "__main__":
    main()
