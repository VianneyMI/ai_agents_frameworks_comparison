"""Main module."""

from llama_index.packs.gradio_agent_chat import GradioAgentChatPack

from technology_scout.agent import create_agent
from technology_scout.tools.query_papers_with_code import (
    search_author_tool,
    get_author_papers_tool,
)
from technology_scout.tools.select_from_db import select_from_db_tool


def main() -> None:
    """Main function."""

    agent = create_agent(
        tools=[
            search_author_tool,
            get_author_papers_tool,
            select_from_db_tool,
        ],
    )

    app = GradioAgentChatPack(agent)
    app.run()


if __name__ == "__main__":
    main()
