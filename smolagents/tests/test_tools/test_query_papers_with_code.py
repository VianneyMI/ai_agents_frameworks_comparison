from technology_scout.agent import create_agent
from technology_scout.tools.query_papers_with_code import (
    search_author,
    get_author_papers,
    search_author_tool,
    get_author_papers_tool,
)
from technology_scout.tools.query_papers_with_code import Author, PaperAuthorPaper


class TestSearchAuthor:
    def test_on_known_author(self) -> None:
        """Tests that the tool returns the correct author data for a known author."""
        result = search_author("Yann LeCun")
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], Author)
        assert "lecun" in result[0].full_name.lower()


class TestGetAuthorPapers:
    def test_on_known_author_id(self) -> None:
        """Tests that the tool returns papers for a known author ID."""
        # First get Yann LeCun's ID
        authors = search_author("Yann LeCun")
        assert len(authors) > 0
        author_id = authors[0].id

        # Then get their papers
        result = get_author_papers(author_id)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], PaperAuthorPaper)
        assert "lecun" in result[0].authors[0].full_name.lower()


class TestSearchAuthorToolUsageByAgent:
    def test_on_simple_task(self) -> None:
        """Tests that the tool is used correctly by the agent."""
        agent = create_agent(
            tools=[search_author_tool],
        )

        result = agent.run("Who is Yann LeCun?")
        assert "lecun" in result.lower(), f"Result: {result}"


class TestGetAuthorPapersToolUsageByAgent:
    def test_on_simple_task(self) -> None:
        """Tests that the tool is used correctly by the agent."""
        agent = create_agent(
            tools=[get_author_papers_tool],
        )

        result = agent.run("What papers has Yann LeCun written?")
        assert "paper" in result.lower(), f"Result: {result}"


def main() -> None:
    """Main function."""
    test_search_author = TestSearchAuthor()
    test_search_author.test_on_known_author()

    test_get_author_papers = TestGetAuthorPapers()
    test_get_author_papers.test_on_known_author_id()

    test_search_author_tool_usage = TestSearchAuthorToolUsageByAgent()
    test_search_author_tool_usage.test_on_simple_task()

    test_get_author_papers_tool_usage = TestGetAuthorPapersToolUsageByAgent()
    test_get_author_papers_tool_usage.test_on_simple_task()


if __name__ == "__main__":
    main()
