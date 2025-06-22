"""Tests for the query_papers_with_code tools."""

import pytest
from technology_scout.tools.query_papers_with_code import (
    search_author,
    get_author_papers,
    search_author_tool,
    get_author_papers_tool,
    Author,
    PaperAuthorPaper,
)


class TestSearchAuthor:
    def test_search_existing_author(self) -> None:
        """Test searching for an existing author."""

        # Search for a well-known researcher
        results = search_author("Yann LeCun")

        assert isinstance(results, list)
        if results:  # If we get results, check they're Author objects
            assert all(isinstance(author, Author) for author in results)
            # Check that at least one result contains the search term
            names = [author.full_name.lower() for author in results]
            assert any("lecun" in name for name in names)

    def test_search_nonexistent_author(self) -> None:
        """Test searching for a non-existent author."""

        results = search_author("NonExistentAuthorXYZ123")

        assert isinstance(results, list)
        # Should be empty or very few results
        assert len(results) <= 1


class TestGetAuthorPapers:
    def test_get_papers_for_valid_author_id(self) -> None:
        """Test getting papers for a valid author ID."""

        # First search for an author to get a valid ID
        authors = search_author("Geoffrey Hinton")

        if authors:
            author_id = int(authors[0].id)
            papers = get_author_papers(author_id)

            assert isinstance(papers, list)
            if papers:  # If we get results, check they're PaperAuthorPaper objects
                assert all(isinstance(paper, PaperAuthorPaper) for paper in papers)
                # Check that papers have required fields
                for paper in papers:
                    assert hasattr(paper, "title")
                    assert hasattr(paper, "abstract")
                    assert hasattr(paper, "authors")

    def test_get_papers_for_invalid_author_id(self) -> None:
        """Test getting papers for an invalid author ID."""

        papers = get_author_papers(999999)  # Assuming this ID doesn't exist

        assert isinstance(papers, list)
        # Should be empty for invalid ID
        assert len(papers) == 0


class TestToolsIntegration:
    def test_search_author_tool_format(self) -> None:
        """Test that the search_author_tool is properly formatted as a LangChain tool."""

        # Check that the tool has the required attributes
        assert hasattr(search_author_tool, "name")
        assert hasattr(search_author_tool, "description")
        assert callable(search_author_tool)

    def test_get_author_papers_tool_format(self) -> None:
        """Test that the get_author_papers_tool is properly formatted as a LangChain tool."""

        # Check that the tool has the required attributes
        assert hasattr(get_author_papers_tool, "name")
        assert hasattr(get_author_papers_tool, "description")
        assert callable(get_author_papers_tool)


def main() -> None:
    """Main function for running tests."""

    print("Running search author tests...")
    test_search = TestSearchAuthor()
    test_search.test_search_existing_author()
    test_search.test_search_nonexistent_author()

    print("Running get author papers tests...")
    test_papers = TestGetAuthorPapers()
    test_papers.test_get_papers_for_valid_author_id()
    test_papers.test_get_papers_for_invalid_author_id()

    print("Running tools integration tests...")
    test_integration = TestToolsIntegration()
    test_integration.test_search_author_tool_format()
    test_integration.test_get_author_papers_tool_format()

    print("All tests completed!")


if __name__ == "__main__":
    main()
