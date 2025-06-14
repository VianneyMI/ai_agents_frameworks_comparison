"""Tools to query the Papers with Code API."""

from typing import Generic, TypeVar

import requests
from pydantic import BaseModel
from smolagents import tool

T = TypeVar("T", bound=BaseModel)


BASE_URL = "https://paperswithcode.com/api/v1"


class ApiResponse(BaseModel, Generic[T]):
    count: int
    next: str | None
    previous: str | None
    results: T


class Author(BaseModel):
    id: str
    full_name: str


class PaperAuthorPaper(BaseModel):
    id: str
    arxiv_id: str
    nips_id: str
    title: str
    url: str
    url_abs: str
    url_pdf: str
    abstract: str
    authors: list[Author]
    published: str
    conference: str
    conference_url_abs: str
    conference_url_pdf: str
    proceeding: str


def search_author(name: str) -> list[Author]:
    """Searches authors by name.

    Args:
        name (str): The name of the author to search for

    Returns:
        ApiResponse[list[Author]]: List of matching authors
    """
    endpoint = f"{BASE_URL}/authors"
    params = {"search": name}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        response_json = response.json()
        response = ApiResponse[list[Author]].model_validate(response_json)

        return response.results

    except requests.exceptions.RequestException as e:
        print(f"Error searching for author: {e}")
        return []


def get_author_papers(author_id: int) -> list[PaperAuthorPaper]:
    """Get papers for a specific author by their ID.

    Args:
        author_id (int): The ID of the author

    Returns:
        List[Dict]: List of papers by the author
    """
    endpoint = f"{BASE_URL}/authors/{author_id}/papers"

    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        response_json = response.json()
        response = ApiResponse[list[PaperAuthorPaper]].model_validate(response_json)

        return response.results

    except requests.exceptions.RequestException as e:
        print(f"Error getting author papers: {e}")
        return []


search_author_tool = tool(search_author)
get_author_papers_tool = tool(get_author_papers)
