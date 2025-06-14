import sqlite3
from pathlib import Path

import pandas as pd
from smolagents import tool

influences_table_name = "influencers"
path_to_database = Path(__file__).parents[4] / "data" / "ai_watch.db"


def select_from_db(query: str) -> pd.DataFrame:
    """Queries the AI personalities database and returns the data as a pandas dataframe.

    Database Description:

    Table:

    - {influences_table_name}:

        - name: str
            - The name of the influencer.
        - rank: int
            - The rank of the influencer in the database.
        - bio: str
            - A short bio of the influencer.
        - twitter_username: str
            - The twitter username of the influencer.
        - nb_twitter_followers: str
            - The number of twitter followers of the influencer.
        - gender: str
            - The gender of the influencer.
        - links: list[str]
            - A list of links to the influencer's website.

    Some of the fields above are optional.


    Args:
        query (str): The raw SQL query to execute.

    Returns:
        pd.DataFrame: The data as a pandas dataframe.

    """

    assert path_to_database.exists(), f"Database file not found at {path_to_database}"

    con = sqlite3.connect(path_to_database)
    df = pd.read_sql_query(query, con)
    return df


select_from_db.__doc__ = select_from_db.__doc__.replace(
    "{influences_table_name}", influences_table_name
)
print(select_from_db.__doc__)


select_from_db_tool = tool(select_from_db)
