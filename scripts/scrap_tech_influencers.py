# from typing import Literal
import asyncio
import json
from pydantic import BaseModel
from browser_use import Agent, Controller
from langchain_google_genai import ChatGoogleGenerativeAI


class TechInfluencer(BaseModel):
    name: str
    bio: str
    twitter_username: str | None = None
    nb_twitter_followers: int
    type: str | None = None  # Literal["Mega", "Macro", "Micro"]
    gender: str | None = None  # Literal["Male", "Female"]
    links: list[str] = []
    rank: int | None = None


class TechInfluencers(BaseModel):
    tech_influencers: list[TechInfluencer] = []


async def main(task: str) -> None:
    controller = Controller(output_model=TechInfluencers)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        # api_key=SecretStr(api_key)
    )

    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        max_actions_per_step=10,
    )

    history = await agent.run(max_steps=20)

    history.save_to_file("history.json")

    result = history.final_result()
    if result:
        parsed: TechInfluencers = TechInfluencers.model_validate_json(result)

        for influencer in parsed.tech_influencers[:5]:
            print("\n--------------------------------")
            print(f"Name: {influencer.name}")
            print(f"Bio: {influencer.bio}")
            print(f"Twitter Username: {influencer.twitter_username}")
            print(f"Number of Twitter Followers: {influencer.nb_twitter_followers}")
            print(f"Type: {influencer.type}")

        with open("tech_influencers2.json", "w") as f:
            json.dump(json.loads(result), f, indent=4)

    else:
        print("No result")


task = """
Go to https://x.feedspot.com/artificial_intelligence_twitter_influencers/
and extract the top 100 of AI influencers displayed on this webpage.
Follow the provided format to extract the data.

For each influencer, extract the following data:
- Name
- Bio
- Twitter Username
- Number of Twitter Followers
- Type (Mega, Macro, Micro, Nano, etc.)
- Gender (Male, Female)
- Links to their website, blog, youtube, etc.

If you are not sure, leave the field empty. Do not make up any data.


""".strip()

if __name__ == "__main__":
    asyncio.run(main(task))
