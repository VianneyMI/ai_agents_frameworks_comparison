"""Main module for LangGraph Technology Scout Agent."""

from technology_scout.agent import create_agent
from technology_scout.tools.select_from_db import select_from_db_tool


def main() -> None:
    """Main function."""

    agent = create_agent(
        tools=[select_from_db_tool],
    )

    # For LangGraph, we can't use GradioUI directly like in smolagents
    # Instead, we'll provide a simple command line interface
    print("Technology Scout Agent (LangGraph)")
    print("Type your questions below. Type 'quit' to exit.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        try:
            # LangGraph agents expect messages in a specific format
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )

            # Extract the last message content
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"\nAgent: {last_message.content}")
                else:
                    print(f"\nAgent: {last_message}")
            else:
                print(f"\nAgent: {response}")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
