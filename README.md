# AI Agent Frameworks Comparison


In this repository, we compare several AI agent frameworks by implmenting the same agent, using the various frameworks.

## Context

This comparison and the design of this agent has been inspired by the [Hugging Face AI Agent Course](https://huggingface.co/learn/ai-agents-course/en/unit0/introduction) where in Unit 2, there is a brief introduction to each of the frameworks.
And then in Unit 3, there is a simple RAGagent that is implemented using the framework of the student's choice.
When working on this project, I first used `llama-index` and got poor results, then switched to `smolagents` and got better results.
Therefore, I decided to deep dive on this topic and compare the frameworks with a more concrete example.


## Agent Overview

The agent is a relatively simple AI RAG agent, that performs AI Technology Watch.
It has access to a database of AI Personalities, it can perform web search and communicate with the Papers with Code API.

## Repository structure

The repository is organized as follows:

- `smolagents/`: Contains the implementation of the agent using `smolagents`.
- `llama-index/`: Contains the implementation of the agent using `llama-index`.
- `langgraph/`: Contains the implementation of the agent using `langgraph`.
- `scripts/`: Contains the scripts used to compare the frameworks.
- `data/`: Contains local data accesible to the agent.


### Agent Implementation Structure

Each agent is implemented in a separate folder, and the implementation is organized as follows:

- `src/`: Contains the implementation of the agent.
    - `technology_scout/`: Contains the implementation of the agent.
        - `tools/`: Contains the tools used by the agent.
        - `agent.py`: Contains the implementation of the agent.
- `tests/`: Contains the tests for the agent.
- `README.md`: Contains the README for the agent.


