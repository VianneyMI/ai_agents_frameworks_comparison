# LangGraph Technology Scout Agent

This is the LangGraph implementation of the Technology Scout Agent, replicating the same structure and functionality as the smolagents implementation but using LangGraph's `create_react_agent` from `langgraph.prebuilt`.

## Structure

The project follows the same structure as the original smolagents implementation:

```
langgraph/
├── src/
│   └── technology_scout/
│       ├── __init__.py
│       ├── agent.py              # Main agent creation using create_react_agent
│       ├── main.py               # Entry point with CLI interface
│       └── tools/
│           ├── __init__.py
│           ├── select_from_db.py    # Database query tool
│           └── query_papers_with_code.py  # Papers with Code API tool
├── tests/
│   ├── __init__.py
│   ├── test_agent.py           # Tests for the agent
│   └── test_tools/
│       ├── __init__.py
│       ├── test_select_from_db.py
│       └── test_query_papers_with_code.py
├── main.py                     # Root entry point
├── pyproject.toml             # Dependencies and project config
└── README.md
```

## Key Components

### Agent (`agent.py`)
- Uses `create_react_agent` from `langgraph.prebuilt`
- Configured with OpenAI's GPT-4o-mini model
- Supports custom tools and system prompts

### Tools
- **select_from_db**: Queries the AI personalities database using SQL
- **query_papers_with_code**: Searches for authors and papers on Papers with Code API

### Main Interface (`main.py`)
- Simple command-line interface for interacting with the agent
- Handles message formatting for LangGraph agents

## Dependencies

- `langgraph>=0.4.8` - Main framework
- `langchain-openai>=0.1.0` - OpenAI integration
- `langchain-core>=0.1.0` - Core LangChain components
- `pandas>=2.0.0` - Data manipulation for database queries
- `requests>=2.25.0` - HTTP requests for API calls
- `pydantic>=2.0.0` - Data validation
- `pytest>=7.0.0` - Testing framework

## Usage

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. Run the agent:
   ```bash
   python main.py
   ```

4. Run tests:
   ```bash
   python -m pytest tests/ -v
   ```

## Implementation Notes

- **Observer functionality**: Omitted for now as requested, but can be added later
- **LangGraph ReAct Agent**: Uses the high-level `create_react_agent` abstraction instead of low-level building blocks
- **Tool Format**: Tools are adapted from smolagents format to LangChain format using the `@tool` decorator
- **Test Structure**: Tests are designed to fail initially for functions not yet fully implemented, following TDD principles

## Differences from Smolagents Implementation

1. **Framework**: Uses LangGraph instead of smolagents
2. **Tool Definition**: Uses LangChain's `@tool` decorator instead of smolagents' `tool` decorator
3. **Agent Creation**: Uses `create_react_agent` instead of `CodeAgent`
4. **Message Format**: Uses LangGraph's message format (`{"messages": [...]}`）
5. **Interface**: CLI instead of Gradio UI (can be enhanced later)

## Next Steps

1. Implement proper error handling
2. Add observer functionality (langfuse integration)
3. Add Gradio UI integration
4. Enhance test coverage
5. Add configuration management
6. Add logging and debugging features
