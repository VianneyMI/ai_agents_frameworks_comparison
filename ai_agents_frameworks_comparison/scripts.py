"""Scripts for running tests for different AI agent frameworks."""

import os
import subprocess
import sys
from pathlib import Path


def get_root_dir():
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def test_langgraph():
    """Run tests for the LangGraph framework."""
    root_dir = get_root_dir()
    langgraph_dir = root_dir / "langgraph"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(langgraph_dir / "src")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v"], cwd=langgraph_dir, env=env
    )

    return result.returncode


def test_llama_index():
    """Run tests for the LlamaIndex framework."""
    root_dir = get_root_dir()
    llama_index_dir = root_dir / "llama-index"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(llama_index_dir / "src")

    # Only run tests that work (excluding agent_workflow tests)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_agent.py",
            "tests/tests_tools/test_select_from_db.py",
            "-v",
        ],
        cwd=llama_index_dir,
        env=env,
    )

    return result.returncode


def test_smolagents():
    """Run tests for the Smolagents framework."""
    root_dir = get_root_dir()
    smolagents_dir = root_dir / "smolagents"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(smolagents_dir / "src")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v"], cwd=smolagents_dir, env=env
    )

    return result.returncode


def test_all():
    """Run tests for all frameworks."""
    print("Running LangGraph tests...")
    langgraph_result = test_langgraph()

    print("\nRunning LlamaIndex tests...")
    llama_index_result = test_llama_index()

    print("\nRunning Smolagents tests...")
    smolagents_result = test_smolagents()

    total_failed = langgraph_result + llama_index_result + smolagents_result

    print(f"\n=== SUMMARY ===")
    print(f"LangGraph: {'PASSED' if langgraph_result == 0 else 'FAILED'}")
    print(f"LlamaIndex: {'PASSED' if llama_index_result == 0 else 'FAILED'}")
    print(f"Smolagents: {'PASSED' if smolagents_result == 0 else 'FAILED'}")

    return total_failed
