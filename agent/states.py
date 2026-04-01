from langgraph.graph import MessagesState
from pydantic import BaseModel
from typing_extensions import Annotated
from typing import Any


class Usage(BaseModel):
    """Usage of the agent system."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    node: str = ""


def reducer(a: list, b: Usage | None) -> list:
    if b is not None:
        return a + [b]
    return a


class State(MessagesState):
    """State for the agent system, extends MessagesState with additional fields."""

    # Runtime Variables
    query: str = ""
    research_topic: str = ""
    observations: list[str] = []
    rag_reference_docs: list[dict] = []
    tool_calls: list[dict] = []
    pending_tool_calls: list[dict] = []  # Tool calls waiting to be executed
    tool_results: dict[str, Any] = {}
    usage: Annotated[list[Usage], reducer] = []
    cot: str = ""
    skill_name: str = ""
    user_memory: str = ""
    agent_memory: str = ""
    last_node: str = ""
