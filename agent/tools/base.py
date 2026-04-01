"""
Base Tool Definition.

All tools (native and MCP) inherit from this base class.
"""
from langchain_core.tools import BaseTool


# Export BaseTool for convenience
__all__ = ["BaseTool"]