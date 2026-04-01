"""
MCP Tool Implementation.

MCPTool inherits from BaseTool and uses session to call MCP server tools.
"""
import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from mcp import ClientSession


class MCPTool(BaseTool):
    """
    A LangChain tool that wraps an MCP server tool.

    Inherits from BaseTool and uses session member to call MCP tools in _arun.
    """

    session: ClientSession = Field(default=None)
    tool_name: str = Field(default="")
    tool_description: str = Field(default="")
    mcp_input_schema: dict = Field(default_factory=dict)

    def __init__(
        self,
        session: ClientSession,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        **kwargs
    ):
        """
        Initialize MCPTool.

        Args:
            session: MCP ClientSession instance
            tool_name: Name of the MCP tool
            tool_description: Tool description from MCP server
            input_schema: Tool input schema from MCP server
        """
        super().__init__(
            name=tool_name,
            description=tool_description,
            session=session,
            tool_name=tool_name,
            tool_description=tool_description,
            mcp_input_schema=input_schema,
            **kwargs
        )

    def _run(self, **kwargs) -> str:
        """Run the tool synchronously (wraps async version)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
        except RuntimeError:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """
        Run the tool asynchronously via MCP session.

        Uses the session member to call the MCP server tool.
        """
        result = await self.session.call_tool(self.tool_name, kwargs)

        if result.content and len(result.content) > 0:
            return result.content[0].text
        return str(result)