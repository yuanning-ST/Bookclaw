"""
Tool Registry - Unified tool management for native and MCP tools.

Usage:
    registry = ToolRegistry()
    registry.register_native_tool(GetWeatherTool())
    await registry.connect_mcp_servers("mcp_config.yaml")

    # Get tool dictionary
    tools = registry.get_tools()  # {"tool_name": tool_instance}

    # Call a tool
    result = await tools["get_weather"].arun({"location": "Beijing"})
"""
import os
import re
import yaml
from pathlib import Path
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .mcp_tool import MCPTool


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str = "uv"  # Command to run (uv, npx, python, etc.)
    args: list[str] = field(default_factory=list)  # Command arguments
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    # For internal Python MCP servers (simplified config)
    path: str = None  # Python module path (e.g., agent/tools/calculator.py)


class ToolRegistry:
    """
    Unified tool registry for native and MCP tools.

    Tool registration order:
    1. Register native tools (local implementations like GetWeatherTool)
    2. Connect to MCP servers and create MCPTool instances

    Returns a dictionary mapping tool names to tool instances.

    Usage:
        registry = ToolRegistry(project_root="/path/to/project")

        # Register native tools first
        registry.register_native_tool(GetWeatherTool())

        # Connect to MCP servers and create MCP tools
        await registry.connect_mcp_servers("mcp_config.yaml")

        # Get all tools as dictionary
        tools = registry.get_tools()

        # Access tool by name and call it
        result = await tools["get_weather"].arun({"location": "Beijing"})
        result = await tools["add"].arun({"a": 1, "b": 2})
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._tools: dict[str, BaseTool] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        self._server_configs: list[ServerConfig] = []

    def register_native_tool(self, tool: BaseTool) -> None:
        """
        Register a native (local implementation) tool.

        Args:
            tool: A BaseTool instance (e.g., GetWeatherTool)
        """
        self._tools[tool.name] = tool

    def register_native_tools(self, tools: list[BaseTool]) -> None:
        """
        Register multiple native tools.

        Args:
            tools: List of BaseTool instances
        """
        for tool in tools:
            self.register_native_tool(tool)

    def load_config(self, config_path: str) -> None:
        """
        Load MCP server configurations from YAML file.

        Supports two formats:
        1. Internal Python MCP servers (simplified):
           - name: calculator
             path: agent/tools/calculator.py
             args: {"--port": "8080"}  # optional

        2. External MCP servers (full control):
           - name: chrome-devtools
             command: npx
             args: ["-y", "chrome-devtools-mcp@latest"]
             env: {"API_KEY": "xxx"}  # optional

        Args:
            config_path: Path to YAML config file
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        for server in config.get("mcp_servers", []):
            if server.get("enabled", True):
                # Check if it's external (has command) or internal (has path)
                if "command" in server:
                    # External MCP server
                    self._server_configs.append(ServerConfig(
                        name=server["name"],
                        command=server["command"],
                        args=server.get("args", []),
                        env=server.get("env", {}),
                        enabled=True,
                    ))
                elif "path" in server:
                    # Internal Python MCP server
                    # Convert dict args to list format
                    args_list = ["run", "-m", server["path"].replace("/", ".").replace(".py", "")]
                    if server.get("args"):
                        for arg_name, arg_value in server["args"].items():
                            args_list.extend([arg_name, arg_value])
                    self._server_configs.append(ServerConfig(
                        name=server["name"],
                        command="uv",
                        args=args_list,
                        env=server.get("env", {}),
                        enabled=True,
                        path=server["path"],
                    ))
                else:
                    print(f"Warning: Server '{server['name']}' missing 'command' or 'path', skipping")

    def _substitute_env(self, value: str) -> str:
        """Substitute ${VAR} patterns with environment variables."""
        pattern = r"\$\{(\w+)\}"

        def replacer(match):
            env_var = match.group(1)
            return os.environ.get(env_var, "")

        return re.sub(pattern, replacer, value)

    async def connect_mcp_servers(self, config_path: str = None) -> None:
        """
        Connect to MCP servers and create MCPTool instances.

        If config_path provided, loads config first.
        Then connects to all configured servers and registers their tools.

        Args:
            config_path: Optional path to YAML config file
        """
        if config_path:
            self.load_config(config_path)

        for server_config in self._server_configs:
            await self._connect_to_server(server_config)

    async def _connect_to_server(self, config: ServerConfig) -> None:
        """Connect to a single MCP server and register its tools."""
        import asyncio

        # Substitute environment variables
        env = {}
        for key, value in config.env.items():
            env[key] = self._substitute_env(value)
        # Merge with current environment for npx/other commands
        if env:
            env = {**os.environ, **env}

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env if env else None,
        )

        exit_stack = AsyncExitStack()

        try:
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport

            # Get the process from stdio_client for cleanup
            # The process is stored in the context
            session = await exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()

            # Store session and exit stack
            self._sessions[config.name] = session
            self._exit_stacks[config.name] = exit_stack

            # Load tools from this server and create MCPTool instances
            await self._load_tools_from_server(session)

        except Exception as e:
            print(f"Failed to connect to server '{config.name}': {e}")
            await exit_stack.aclose()
            raise

    async def _load_tools_from_server(self, session: ClientSession) -> None:
        """Load tools from MCP server and create MCPTool instances."""
        response = await session.list_tools()

        for tool in response.tools:
            mcp_tool = MCPTool(
                session=session,
                tool_name=tool.name,
                tool_description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
            )
            # Register tool with server context (for debugging/repr)
            self._tools[tool.name] = mcp_tool

    def get_tools(self) -> dict[str, BaseTool]:
        """
        Get all registered tools as a dictionary.

        Returns:
            Dictionary mapping tool name to tool instance
        """
        return self._tools.copy()

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """
        Get a specific tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def get_tool_names(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    async def close(self) -> None:
        """Close all MCP server connections.

        Note: Due to known issues with anyio task groups and asyncio subprocess cleanup,
        we don't explicitly close the exit stacks. The subprocess will be cleaned up
        by the OS when the parent process exits.
        See: https://github.com/modelcontextprotocol/python-sdk/issues/81
        """
        # Just clear references - let OS clean up subprocesses
        self._sessions.clear()
        self._exit_stacks.clear()

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())}, servers={list(self._sessions.keys())})"


async def create_tool_registry(
    config_path: str,
    native_tools: list[BaseTool] = None,
    project_root: str = None
) -> ToolRegistry:
    """
    Create and initialize a ToolRegistry with native and MCP tools.

    Args:
        config_path: Path to YAML config file
        native_tools: Optional list of native tool instances to register first
        project_root: Optional project root path

    Returns:
        Initialized ToolRegistry instance

    Usage:
        from agent.tools.get_weather import GetWeatherTool

        registry = await create_tool_registry(
            "mcp_config.yaml",
            native_tools=[GetWeatherTool()],
        )

        tools = registry.get_tools()
        result = await tools["get_weather"].arun({"location": "Beijing"})
    """
    registry = ToolRegistry(project_root=project_root)

    # Register native tools first
    if native_tools:
        registry.register_native_tools(native_tools)

    # Connect to MCP servers and create MCP tools
    await registry.connect_mcp_servers(config_path)

    return registry