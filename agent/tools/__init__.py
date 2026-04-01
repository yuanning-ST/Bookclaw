"""
Tools for the agent.

This module provides:
- BaseTool: Base class for all tools
- GetWeatherTool: Example native tool
- GetSkillTool: Tool to retrieve skill information
- CurrentTimeTool: Tool to get current time
- SubagentTool: Tool to call specialized subagents (rag, websearch)
- SubagentManager: Manager for subagent execution
- SubagentGraph: Subagent implementation with think-act loop
- SubagentConfig: Subagent configuration
- MCPTool: MCP server tool wrapper
- ToolRegistry: Unified tool registry for native and MCP tools
"""
from .base import BaseTool
from .get_weather import GetWeatherTool
from .get_skill import GetSkillTool
from .current_time import CurrentTimeTool
from .subagent import (
    SubagentTool,
    SubagentManager,
    SubagentGraph,
    SubagentConfig,
    SUBAGENT_CONFIGS,
)
from .mcp_tool import MCPTool
from .register import ToolRegistry, create_tool_registry

__all__ = [
    "BaseTool",
    "GetWeatherTool",
    "GetSkillTool",
    "CurrentTimeTool",
    "SubagentTool",
    "SubagentManager",
    "SubagentGraph",
    "SubagentConfig",
    "SUBAGENT_CONFIGS",
    "MCPTool",
    "ToolRegistry",
    "create_tool_registry",
]