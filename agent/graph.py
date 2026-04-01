from pathlib import Path
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from .states import State
from .tools.register import ToolRegistry
from .tools.get_skill import GetSkillTool
from .tools.subagent import SubagentTool, SubagentManager
from .context import ContextManager


# 默认 LLM 配置
DEFAULT_CONFIG = {
    "base_url": "http://10.151.140.16:32727/v1",
    "api_key": "sk-placeholder",
    "model": "SenseNova-235B-A22B-Instruct",
    "temperature": 0.3,
    "presence_penalty": 0,
    "top_p": 1.0,
}


def get_llm(config: dict, tools: list[BaseTool] = None) -> ChatOpenAI:
    """Create LLM instance with optional tool binding."""
    llm = ChatOpenAI(
        base_url=config.get("base_url", ""),
        api_key=config.get("api_key", ""),
        model=config.get("model", ""),
        temperature=config.get("temperature", 0.3),
        presence_penalty=config.get("presence_penalty", 0),
        top_p=config.get("top_p", 1.0),
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


class AgentGraph:
    """LangGraph-based agent with think + act nodes."""

    def __init__(
        self,
        mcp_config_path: str = None,
        native_tools: list[BaseTool] = None,
        llm_config: dict = None,
        context_dir: str = None,
    ):
        """
        Initialize agent.

        Args:
            mcp_config_path: Path to MCP config YAML file
            native_tools: Optional list of native tool instances
            llm_config: Optional LLM config override
            context_dir: Optional path to context directory (skills/memory)
        """
        self.config = {**DEFAULT_CONFIG, **(llm_config or {})}
        self.registry = ToolRegistry(project_root=str(Path(__file__).parent.parent))
        self.context = ContextManager(context_dir)

        # Register native tools
        if native_tools:
            self.registry.register_native_tools(native_tools)

        # Tools will be loaded when connect_mcp_servers is called
        self._tools_loaded = False
        self._mcp_config_path = mcp_config_path

        # Build the graph
        self.graph = self._build_graph()

    async def initialize(self) -> None:
        """Connect to MCP servers and load tools and context."""
        # Load context (skills, memory)
        self.context.load()
        self.system_prompt = self.context.build_system_prompt()
        print(f"Loaded context: skills={list(self.context.get_skills().keys())}, memory={list(self.context.get_memory().keys())}")
        print(f"\n=== System Prompt ===\n{self.system_prompt}\n")

        # Register GetSkillTool with context manager
        get_skill_tool = GetSkillTool()
        get_skill_tool.context_manager = self.context
        self.registry.register_native_tool(get_skill_tool)

        if self._mcp_config_path:
            await self.registry.connect_mcp_servers(self._mcp_config_path)

        # 创建 SubagentManager 并注入到 SubagentTool
        self.subagent_manager = SubagentManager(
            tool_registry=self.registry,
            llm_config=self.config,
        )
        subagent_tool = SubagentTool()
        subagent_tool.manager = self.subagent_manager

        self.registry.register_native_tool(subagent_tool)

        tools = self.registry.get_tools()
        print(f"Agent tools: {list(tools.keys())}")

        # Initialize LLM with tool binding
        self.llm = get_llm(self.config, list(tools.values()))
        self._tools_loaded = True

    async def close(self) -> None:
        """Close MCP server connections."""
        await self.registry.close()

    def _build_graph(self):
        """Build the LangGraph workflow with think + act nodes."""
        workflow = StateGraph(State)
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)

        # Entry point -> think
        workflow.add_edge(START, "think")

        # After act -> back to think (loop until no more tool calls)
        workflow.add_edge("act", "think")

        return workflow.compile()

    async def _think_node(self, state: State) -> Command:
        """
        Think node: invoke LLM, decide whether to call tools.

        If tool calls detected: return Command to go to act node.
        If no tool calls: return Command to end (final answer).
        """
        messages = state.get("messages", [])

        # Add system prompt if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        response = await self.llm.ainvoke(messages)

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", [])

        if tool_calls:
            # Store tool calls in state for act node to process
            return Command(
                goto="act",
                update={
                    "messages": [response],
                    "tool_calls": tool_calls,
                }
            )
        else:
            # No tool calls, we're done
            return Command(
                goto=END,
                update={
                    "messages": [response],
                }
            )

    async def _act_node(self, state: State) -> Command:
        """
        Act node: execute pending tool calls, store results.

        After execution, loop back to think node to process results.
        """
        tool_calls = state.get("tool_calls", [])

        if not tool_calls:
            return Command(goto="think", update={})

        tools = self.registry.get_tools()

        # Execute all pending tools
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", None)

            # Handle kwargs wrapper (some LLMs wrap args in kwargs)
            if "kwargs" in tool_args and isinstance(tool_args["kwargs"], dict):
                tool_args = tool_args["kwargs"]

            print(f"[Tool Call] {tool_name}({tool_args})")

            if tool_name in tools:
                tool_func = tools[tool_name]
                try:
                    result = await tool_func.arun(tool_args)
                    print(f"[Tool Result] {tool_name}: {result}")
                    tool_results.append(
                        ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
                    )
                except Exception as e:
                    print(f"[Tool Error] {tool_name}: {e}")
                    tool_results.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tool_id, name=tool_name)
                    )
            else:
                print(f"[Tool Error] {tool_name}: not found")
                tool_results.append(
                    ToolMessage(content=f"Tool '{tool_name}' not found", tool_call_id=tool_id, name=tool_name)
                )

        # Clear tool_calls and add results to messages
        return Command(
            goto="think",
            update={
                "messages": tool_results,
                "tool_calls": [],
            }
        )

    async def run(self, messages: list) -> str:
        """
        Main entry point: receive user input, run the think-act loop, return final response.

        Args:
            messages: List of message objects (e.g., [HumanMessage(content="hello")])

        Returns:
            Final response string from the agent
        """
        initial_state = {
            "messages": messages,
            "tool_calls": [],
        }

        final_state = await self.graph.ainvoke(initial_state)
        response = final_state.get("messages", [])[-1].content if final_state.get("messages") else None
        return response


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = AgentGraph(
            mcp_config_path="mcp_config.yaml",
            native_tools=[],  # Add native tools here
        )
        await agent.initialize()
        print(f"Tools: {list(agent.registry.get_tools().keys())}")
        await agent.close()

    asyncio.run(test())
