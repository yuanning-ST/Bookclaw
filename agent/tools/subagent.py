"""
Subagent - 子 agent 实现。

结构与主 AgentGraph 一致，但：
- 不需要 skill 和 memory
- 工具通过 config 从主 agent 筛选传入
- 不需要连接 MCP

用法：
    config = SubagentConfig(
        agent_type="rag",
        tool_names=["rag_search", "rag_stats", "rag_list_files"],
        system_prompt="你是知识库检索专家...",
    )
    subagent = SubagentGraph(config, llm_config, tools)
    result = await subagent.run([HumanMessage(content="搜索...")])
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from ..states import State  # 从 agent.states 导入

if TYPE_CHECKING:
    from .register import ToolRegistry


# ANSI 颜色代码
class Colors:
    """终端颜色"""
    RAG = "\033[36m"      # 青色
    WEBSEARCH = "\033[35m"  # 紫色
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def get_color(cls, agent_type: str) -> str:
        return {
            "rag": cls.RAG,
            "websearch": cls.WEBSEARCH,
        }.get(agent_type, "")


@dataclass
class SubagentConfig:
    """子 agent 配置。"""
    agent_type: str  # rag, websearch
    tool_names: list[str]  # 允许使用的工具名称
    system_prompt: str  # 系统提示
    max_iterations: int = 10  # 最大迭代次数


# 预定义的子 agent 配置
SUBAGENT_CONFIGS = {
    "rag": SubagentConfig(
        agent_type="rag",
        tool_names=["rag_search", "rag_stats", "rag_list_files"],
        system_prompt="""# RAG Agent
你是知识库检索专家。使用检索工具在知识库中搜索相关内容。并返回简洁的检索结果摘要。
最佳实践：
1. 在查询知识库前，你要先通过 rag_list_files 工具列一下内部知识库的文档，看看是否有相关的文档可以查阅来帮助解决问题
2. 如果你认为有相关文档，那么针对你要解决的问题：
   - 基于问题来思考一个查询语句（查询改写）
   - 如果你认为一个查询语句无法覆盖，那需要拆解为多个查询语句依次查询并汇总。
3. 可以用 rag_search 发起一次查询""",
    ),
    "websearch": SubagentConfig(
        agent_type="websearch",
        tool_names=[
            "click", "close_page", "drag", "emulate", "evaluate_script", "fill",
            "fill_form", "get_console_message", "get_network_request", "handle_dialog",
            "hover", "lighthouse_audit", "list_console_messages", "list_network_requests",
            "list_pages", "navigate_page", "new_page", "performance_analyze_insight",
            "performance_start_trace", "performance_stop_trace", "press_key", "resize_page",
            "select_page", "take_memory_snapshot", "take_screenshot", "take_snapshot",
            "type_text", "upload_file", "wait_for",
        ],
        system_prompt="""# Web Search Agent

你是网页搜索专家。
使用 Chrome DevTools 工具浏览网页、搜索信息。

首先，针对你要解决的问题：
   - 基于问题来思考一个查询语句（查询改写）
   - 如果你认为一个查询语句无法覆盖，那需要拆解为多个查询语句依次查询并汇总。

最佳实践：

步骤 1: 打开新页面并导航到搜索引擎

工具: new_page
参数:
  url: ""
  background: false  

说明: 创建一个新的浏览器标签页，并导航到百度首页。

步骤 2: 获取页面快照，定位搜索框

工具: take_snapshot
参数:
  verbose: false  # 使用简洁模式即可
返回示例:

text
- textbox "搜索" [uid=1_12]
- button "百度一下" [uid=1_13]
- link "新闻" [uid=1_5]
...
说明: 通过快照获取页面上所有可交互元素及其 uid。从结果中，我们能识别出：

搜索框的 uid 为 1_12
搜索按钮的 uid 为 1_13
步骤 3: 在搜索框中输入关键词

工具: fill
参数:
  uid: "1_12"  # 从快照中获取的搜索框uid
  value: "人工智能最新进展"
  includeSnapshot: false  # 不需要立即获取新快照
说明: 将搜索关键词填入搜索框。

步骤 4: 点击搜索按钮提交搜索

工具: click
参数:
  uid: "1_13"  # 从快照中获取的"百度一下"按钮uid
  includeSnapshot: true  # 点击后获取新快照，确认页面已跳转
说明:

点击搜索按钮触发搜索
设置 includeSnapshot: true，这样点击后会自动返回搜索结果页的快照
步骤 5: 等待搜索结果加载（可选）

如果搜索结果加载较慢，可以使用等待工具：

工具: wait_for
参数:
  text: 
    - "人工智能"
    - "搜索结果"
  timeout: 5000  # 最多等待5秒
说明: 等待页面出现预期的文本，确保结果已加载完成。

步骤 6: 获取搜索结果页面快照

工具: take_snapshot
参数:
  filePath: "./search_results.txt"  # 可选，保存到文件
  verbose: false
说明:

获取搜索结果页的结构快照
可以从中提取搜索结果链接、标题、摘要等信息
步骤 7: 提取搜索结果信息（可选）

使用 JavaScript 提取想要的搜索结果：

工具: evaluate_script
参数:
  function: |
    () => {
      const results = [];
      const items = document.querySelectorAll('.result, .c-container');
      items.forEach((item, index) => {
        const title = item.querySelector('h3')?.innerText || '';
        const link = item.querySelector('a')?.href || '';
        const summary = item.querySelector('.c-abstract, .content')?.innerText || '';
        if (title) {
          results.push({ index: index + 1, title, link, summary });
        }
      });
      return results.slice(0, 5);  // 返回前5条结果
    }
返回示例:

json
[
  {
    "index": 1,
    "title": "人工智能最新进展：2024年重大突破盘点",
    "link": "https://example.com/ai-news-1",
    "summary": "2024年人工智能领域迎来多项重大突破..."
  },
  {
    "index": 2,
    "title": "OpenAI发布新一代AI模型",
    "link": "https://example.com/openai-new-model",
    "summary": "OpenAI今日宣布推出..."
  }
]
步骤 8: 截屏保存搜索结果（可选）

工具: take_screenshot
参数:
  filePath: "./search_results.png"
  fullPage: true  # 截取整个搜索结果页面
  format: "png"
说明: 保存整个搜索结果页面的截图作为记录。

""",
    ),
}


class SubagentGraph:
    """
    子 agent - 结构与主 AgentGraph 一致。

    不需要 skill/memory，工具从主 agent 筛选传入。
    """

    def __init__(
        self,
        config: SubagentConfig,
        llm_config: dict,
        tools: dict[str, BaseTool],
    ):
        """
        Args:
            config: 子 agent 配置
            llm_config: LLM 配置
            tools: 主 agent 的工具字典（会筛选出允许的工具）
        """
        self.config = config
        self.llm_config = llm_config
        self.color = Colors.get_color(config.agent_type)

        # 筛选工具
        self.tools = {
            name: tool for name, tool in tools.items()
            if name in config.tool_names
        }

        # 创建 LLM 并绑定工具
        self.llm = ChatOpenAI(
            base_url=llm_config.get("base_url", ""),
            api_key=llm_config.get("api_key", ""),
            model=llm_config.get("model", ""),
            temperature=llm_config.get("temperature", 0.3),
        )
        if self.tools:
            self.llm = self.llm.bind_tools(list(self.tools.values()))

        # 构建 graph
        self.graph = self._build_graph()

    def _log(self, msg: str) -> None:
        """打印带颜色的日志。"""
        print(f"{self.color}{Colors.BOLD}[{self.config.agent_type}]{Colors.RESET} {msg}")

    def _log_tool(self, tool_name: str, args: dict, result: str = None) -> None:
        """打印工具调用日志。"""
        args_str = str(args)[:50] + ("..." if len(str(args)) > 50 else "")
        if result:
            result_str = result[:80] + ("..." if len(result) > 80 else "")
            print(f"{self.color}  └─ {tool_name}({args_str}){Colors.RESET}")
            print(f"{Colors.DIM}     → {result_str}{Colors.RESET}")
        else:
            print(f"{self.color}  └─ {tool_name}({args_str}){Colors.RESET}")

    def _build_graph(self):
        """构建 LangGraph 工作流（think + act 节点）。"""
        workflow = StateGraph(State)
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_edge(START, "think")
        workflow.add_edge("act", "think")
        return workflow.compile()

    async def _think_node(self, state: State) -> Command:
        """Think 节点：调用 LLM，决定是否调用工具。"""
        messages = state.get("messages", [])

        # 添加系统提示
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.config.system_prompt)] + messages

        response = await self.llm.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", [])

        if tool_calls:
            return Command(
                goto="act",
                update={"messages": [response], "tool_calls": tool_calls}
            )
        else:
            return Command(
                goto=END,
                update={"messages": [response]}
            )

    async def _act_node(self, state: State) -> Command:
        """Act 节点：执行工具调用。"""
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return Command(goto="think", update={})

        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            if "kwargs" in tool_args and isinstance(tool_args["kwargs"], dict):
                tool_args = tool_args["kwargs"]

            # 打印工具调用
            self._log_tool(tool_name, tool_args)

            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].arun(tool_args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Tool '{tool_name}' not found"

            # 打印工具结果
            self._log_tool(tool_name, tool_args, str(result))

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
            )

        return Command(
            goto="think",
            update={"messages": tool_results, "tool_calls": []}
        )

    async def run(self, messages: list) -> str:
        """
        执行子 agent。

        Args:
            messages: 消息列表

        Returns:
            最终响应
        """
        # 打印开始日志
        task = messages[-1].content if messages else ""
        task_preview = task[:50] + ("..." if len(task) > 50 else "")
        self._log(f"开始执行: {task_preview}")

        initial_state = {"messages": messages, "tool_calls": []}
        final_state = await self.graph.ainvoke(initial_state, config={"recursion_limit": 50})

        if final_state.get("messages"):
            result = final_state["messages"][-1].content
            result_preview = result[:100] + ("..." if len(result) > 100 else "")
            self._log(f"完成: {result_preview}")
            return result
        return "任务完成，但没有生成响应。"


class SubagentManager:
    """管理子 agent 的创建和执行。"""

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        llm_config: dict,
    ):
        """
        Args:
            tool_registry: 主 agent 的工具注册表
            llm_config: LLM 配置
        """
        self.tool_registry = tool_registry
        self.llm_config = llm_config

    async def run(self, agent_type: str, task: str) -> str:
        """
        执行子 agent。

        Args:
            agent_type: 子 agent 类型 ("rag" 或 "websearch")
            task: 任务描述

        Returns:
            执行结果
        """
        # 1. 获取配置
        config = SUBAGENT_CONFIGS.get(agent_type)
        if not config:
            return f"Error: 未知的子 agent 类型 '{agent_type}'，可选: {list(SUBAGENT_CONFIGS.keys())}"

        # 2. 从主 agent 获取工具
        tools = self.tool_registry.get_tools()

        # 3. 创建子 agent
        subagent = SubagentGraph(
            config=config,
            llm_config=self.llm_config,
            tools=tools,
        )

        # 4. 执行
        from langchain_core.messages import HumanMessage
        result = await subagent.run([HumanMessage(content=task)])

        return result


class SubagentTool(BaseTool):
    """调用子 agent 执行特定任务。"""

    name: str = "subagent"
    description: str = (
        "调用专门的子 agent 执行任务。"
        "可用类型："
        "rag: 知识库检索专家"
        "websearch: 网页搜索专家"
    )

    manager: object = None

    def _run(self, agent_type: str, task: str) -> str:
        raise NotImplementedError("使用 arun 进行异步执行")

    async def _arun(self, agent_type: str, task: str) -> str:
        """    
        Args:
            agent_type: 子 agent 类型 ("rag" 或 "websearch")
            task: 任务描述
        Returns:
            执行结果

        """
        if not self.manager:
            return "Error: SubagentManager 未配置"
        return await self.manager.run(agent_type=agent_type, task=task)