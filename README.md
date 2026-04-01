# BookClaw LangChain Agent

基于 LangGraph 的智能 Agent 系统，支持 MCP 工具集成和子 Agent 架构。

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户请求                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Main Agent (AgentGraph)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Context Manager                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Skills     │  │   Memory     │  │ System Prompt│               │   │
│  │  │ - calculator │  │ - user_prefs │  │   (Jinja2)   │               │   │
│  │  │ - reporter   │  │              │  │              │               │   │
│  │  │ - document   │  │              │  │              │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LangGraph (Think + Act)                          │   │
│  │                                                                      │   │
│  │    START ──▶ Think ──▶ [Tool Calls?] ──▶ Act ──▶ Think ──▶ END     │   │
│  │                 │                        │                           │   │
│  │                 │                        ▼                           │   │
│  │                 └──────────────────▶ [No Tools] ──▶ END             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   Native Tools      │   │    MCP Servers      │   │    SubagentTool     │
│  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │
│  │ get_weather   │  │   │  │  calculator   │  │   │  │ subagent()    │  │
│  │ current_time  │  │   │  │  rag          │  │   │  │ agent_type    │  │
│  │ get_skill     │  │   │  │chrome-devtools│  │   │  │ task          │  │
│  └───────────────┘  │   │  └───────────────┘  │   │  └───────┬───────┘  │
└─────────────────────┘   └─────────────────────┘   └──────────┼──────────┘
                                                               │
                          ┌────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│    RAG Subagent     │       │  WebSearch Subagent │
│  ┌───────────────┐  │       │  ┌───────────────┐  │
│  │ rag_search    │  │       │  │ new_page      │  │
│  │ rag_stats     │  │       │  │ take_snapshot │  │
│  │ rag_list_files│  │       │  │ click         │  │
│  └───────────────┘  │       │  │ fill          │  │
│                     │       │  │ ...           │  │
│  [上下文隔离]        │       │  └───────────────┘  │
│  独立 think-act     │       │                     │
└─────────────────────┘       │  [上下文隔离]        │
                              │  独立 think-act     │
                              └─────────────────────┘
```

## 目录结构

```
BookClaw_langchain/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # 主 Agent (LangGraph think+act)
│   ├── states.py             # State 定义
│   ├── context.py            # Context Manager (skills, memory)
│   ├── prompts/
│   │   └── system_prompt.md  # Jinja2 系统提示模板
│   └── tools/
│       ├── __init__.py
│       ├── base.py           # BaseTool 导出
│       ├── register.py       # ToolRegistry (统一工具管理)
│       ├── mcp_tool.py       # MCP 工具包装器
│       ├── get_weather.py    # Native 工具示例
│       ├── current_time.py   # Native 工具示例
│       ├── get_skill.py      # 技能获取工具
│       └── subagent.py       # 子 Agent 实现
├── skills/                   # 技能定义
│   ├── calculator.md
│   ├── reporter.md
│   └── document_search.md
├── memory/                   # 记忆模块
│   └── user_preferences.md
├── rag/                      # RAG 模块
│   ├── __init__.py
│   ├── processor.py          # RAG 处理器
│   ├── vector_store.py       # Milvus 向量存储
│   ├── embedding.py          # BGE-M3 Embeddings
│   ├── file_reader.py        # 文件读取
│   └── text_chunker.py       # 中文分块
├── mcp_config.yaml           # MCP 服务器配置
├── main.py                   # 入口文件
└── pyproject.toml
```

## 核心组件

### 1. AgentGraph (主 Agent)

基于 LangGraph 的 think-act 循环：

```python
from agent.graph import AgentGraph
from agent.tools import GetWeatherTool, CurrentTimeTool

agent = AgentGraph(
    mcp_config_path="mcp_config.yaml",
    native_tools=[GetWeatherTool(), CurrentTimeTool()],
)
await agent.initialize()

response = await agent.run([HumanMessage(content="今天天气怎么样？")])
```

### 2. ToolRegistry (工具注册)

统一管理 Native 工具和 MCP 工具：

```python
registry = ToolRegistry()
registry.register_native_tool(GetWeatherTool())
await registry.connect_mcp_servers("mcp_config.yaml")

tools = registry.get_tools()  # {"tool_name": tool_instance}
```

### 3. Subagent (子 Agent)

上下文隔离的子 Agent：

```python
# 主 Agent 调用子 Agent
subagent(agent_type="rag", task="搜索上海红色景点")
subagent(agent_type="websearch", task="搜索今日股价")
```

| 子 Agent | 工具 | 用途 |
|----------|------|------|
| `rag` | rag_search, rag_stats, rag_list_files | 知识库检索 |
| `websearch` | Chrome DevTools (click, fill, take_snapshot...) | 网页搜索 |

### 4. MCP 工具集成

```yaml
# mcp_config.yaml
mcp_servers:

  - name: calculator
    path: agent/tools/calculator.py
    enabled: true

  - name: chrome-devtools
    command: npx
    args: ["-y", "chrome-devtools-mcp@latest"]
    enabled: true
```


### 5. Skills

在 `skills/` 目录创建 Markdown 文件：

```markdown
# calculator

**描述**: 执行数学计算
**触发条件**: 用户需要进行数学运算

## 指令
使用 add, subtract, multiply, divide 工具完成计算...
```

### 6. Memory

在 `memory/` 目录创建 Markdown 文件：

```markdown
# user_preferences


## 快速开始

### 安装依赖

```bash
uv sync
```

### 运行

```bash
uv run python main.py
```

### 调试 MCP Server

```bash
# 使用 MCP Inspector 调试
npx -y @modelcontextprotocol/inspector uv run agent/tools/calculator.py
```

## 配置

### LLM 配置

```yaml
# mcp_config.yaml
llm:
  base_url: https://api.example.com/v1
  api_key: "your-api-key"
  model: gpt-4
  temperature: 0.3
```

