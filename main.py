"""
Main entry point for BookClaw LangChain Agent.

Usage:
    python main.py
"""
import asyncio
from langchain_core.messages import HumanMessage,AIMessage
from agent.graph import AgentGraph
from agent.tools import GetWeatherTool, CurrentTimeTool


async def main():
    # 创建 agent
    agent = AgentGraph(
        mcp_config_path="mcp_config.yaml",
        native_tools=[GetWeatherTool(), CurrentTimeTool()],
    )

    # 初始化（连接 MCP 服务器，加载工具）
    await agent.initialize()
    messages = []
    while True:
        # 运行 agent
        user_input = input("\n请输入您的问题（或输入 'exit' 退出）：")
        if user_input.lower() == "exit":
            print("退出程序。")
            break
        messages.append(HumanMessage(content=user_input))
        response , messages = await agent.run(messages)
        print(f"\nAssistant: {response}")
    await agent.close()


def run():
    """Run the async main with proper cleanup for MCP stdio clients."""
    # Set custom exception handler to suppress MCP cleanup errors
    def exception_handler(loop, context):
        # Suppress known MCP stdio cleanup errors
        exc = context.get("exception")
        if exc:
            msg = str(exc)
            if "cancel scope" in msg or "CancelledError" in type(exc).__name__:
                return
        # Log other exceptions
        loop.default_exception_handler(context)

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(exception_handler)
    try:
        loop.run_until_complete(main())
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    run()