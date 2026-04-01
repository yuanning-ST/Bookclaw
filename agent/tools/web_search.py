"""
Web Search MCP Server using FastMCP.

Uses Zhipu AI's web-search-pro API for internet search.
"""
import httpx
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务器
mcp = FastMCP("Web Search")


@mcp.tool()
async def web_search(query: str) -> str:
    """
    搜索互联网内容

    Args:
        query: 要搜索的内容

    Returns:
        搜索结果的总结
    """
    api_key = "YOUR_ZHIPU_API_KEY"  # 替换成你自己的 API Key

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://open.bigmodel.cn/api/paas/v4/tools",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "tool": "web-search-pro",
                "messages": [{"role": "user", "content": query}],
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for choice in data.get("choices", []):
            for msg in choice.get("message", {}).get("tool_calls", []):
                for result in msg.get("search_result", []):
                    results.append(result["content"])

        return "\n\n".join(results) if results else "未找到相关搜索结果"


if __name__ == "__main__":
    mcp.run(transport="stdio")
