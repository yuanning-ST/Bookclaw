"""
Calculator MCP Server using FastMCP.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """两个数相加"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """两个数相减"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """两个数相除"""
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
