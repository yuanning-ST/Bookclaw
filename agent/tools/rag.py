"""
RAG MCP Server using FastMCP.

Usage:
    uv run -m agent.tools.rag
"""
import json
from mcp.server.fastmcp import FastMCP
from rag.processor import RAGProcessor

# Initialize RAG processor
rag = RAGProcessor(
    files_dir="./books",
    milvus_uri="./milvus_data.db",
    embedding_api_base="http://10.151.140.16:31289/bge-m3/v1/embed",
    collection_name="bookclaw_test",
)

# Create FastMCP server
mcp = FastMCP("rag")


@mcp.tool()
def rag_search(query: str, limit: int = 5) -> str:
    """
    在知识库中搜索相关内容。

    Args:
        query: 搜索查询词
        limit: 返回结果数量（默认 5）

    Returns:
        JSON 字符串，包含相关文档块和置信度分数
    """
    results = rag.search(query=query, limit=limit)

    if not results:
        return json.dumps({"results": [], "message": "未找到相关内容"}, ensure_ascii=False)

    output = {
        "query": query,
        "total": len(results),
        "results": [
            {
                "chunk_text": r.get("chunk_text", ""),
                "file_path": r.get("file_path", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": round(r.get("score", 0), 4),
            }
            for r in results
        ],
    }

    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool()
def rag_stats() -> str:
    """
    获取知识库统计信息。

    Returns:
        JSON 字符串，包含文件数和分块数
    """
    stats = rag.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


@mcp.tool()
def rag_list_files() -> str:
    """
    列出知识库中的所有文件。

    Returns:
        JSON 字符串，包含文件列表
    """
    files = rag.list_files()
    return json.dumps({"files": files}, ensure_ascii=False, indent=2)


@mcp.tool()
def rag_update(include_modified: bool = False) -> str:
    """
    增量更新知识库，检测并处理新增的图书文件。

    Args:
        include_modified: 是否也处理修改过的文件（默认只处理新增文件）

    Returns:
        JSON 字符串，包含更新结果统计
    """
    result = rag.ingest_incremental(include_modified=include_modified)

    output = {
        "success": True,
        "new_files_count": len(result["new_files_processed"]),
        "modified_files_count": len(result["modified_files_processed"]),
        "deleted_files_count": len(result["deleted_files_cleaned"]),
        "errors_count": len(result["errors"]),
        "new_files": [
            {"file_path": f["file_path"], "chunk_count": f["chunk_count"]}
            for f in result["new_files_processed"]
        ],
        "modified_files": [
            {"file_path": f["file_path"], "chunk_count": f["chunk_count"]}
            for f in result["modified_files_processed"]
        ],
        "deleted_files": result["deleted_files_cleaned"],
        "errors": result["errors"],
    }

    return json.dumps(output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")