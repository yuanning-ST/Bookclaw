"""
RAG FastAPI 服务 - 提供向量检索 API

Usage:
    uv run python -m rag.api_server
    或
    uv run python rag/api_server.py
"""
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .processor import RAGProcessor


# 请求模型
class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    file_path: Optional[str] = None


class IngestRequest(BaseModel):
    file_path: str


# 响应模型
class SearchResult(BaseModel):
    file_path: str
    chunk_text: str
    chunk_index: int
    score: float
    metadata: dict


# 全局 RAG 处理器
rag_processor: Optional[RAGProcessor] = None


def create_app(
    files_dir: str = "./books",
    milvus_uri: str = "./milvus_data.db",
    embedding_api_base: str = "http://10.151.140.16:31289/bge-m3/v1/embed",
    collection_name: str = "bookclaw_test",
) -> FastAPI:
    """创建 FastAPI 应用"""
    global rag_processor

    app = FastAPI(
        title="RAG API",
        description="向量检索 API 服务",
        version="1.0.0",
    )

    # 初始化 RAG 处理器
    rag_processor = RAGProcessor(
        files_dir=files_dir,
        milvus_uri=milvus_uri,
        embedding_api_base=embedding_api_base,
        collection_name=collection_name,
    )

    @app.get("/")
    async def root():
        """健康检查"""
        return {"status": "ok", "service": "RAG API"}

    @app.get("/stats")
    async def get_stats():
        """获取统计信息"""
        return rag_processor.get_stats()

    @app.get("/files")
    async def list_files():
        """列出已存储的文件"""
        return {"files": rag_processor.list_files()}

    @app.post("/search", response_model=list[SearchResult])
    async def search(request: SearchRequest):
        """
        向量检索

        输入查询文本，返回相关文档块和置信度分数
        """
        try:
            results = rag_processor.search(
                query=request.query,
                limit=request.limit,
                file_path=request.file_path,
            )
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search/json")
    async def search_json(request: SearchRequest):
        """
        向量检索 (返回 JSON 字符串)

        输入查询文本，返回 JSON 字符串格式的结果
        """
        try:
            results = rag_processor.search(
                query=request.query,
                limit=request.limit,
                file_path=request.file_path,
            )
            return JSONResponse(content=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest")
    async def ingest_file(request: IngestRequest):
        """导入文件到向量库"""
        try:
            result = rag_processor.ingest_file(request.file_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/files/{file_path:path}")
    async def delete_file(file_path: str):
        """删除文件"""
        try:
            count = rag_processor.delete_file(file_path)
            return {"deleted_chunks": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# 默认应用
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    files_dir: str = "./books",
    milvus_uri: str = "./milvus_data.db",
    embedding_api_base: str = "http://10.151.140.16:31289/bge-m3/v1/embed",
):
    """运行服务器"""
    global app
    app = create_app(
        files_dir=files_dir,
        milvus_uri=milvus_uri,
        embedding_api_base=embedding_api_base,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8001, help="Port")
    parser.add_argument("--files-dir", default="./books", help="Files directory")
    parser.add_argument("--milvus-uri", default="./milvus_data.db", help="Milvus URI")
    parser.add_argument("--embedding-api", default="http://10.151.140.16:31289/bge-m3/v1/embed", help="Embedding API URL")

    args = parser.parse_args()

    print(f"Starting RAG API Server on {args.host}:{args.port}")
    run_server(
        host=args.host,
        port=args.port,
        files_dir=args.files_dir,
        milvus_uri=args.milvus_uri,
        embedding_api_base=args.embedding_api,
    )