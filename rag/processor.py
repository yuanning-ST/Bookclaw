"""
RAG 处理器 - 整合文档读取、分块、存储和检索
"""
import os
from typing import List, Dict, Any, Optional
from .file_reader import FileReader
from .text_chunker import ChineseTextChunker
from .vector_store import MilvusVectorStore
from .incremental import IncrementalUpdater


class RAGProcessor:
    """RAG 处理器，整合文档读取、分块、存储和检索"""

    def __init__(
        self,
        files_dir: str = "./books",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        milvus_uri: str = "./milvus_data.db",  # 默认使用本地文件模式
        collection_name: str = "bookclaw_chunks",
        embedding_api_base: Optional[str] = None,
        hash_file: str = "./rag_data/file_hashes.json",
    ):
        """
        初始化 RAG 处理器

        Args:
            files_dir: 文档目录
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            milvus_uri: Milvus 地址 (本地文件模式: "./milvus_data.db", 服务模式: "http://localhost:19530")
            collection_name: 集合名称
            embedding_api_base: Embedding API 地址
            hash_file: 哈希记录文件路径
        """
        self.files_dir = files_dir
        self.file_reader = FileReader(files_dir)
        self.chunker = ChineseTextChunker(chunk_size=chunk_size, overlap=chunk_overlap)

        self.vector_store = MilvusVectorStore(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            embedding_api_base=embedding_api_base,
        )

        self.incremental_updater = IncrementalUpdater(
            files_dir=files_dir,
            hash_file=hash_file,
        )

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个文件并存储到向量库

        Args:
            file_path: 文件路径（相对路径）

        Returns:
            Dict: 处理结果
        """
        # 读取文件
        results = self.file_reader.read_files(recursive=False)

        # 找到目标文件
        content = None
        for path, file_content in results:
            if path == file_path:
                content = file_content
                break

        if content is None:
            return {"success": False, "error": f"文件不存在：{file_path}"}

        # 分块
        chunks = self.chunker.chunk_text_to_strings(content)

        # 存储到向量库
        metadata = {
            "file_path": file_path,
            "chunk_count": len(chunks),
            "content_length": len(content),
        }
        ids = self.vector_store.add_chunks(file_path, chunks, metadata)

        return {
            "success": True,
            "file_path": file_path,
            "chunk_count": len(chunks),
            "content_length": len(content),
            "inserted_ids": ids,
        }

    def ingest_all(self) -> List[Dict[str, Any]]:
        """
        处理所有文件并存储到向量库

        Returns:
            List[Dict]: 每个文件的处理结果
        """
        results = self.file_reader.read_files(recursive=True)

        processed_files = set()
        ingest_results = []

        for file_path, content in results:
            if file_path in processed_files:
                continue

            # 分块
            chunks = self.chunker.chunk_text_to_strings(content)

            # 先删除旧数据（如果有）
            self.vector_store.delete_by_file_path(file_path)

            # 存储新数据
            metadata = {
                "file_path": file_path,
                "chunk_count": len(chunks),
                "content_length": len(content),
            }
            ids = self.vector_store.add_chunks(file_path, chunks, metadata)

            ingest_results.append({
                "success": True,
                "file_path": file_path,
                "chunk_count": len(chunks),
                "content_length": len(content),
                "inserted_ids": list(ids) if ids else [],
            })
            processed_files.add(file_path)

        return ingest_results

    def search(self, query: str, limit: int = 5, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索相关分块

        Args:
            query: 查询文本
            limit: 返回结果数量
            file_path: 可选的文件路径过滤

        Returns:
            List[Dict]: 搜索结果
        """
        return self.vector_store.search(query=query, limit=limit, filter_file_path=file_path)

    def list_files(self) -> List[str]:
        """列出所有已存储的文件"""
        return self.vector_store.list_files()

    def get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """获取指定文件的所有分块"""
        return self.vector_store.get_chunks_by_file(file_path)

    def delete_file(self, file_path: str) -> int:
        """删除指定文件的所有分块"""
        return self.vector_store.delete_by_file_path(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        files = self.list_files()
        return {
            "total_files": len(files),
            "total_chunks": self.vector_store.count(),
            "files": files,
        }

    def detect_new_files(self) -> Dict[str, List[str]]:
        """
        检测文件变化（新增、修改、未变化、删除）

        Returns:
            Dict: {"new": 新增文件, "modified": 修改文件, "unchanged": 未变化文件, "deleted": 删除文件}
        """
        return self.incremental_updater.detect_all_changes()

    def ingest_incremental(self, include_modified: bool = False) -> Dict[str, Any]:
        """
        增量更新：只处理新增的文件

        Args:
            include_modified: 是否也处理修改过的文件（默认只处理新增文件）

        Returns:
            Dict: 更新结果
        """
        changes = self.detect_new_files()
        new_files = changes["new"]
        modified_files = changes["modified"]
        deleted_files = changes["deleted"]

        # 要处理的文件列表
        files_to_process = new_files
        if include_modified:
            files_to_process = new_files + modified_files

        results = {
            "new_files_processed": [],
            "modified_files_processed": [],
            "deleted_files_cleaned": [],
            "errors": [],
        }

        # 处理新增/修改的文件
        for file_path in files_to_process:
            try:
                # 读取文件内容
                full_path = os.path.join(self.files_dir, file_path)
                content = self.file_reader.read_file_by_path(full_path)

                if not content or content.startswith("[无法读取"):
                    results["errors"].append({
                        "file_path": file_path,
                        "error": "无法读取文件内容",
                    })
                    continue

                # 分块
                chunks = self.chunker.chunk_text_to_strings(content)

                # 删除旧数据（如果有）
                self.vector_store.delete_by_file_path(file_path)

                # 存储新数据
                metadata = {
                    "file_path": file_path,
                    "chunk_count": len(chunks),
                    "content_length": len(content),
                }
                ids = self.vector_store.add_chunks(file_path, chunks, metadata)

                # 标记文件已处理（更新哈希记录）
                self.incremental_updater.mark_file_processed(file_path, full_path)

                if file_path in new_files:
                    results["new_files_processed"].append({
                        "file_path": file_path,
                        "chunk_count": len(chunks),
                        "inserted_ids": list(ids) if ids else [],
                    })
                else:
                    results["modified_files_processed"].append({
                        "file_path": file_path,
                        "chunk_count": len(chunks),
                        "inserted_ids": list(ids) if ids else [],
                    })

            except Exception as e:
                results["errors"].append({
                    "file_path": file_path,
                    "error": str(e),
                })

        # 清理已删除文件的哈希记录
        for file_path in deleted_files:
            self.incremental_updater.remove_deleted_records([file_path])
            # 也删除向量库中的数据
            self.vector_store.delete_by_file_path(file_path)
            results["deleted_files_cleaned"].append(file_path)

        return results

    def get_incremental_stats(self) -> Dict[str, Any]:
        """获取增量更新统计信息"""
        changes = self.detect_new_files()
        return {
            "vector_store": self.get_stats(),
            "file_changes": {
                "new_count": len(changes["new"]),
                "modified_count": len(changes["modified"]),
                "unchanged_count": len(changes["unchanged"]),
                "deleted_count": len(changes["deleted"]),
                "new_files": changes["new"],
                "modified_files": changes["modified"],
            },
            "hash_records": self.incremental_updater.get_stats(),
        }