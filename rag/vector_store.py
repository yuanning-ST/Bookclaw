"""
Milvus 向量存储 - 负责 embedding 计算和向量存储
"""
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from .embedding import BGEM3Embeddings


class MilvusVectorStore:
    """Milvus 向量存储，支持文档分块的 embedding 和检索"""

    def __init__(
        self,
        collection_name: str = "bookclaw_chunks",
        milvus_uri: str = "",
        embedding_api_base: Optional[str] = None,
        dimension: int = 1024,  # BGE-M3 dimension
    ):
        """
        初始化 Milvus 向量存储

        Args:
            collection_name: Milvus 集合名称
            milvus_uri: Milvus 服务地址
            embedding_api_base: Embedding API 地址
            dimension: embedding 维度 (BGE-M3: 1024)
        """
        self.collection_name = collection_name
        self.dimension = dimension

        # 初始化 embeddings
        self.embeddings = BGEM3Embeddings(api_base=embedding_api_base) if embedding_api_base else BGEM3Embeddings()

        # 连接 Milvus
        connections.connect("default", uri=milvus_uri)

        # 初始化 collection
        self.collection = self._init_collection()

    def _init_collection(self) -> Collection:
        """初始化或打开 collection"""
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        # 创建 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, "Bookclaw Chunks Collection")
        collection = Collection(self.collection_name, schema)

        # 创建索引 (Milvus-Lite 只支持 FLAT, IVF_FLAT, AUTOINDEX)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",  # 本地模式使用 AUTOINDEX
        }
        collection.create_index("embedding", index_params)

        return collection

    def compute_embedding(self, text: str) -> List[float]:
        """计算单个文本的 embedding"""
        return self.embeddings.embed_query(text)

    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量计算 embedding"""
        return self.embeddings.embed_documents(texts)

    def add_chunks(
        self,
        file_path: str,
        chunks: List[str],
        metadata: Optional[Dict] = None,
    ) -> List[int]:
        """
        添加文档分块到向量库

        Args:
            file_path: 文件路径
            chunks: 文本分块列表
            metadata: 额外元数据

        Returns:
            List[int]: 插入的 ID 列表
        """
        if not chunks:
            return []

        # 计算 embeddings
        embeddings = self.compute_embeddings(chunks)

        # 准备数据
        entities = [
            [file_path] * len(chunks),  # file_path
            chunks,  # chunk_text
            list(range(len(chunks))),  # chunk_index
            embeddings,  # embedding
            [metadata or {}] * len(chunks),  # metadata
        ]

        # 插入数据
        result = self.collection.insert(entities)
        self.collection.flush()

        return result.primary_keys

    def search(
        self,
        query: str,
        limit: int = 5,
        filter_file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query: 查询文本
            limit: 返回结果数量
            filter_file_path: 可选的文件路径过滤

        Returns:
            List[Dict]: 搜索结果，包含 chunk_text, file_path, score 等
        """
        # 计算 query embedding
        query_embedding = self.compute_embedding(query)

        # 加载 collection
        self.collection.load()

        # 构建搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {},  # AUTOINDEX 不需要额外参数
        }

        # 构建表达式过滤
        expr = None
        if filter_file_path:
            expr = f'file_path == "{filter_file_path}"'

        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["file_path", "chunk_text", "chunk_index", "metadata"],
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "file_path": hit.entity.get("file_path"),
                    "chunk_text": hit.entity.get("chunk_text"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score,
                })

        return formatted_results

    def delete_by_file_path(self, file_path: str) -> int:
        """删除指定文件的所有分块"""
        expr = f'file_path == "{file_path}"'
        result = self.collection.delete(expr)
        self.collection.flush()
        return result.delete_count

    def list_files(self) -> List[str]:
        """列出所有已存储的文件"""
        self.collection.load()
        result = self.collection.query(expr="id >= 0", output_fields=["file_path"])
        files = set(item.get("file_path") for item in result)
        return sorted(list(files))

    def get_chunks_by_file(self, file_path: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定文件的所有分块"""
        expr = f'file_path == "{file_path}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["file_path", "chunk_text", "chunk_index", "metadata"],
            limit=limit,
        )
        return results

    def count(self) -> int:
        """返回总分块数"""
        return self.collection.num_entities

    def drop(self):
        """删除 collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)