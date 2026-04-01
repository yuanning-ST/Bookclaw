"""
Embedding 模型 - BGE-M3 API 接口
"""
from typing import List, Optional
import requests
import aiohttp

# Embedding API 配置
EMBEDDING_URL = "http://10.151.140.16:31289/bge-m3/v1/embed"


class BGEM3Embeddings:
    """BGE-M3 Embedding 模型（HTTP API）"""

    def __init__(
        self,
        api_base: str = EMBEDDING_URL,
        model_name: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_base = api_base
        self.model_name = model_name or "bge-m3"
        self.timeout = timeout

    def embed_query(self, text: str) -> List[float]:
        """计算单个文本的 embedding"""
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量计算 embedding"""
        return self.embed_texts(texts)

    def embed_text(self, text: str) -> List[float]:
        """计算单个文本的 embedding"""
        body = {"inputs": text}
        response = requests.post(self.api_base, json=body, timeout=self.timeout)
        if response.status_code != 200:
            raise Exception(f"Failed to embed text: {response.text}")
        return response.json()[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量计算 embedding"""
        results = []
        for text in texts:
            results.append(self.embed_text(text))
        return results

    async def a_embed_text(self, text: str) -> List[float]:
        """异步计算单个文本的 embedding"""
        body = {"inputs": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_base, json=body, timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to embed text: {error_text}")
                result = await response.json()
                return result[0]

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """异步批量计算 embedding"""
        results = []
        for text in texts:
            result = await self.a_embed_text(text)
            results.append(result)
        return results