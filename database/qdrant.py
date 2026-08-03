# 1. 原生客户端 —用于手动管理
import os
import atexit

# 修复 Windows 系统代理干扰 httpx 连接 localhost 导致 502 的问题
# httpx 的 trust_env=True（默认）会读取 Windows 系统代理设置
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

from qdrant_client import QdrantClient
from config import config

collection_name = config.qdrant.collection_name

# 懒初始化：首次调用 get_client() 时才创建并连接 Qdrant，
# 避免模块 import 即连库（单元测试、仅加载配置时不依赖 Qdrant 可用）。
_client = None


def get_client() -> QdrantClient:
    """线程安全的懒加载单例：返回唯一的 Qdrant 客户端。"""
    global _client
    if _client is None:
        _client = QdrantClient(url=config.qdrant.url)
        # 进程退出时关闭连接，避免 ResourceWarning
        atexit.register(lambda: _client.close() if _client is not None and hasattr(_client, "close") else None)
    return _client


# 兼容旧用法 `from database.qdrant import client`（首次访问才真正连接）
def __getattr__(name):
    if name == "client":
        return get_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 2. LlamaIndex 包装 —给 VectorStoreIndex 用
from llama_index.vector_stores.qdrant import QdrantVectorStore

def get_vector_store():
    vector_store = QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name,
    )
    return vector_store
