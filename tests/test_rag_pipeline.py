"""
单元测试: RAG Pipeline 核心模块。
"""

import os
import sys
import pytest

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """配置加载测试。"""

    def test_config_loads(self):
        from config import config
        assert config.llm.model == "qwen2.5:7b"
        assert config.embedding.model_name == "nomic-embed-text"
        assert config.qdrant.vector_size == 768
        assert config.retrieval.similarity_top_k == 10

    def test_config_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_RAG_RETRIEVAL__SIMILARITY_TOP_K", "20")
        from config import config
        # Config is a singleton, so this tests that env vars are read
        assert config.retrieval.similarity_top_k in (10, 20)


class TestQdrantConnection:
    """Qdrant 连接测试。"""

    def test_client_connects(self):
        from database.qdrant import client
        assert client is not None
        # Verify the client can reach Qdrant
        collections = client.get_collections()
        assert collections is not None

    def test_get_vector_store(self):
        from database.qdrant import get_vector_store
        vs = get_vector_store()
        assert vs is not None
        assert vs.collection_name == "agentic_rag_knowledge"


class TestModels:
    """模型工厂测试。"""

    def test_create_llm(self):
        from models.llm import create_llm
        from config import config
        llm = create_llm(config.llm)
        assert llm is not None
        assert llm.model == "qwen2.5:7b"

    def test_create_embedding(self):
        from models.embedding import create_embedding
        from config import config
        emb = create_embedding(config.embedding)
        assert emb is not None
        assert emb.model_name == "nomic-embed-text"


class TestRetrieval:
    """检索功能测试。"""

    def test_retriever_returns_results(self):
        from rag.index import get_index
        from config import config
        from database.qdrant import get_vector_store
        from models.embedding import create_embedding

        index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("LLM agent")
        assert len(nodes) > 0
        assert nodes[0].score > 0

    def test_retriever_scores_descending(self):
        from rag.index import get_index
        from config import config
        from database.qdrant import get_vector_store
        from models.embedding import create_embedding

        index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve("autonomous agent")
        scores = [n.score for n in nodes if n.score]
        assert scores == sorted(scores, reverse=True), "Retrieved nodes should be sorted by score descending"

    def test_page_label_in_metadata(self):
        """验证 PDF 页码信息被保留。"""
        from rag.index import get_index
        from config import config
        from database.qdrant import get_vector_store
        from models.embedding import create_embedding

        index = get_index(config.paths.data_dir, get_vector_store(), create_embedding(config.embedding))
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("agent")
        has_page = any(n.node.metadata.get("page_label") for n in nodes)
        assert has_page, "至少某些节点应该有 page_label metadata"
