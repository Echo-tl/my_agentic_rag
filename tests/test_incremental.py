"""
增量摄取测试（集成测试，需要 Qdrant 在线）。

验证哈希检测下：新增/变更/删除文档时只做必要的 embedding，绝不重复处理全部文档。
使用隔离的临时 data/、storage/ 和独立 Qdrant collection，测试结束自动清理。
"""

import os
import sys
import uuid
from pathlib import Path

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

TEXT1 = "LangGraph is a library for building stateful, multi-agent applications with LLMs."
TEXT2 = "PyTorch is a deep learning framework developed by Meta."
TEXT2_NEW = "PyTorch is a deep learning framework by Meta, used widely for research."


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """隔离环境：临时 data/storage + 独立 Qdrant collection，测试后清理。"""
    import database.qdrant as qdrant_mod
    import rag.index as index_mod
    from config import config as settings

    data_dir = tmp_path / "data"
    persist_dir = tmp_path / "storage"
    data_dir.mkdir()
    coll = "test_incr_" + uuid.uuid4().hex[:8]

    monkeypatch.setattr(settings.paths, "data_dir", data_dir)
    monkeypatch.setattr(settings.paths, "persist_dir", persist_dir)
    monkeypatch.setattr(settings.qdrant, "collection_name", coll)
    monkeypatch.setattr(qdrant_mod, "collection_name", coll)
    monkeypatch.setattr(index_mod, "collection_name", coll)

    yield data_dir, persist_dir, coll

    from database.qdrant import get_client
    try:
        get_client().delete_collection(collection_name=coll)
    except Exception:
        pass


def _vs():
    from database.qdrant import get_client, collection_name
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    return QdrantVectorStore(client=get_client(), collection_name=collection_name)


def _embed():
    from models.embedding import create_embedding
    from config import config
    return create_embedding(config.embedding)


def _count(coll):
    from database.qdrant import get_client
    return get_client().count(collection_name=coll).count


def _file_names(coll):
    from database.qdrant import get_client
    pts, _ = get_client().scroll(
        collection_name=coll, limit=1000,
        with_payload=["file_name"], with_vectors=False,
    )
    return sorted({p.payload.get("file_name") for p in pts})


def _hash_record():
    from rag.index import _load_hash_record
    return _load_hash_record()


class TestIncrementalIngest:
    def test_lifecycle(self, isolated_env):
        data_dir, persist_dir, coll = isolated_env
        from rag.index import get_index

        # 1) 首次构建：doc1
        (data_dir / "doc1.txt").write_text(TEXT1, encoding="utf-8")
        get_index(data_dir, _vs(), _embed())
        assert _file_names(coll) == ["doc1.txt"]
        assert _count(coll) > 0

        # 2) 新增 doc2：只摄取 doc2，doc1 不重新 embedding
        (data_dir / "doc2.txt").write_text(TEXT2, encoding="utf-8")
        before = _count(coll)
        get_index(data_dir, _vs(), _embed())
        after = _count(coll)
        assert _file_names(coll) == ["doc1.txt", "doc2.txt"]
        assert after > before  # 只增加了 doc2 的分块

        # 3) 无变化：points 不变（不重新 embedding）
        before = _count(coll)
        get_index(data_dir, _vs(), _embed())
        assert _count(coll) == before

        # 4) 变更 doc2：旧分块替换为新分块，数量稳定
        (data_dir / "doc2.txt").write_text(TEXT2_NEW, encoding="utf-8")
        before = _count(coll)
        get_index(data_dir, _vs(), _embed())
        after = _count(coll)
        assert _file_names(coll) == ["doc1.txt", "doc2.txt"]
        assert after == before  # 删除旧的 + 插入新的，净增 0

        # 5) 删除 doc2：分块被清理
        (data_dir / "doc2.txt").unlink()
        before = _count(coll)
        get_index(data_dir, _vs(), _embed())
        after = _count(coll)
        assert _file_names(coll) == ["doc1.txt"]
        assert after < before

        # 6) 哈希记录只剩 doc1
        assert set(_hash_record().keys()) == {"doc1.txt"}
