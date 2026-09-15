"""
单元测试: 文档元数据目录（persistence/doc_registry）。

关键设计断言：**增量摄取的判定必须读本地 JSON，不能读 MySQL**——MySQL 写入走的是
「可能丢弃」的后台队列，用一份可能丢数据的副本决定「这个文件要不要重新 embedding」
是危险的（最坏情况：文件变了却被判为未变，检索结果静默过期）。
MySQL 只在本地文件缺失时兜底。

不需要 Qdrant / Ollama。
"""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MD5_A = "a" * 32
MD5_B = "b" * 32


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """把 persist_dir / data_dir 指到临时目录，避免污染真实 storage/。"""
    import persistence.doc_registry as dr

    persist = tmp_path / "storage"
    data = tmp_path / "data"
    persist.mkdir()
    data.mkdir()
    monkeypatch.setattr(dr.config.paths, "persist_dir", persist)
    monkeypatch.setattr(dr.config.paths, "data_dir", data)
    return dr


class TestHashRecord:
    def test_roundtrip(self, registry):
        registry.save_hash_record({"a.pdf": MD5_A})
        assert registry.load_hash_record() == {"a.pdf": MD5_A}

    def test_missing_file_returns_empty(self, registry):
        assert registry.load_hash_record() == {}

    def test_corrupt_file_returns_empty_not_crash(self, registry):
        (registry.config.paths.persist_dir / "indexed_files.json").write_text(
            "{ 不是合法 JSON", encoding="utf-8")
        assert registry.load_hash_record() == {}

    def test_save_is_atomic(self, registry):
        """先写临时文件再替换：写到一半崩了不能留下半截 JSON。"""
        registry.save_hash_record({"a.pdf": MD5_A})
        registry.save_hash_record({"b.pdf": MD5_B})

        assert registry.load_hash_record() == {"b.pdf": MD5_B}
        assert not (registry.config.paths.persist_dir / "indexed_files.json.tmp").exists()

    def test_mirror_does_not_rewrite_local(self, registry, monkeypatch):
        """mirror 只投递 MySQL 副本，不碰本地判定真源。"""
        from persistence import writer

        sent = []
        monkeypatch.setattr(writer, "enqueue",
                            lambda kind, payload: sent.append((kind, payload)) or True)
        registry.mirror_hash_record({"a.pdf": MD5_A})

        assert sent == [("doc_snapshot", {"record": {"a.pdf": MD5_A}})]
        assert registry.load_hash_record() == {}   # 本地文件没被凭空造出来

    def test_hydrates_from_mysql_when_local_missing(self, registry, sqlite_orm):
        """换机器 / 清了 storage 但 MySQL 还在 → 不能全量重新 embedding。"""
        from persistence import repo
        from persistence.db import session_scope

        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "old.pdf", "file_hash": MD5_A,
                                   "status": "indexed"})

        assert registry.load_hash_record() == {"old.pdf": MD5_A}
        # 灌回后本地也有了，后续不再依赖 MySQL
        assert json.loads(
            (registry.config.paths.persist_dir / "indexed_files.json").read_text("utf-8")
        ) == {"old.pdf": MD5_A}

    def test_local_wins_over_mysql(self, registry, sqlite_orm):
        """本地是判定真源——MySQL 里的旧值不能覆盖它。"""
        from persistence import repo
        from persistence.db import session_scope

        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "a.pdf", "file_hash": MD5_B,
                                   "status": "indexed"})
        registry.save_hash_record({"a.pdf": MD5_A})

        assert registry.load_hash_record() == {"a.pdf": MD5_A}

    def test_removed_documents_excluded_from_hydration(self, registry, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope

        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "gone.pdf", "file_hash": MD5_A,
                                   "status": "removed"})
        assert registry.load_hash_record() == {}


class TestKbVersion:
    def test_notify_bumps_version_on_change(self, registry, fake_redis):
        from persistence import qa_cache

        assert qa_cache.get_kb_version() == 0
        assert registry.notify_kb_changed(added=["new.pdf"], total=3) == 1
        assert qa_cache.get_kb_version() == 1

    def test_notify_noop_without_changes(self, registry, fake_redis):
        assert registry.notify_kb_changed() == 0
        assert registry.notify_kb_changed(added=[], changed=[], removed=[]) == 0

    def test_notify_safe_without_redis(self, registry, no_backend):
        assert registry.notify_kb_changed(added=["x.pdf"]) == 0


class TestDocumentWrites:
    def test_upsert_and_list(self, registry, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope

        repo.register_all()
        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "a.pdf", "file_hash": MD5_A,
                                   "status": "indexed", "chunk_count": 12,
                                   "page_count": 3, "file_size": 2048})

        docs = registry.list_documents()
        assert len(docs) == 1
        assert docs[0]["chunk_count"] == 12 and docs[0]["file_size"] == 2048

    def test_list_filters_by_status(self, registry, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope

        repo.register_all()
        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "ok.pdf", "file_hash": MD5_A,
                                   "status": "indexed"})
            repo.on_doc_upsert(s, {"file_name": "bad.pdf", "file_hash": MD5_B,
                                   "status": "failed", "error": "解析失败"})

        assert [d["file_name"] for d in registry.list_documents(status="failed")] == ["bad.pdf"]
        assert len(registry.list_documents()) == 2

    def test_list_empty_without_mysql(self, registry, no_backend):
        assert registry.list_documents() == []

    def test_mark_removed_keeps_row(self, registry, sqlite_orm):
        """不物理删除：留一行 removed 供审计「这份文档曾经在库里」。"""
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import Document

        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "a.pdf", "file_hash": MD5_A,
                                   "status": "indexed"})
        with session_scope() as s:
            repo.on_doc_removed(s, {"file_name": "a.pdf", "kb_version": 5})

        with session_scope() as s:
            row = s.query(Document).filter_by(file_name="a.pdf").one()
            status, kb_version = row.status, row.kb_version
        assert status == "removed" and kb_version == 5

    def test_apply_snapshot_merges_without_duplicates(self, registry, sqlite_orm):
        """整份哈希记录镜像进 documents 时不能产生重复行。"""
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import Document

        with session_scope() as s:
            repo.on_doc_snapshot(s, {"record": {"a.pdf": MD5_A, "b.pdf": MD5_B}})
        with session_scope() as s:
            repo.on_doc_snapshot(s, {"record": {"a.pdf": MD5_A, "b.pdf": MD5_A}})

        with session_scope() as s:
            rows = s.query(Document).all()
        assert len(rows) == 2
        assert {r.file_name: r.file_hash for r in rows} == {"a.pdf": MD5_A, "b.pdf": MD5_A}

    def test_submit_helpers_safe_without_mysql(self, registry, no_backend):
        """MySQL 关闭时所有投递都是 no-op，不该抛异常。"""
        registry.upsert("a.pdf", MD5_A)
        registry.mark_removed("a.pdf")
        registry.mark_failed("a.pdf", MD5_A, "boom")
        assert registry.notify_kb_changed(added=["a.pdf"]) == 0


class TestIndexIntegration:
    def test_index_delegates_to_registry(self, registry, monkeypatch):
        """rag/index.py 的哈希读写必须走 registry（单点真源，不各写一份）。"""
        import rag.index as idx

        saved = {}
        monkeypatch.setattr("persistence.doc_registry.load_hash_record",
                            lambda: {"from_registry.pdf": MD5_A}, raising=False)
        monkeypatch.setattr("persistence.doc_registry.save_hash_record",
                            lambda r: saved.update(r), raising=False)

        assert idx._load_hash_record() == {"from_registry.pdf": MD5_A}
        idx._save_hash_record({"x.pdf": MD5_B})
        assert saved == {"x.pdf": MD5_B}

    def test_ingest_notifies_kb_change(self, registry, fake_redis, monkeypatch, tmp_path):
        """增量摄取结束后必须 bump kb_version，否则改了知识库缓存还在命中。"""
        import rag.index as idx
        from persistence import qa_cache

        data = registry.config.paths.data_dir
        (data / "new.pdf").write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(idx, "_scan_data_files", lambda d: {"new.pdf": data / "new.pdf"})

        before = qa_cache.get_kb_version()
        idx._notify_kb_changed(["new.pdf"], [], [], {"new.pdf": Path("x")})
        assert qa_cache.get_kb_version() == before + 1

    def test_no_change_run_still_seeds_mysql_mirror(self, registry, monkeypatch):
        """无变更的启动路径也要补一次 MySQL 镜像。

        否则：老部署本地 JSON 有记录、首次开 MySQL 时 documents 表是空的，
        而增量摄取发现无变更就直接 return —— /documents 会一直空到某个文件真的变了。
        """
        import rag.index as idx

        data = registry.config.paths.data_dir
        path = data / "a.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        registry.save_hash_record({"a.pdf": idx._hash_file(path)})

        monkeypatch.setattr(idx, "_scan_data_files", lambda d: {"a.pdf": path})
        seen = []
        monkeypatch.setattr("persistence.doc_registry.mirror_hash_record",
                            lambda rec: seen.append(dict(rec)))

        assert idx.incremental_ingest(None, data, None) is None
        assert seen == [{"a.pdf": idx._hash_file(path)}]

    def test_notify_never_raises(self, registry, monkeypatch):
        """缓存失效失败绝不能影响摄取结果。"""
        import rag.index as idx

        def boom(**kwargs):
            raise RuntimeError("redis 挂了")

        monkeypatch.setattr("persistence.doc_registry.notify_kb_changed", boom)
        idx._notify_kb_changed(["a.pdf"], [], [])   # 不应抛
