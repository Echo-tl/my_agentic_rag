"""
单元测试: 会话状态（persistence/session_store）+ 落库处理器（persistence/repo）。

两条主线：
1. Redis 可用时窗口/摘要的行为正确（滑动过期、LTRIM 裁剪）
2. **Redis/MySQL 全挂时主流程照常**——这是整个扩展层的降级底线
无需 Qdrant / Ollama。
"""

import os
import sys

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSessionStoreWithRedis:
    def test_touch_creates_session(self, fake_redis):
        from persistence import session_store

        meta = session_store.touch("s1", user_id=7)
        assert meta["session_id"] == "s1"
        assert meta["user_id"] == 7
        assert meta["turn_count"] == 1

    def test_touch_increments_turn_count(self, fake_redis):
        from persistence import session_store

        session_store.touch("s1")
        meta = session_store.touch("s1")
        assert meta["turn_count"] == 2
        assert meta["created_at"] < meta["last_seen"] or meta["created_at"] <= meta["last_seen"]

    def test_append_turn_trims_to_window(self, fake_redis, monkeypatch):
        """窗口必须有界——这是 checkpoints.db 无界增长之外的第二个膨胀点。"""
        from persistence import session_store

        monkeypatch.setattr(session_store.config.redis, "window_size", 4)
        for i in range(10):
            session_store.append_turn("s1", f"q{i}", f"a{i}")

        window = session_store.get_window("s1")
        assert len(window) == 4
        # 保留的是最近的：q8/a8, q9/a9
        assert window[0]["content"] == "q8"
        assert window[-1]["content"] == "a9"

    def test_window_is_chronological(self, fake_redis):
        from persistence import session_store

        session_store.append_turn("s1", "第一个问题", "第一个回答")
        session_store.append_turn("s1", "第二个问题", "第二个回答")

        window = session_store.get_window("s1")
        assert [m["role"] for m in window] == ["user", "assistant", "user", "assistant"]
        assert window[0]["content"] == "第一个问题"

    def test_summary_roundtrip(self, fake_redis):
        from persistence import session_store

        assert session_store.get_summary("s1") is None
        session_store.set_summary("s1", "用户先问了 AutoGen，又追问了与 ReAct 的区别。")
        assert "AutoGen" in session_store.get_summary("s1")

    def test_should_summarize_at_trigger(self, fake_redis, monkeypatch):
        from persistence import session_store

        monkeypatch.setattr(session_store.config.redis, "summary_trigger", 4)
        assert session_store.should_summarize("s1") is False
        session_store.append_turn("s1", "q1", "a1")
        session_store.append_turn("s1", "q2", "a2")
        assert session_store.should_summarize("s1") is True

    def test_delete_clears_state(self, fake_redis):
        from persistence import session_store

        session_store.touch("s1")
        session_store.append_turn("s1", "q", "a")
        session_store.set_summary("s1", "摘要")

        assert session_store.delete("s1") is True
        assert session_store.get_window("s1") == []
        assert session_store.get_summary("s1") is None

    def test_window_feeds_cache_fingerprint(self, fake_redis):
        """两个模块的接缝：session_store 写的窗口要能被 qa_cache 用来算指纹。"""
        from persistence import qa_cache, session_store

        session_store.append_turn("s1", "AutoGen是什么", "……")
        fp = qa_cache.build_fp("那它的缺点呢", session_store.get_window("s1"))
        assert fp is not None and fp.startswith("ctx:")


class TestSessionStoreDegradation:
    def test_all_operations_safe_without_redis(self, no_backend):
        """Redis 挂掉：读返回空/None，写静默 no-op，一个异常都不许冒出来。"""
        from persistence import session_store

        assert session_store.touch("s1") is None
        assert session_store.get_session("s1") is None
        assert session_store.get_window("s1") == []
        assert session_store.message_count("s1") == 0
        assert session_store.get_summary("s1") is None
        assert session_store.should_summarize("s1") is False
        assert session_store.list_sessions() == []
        assert session_store.delete("s1") is False
        session_store.append_turn("s1", "q", "a")   # 不应抛
        session_store.set_summary("s1", "摘要")      # 不应抛


class TestUserStore:
    def test_resolve_creates_guest(self, sqlite_orm):
        from persistence import user_store

        user_store.reset()
        uid = user_store.resolve(None)
        assert uid is not None

        info = user_store.get_info(uid)
        assert info["is_guest"] is True
        assert info["username"] == user_store.GUEST_USERNAME

    def test_resolve_is_memoized(self, sqlite_orm):
        """同一外部 id 只查一次库——resolve 在请求路径上。"""
        from persistence import user_store

        user_store.reset()
        assert user_store.resolve("alice") == user_store.resolve("alice")
        assert user_store.resolve("alice") != user_store.resolve("bob")

    def test_external_id_truncated_not_rejected(self, sqlite_orm):
        """超长外部 id 截断处理，不该 500。"""
        from persistence import user_store

        user_store.reset()
        assert user_store.resolve("x" * 300) is not None

    def test_resolve_degraded_without_mysql(self, no_backend):
        from persistence import user_store

        user_store.reset()
        assert user_store.resolve("alice") is None
        assert user_store.get_info(1) is None


class TestRepoWrites:
    def test_turn_insert_autocreates_session(self, sqlite_orm):
        """qa_records 有外键指向 sessions —— 处理器必须先建会话行，否则 MySQL 报 1452。"""
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import QARecord, Session

        with session_scope() as s:
            repo.on_turn(s, {
                "session_id": "sess-a",
                "question": "AutoGen 是什么",
                "question_hash": "a" * 40,
                "answer": "AutoGen 是……",
                "task_type": "paper_summary",
                "cached": False,
                "kb_version": 1,
            })

        with session_scope() as s:
            sess_row = s.get(Session, "sess-a")
            qa_count = s.query(QARecord).count()
        assert sess_row is not None
        assert qa_count == 1

    def test_turn_index_increments_from_db(self, sqlite_orm):
        """轮次序号以库里的 turn_count 为准，不用内存计数（多 worker 会各数各的）。"""
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import QARecord

        for i in range(3):
            with session_scope() as s:
                repo.on_turn(s, {
                    "session_id": "sess-b",
                    "question": f"q{i}",
                    "question_hash": "b" * 40,
                    "answer": "a",
                })

        with session_scope() as s:
            indices = [r.turn_index for r in
                       s.query(QARecord).order_by(QARecord.id).all()]
        assert indices == [1, 2, 3]

    def test_first_question_becomes_title(self, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import Session

        with session_scope() as s:
            repo.on_turn(s, {"session_id": "s", "question": "什么是 ReAct",
                             "question_hash": "c" * 40, "answer": "x"})

        with session_scope() as s:
            title = s.get(Session, "s").title
        assert title == "什么是 ReAct"

    def test_summary_updates_session(self, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import Session

        with session_scope() as s:
            repo._ensure_session(s, "s")
        with session_scope() as s:
            repo.on_summary(s, {"session_id": "s", "summary": "滚动摘要正文",
                                "summary_upto": 4})

        with session_scope() as s:
            row = s.get(Session, "s")
            summary, upto = row.summary, row.summary_upto
        assert summary == "滚动摘要正文" and upto == 4

    def test_trace_upsert_is_idempotent(self, sqlite_orm):
        """trace 分两次上报（开始/结束），必须 upsert 而不是插两行。"""
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import ExecutionTrace

        with session_scope() as s:
            repo.on_trace(s, {"trace_id": "t1", "query": "问题", "session_id": "s"})
        with session_scope() as s:
            repo.on_trace(s, {"trace_id": "t1", "query": "问题", "elapsed_ms": 123,
                              "tool_count": 2, "node_path": ["intent", "agent"]})

        with session_scope() as s:
            rows = s.query(ExecutionTrace).all()
        assert len(rows) == 1 and rows[0].elapsed_ms == 123

    def test_document_lifecycle(self, sqlite_orm):
        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import Document

        with session_scope() as s:
            repo.on_doc_upsert(s, {"file_name": "a.pdf", "file_hash": "d" * 32,
                                   "status": "indexed", "chunk_count": 42})
        with session_scope() as s:
            row = s.query(Document).filter_by(file_name="a.pdf").one()
            status, chunks, indexed_at = row.status, row.chunk_count, row.indexed_at
        assert status == "indexed" and chunks == 42
        assert indexed_at is not None

        with session_scope() as s:
            repo.on_doc_removed(s, {"file_name": "a.pdf"})
        with session_scope() as s:
            status = s.query(Document).filter_by(file_name="a.pdf").one().status
        assert status == "removed"

    def test_read_apis(self, sqlite_orm):
        from persistence import repo, user_store

        user_store.reset()
        uid = user_store.resolve("alice")
        for i in range(3):
            with sqlite_orm() as s:
                repo.on_turn(s, {"session_id": "s", "question": f"q{i}",
                                 "question_hash": "e" * 40, "answer": f"a{i}",
                                 "user_id": uid})
                s.commit()

        qa = repo.list_session_qa("s", limit=2)
        assert len(qa) == 2
        # 倒序：最新在前
        assert qa[0]["question"] == "q2"
        assert [r["turn_index"] for r in repo.list_session_qa("s")] == [3, 2, 1]

        sessions = repo.list_user_sessions(uid)
        assert len(sessions) == 1 and sessions[0]["session_id"] == "s"

    def test_read_apis_degraded_without_mysql(self, no_backend):
        from persistence import repo

        assert repo.list_session_qa("s") == []
        assert repo.list_user_sessions(1) == []


class TestWriter:
    def test_enqueue_is_noop_without_mysql(self, no_backend):
        from persistence import repo, writer

        assert repo.submit_turn("s", "q", "a") is False
        assert writer.enqueue("turn", {}) is False

    def test_flush_dispatches_to_registered_handler(self, sqlite_orm, monkeypatch):
        """队列的消费端：一批里的每条都应落到对应 handler。"""
        from persistence import repo, writer

        repo.register_all()
        seen = []

        def fake_handler(session, payload):
            seen.append(payload["n"])

        monkeypatch.setitem(writer._HANDLERS, "test_kind", fake_handler)
        writer._flush([("test_kind", {"n": 1}), ("test_kind", {"n": 2})])
        assert seen == [1, 2]

    def test_one_bad_entry_does_not_kill_the_batch(self, sqlite_orm, monkeypatch):
        """一条脏数据不能连累同批其它条。"""
        from persistence import repo, writer

        repo.register_all()
        seen = []

        def boom(session, payload):
            raise RuntimeError("坏数据")

        def ok(session, payload):
            seen.append(payload["n"])

        monkeypatch.setitem(writer._HANDLERS, "bad", boom)
        monkeypatch.setitem(writer._HANDLERS, "good", ok)
        writer._flush([("bad", {}), ("good", {"n": 9})])
        assert seen == [9]

    def test_real_db_failure_does_not_take_down_the_batch(self, sqlite_orm, monkeypatch):
        """**真·写库失败**（唯一键冲突）不能连累同批的其它记录。

        上面那个用例其实证明不了什么：`boom` 只 raise、不碰 session，
        所以没有 SAVEPOINT 也能过。真正的失败模式是 SQLAlchemy 在一次写库异常后
        把整个 session 标成「必须 rollback」——此后每条 handler 都抛
        PendingRollbackError，末尾 commit 也失败，**整批（含无关会话的记录）一起丢**。
        线上对应：某一行超长被 MySQL 拒绝 → 同一批里别人的问答记录跟着消失。
        """
        from persistence import repo, writer
        from persistence.db import session_scope
        from persistence.models import Document, QARecord

        repo.register_all()

        def conflict(s, payload):
            # 同一事务里插两份同名文档 → 唯一键冲突，DB 层真异常
            s.add(Document(file_name="dup.pdf", file_hash="a" * 32))
            s.flush()
            s.add(Document(file_name="dup.pdf", file_hash="b" * 32))
            s.flush()

        monkeypatch.setitem(writer._HANDLERS, "conflict", conflict)
        writer._flush([
            ("conflict", {}),
            ("turn", {"session_id": "s-ok", "question": "同批的好记录",
                      "question_hash": "c" * 40, "answer": "答案"}),
        ])

        with session_scope() as s:
            landed = s.query(QARecord).filter_by(session_id="s-ok").count()
        assert landed == 1

    def test_writer_restarts_after_shutdown(self, sqlite_orm, monkeypatch):
        """shutdown 之后必须能重新启动。

        否则同进程内第二次 lifespan（TestClient 复用一个 app、或同进程再起一个
        实例）之后，所有写入都会静静进队列、无人消费：不报错，只是数据没了。
        """
        import time

        from persistence import repo, writer
        from persistence.db import session_scope
        from persistence.models import QARecord

        repo.register_all()
        monkeypatch.setattr(writer.config.mysql, "enabled", True)

        writer.shutdown(wait=True, timeout=2.0)
        assert repo.submit_turn("s-restart", "重启之后", "答案", question_hash="9" * 40)

        n = 0
        deadline = time.time() + 5
        while time.time() < deadline:
            with session_scope() as s:
                n = s.query(QARecord).filter_by(session_id="s-restart").count()
            if n:
                break
            time.sleep(0.05)
        assert n == 1

    def test_end_to_end_through_queue(self, sqlite_orm, monkeypatch):
        """真正的异步路径：enqueue → 后台线程 → 落库。"""
        import time

        from persistence import repo, writer
        from persistence.db import session_scope
        from persistence.models import QARecord

        repo.register_all()
        monkeypatch.setattr(writer.config.mysql, "enabled", True)

        assert repo.submit_turn("q-sess", "写队列落库", "答案", question_hash="f" * 40)

        deadline = time.time() + 5
        while time.time() < deadline:
            with session_scope() as s:
                if s is not None and s.query(QARecord).count() > 0:
                    break
            time.sleep(0.05)

        with session_scope() as s:
            total = s.query(QARecord).count()
        assert total == 1
