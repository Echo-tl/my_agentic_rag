"""
单元测试: API 层的缓存/会话接线（api/server 的 _prepare_turn / _finalize）。

这里覆盖的是**两个端点共用的那段逻辑**，也就是最容易在两边各写一遍后漂移的地方。
不启动 HTTP 服务、不调 LLM、不需要 Qdrant。
"""

import os
import sys

import pytest

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402


@pytest.fixture
def api(fake_redis, monkeypatch):
    """导入 api.server；同时把 cache_min_hits 降到 1，省去三次往返。"""
    import api.server as server
    from persistence import qa_cache

    monkeypatch.setattr(qa_cache.config.redis, "cache_min_hits", 1)
    return server


def _req(server, question, session_id=None, user_id=None):
    return server.QueryRequest(question=question, session_id=session_id, user_id=user_id)


class TestTurnPreparation:
    def test_generates_session_id_when_absent(self, api):
        turn = api._prepare_turn(_req(api, "AutoGen 是什么"))
        assert turn.sid and len(turn.sid) == 8

    def test_keeps_provided_session_id(self, api):
        turn = api._prepare_turn(_req(api, "AutoGen 是什么", session_id="sess-42"))
        assert turn.sid == "sess-42"

    def test_window_is_read_before_this_turn_is_appended(self, api, monkeypatch):
        """核心回归：指纹取自**本轮之前**的上下文。

        如果先 append_turn 再读窗口，指纹里就混进了本轮问题自身，
        同一句追问的指纹每轮都不同 → 多轮缓存命中率归零。
        """
        from persistence import session_store

        api._finalize(api._prepare_turn(_req(api, "AutoGen是什么", "s1")),
                      "AutoGen 是微软的多智能体框架……")

        # 第二轮的窗口里必须已经有第一轮，且不含本轮
        turn2 = api._prepare_turn(_req(api, "那它的缺点呢", "s1"))
        assert [m["content"] for m in turn2.window] == ["AutoGen是什么", "AutoGen 是微软的多智能体框架……"]

    def test_same_followup_hits_after_a_repeat(self, api, monkeypatch):
        """刷新页面重问同一句追问，第二次必须命中（指纹不该被自己污染）。"""
        q1, q2 = "AutoGen是什么", "那它的缺点呢"
        answer = "主要缺点是依赖大量人工设计的交互规则，扩展成本高……"

        first = api._prepare_turn(_req(api, q2, "s1"))
        assert first.cache.hit is False

        # 补上第一轮上下文后再问同一句
        api._finalize(api._prepare_turn(_req(api, q1, "s1")), "AutoGen 是……" * 5)
        second = api._prepare_turn(_req(api, q2, "s1"))
        assert second.cache.hit is False          # 首次仍需生成
        api._finalize(second, answer)

        third = api._prepare_turn(_req(api, q2, "s1"))
        assert third.cache.hit is True
        assert third.cache.answer == answer

    def test_different_context_does_not_hit(self, api):
        """两个会话各自聊不同论文，同一句追问绝不能互相命中。"""
        answer = "Voyager 的局限在于强依赖 Minecraft 环境……"
        api._finalize(api._prepare_turn(_req(api, "Voyager是什么", "sB")), "Voyager 是……" * 5)
        api._finalize(api._prepare_turn(_req(api, "那它的缺点呢", "sB")), answer)

        # 另一个会话：上下文是 AutoGen
        api._finalize(api._prepare_turn(_req(api, "AutoGen是什么", "sA")), "AutoGen 是……" * 5)
        other = api._prepare_turn(_req(api, "那它的缺点呢", "sA"))
        assert other.cache.hit is False


class TestFinalize:
    def test_cache_hit_also_appends_to_window(self, api):
        """命中缓存时也要写窗口，否则后续轮次的上下文指纹会缺一条。"""
        from persistence import session_store

        q = "ReAct 的核心机制是什么"
        turn = api._prepare_turn(_req(api, q, "s2"))
        api._finalize(turn, "ReAct 交替进行推理与行动……" * 3)

        # 命中路径
        hit = api._prepare_turn(_req(api, q, "s2"))
        assert hit.cache.hit is True
        api._finalize(hit, hit.cache.answer, cached=True)

        contents = [m["content"] for m in session_store.get_window("s2")]
        assert contents.count(q) == 2   # 两轮都被记进窗口

    def test_finalize_writes_turn_to_db(self, api, sqlite_orm):
        """完整异步链路：_finalize → 写队列 → 后台线程 → qa_records 落库。"""
        import time

        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import QARecord, Session

        repo.register_all()

        turn = api._prepare_turn(_req(api, "落库验证的问题", "s3", user_id="alice"))
        api._finalize(turn, "落库验证的答案" * 3, intent={"task_type": "paper_summary"},
                      citations=[{"file_name": "AutoGen.pdf", "page": 3, "score": 0.9}])

        deadline = time.time() + 5
        while time.time() < deadline:
            with session_scope() as s:
                if s is not None and s.query(QARecord).count() > 0:
                    break
            time.sleep(0.05)

        # 断言必须写在 session_scope **外面**：它按设计会吞掉 DB 层异常，
        # 留在里面的话连 AssertionError 一起吞，用例会假装通过
        with session_scope() as s:
            row = s.query(QARecord).one()
            question, task_type = row.question, row.task_type
            citation = row.citations[0]["file_name"]
            has_session = s.get(Session, "s3") is not None

        assert question == "落库验证的问题"
        assert task_type == "paper_summary"
        assert citation == "AutoGen.pdf"
        assert has_session

    def test_finalize_records_cache_hit_as_cached(self, api, sqlite_orm):
        """命中缓存的那一轮也要落库，且 cached=True —— 否则命中率无从统计。"""
        import time

        from persistence import repo
        from persistence.db import session_scope
        from persistence.models import QARecord

        repo.register_all()

        # 问题必须自足才会走 'self' 指纹：无实体、含指代词或短于 8 字的问句
        # 都会被判为不自足（缺上下文）→ 本轮不缓存
        q = "向量数据库的选型标准"
        api._finalize(api._prepare_turn(_req(api, q, "s4")), "缓存命中的答案" * 3)
        hit = api._prepare_turn(_req(api, q, "s4"))
        assert hit.cache.hit is True
        api._finalize(hit, hit.cache.answer, cached=True)

        deadline = time.time() + 5
        count = 0
        while time.time() < deadline:
            with session_scope() as s:
                count = s.query(QARecord).count()
            if count >= 2:
                break
            time.sleep(0.05)

        with session_scope() as s:
            cached_flags = [r.cached for r in s.query(QARecord).order_by(QARecord.id).all()]
        assert cached_flags == [False, True]

    def test_finalize_survives_no_backend(self, no_backend, monkeypatch):
        """Redis/MySQL 全挂时，_finalize 必须安静地什么都不做。"""
        import api.server as server

        turn = server._prepare_turn(_req(server, "AutoGen 是什么", "s9"))
        server._finalize(turn, "答案" * 20)      # 不应抛


class TestMessageHelpers:
    def test_final_answer_skips_tool_calls(self, api):
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "args": {}, "id": "1"}]),
            ToolMessage(content="检索结果", tool_call_id="1"),
            AIMessage(content="最终答案"),
        ]
        assert api._final_answer(msgs) == "最终答案"

    def test_final_answer_empty_when_no_ai_message(self, api):
        assert api._final_answer([HumanMessage(content="问题")]) == ""
        assert api._final_answer([]) == ""

    def test_extract_citations_parses_tool_output(self, api):
        msg = ToolMessage(
            content="[来源: AutoGen.pdf | 第3页 | 相关度: 0.87]\n"
                    "[来源: ReAct.pdf | 第1页 | 相关度: 0.61]\n"
                    "[来源: AutoGen.pdf | 第3页 | 相关度: 0.87]",   # 重复项
            tool_call_id="1",
        )
        cites = api._extract_citations([msg])
        assert len(cites) == 2
        assert {"file_name": "AutoGen.pdf", "page": 3, "score": 0.87} in cites

    def test_extract_citations_ignores_non_tool_messages(self, api):
        assert api._extract_citations([AIMessage(content="[来源: x.pdf | 第1页 | 相关度: 0.9]")]) == []

    def test_extract_citations_tolerates_empty(self, api):
        assert api._extract_citations(None) == []
        assert api._extract_citations([]) == []
