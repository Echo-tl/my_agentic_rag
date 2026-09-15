"""
单元测试: memory_node 的滚动摘要与历史裁剪。

裁剪是**不可逆**的（RemoveMessage 一旦生效，那段对话就没了），所以这里的断言
重点不是「有没有裁」，而是三条安全边界：
1. 摘要失败时**必须不裁**——不能删了却没有语义留存
2. 本轮的提问消息**必须留下**——否则 Agent 失去提问对象
3. 裁剪后 `_question_of` 仍能拿到本轮问题（不能退化成扫消息列表）

不触发真实 LLM：summarize 一律 monkeypatch 成桩。
无需 Redis / Qdrant / Ollama。
"""

import os
import sys

import pytest

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage  # noqa: E402


def _msgs(n: int) -> list:
    """构造 n 条交替消息，每条都带 id（RemoveMessage 需要 id 才能定位）。"""
    out = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(content=f"消息{i}", id=f"m{i}"))
    return out


@pytest.fixture
def stub_summary(monkeypatch):
    """把 summarizer 换成可编程桩，避免测试里真的调 Ollama。"""
    import memory.summarizer as sm

    calls = []

    def fake(messages, prev_summary=""):
        calls.append({"n": len(messages), "prev": prev_summary})
        return "（桩摘要）"

    monkeypatch.setattr(sm, "summarize", fake)
    return calls


class TestMemoryNode:
    def test_below_trigger_does_not_trim(self, stub_summary, monkeypatch):
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 12)
        out = graph.memory_node({"messages": _msgs(6), "summary": ""}, None)

        assert "messages" not in out          # 没有 RemoveMessage
        assert stub_summary == []             # 也根本没调摘要

    def test_above_trigger_trims_and_summarizes(self, stub_summary, monkeypatch):
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 4)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 2)

        out = graph.memory_node({"messages": _msgs(10), "summary": ""}, None)

        removed = out["messages"]
        assert all(isinstance(m, RemoveMessage) for m in removed)
        assert len(removed) == 8              # 10 - 保留 2
        assert {m.id for m in removed} == {f"m{i}" for i in range(8)}
        assert out["summary"] == "（桩摘要）"
        # 摘要只喂被裁掉的那 8 条，不是全部
        assert stub_summary[0]["n"] == 8

    def test_current_question_always_survives(self, stub_summary, monkeypatch):
        """本轮提问必须是最后一条 → 裁剪绝不能碰到它。"""
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 2)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 1)

        msgs = _msgs(6)
        msgs.append(HumanMessage(content="本轮问题", id="current"))

        out = graph.memory_node(
            {"messages": msgs, "current_question": "本轮问题"}, None
        )
        assert "current" not in {m.id for m in out["messages"]}

    def test_keep_recent_zero_still_protects_question(self, stub_summary, monkeypatch):
        """配置成 0 是误配，但绝不能因此把本轮问题删掉（max(1, …) 兜底）。"""
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 1)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 0)

        msgs = _msgs(4)
        msgs.append(HumanMessage(content="本轮问题", id="current"))

        out = graph.memory_node({"messages": msgs, "current_question": "本轮问题"}, None)
        assert "current" not in {m.id for m in out["messages"]}

    def test_summary_failure_does_not_trim(self, monkeypatch):
        """摘要失败（空串）→ 一条消息都不能删。裁剪不可逆，必须保守。"""
        import memory.summarizer as sm
        from workflows import graph

        monkeypatch.setattr(sm, "summarize", lambda messages, prev_summary="": "")
        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 2)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 1)

        out = graph.memory_node(
            {"messages": _msgs(6), "summary": "旧摘要"}, None
        )
        assert "messages" not in out
        assert out.get("summary", "旧摘要") == "旧摘要"

    def test_summary_disabled_never_trims(self, stub_summary, monkeypatch):
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_enabled", False)
        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 1)

        out = graph.memory_node({"messages": _msgs(10), "summary": ""}, None)
        assert "messages" not in out
        assert stub_summary == []

    def test_prev_summary_is_passed_to_summarizer(self, stub_summary, monkeypatch):
        """滚动摘要必须能续写——每一轮都把上一版摘要带进去。"""
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 2)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 1)

        graph.memory_node({"messages": _msgs(6), "summary": "第一版摘要"}, None)
        assert stub_summary[0]["prev"] == "第一版摘要"

    def test_messages_without_id_are_skipped(self, stub_summary, monkeypatch):
        """无 id 的消息 RemoveMessage 定位不到，只能跳过而不是崩掉。"""
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 2)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 1)

        msgs = [HumanMessage(content="无 id 消息"), HumanMessage(content="有 id", id="x")]
        msgs += _msgs(4)
        out = graph.memory_node({"messages": msgs, "summary": ""}, None)
        assert "messages" in out
        assert all(m.id != "无 id 消息" for m in out["messages"])

    def test_redis_failure_does_not_break_node(self, no_backend, stub_summary, monkeypatch):
        from workflows import graph

        monkeypatch.setattr(graph.app_config.redis, "summary_trigger", 2)
        monkeypatch.setattr(graph.app_config.redis, "summary_keep_recent", 1)

        out = graph.memory_node(
            {"messages": _msgs(6), "summary": ""},
            {"configurable": {"thread_id": "t1"}},
        )
        assert out["summary"] == "（桩摘要）"


class TestQuestionResolution:
    def test_prefers_current_question_over_messages(self):
        """核心回归：裁剪后「第一条 HumanMessage」已不是本轮问题，必须靠状态字段。"""
        from workflows import graph

        state = {
            "messages": [HumanMessage(content="上一轮的问题"), HumanMessage(content="本轮问题")],
            "current_question": "本轮问题",
        }
        assert graph._question_of(state) == "本轮问题"

    def test_falls_back_to_messages_when_absent(self):
        from workflows import graph

        state = {"messages": [HumanMessage(content="只有消息列表")]}
        assert graph._question_of(state) == "只有消息列表"

    def test_ignores_reflection_feedback(self):
        from workflows import graph

        state = {"messages": [
            HumanMessage(content="真正的问题"),
            HumanMessage(content="[Reflection Feedback] 请改进"),
        ]}
        assert graph._question_of(state) == "真正的问题"

    def test_empty_state_is_safe(self):
        from workflows import graph

        assert graph._question_of({}) == "unknown"

    def test_thread_id_extraction(self):
        from workflows import graph

        assert graph._thread_id({"configurable": {"thread_id": "abc"}}) == "abc"
        assert graph._thread_id(None) == ""
        assert graph._thread_id({}) == ""


class TestSummarizer:
    def test_render_includes_roles_and_truncates(self):
        from memory.summarizer import _render

        long = AIMessage(content="x" * 2000)
        text = _render([HumanMessage(content="问题"), long], per_msg_limit=100)
        assert "[用户] 问题" in text
        assert "截断" in text and len(text) < 2000

    def test_empty_input_returns_empty_string(self):
        from memory.summarizer import summarize

        assert summarize([], "旧摘要") == ""

    def test_render_for_prompt_mentions_recency(self):
        from memory.summarizer import render_for_prompt

        out = render_for_prompt("用户问过 AutoGen")
        assert "AutoGen" in out and "历史对话摘要" in out


class TestCheckpoint:
    def test_backend_degrades_to_sqlite_without_redis(self, no_backend, monkeypatch):
        """未启用 Redis 时自动退回 SQLite——不配环境变量的用户行为与引入前一致。"""
        import memory.checkpoint as ck

        monkeypatch.setattr(ck.config.redis, "checkpoint_backend", "redis")
        monkeypatch.setattr(ck.config.memory, "checkpoint_db_path", ":memory:")
        ck.reset()
        try:
            saver = ck.get_checkpointer()
            assert saver is not None
            assert ck.backend() == "sqlite"
        finally:
            ck.reset()

    def test_backend_none_returns_no_checkpointer(self, no_backend, monkeypatch):
        import memory.checkpoint as ck

        monkeypatch.setattr(ck.config.redis, "checkpoint_backend", "none")
        ck.reset()
        try:
            assert ck.get_checkpointer() is None
            assert ck.backend() == "none"
            # 无 checkpointer 时这些操作必须安全 no-op
            assert ck.prune(["t1"]) == 0
            assert ck.delete_thread("t1") is False
        finally:
            ck.reset()

    def test_checkpointer_is_singleton(self, monkeypatch, tmp_path):
        """必须单例：多线程各自持连接会 database is locked。"""
        import memory.checkpoint as ck

        monkeypatch.setattr(ck.config.redis, "checkpoint_backend", "sqlite")
        monkeypatch.setattr(ck.config.memory, "checkpoint_db_path",
                            str(tmp_path / "ck.db"))
        ck.reset()
        try:
            assert ck.get_checkpointer() is ck.get_checkpointer()
        finally:
            ck.reset()

    def test_prune_tolerates_bad_input(self, no_backend):
        import memory.checkpoint as ck

        assert ck.prune(None) == 0
        assert ck.prune([]) == 0
