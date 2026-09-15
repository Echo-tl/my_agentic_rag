"""
单元测试: 热点问答缓存（persistence/qa_cache）。

最重要的一组断言是「多轮指代不能串答案」——这是缓存方案唯一会答错的地方，
也是决定用上下文指纹而不是裸问题哈希的原因。
无需 Qdrant / Ollama；Redis 用 fakeredis。
"""

import os
import sys

import pytest

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


WINDOW_AUTOGEN = [
    {"role": "user", "content": "AutoGen是什么"},
    {"role": "assistant", "content": "AutoGen 是微软的多智能体对话框架…"},
]
WINDOW_VOYAGER = [
    {"role": "user", "content": "Voyager是什么"},
    {"role": "assistant", "content": "Voyager 是 Minecraft 中的终身学习智能体…"},
]


class TestNormalize:
    def test_chinese_spacing_insensitive(self):
        from persistence.qa_cache import normalize

        assert normalize("ReAct 和 AutoGen 有什么区别？") == normalize("react和autogen有什么区别")

    def test_fullwidth_and_polite_prefix(self):
        from persistence.qa_cache import normalize

        assert normalize("　请问  AutoGen 是什么？ ") == normalize("autogen是什么")

    def test_english_word_spacing_preserved(self):
        """英文词间空格必须保留——否则 'whatisreact' 与 'what is react' 会误合并。"""
        from persistence.qa_cache import normalize

        assert normalize("what is react") != normalize("whatisreact")

    def test_question_hash_is_stable(self):
        from persistence.qa_cache import question_hash

        assert question_hash("AutoGen 是什么？") == question_hash("autogen是什么")
        assert len(question_hash("x")) == 40


class TestFingerprint:
    def test_followup_does_not_share_cache_across_contexts(self):
        """核心正确性断言：同一句追问在不同上下文下必须得到不同指纹。"""
        from persistence.qa_cache import build_fp

        q = "和ReAct有什么区别"
        fp_autogen = build_fp(q, WINDOW_AUTOGEN)
        fp_voyager = build_fp(q, WINDOW_VOYAGER)

        assert fp_autogen is not None and fp_voyager is not None
        assert fp_autogen != fp_voyager, "不同上下文的同一句追问绝不能共用缓存"
        assert fp_autogen.startswith("ctx:")

    def test_self_contained_shares_across_sessions(self):
        """含明确实体的问题自足 → 跨会话共享，这是缓存收益的主要来源。"""
        from persistence.qa_cache import build_fp

        q = "AutoGen 和 ReAct 有什么区别"
        assert build_fp(q, []) == "self"
        assert build_fp(q, WINDOW_AUTOGEN) == "self"

    def test_first_turn_anaphora_not_cached(self):
        """首轮就问指代问题 → 无上下文可依据 → 禁止缓存。"""
        from persistence.qa_cache import build_fp

        assert build_fp("那它的缺点呢", []) is None
        assert build_fp("和ReAct有什么区别", []) is None

    def test_contrast_needs_two_entities(self):
        """对比句必须点名两个实体才自足——只点名一个说明另一方来自上下文。"""
        from persistence.qa_cache import is_self_contained

        assert is_self_contained("AutoGen 和 ReAct 有什么区别") is True   # 两个实体
        assert is_self_contained("和ReAct有什么区别") is False           # 缺对比对象
        assert is_self_contained("对比一下") is False                    # 无实体

    def test_repeating_a_followup_keeps_the_same_fingerprint(self):
        """重复问同一句指代问题必须得到同一指纹，否则刷新页面就必然 MISS。

        窗口里已经存在同一句提问时，它不能参与自己的指纹计算——
        指代句本身不携带「它在指谁」的信息。
        """
        from persistence.qa_cache import build_fp

        q = "那它的缺点呢"
        first = build_fp(q, WINDOW_AUTOGEN)
        # 第二轮：窗口里多了上一次的同一句提问
        repeated = build_fp(q, WINDOW_AUTOGEN + [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "上一轮的答案"},
        ])
        assert first == repeated

    def test_repeat_does_not_leak_across_contexts(self):
        """剔除同句不能把「不同上下文」也合并了——安全性优先。"""
        from persistence.qa_cache import build_fp

        q = "那它的缺点呢"
        fp_autogen = build_fp(q, WINDOW_AUTOGEN + [{"role": "user", "content": q}])
        fp_voyager = build_fp(q, WINDOW_VOYAGER + [{"role": "user", "content": q}])
        assert fp_autogen != fp_voyager

    def test_context_uses_only_user_turns(self):
        """指纹取自用户问题（确定性），不含 assistant 回答（LLM 输出会漂移）。"""
        from persistence.qa_cache import build_fp

        q = "那它的缺点呢"
        fp_a = build_fp(q, WINDOW_AUTOGEN)
        fp_b = build_fp(q, [
            {"role": "user", "content": "AutoGen是什么"},
            {"role": "assistant", "content": "换了完全不同的回答内容"},
        ])
        assert fp_a == fp_b


class TestIsCacheable:
    @pytest.mark.parametrize("answer,intent,reflection,reason", [
        ("", {"task_type": "paper_summary", "confidence": 0.9}, 0, "answer_too_short"),
        ("x" * 100, {"task_type": "clarification", "confidence": 0.9}, 0, "clarification"),
        ("x" * 100, {"task_type": "web_search", "confidence": 0.9}, 0, "time_sensitive"),
        ("x" * 100, {"task_type": "paper_summary", "confidence": 0.3}, 0, "low_confidence"),
        ("x" * 100, {"task_type": "paper_summary", "confidence": 0.9}, 1, "reflection_retried"),
    ])
    def test_rejected(self, answer, intent, reflection, reason):
        from persistence.qa_cache import is_cacheable

        ok, why = is_cacheable(answer, intent, reflection)
        assert not ok and why == reason

    def test_accepted(self):
        from persistence.qa_cache import is_cacheable

        ok, why = is_cacheable(
            "AutoGen 的核心贡献是……" * 3,
            {"task_type": "paper_summary", "confidence": 0.9},
            0,
        )
        assert ok and why == "ok"


class TestCacheRoundTrip:
    def test_hit_only_after_min_hits(self, fake_redis, monkeypatch):
        """热点门槛：第 1 次不缓存，第 2 次才写入，第 3 次命中。"""
        from persistence import qa_cache

        monkeypatch.setattr(qa_cache.config.redis, "cache_min_hits", 2)
        q = "AutoGen 的核心方法是什么"
        intent = {"task_type": "paper_summary", "confidence": 0.9}
        answer = "AutoGen 通过可对话的智能体编排实现多智能体协作……"

        # 第 1 次：未命中，热度 1 < 2 → 不写入
        look1 = qa_cache.lookup(q, [])
        assert not look1.hit and look1.key is not None and look1.locked
        assert qa_cache.observe_and_store(look1, q, answer, intent, 0) is False

        # 第 2 次：仍未命中，热度 2 >= 2 → 写入
        look2 = qa_cache.lookup(q, [])
        assert not look2.hit
        assert qa_cache.observe_and_store(look2, q, answer, intent, 0) is True

        # 第 3 次：命中
        look3 = qa_cache.lookup(q, [])
        assert look3.hit and look3.answer == answer

    def test_lock_released_after_store(self, fake_redis, monkeypatch):
        """锁必须主动释放：只靠 30s TTL 会让相邻两次提问抢不到锁，缓存永远回填不了。"""
        from persistence import qa_cache

        monkeypatch.setattr(qa_cache.config.redis, "cache_min_hits", 1)
        q = "ReAct 的核心思想是什么"
        intent = {"task_type": "paper_summary", "confidence": 0.9}

        answer = "ReAct 让模型交替进行推理与行动，把思维链与工具调用交织在一起……"
        look = qa_cache.lookup(q, [])
        assert look.locked is True
        assert qa_cache.observe_and_store(look, q, answer, intent, 0) is True

        # 锁已释放 → 下一次同问仍能抢到锁（而不是被 30s TTL 挡住）
        assert qa_cache.lookup(q, []).hit is True

    def test_kb_version_change_invalidates(self, fake_redis, monkeypatch):
        """知识库版本一涨，老缓存不可达。"""
        from persistence import qa_cache

        monkeypatch.setattr(qa_cache.config.redis, "cache_min_hits", 1)
        q = "Voyager 的终身学习机制是什么"
        intent = {"task_type": "paper_summary", "confidence": 0.9}
        answer = "Voyager 通过技能库与自动课程实现终身学习……"

        look = qa_cache.lookup(q, [])
        qa_cache.observe_and_store(look, q, answer, intent, 0)
        assert qa_cache.lookup(q, []).hit

        qa_cache.bump_kb_version(added=["new.pdf"], total=13)
        assert qa_cache.lookup(q, []).hit is False

    def test_context_scoped_cache_serves_same_context(self, fake_redis, monkeypatch):
        """同一上下文下的重复追问应命中（用户刷新/重试场景）。"""
        from persistence import qa_cache

        monkeypatch.setattr(qa_cache.config.redis, "cache_min_hits", 1)
        q = "那它的缺点呢"
        intent = {"task_type": "paper_summary", "confidence": 0.9}
        answer = "主要缺点是依赖大量人工设计的多智能体交互规则……"

        look = qa_cache.lookup(q, WINDOW_AUTOGEN)
        qa_cache.observe_and_store(look, q, answer, intent, 0)
        assert qa_cache.lookup(q, WINDOW_AUTOGEN).answer == answer
        # 换成另一段上下文 → 不命中
        assert qa_cache.lookup(q, WINDOW_VOYAGER).hit is False


class TestDegradation:
    def test_no_redis_disables_cache(self, no_backend):
        """Redis 不可用时缓存整体禁用，主流程不受影响。"""
        from persistence import qa_cache

        look = qa_cache.lookup("AutoGen 是什么", [])
        assert look.key is None and look.hit is False
        assert qa_cache.observe_and_store(look, "AutoGen 是什么", "x" * 50, {}, 0) is False
        assert qa_cache.get_kb_version() == 0
        assert qa_cache.bump_kb_version() == 0
        assert qa_cache.stats()["redis_available"] is False

    def test_build_fp_still_works_without_redis(self):
        """指纹计算是纯函数，不依赖 Redis。"""
        from persistence.qa_cache import build_fp

        assert build_fp("AutoGen 是什么", []) == "self"
