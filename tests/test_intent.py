"""
意图识别（IntentRouter）单元测试。
不依赖 Ollama：测试关键词降级、LLM 失败回退、论文实体过滤与路由逻辑。
"""

import os
import sys

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHeuristicFallback:
    def _fb(self, q):
        from rag.intent import _heuristic_fallback
        return _heuristic_fallback(q)

    def test_comparison(self):
        assert self._fb("AutoGen和ReAct有什么区别").task_type == "paper_comparison"
        assert self._fb("compare ReAct and Reflexion").task_type == "paper_comparison"

    def test_summary(self):
        assert self._fb("总结Reflexion的核心方法").task_type == "paper_summary"
        assert self._fb("summarize AutoGen").task_type == "paper_summary"

    def test_web_search(self):
        assert self._fb("最新的大模型Agent进展").task_type == "web_search"
        assert self._fb("latest LLM agent news").task_type == "web_search"

    def test_retrieval(self):
        assert self._fb("Voyager在Minecraft中如何探索").task_type == "literature_retrieval"
        assert self._fb("what is ReAct framework").task_type == "literature_retrieval"

    def test_clarification(self):
        assert self._fb("你好").task_type == "clarification"


class _FailLLM:
    """结构化输出必然失败的 LLM，触发降级。"""

    def as_structured_llm(self, output_cls=None):
        raise RuntimeError("llm unavailable")


class TestClassifyIntent:
    def test_falls_back_when_llm_fails(self):
        from rag.intent import classify_intent
        r = classify_intent("AutoGen和ReAct有什么区别", llm=_FailLLM())
        assert r.task_type == "paper_comparison"
        assert r.confidence <= 0.5  # 降级后置信度低，由路由层决定是否澄清

    def test_papers_filtered_to_known(self):
        from rag.intent import classify_intent, IntentResult

        class Resp:
            raw = IntentResult(task_type="paper_comparison",
                               papers=["AutoGen", "FakePaper123"],
                               confidence=0.9)

        class FakeStructured:
            def complete(self, prompt):
                return Resp()

        class FakeLLM:
            def as_structured_llm(self, output_cls=None):
                return FakeStructured()

        r = classify_intent("AutoGen vs FakePaper123", llm=FakeLLM())
        assert r.papers == ["AutoGen"]  # FakePaper123 不在知识库，被过滤


class TestRouting:
    def test_route_after_intent(self):
        from workflows.graph import route_after_intent
        assert route_after_intent({"intent": {"task_type": "clarification"}}) == "clarification"
        assert route_after_intent({"intent": {"task_type": "paper_summary"}}) == "agent"
        assert route_after_intent({"intent": {}}) == "agent"

    def test_graph_compiles_with_intent_routing(self):
        from workflows.graph import agent
        assert agent is not None
