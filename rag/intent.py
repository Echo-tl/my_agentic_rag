"""
IntentRouter: 查询理解与工作流路由。

基于 LLM Structured Output 抽取任务类型、论文实体与检索约束，
供 LangGraph 条件边将请求路由到 检索 / 总结 / 对比 / 联网搜索 / 澄清 等流程。
结构化输出失败时降级到关键词启发式，低置信度 / 未支持意图由路由层触发澄清。
"""

import logging
from typing import Dict, List, Literal, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM

from config import config
from models.llm import create_llm

logger = logging.getLogger("agentic_rag")

TaskType = Literal[
    "literature_retrieval",  # 文献检索
    "paper_summary",         # 论文总结
    "paper_comparison",      # 跨文献对比
    "web_search",            # 联网搜索
    "clarification",         # 需澄清 / 不支持
]

# 本地知识库中可识别的论文实体
KNOWN_PAPERS = ["AutoGen", "ReAct", "Reflexion", "AMOR", "Voyager"]


class IntentResult(BaseModel):
    """结构化意图识别结果。"""

    task_type: TaskType
    papers: List[str] = Field(default_factory=list, description="涉及的论文实体")
    constraints: Dict[str, object] = Field(default_factory=dict, description="检索约束")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度 0-1")
    reason: str = Field(default="", description="判断理由（简短）")


INTENT_PROMPT = """你是查询理解路由器。分析用户对本地论文知识库的查询，抽取意图。

任务类型（只能选一个）：
- literature_retrieval: 文献检索——查找某主题 / 某论文的相关内容
- paper_summary: 论文总结——概括某篇 / 某些论文的核心方法、贡献
- paper_comparison: 跨文献对比——比较多篇论文的差异 / 优劣
- web_search: 联网搜索——需要互联网最新 / 时效性信息
- clarification: 查询模糊、缺少关键信息，或不属于以上任何类型

本地论文库包含：AutoGen, ReAct, Reflexion, AMOR, Voyager。
papers 字段只填查询中明确提到的论文名（没提到就留空）。
constraints 可含 language、time_range、multiple_papers 等检索约束。

用户查询：{query}

输出任务类型、论文实体、检索约束、置信度(0-1)与简短理由。"""


_llm: Optional[LLM] = None


def get_intent_llm() -> LLM:
    """懒加载意图识别用的 LLM（与 Agent 共用同一个 Ollama 模型）。"""
    global _llm
    if _llm is None:
        _llm = create_llm(config.llm)
    return _llm


def _heuristic_fallback(query: str) -> IntentResult:
    """结构化输出失败时的关键词降级（低置信度，由路由层决定是否澄清）。"""
    q = query.lower()
    if any(k in q for k in ["对比", "区别", "比较", "差异", "优劣", "vs", "versus",
                            "compare", "comparison", "difference"]):
        return IntentResult(task_type="paper_comparison", confidence=0.4,
                            reason="heuristic: comparison keywords")
    if any(k in q for k in ["总结", "概括", "摘要", "核心方法", "贡献", "summar",
                            "overview", "takeaway"]):
        return IntentResult(task_type="paper_summary", confidence=0.4,
                            reason="heuristic: summary keywords")
    if any(k in q for k in ["最新", "新闻", "互联网", "实时", "在线", "news", "latest",
                            "2025", "2026", "online", "web", "internet"]):
        return IntentResult(task_type="web_search", confidence=0.4,
                            reason="heuristic: web/recency keywords")
    if any(k in q for k in ["是什么", "如何", "怎么", "原理", "机制", "what", "how",
                            "retriev", "search", "检索", "查找", "find", "framework", "method"]):
        return IntentResult(task_type="literature_retrieval", confidence=0.4,
                            reason="heuristic: retrieval keywords")
    return IntentResult(task_type="clarification", confidence=0.2,
                        reason="heuristic: no clear intent")


def classify_intent(query: str, llm: Optional[LLM] = None) -> IntentResult:
    """LLM Structured Output 解析意图；失败降级到关键词启发式。

    Args:
        query: 用户原始问题
        llm: 可注入的 LLM（默认懒加载），便于测试 mock

    Returns:
        IntentResult
    """
    llm = llm or get_intent_llm()
    try:
        structured = llm.as_structured_llm(output_cls=IntentResult)
        resp = structured.complete(INTENT_PROMPT.format(query=query[:500]))
        result = resp.raw
        if isinstance(result, IntentResult) and result.task_type:
            # 只保留知识库内可识别的论文实体，避免幻觉实体传给下游
            result.papers = [p for p in result.papers if p in KNOWN_PAPERS]
            return result
    except Exception as e:
        logger.warning(f"[intent] LLM 结构化输出失败，降级到启发式: {e}")
    return _heuristic_fallback(query)


# 各任务类型给 Agent 的引导提示（注入到当轮系统消息）
INTENT_HINTS = {
    "literature_retrieval": "任务类型：文献检索。请用不同角度的英文关键词多次检索知识库，整理相关论文。",
    "paper_summary": "任务类型：论文总结。请先检索目标论文，再基于检索到的内容概括其核心方法与贡献，不要编造。",
    "paper_comparison": "任务类型：跨文献对比。请分别检索涉及的各篇论文，再按方法、贡献、差异逐点对比。",
    "web_search": "任务类型：联网搜索。请调用 search_web 获取最新信息。",
}
