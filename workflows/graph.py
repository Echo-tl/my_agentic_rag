import atexit
import logging
from datetime import datetime
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage,
)

from langchain_ollama import ChatOllama

# 别名导入：节点函数签名里的 config 参数会遮蔽这个名字（tools_node 已踩过这个坑）
from config import config, config as app_config
from tools.search.web_search import search_web
from tools.rag.llamaindex_tool import search_knowledge_base
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from observability.tracing import record_node
from rag.intent import classify_intent, INTENT_HINTS

logger = logging.getLogger("agentic_rag")

# ── Agent State ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reflection_count: int
    intent: dict  # 意图识别结果（IntentResult 的 dict）

    # current_question 必须单独存一份，**不能**靠扫 messages 找第一条用户消息：
    # memory_node 会裁剪历史，裁掉之后「第一条 HumanMessage」就变成了别的轮次的问题。
    current_question: str
    summary: str  # 滚动短期摘要（被裁掉的旧对话的语义留存）


# ── Model ───────────────────────────────────────────────────
model = ChatOllama(
    model=config.llm.model,
    base_url=config.llm.base_url,
    temperature=config.llm.temperature,
    num_ctx=config.llm.context_window,
    num_predict=config.llm.max_tokens,
    client_kwargs={"timeout": config.llm.request_timeout},
)

def _cleanup_chat_ollama():
    try:
        if hasattr(model, '_client') and model._client and hasattr(model._client, '_client'):
            model._client._client.close()
    except Exception:
        pass
atexit.register(_cleanup_chat_ollama)

tools = [search_knowledge_base, search_web]
model_with_tools = model.bind_tools(tools)
# 联网搜索意图时只暴露 search_web，避免误用知识库
model_with_tools_web = model.bind_tools([search_web])

# ── System Prompt（使用 SystemMessage 而非 HumanMessage）─────
SYSTEM_MESSAGE = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)


# ── Dynamic Date Injection ──────────────────────────────────
DATE_REMINDER_TAG = "<system-reminder>"


def _inject_current_date(messages: list) -> list:
    """首次调用时找到第一条用户消息，在前面注入当前日期。跳过 SystemMessage。"""
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str) and DATE_REMINDER_TAG in content:
            return list(messages)  # 已注入

    current_date = datetime.now().strftime("%Y-%m-%d %A")
    reminder = f"<system-reminder>\nCurrent date: {current_date}\n</system-reminder>\n\n"

    result = []
    injected = False
    for msg in messages:
        # 只注入到用户消息（非 SystemMessage、非 Reflection 反馈）
        if (not injected and isinstance(msg, HumanMessage)
                and not isinstance(msg, SystemMessage)
                and hasattr(msg, "content")):
            content = msg.content or ""
            if not content.startswith("[Reflection"):
                result.append(HumanMessage(content=reminder + content))
                injected = True
                continue
        result.append(msg)
    return result


# ── Reflection Prompt ───────────────────────────────────────
REFLECTION_PROMPT = """Evaluate the answer above for quality and completeness.

Check:
1. Does it directly answer the user's question?
2. Does it cite specific sources or papers from the knowledge base?
3. Is the information accurate and well-structured?
4. Is anything important missing?

Reply with ONLY one line:
- "GRADE: PASS" if the answer is satisfactory
- "GRADE: FAIL | <reason>" if the answer needs improvement

Your evaluation:"""


def _has_system_message(messages) -> bool:
    """检查消息列表中是否已包含 SystemMessage 类型的系统提示。"""
    return any(isinstance(m, SystemMessage) and m.content == SUPERVISOR_SYSTEM_PROMPT
               for m in messages)


def _get_user_question(messages) -> str:
    """提取原始用户问题（跳过 SystemMessage 和 Reflection 反馈）。

    仅在 state 里拿不到 current_question 时才用（单测、直接构造 state 的场景）。
    正常路径请走 `_question_of(state)`——扫 messages 在历史被裁剪后会返回错误的轮次。
    """
    for m in messages:
        if isinstance(m, HumanMessage) and not isinstance(m, SystemMessage):
            content = m.content or ""
            if not content.startswith("[Reflection"):
                return content[:500]
    return "unknown"


def _question_of(state: dict) -> str:
    """本轮用户问题。优先取 state 上的 current_question，退化才扫消息列表。"""
    q = (state.get("current_question") or "").strip()
    if q:
        return q[:500]
    return _get_user_question(state.get("messages") or [])


def _thread_id(conf) -> str:
    """从 LangGraph 运行时 config 里取 thread_id（LangGraph 1.x 叫 thread_id）。"""
    if not conf:
        return ""
    cfg = conf.get("configurable", {}) if hasattr(conf, "get") else {}
    return cfg.get("thread_id") or cfg.get("thread_ts") or ""


# ── Agent Node ──────────────────────────────────────────────
def agent_node(state: AgentState, config=None) -> dict:
    """Agent reasoning: call LLM, decide tools or answer.

    意图路由后按任务类型限制工具，并注入任务引导提示。
    """
    record_node("agent")
    messages = state["messages"]

    # 首次：添加系统提示（按消息类型检查，避免重复）
    if not _has_system_message(messages):
        messages = [SYSTEM_MESSAGE] + list(messages)

    messages = _inject_current_date(messages)

    # 注入滚动摘要：被裁剪掉的旧对话只剩这段语义留存，指代消解全靠它
    summary = state.get("summary") or ""
    if summary:
        from memory.summarizer import render_for_prompt

        messages = list(messages) + [SystemMessage(content=render_for_prompt(summary))]

    # 按意图路由：web_search 只给联网工具，其余给完整工具
    task_type = (state.get("intent") or {}).get("task_type")
    bound_model = model_with_tools_web if task_type == "web_search" else model_with_tools

    # 注入任务引导提示（当轮有效，不改全局系统提示）
    hint = INTENT_HINTS.get(task_type)
    if hint:
        papers = (state.get("intent") or {}).get("papers") or []
        if papers:
            hint += f"\n用户可能涉及论文: {', '.join(papers)}"
        messages = list(messages) + [SystemMessage(content=f"[任务引导] {hint}")]

    response = bound_model.invoke(messages)
    return {"messages": [response]}


# ── Tools Node ──────────────────────────────────────────────
_tools_node_impl = ToolNode(tools)


def tools_node(state: AgentState, config=None) -> dict:
    record_node("tools")
    return _tools_node_impl.invoke(state, config)


# ── Reflection Node ─────────────────────────────────────────
def reflection_node(state: AgentState) -> dict:
    """Evaluate the agent's final answer. If inadequate, request retry."""
    record_node("reflection")
    if not config.retrieval.enable_reflection:
        return {}

    messages = state["messages"]
    retries = state.get("reflection_count", 0)

    if retries >= config.retrieval.max_reflection_retries:
        return {}

    last_ai = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            last_ai = m
            break

    if last_ai is None:
        return {}

    reflection_input = [
        HumanMessage(content=f"User question: {_question_of(state)}\n\n"
                             f"Answer to evaluate:\n{last_ai.content[:3000]}\n\n"
                             f"{REFLECTION_PROMPT}")
    ]
    grade_response = model.invoke(reflection_input)
    grade = str(grade_response.content).strip()

    if grade.startswith("GRADE: PASS"):
        return {}

    feedback = grade.replace("GRADE: FAIL", "").strip().lstrip("|").strip()
    if not feedback:
        feedback = "The answer needs more specific details and citations from the knowledge base."

    return {
        "messages": [HumanMessage(
            content=f"[Reflection Feedback] {feedback}\n\n"
                    f"Please improve your answer based on this feedback."
        )],
        "reflection_count": retries + 1,
    }


# ── Intent Routing（查询理解与工作流路由）──────────────────────
CLARIFICATION_PROMPT = (
    "抱歉，你的问题不够明确，我无法判断你想检索、总结还是对比。"
    "请补充：1) 你想查的主题或目标论文（如 AutoGen、ReAct、Reflexion、AMOR、Voyager）；"
    "2) 你需要的是文献检索、论文总结，还是多篇论文的对比。"
    "\n例如：“AutoGen 和 ReAct 的核心区别是什么？”"
)


def intent_node(state: AgentState) -> dict:
    """查询理解：LLM 结构化输出抽取意图（任务类型 / 论文实体 / 约束 / 置信度）。"""
    record_node("intent")
    query = _question_of(state)
    result = classify_intent(query)
    return {"intent": result.model_dump()}


# ── Memory Node（滚动摘要 + 历史裁剪）────────────────────────
def memory_node(state: AgentState, config=None) -> dict:
    """把被窗口挤出去的旧对话压成摘要，并用 RemoveMessage 真正从状态里裁掉。

    这个节点同时解决两个问题：
    - 上下文成本：每轮把整段历史喂给 LLM，token 线性增长
    - 存储膨胀：checkpoint 每轮重写整个消息列表，SqliteSaver 的 .db 无界增长

    摘要走 LLM，**失败就不裁剪**——裁剪不可逆，摘要没生成出来却把消息删了等于彻底
    丢失那段对话。宁可多留一轮历史的 token，也不能删得没有语义留存。
    """
    record_node("memory")

    cfg_redis = app_config.redis
    sid = _thread_id(config)
    messages = state["messages"]

    prev_summary = state.get("summary") or ""
    if not prev_summary and sid:
        # checkpoint 里没摘要（换了后端 / 裁剪后首轮）→ 从 Redis 或 MySQL 的持久副本捞回来
        from persistence import session_store

        prev_summary = session_store.get_summary(sid) or ""

    if not cfg_redis.summary_enabled or len(messages) <= cfg_redis.summary_trigger:
        if prev_summary and prev_summary != state.get("summary"):
            return {"summary": prev_summary}
        return {}

    # 至少保留 1 条：否则配置成 0 时会把本轮问题一起删掉，Agent 直接失去提问
    keep = max(1, cfg_redis.summary_keep_recent)
    old, recent = messages[:-keep], messages[-keep:]

    from memory.summarizer import summarize

    summary = summarize(old, prev_summary=prev_summary)

    if not summary:
        # 摘要失败：保留旧摘要，本轮不动消息
        if prev_summary and prev_summary != state.get("summary"):
            return {"summary": prev_summary}
        return {}

    if sid and summary != prev_summary:
        from persistence import session_store

        session_store.set_summary(sid, summary)

    removed = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
    if not removed:
        # 没有可裁剪的消息（都已无 id）→ 只更新摘要，别返回空的 messages 更新
        return {"summary": summary}

    logger.info(f"[memory] 裁剪 {len(removed)} 条历史（保留最近 {len(recent)} 条），"
                f"摘要 {len(summary)} 字")
    return {"summary": summary, "messages": removed}


def clarification_node(state: AgentState) -> dict:
    """低置信度 / 未支持意图：请求澄清，结束本轮回合。"""
    record_node("clarification")
    return {"messages": [AIMessage(content=CLARIFICATION_PROMPT)]}


def route_after_intent(state: AgentState) -> Literal["memory", "clarification"]:
    """按意图条件路由：澄清意图走澄清节点，其余先过 memory 再进 Agent。

    澄清轮跳 memory 是刻意的——澄清是模板化短回复，不进 LLM 也不值得为它付
    一次摘要成本；它只往状态里加一条 AIMessage，裁剪推迟到真正对话的轮次做。
    """
    intent = state.get("intent") or {}
    if intent.get("task_type") == "clarification":
        return "clarification"
    return "memory"


# ── Routing ─────────────────────────────────────────────────
def route_after_agent(state: AgentState) -> Literal["tools", "reflection", "__end__"]:
    messages = state["messages"]
    last_msg = messages[-1] if messages else None

    if isinstance(last_msg, AIMessage):
        if last_msg.tool_calls:
            return "tools"
        if last_msg.content and config.retrieval.enable_reflection:
            return "reflection"
        return "__end__"

    return "__end__"


def route_after_reflection(state: AgentState) -> Literal["agent", "__end__"]:
    messages = state["messages"]
    last_msg = messages[-1] if messages else None

    if isinstance(last_msg, HumanMessage) and not isinstance(last_msg, SystemMessage):
        if last_msg.content and last_msg.content.startswith("[Reflection"):
            return "agent"

    return "__end__"


# ── Build Graph ─────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("intent", intent_node)
workflow.add_node("memory", memory_node)
workflow.add_node("clarification", clarification_node)

if config.retrieval.enable_intent_routing:
    # 意图识别 → 按任务类型条件路由；澄清意图直接结束
    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", route_after_intent, {
        "memory": "memory",
        "clarification": "clarification",
    })
    workflow.add_edge("clarification", END)
    workflow.add_edge("memory", "agent")
else:
    # 无意图路由时 memory 仍要跑，否则裁剪功能整条路径都失效
    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "agent")

workflow.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools", "reflection": "reflection", "__end__": END,
})
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("reflection", route_after_reflection, {
    "agent": "agent", "__end__": END,
})

if config.redis.checkpoint_backend == "none":
    # 显式关掉持久化：单轮可用，但没有多轮记忆
    agent = workflow.compile()
else:
    from memory.checkpoint import get_checkpointer

    _checkpointer = get_checkpointer()
    agent = workflow.compile(checkpointer=_checkpointer) if _checkpointer else workflow.compile()

