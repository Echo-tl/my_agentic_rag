import atexit
from datetime import datetime
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import config
from tools.search.web_search import search_web
from tools.rag.llamaindex_tool import search_knowledge_base
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from observability.tracing import record_node
from rag.intent import classify_intent, INTENT_HINTS

# ── Agent State ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reflection_count: int
    intent: dict  # 意图识别结果（IntentResult 的 dict）


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
    """提取原始用户问题（跳过 SystemMessage 和 Reflection 反馈）。"""
    for m in messages:
        if isinstance(m, HumanMessage) and not isinstance(m, SystemMessage):
            content = m.content or ""
            if not content.startswith("[Reflection"):
                return content[:500]
    return "unknown"


# ── Agent Node ──────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """Agent reasoning: call LLM, decide tools or answer.

    意图路由后按任务类型限制工具，并注入任务引导提示。
    """
    record_node("agent")
    messages = state["messages"]

    # 首次：添加系统提示（按消息类型检查，避免重复）
    if not _has_system_message(messages):
        messages = [SYSTEM_MESSAGE] + list(messages)

    messages = _inject_current_date(messages)

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
        HumanMessage(content=f"User question: {_get_user_question(messages)}\n\n"
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
    query = _get_user_question(state["messages"])
    result = classify_intent(query)
    return {"intent": result.model_dump()}


def clarification_node(state: AgentState) -> dict:
    """低置信度 / 未支持意图：请求澄清，结束本轮回合。"""
    record_node("clarification")
    return {"messages": [AIMessage(content=CLARIFICATION_PROMPT)]}


def route_after_intent(state: AgentState) -> Literal["agent", "clarification"]:
    """按意图条件路由：澄清意图走澄清节点，其余进入 Agent 工作流。"""
    intent = state.get("intent") or {}
    if intent.get("task_type") == "clarification":
        return "clarification"
    return "agent"


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
workflow.add_node("clarification", clarification_node)

if config.retrieval.enable_intent_routing:
    # 意图识别 → 按任务类型条件路由；澄清意图直接结束
    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", route_after_intent, {
        "agent": "agent",
        "clarification": "clarification",
    })
    workflow.add_edge("clarification", END)
else:
    workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools", "reflection": "reflection", "__end__": END,
})
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("reflection", route_after_reflection, {
    "agent": "agent", "__end__": END,
})

if config.memory.checkpoint_db_path:
    from memory.checkpoint import get_checkpointer
    agent = workflow.compile(checkpointer=get_checkpointer())
else:
    agent = workflow.compile()
