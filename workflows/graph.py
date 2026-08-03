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

# ── Agent State ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reflection_count: int


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
    """Agent reasoning: call LLM, decide tools or answer."""
    record_node("agent")
    messages = state["messages"]

    # 首次：添加系统提示（按消息类型检查，避免重复）
    if not _has_system_message(messages):
        messages = [SYSTEM_MESSAGE] + list(messages)

    messages = _inject_current_date(messages)

    response = model_with_tools.invoke(messages)
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
