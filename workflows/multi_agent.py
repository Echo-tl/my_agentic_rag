"""
Multi-Agent 系统: Supervisor + Research Agent + Writer Agent。

Supervisor 规划任务 → Research Agent 搜索信息 → Writer Agent 撰写报告。

使用 agents/ 工厂函数创建各 Agent，路由通过 state.next_agent 字段而非字符串匹配。
"""

import atexit
import re
from datetime import datetime
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)

from config import config
from tools.rag.llamaindex_tool import search_knowledge_base
from prompts.researcher import RESEARCHER_FIRST_STEP_PROMPT
from observability.tracing import record_node

# ── 使用 agents/ 工厂函数 ─────────────────────────────────────
from agents.supervisor import create_supervisor
from agents.research_agent import create_research_agent
from agents.writer_agent import create_writer_agent

supervisor_model, SUPERVISOR_SYSTEM_PROMPT = create_supervisor()
research_model, RESEARCHER_SYSTEM_PROMPT = create_research_agent()
writer_model, WRITER_SYSTEM_PROMPT = create_writer_agent()

def _cleanup():
    for m in [supervisor_model, research_model, writer_model]:
        try:
            if hasattr(m, '_client') and m._client and hasattr(m._client, '_client'):
                m._client._client.close()
        except Exception:
            pass
atexit.register(_cleanup)

# ── State（next_agent 替代字符串路由）─────────────────────────
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str  # "research" | "writer" | "finish"


# ── System Prompts（使用 SystemMessage）───────────────────────
S_MSG = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
R_MSG = SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)
W_MSG = SystemMessage(content=WRITER_SYSTEM_PROMPT)

DATE_REMINDER_TAG = "<system-reminder>"
ROUTE_PATTERN = re.compile(r"ROUTE:\s*(research_agent|writer_agent|FINISH)", re.IGNORECASE)


def _inject_date(messages: list) -> list:
    for msg in messages:
        c = msg.content if hasattr(msg, "content") else ""
        if isinstance(c, str) and DATE_REMINDER_TAG in c:
            return list(messages)
    current_date = datetime.now().strftime("%Y-%m-%d %A")
    reminder = f"<system-reminder>\nCurrent date: {current_date}\n</system-reminder>\n\n"
    result, done = [], False
    for msg in messages:
        if (not done and isinstance(msg, HumanMessage)
                and not isinstance(msg, SystemMessage)
                and hasattr(msg, "content")):
            content = msg.content or ""
            if not content.startswith("[Reflection"):
                result.append(HumanMessage(content=reminder + content))
                done = True
                continue
        result.append(msg)
    return result


def _parse_route(content: str) -> str:
    """从 supervisor 输出中提取路由目标。"""
    m = ROUTE_PATTERN.search(content)
    if not m:
        return "finish"
    route = m.group(1).upper()
    if route == "RESEARCH_AGENT":
        return "research"
    elif route == "WRITER_AGENT":
        return "writer"
    return "finish"


# ── Supervisor Node ──────────────────────────────────────────
def supervisor_node(state: MultiAgentState) -> dict:
    record_node("supervisor")
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) and m.content == SUPERVISOR_SYSTEM_PROMPT
               for m in messages):
        messages = [S_MSG] + list(messages)
    messages = _inject_date(messages)

    response = supervisor_model.invoke(messages)
    route = _parse_route(str(response.content))
    return {"messages": [response], "next_agent": route}


# ── Research Node ────────────────────────────────────────────
def research_agent_node(state: MultiAgentState) -> dict:
    record_node("research")
    messages = state["messages"]

    supervisor_instruction = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and "route: research_agent" in str(m.content).lower():
            supervisor_instruction = str(m.content)
            break

    research_messages = [R_MSG]
    if supervisor_instruction:
        research_messages.append(HumanMessage(
            content=f"请根据以下指令进行研究：\n\n{supervisor_instruction}\n\n{RESEARCHER_FIRST_STEP_PROMPT}"
        ))

    all_results = []
    for _ in range(3):
        response = research_model.invoke(research_messages)
        if not response.tool_calls:
            combined = "\n\n---\n\n".join(all_results) if all_results else "未找到相关信息。"
            return {
                "messages": [AIMessage(content=f"[Research Agent 完成]\n\n{combined}")],
                "next_agent": "finish",
            }

        research_messages.append(response)
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")
            try:
                if tool_name == "search_knowledge_base":
                    result = search_knowledge_base.invoke(tool_args)
                    all_results.append(f"【{tool_name}: {tool_args.get('query', '')}】\n{str(result)[:5000]}")
                    research_messages.append(ToolMessage(
                        content=f"结果已收集（{len(str(result))} 字符）",
                        tool_call_id=tool_call_id,
                    ))
            except Exception as e:
                research_messages.append(ToolMessage(
                    content=f"Error: {e}", tool_call_id=tool_call_id,
                ))

    combined = "\n\n---\n\n".join(all_results) if all_results else "未找到相关信息。"
    return {
        "messages": [AIMessage(content=f"[Research Agent 完成]\n\n{combined}")],
        "next_agent": "finish",
    }


# ── Writer Node ──────────────────────────────────────────────
def writer_agent_node(state: MultiAgentState) -> dict:
    record_node("writer")
    messages = state["messages"]

    findings = []
    for m in messages:
        if isinstance(m, AIMessage) and "[Research Agent 完成]" in str(m.content):
            findings.append(str(m.content))
        elif isinstance(m, ToolMessage):
            findings.append(str(m.content)[:3000])

    research_text = "\n\n---\n\n".join(findings) if findings else "无研究结果。"

    writer_messages = [
        W_MSG,
        HumanMessage(content=f"请根据以下研究材料撰写结构化报告：\n\n{research_text[:8000]}")
    ]
    response = writer_model.invoke(writer_messages)
    return {
        "messages": [AIMessage(content=f"[Writer Agent 完成]\n\n{response.content}")],
        "next_agent": "finish",
    }


# ── Routing（基于 state.next_agent 字段）──────────────────────
def route_supervisor(state: MultiAgentState) -> Literal["research", "writer", "__end__"]:
    target = state.get("next_agent", "finish")
    if target in ("research", "writer"):
        return target
    return "__end__"


# ── Build Graph ──────────────────────────────────────────────
workflow = StateGraph(MultiAgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research", research_agent_node)
workflow.add_node("writer", writer_agent_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges("supervisor", route_supervisor, {
    "research": "research", "writer": "writer", "__end__": END,
})
workflow.add_edge("research", "supervisor")
workflow.add_edge("writer", "supervisor")

agent = workflow.compile()
