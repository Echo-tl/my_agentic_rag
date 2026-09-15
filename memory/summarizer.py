"""
滚动短期摘要：把被窗口挤出去的旧对话压成一段可复用的背景。

为什么需要它：裁剪掉的消息是真删（RemoveMessage 不可逆），如果什么都不留，
用户第三轮问「刚才那篇论文的第三点再说说」就彻底失忆了。摘要保留**语义**，
裁剪回收**空间**，两者必须配套。

**失败返回空串，而不是旧摘要**：摘要走 LLM，而 LLM 会超时、会返回空。
调用方（memory_node）靠「返回值是否为空」判断该不该裁剪——因为裁剪是**不可逆**的，
「摘要没生成出来却把消息删了」等于彻底丢失那段对话。返回空串让这个失败可观测。
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import config

logger = logging.getLogger("agentic_rag")

SUMMARY_PROMPT = """你在维护一段多轮技术对话的滚动摘要。下面已有「已有摘要」，以及若干条被窗口挤出去的新对话。

请把两者合并成**一份**新的摘要，要求：
1. 保留具体实体：论文名、方法名、术语、数字。这些是后续追问的关键，绝不可以用「某篇论文」代替
2. 保留用户已表达的偏好与约束（关注点、语言、深度、已排除的方向）
3. 按时间顺序组织；已被后续对话推翻的信息直接删掉，不要保留互相矛盾的两版
4. 不要写「用户问了…助手答了…」这类过程描述，直接写**结论与事实**
5. 纯文本，不用 Markdown 标题；中文回答；控制在 {max_chars} 字以内

只输出摘要正文，不要任何前言或解释。"""


_model = None


def _get_model():
    """独立于 graph 的 LLM 实例——graph import 本模块，反向 import 会成环。

    懒创建：不触发摘要的话不该额外建一个客户端。
    """
    global _model
    if _model is None:
        _model = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url,
            temperature=0.0,          # 摘要是压缩不是创作，要可复现
            num_ctx=config.llm.context_window,
            num_predict=min(1024, config.llm.max_tokens),
            client_kwargs={"timeout": config.llm.request_timeout},
        )
    return _model


def _render(messages, per_msg_limit: int = 800) -> str:
    """把消息列表渲染成待压缩的文本。单条截断，防一条长回答吃满 num_ctx。"""
    lines = []
    for m in messages:
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        role = "用户" if isinstance(m, HumanMessage) and not _is_system(m) else "助手"
        text = content.strip()
        if len(text) > per_msg_limit:
            text = text[:per_msg_limit] + "…（截断）"
        lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


def _is_system(m) -> bool:
    from langchain_core.messages import SystemMessage

    return isinstance(m, SystemMessage)


def summarize(messages, prev_summary: str = "") -> str:
    """把 older messages 压进 prev_summary，返回新摘要。

    失败 / 无可用内容 → 返回 **空串**（绝不抛异常）。调用方据此决定不裁剪。
    """
    if not messages:
        return ""

    body = _render(messages)

    # 反思反馈是 Agent 的自我纠错噪声，对「用户聊过什么」没有信息量
    body = "\n".join(
        line for line in body.split("\n")
        if not line.startswith("[助手] [Reflection")
    ).strip()

    if not body:
        return ""

    max_chars = config.redis.summary_max_chars
    prompt = (
        f"{SUMMARY_PROMPT.format(max_chars=max_chars)}\n\n"
        f"## 已有摘要\n{prev_summary or '（无）'}\n\n"
        f"## 新增对话\n{body}\n\n"
        f"## 合并后的摘要"
    )

    try:
        resp = _get_model().invoke([HumanMessage(content=prompt)])
        summary = str(resp.content or "").strip()
    except Exception as e:
        logger.warning(f"[summary] 生成失败，本轮不裁剪: {e}")
        return ""

    if not summary:
        logger.warning("[summary] LLM 返回空摘要，本轮不裁剪")
        return ""

    # 硬截断兜底：提示词里的字数要求不保证被遵守
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "…"

    return summary


def render_for_prompt(summary: str) -> str:
    """把摘要包装成注入 agent 的 SystemMessage 内容。"""
    return (
        "[历史对话摘要]\n"
        f"{summary}\n\n"
        "以上是本次会话较早内容的压缩摘要，用于理解用户提到的指代与已确认的结论；"
        "如与最近几轮消息冲突，以最近的消息为准。"
    )


def reset():
    """丢弃缓存的 LLM 实例（测试用）。"""
    global _model
    _model = None
