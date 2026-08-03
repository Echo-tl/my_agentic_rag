"""
Supervisor Agent: 任务规划 + 分发。
"""

from langchain_ollama import ChatOllama
from config import config
from prompts.supervisor_agent import SUPERVISOR_SYSTEM_PROMPT


def create_supervisor():
    """创建 Supervisor Agent 模型（纯调度，无工具）。"""
    return ChatOllama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=0.0,
        num_ctx=config.llm.context_window,
        num_predict=config.llm.max_tokens,
        client_kwargs={"timeout": config.llm.request_timeout},
    ), SUPERVISOR_SYSTEM_PROMPT
