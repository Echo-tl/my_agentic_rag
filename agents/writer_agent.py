"""
Writer Agent: 报告撰写。
"""

from langchain_ollama import ChatOllama
from config import config
from prompts.writer import WRITER_SYSTEM_PROMPT


def create_writer_agent():
    """创建 Writer Agent（无工具，纯写作）。"""
    model = ChatOllama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=0.3,  # 写作可稍有温度
        num_ctx=config.llm.context_window,
        num_predict=config.llm.max_tokens,
        client_kwargs={"timeout": config.llm.request_timeout},
    )
    return model, WRITER_SYSTEM_PROMPT
