"""
Research Agent: 信息检索。
"""

from langchain_ollama import ChatOllama
from config import config
from prompts.researcher import RESEARCHER_SYSTEM_PROMPT
from tools.rag.llamaindex_tool import search_knowledge_base


def create_research_agent():
    """创建 Research Agent（只绑定知识库检索工具，不绑定 web_search 防止编造）。"""
    model = ChatOllama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=0.0,
        num_ctx=config.llm.context_window,
        num_predict=config.llm.max_tokens,
        client_kwargs={"timeout": config.llm.request_timeout},
    )
    return model.bind_tools([search_knowledge_base]), RESEARCHER_SYSTEM_PROMPT
