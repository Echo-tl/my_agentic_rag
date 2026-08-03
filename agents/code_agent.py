"""
Code Analysis Agent: 代码分析（预留）。

待实现: 代码审查、安全分析、依赖扫描等。
"""

CODE_AGENT_SYSTEM_PROMPT = """你是一个代码分析专家。你的职责是分析代码、识别安全问题、提供优化建议。

当前状态: 待实现。请在需要时激活此 Agent。
"""


def create_code_agent():
    """创建 Code Agent（预留）。"""
    from langchain_ollama import ChatOllama
    from config import config

    model = ChatOllama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=0.0,
        num_ctx=config.llm.context_window,
        client_kwargs={"timeout": config.llm.request_timeout},
    )
    return model, CODE_AGENT_SYSTEM_PROMPT
