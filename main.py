"""
Agentic RAG 入口 —— 支持单次查询和交互式多轮对话。
"""

import sys
import uuid
from workflows.graph import agent
from observability.tracing import trace_query

# 生成唯一会话 ID（用于 Memory 中区分不同对话）
SESSION_ID = str(uuid.uuid4())[:8]


def ask(question: str) -> str:
    """单次查询。Memory 会自动记住上下文。每次查询都会写入一条 execution trace。"""
    with trace_query(question):
        result = agent.invoke(
            {"messages": [("user", question)], "reflection_count": 0},
            config={"configurable": {"thread_id": SESSION_ID}},
        )
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return "无响应"


def interactive():
    """交互式多轮对话模式。"""
    global SESSION_ID
    print("=" * 60)
    print(f"Agentic RAG 交互模式 (会话: {SESSION_ID})")
    print("输入问题开始对话，输入 /exit 退出，输入 /new 新会话")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出")
            break

        if not user_input:
            continue
        if user_input.lower() == "/exit":
            break
        if user_input.lower() == "/new":
            SESSION_ID = str(uuid.uuid4())[:8]
            print(f"新会话: {SESSION_ID}")
            continue

        print("\nAgent: ", end="", flush=True)
        answer = ask(user_input)
        print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行传参模式: python main.py "你的问题"
        print(ask(sys.argv[1]))
    else:
        # 交互模式
        interactive()
