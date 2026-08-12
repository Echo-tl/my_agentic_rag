"""
FastAPI 服务 —— 对外暴露 Agentic RAG 查询接口。
"""

import sys
import json
import uuid
import os
from pathlib import Path

# 必须在导入 agent 之前设置，否则 Qdrant 连接 502
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

# 支持 `python api/server.py` 直接运行：把项目根目录加入 sys.path，
# 否则 Python 只把脚本所在目录（api/）加入 path，找不到 workflows 等顶层模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk
from workflows.graph import agent
from evaluation.rag_eval import evaluate_retrieval
from observability.tracing import trace_query, get_recent_traces, clear_traces

app = FastAPI(title="Agentic RAG API", version="1.0.0")

# 网页聊天前端（单文件，无构建依赖）
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """网页聊天界面。"""
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── 请求/响应模型 ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None  # 可选，不传则自动生成新会话


class QueryResponse(BaseModel):
    answer: str
    session_id: str


class EvalResponse(BaseModel):
    top_k_5: dict
    top_k_10: dict
    top_k_20: dict


# ── API 端点 ──────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """查询知识库。支持多轮对话（传入 session_id 保持上下文）。"""
    sid = req.session_id or str(uuid.uuid4())[:8]
    try:
        # 用 trace_query 包裹真实查询：结束后写入 execution trace，供 /traces 查询
        with trace_query(req.question):
            result = agent.invoke(
                {"messages": [("user", req.question)], "reflection_count": 0},
                config={"configurable": {"thread_id": sid}},
            )
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                return QueryResponse(answer=msg.content, session_id=sid)
        return QueryResponse(answer="无响应", session_id=sid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """SSE 流式回答：意图/工具状态 + 答案 token 逐段推送。

    事件类型：
      status  -> 进度提示（正在理解意图/检索知识库/联网搜索）
      token   -> 回答文本增量
      reset   -> 反思重答，清空当前回答重新输出
      error   -> 出错
      done    -> 结束，携带 session_id
    """
    sid = req.session_id or str(uuid.uuid4())[:8]

    def _event_gen():
        first_answer = True
        streaming = False
        last_status = None
        final_state = None

        def _status(msg: str):
            nonlocal last_status
            if msg != last_status:
                last_status = msg
                return _sse({"type": "status", "message": msg})
            return None

        yield _sse({"type": "status", "message": "正在理解意图…"})
        try:
            with trace_query(req.question):
                # 同时流式返回 message token 与最终状态（用于非流式路径的完整答案）
                for mode, data in agent.stream(
                    {"messages": [("user", req.question)], "reflection_count": 0, "intent": {}},
                    config={"configurable": {"thread_id": sid}},
                    stream_mode=["messages", "values"],
                ):
                    if mode == "values":
                        final_state = data
                        continue

                    chunk, meta = data
                    node = meta.get("langgraph_node")

                    # 工具调用决策：提示正在调用什么工具
                    if node == "agent" and isinstance(chunk, AIMessageChunk) and chunk.tool_call_chunks:
                        names = {c.name for c in chunk.tool_call_chunks if getattr(c, "name", None)}
                        s = _status("正在联网搜索…" if "search_web" in names else "正在检索知识库…")
                        if s:
                            yield s
                        streaming = False
                        continue

                    # 答案 token：agent 节点、非工具调用、有内容
                    is_token = (
                        node == "agent"
                        and isinstance(chunk, AIMessageChunk)
                        and not getattr(chunk, "tool_call_chunks", None)
                        and chunk.content
                    )
                    if not is_token:
                        streaming = False
                        continue
                    if not streaming:
                        if not first_answer:
                            yield _sse({"type": "reset"})  # 反思重答，清空旧答案
                        first_answer = False
                        streaming = True
                    yield _sse({"type": "token", "token": chunk.content})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

        # 从最终状态提取完整答案（澄清等非流式路径兜底）
        answer = ""
        if final_state:
            for m in reversed(final_state.get("messages", [])):
                if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                    answer = m.content
                    break
        yield _sse({"type": "done", "session_id": sid, "answer": answer})

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/eval", response_model=EvalResponse)
def eval_retrieval():
    """运行检索质量评估。"""
    return EvalResponse(
        top_k_5=evaluate_retrieval(5),
        top_k_10=evaluate_retrieval(10),
        top_k_20=evaluate_retrieval(20),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/traces")
def traces(limit: int = 20):
    """查看最近 N 条 execution trace（耗时、工具调用链、节点流转）。"""
    return get_recent_traces(limit)


@app.delete("/traces")
def traces_clear():
    """清空 execution trace 记录。"""
    clear_traces()
    return {"cleared": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
