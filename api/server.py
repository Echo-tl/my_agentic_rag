"""
FastAPI 服务 —— 对外暴露 Agentic RAG 查询接口。
"""

import uuid
import os

# 必须在导入 agent 之前设置，否则 Qdrant 连接 502
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflows.graph import agent
from evaluation.rag_eval import evaluate_retrieval
from observability.tracing import trace_query, get_recent_traces, clear_traces

app = FastAPI(title="Agentic RAG API", version="1.0.0")


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
