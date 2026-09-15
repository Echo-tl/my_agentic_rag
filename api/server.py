"""
FastAPI 服务 —— 对外暴露 Agentic RAG 查询接口。
"""

import sys
import json
import uuid
import os
import re
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# 必须在导入 agent 之前设置，否则 Qdrant 连接 502
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

# 支持 `python api/server.py` 直接运行：把项目根目录加入 sys.path，
# 否则 Python 只把脚本所在目录（api/）加入 path，找不到 workflows 等顶层模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from config import config
from workflows.graph import agent
from evaluation.rag_eval import evaluate_retrieval
from observability.tracing import trace_query, get_recent_traces, clear_traces

logger = logging.getLogger("agentic_rag")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时挂上后台写队列的处理器，退出时尽量把队列冲干净。

    Redis/MySQL 未启用时这两个调用都是 no-op。
    """
    from persistence import ensure_ready

    ensure_ready()
    yield

    try:
        from persistence import writer

        writer.shutdown(wait=True, timeout=5)
    except Exception:
        pass


app = FastAPI(title="Agentic RAG API", version="1.1.0", lifespan=_lifespan)

# 网页聊天前端（单文件，无构建依赖）
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """网页聊天界面。"""
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── 请求/响应模型 ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    # 只做透传：不传就是浏览器的游客身份，前端零改动
    question: str
    session_id: str | None = None
    user_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    cached: bool = False


class EvalResponse(BaseModel):
    top_k_5: dict
    top_k_10: dict
    top_k_20: dict


# ============================================================
# 一轮请求的公共前置 / 收尾
#
# 抽出来是因为 /query 与 /query/stream 的差异只在「怎么产出答案」，
# 会话登记、缓存查、缓存写、落库这四件事必须完全一致——两边各写一遍必然会漂移。
# ============================================================
# 工具输出里的引用格式：`[来源: <file> | 第<N>页 | 相关度: <x>]`
_CITATION_RE = re.compile(r"\[来源: (.+?) \| 第(\d+)页 \| 相关度: ([\d.]+)\]")


class TurnContext:
    """一轮请求的可变上下文，在前后置之间传递。"""

    def __init__(self, sid: str, question: str, user_id, window: list, cache):
        self.sid = sid
        self.question = question
        self.user_id = user_id
        self.window = window
        self.cache = cache
        self.started = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


def _prepare_turn(req: QueryRequest, allow_wait: bool = False) -> TurnContext:
    """会话登记 + 缓存查找。**必须在写本轮问答之前调用**。"""
    sid = req.session_id or str(uuid.uuid4())[:8]
    question = req.question or ""

    # 窗口必须在 append_turn 之前读：否则指纹会包含本轮问题本身，同一句追问
    # 永远算出不同指纹 → 多轮缓存命中率归零
    from persistence import qa_cache, session_store, user_store

    window = session_store.get_window(sid)

    user_id = user_store.resolve(req.user_id)
    try:
        session_store.touch(sid, user_id)
        from persistence import repo

        repo.submit_session(sid, user_id)
    except Exception as e:
        logger.debug(f"[api] 会话登记降级: {e}")

    look = qa_cache.lookup(question, window)

    # 非流式端点可以短暂等并发邻居回填（SSE 端点不能——同步生成器里 sleep
    # 会白占一个 threadpool 线程，用户那边表现为卡顿）
    if allow_wait and look.key and not look.hit and not look.locked:
        peer = qa_cache.wait_for_peer(look.key, timeout=2.0)
        if peer:
            look.answer = peer

    return TurnContext(sid, question, user_id, window, look)


def _extract_citations(messages) -> list:
    """从工具输出里解析引用来源，供 qa_records.citations 归档。"""
    out, seen = [], set()
    for m in messages or []:
        if not isinstance(m, ToolMessage):
            continue
        for file_name, page, score in _CITATION_RE.findall(str(m.content or "")):
            key = (file_name, page)
            if key in seen:
                continue
            seen.add(key)
            out.append({"file_name": file_name, "page": int(page),
                        "score": float(score)})
    return out


def _finalize(turn: TurnContext, answer: str, *, intent=None, citations=None,
              tool_calls=None, reflection_count: int = 0, cached: bool = False,
              trace_id: str = None, error: str = None, cacheable: bool = True,
              model: str = None):
    """缓存回填 + 会话窗口 + 落库。两个端点共用。"""
    answer = answer or ""

    try:
        from persistence import qa_cache, session_store

        # 回填缓存：只有抢到锁、答案合格、且达到热度门槛才真的写进去
        if cacheable and not cached:
            qa_cache.observe_and_store(turn.cache, turn.question, answer,
                                       intent, reflection_count, citations)

        # 会话窗口：**缓存命中时也要写**，否则下一轮的上下文指纹会缺一条，
        # 同一段对话里"命中过缓存的那一轮"前后的追问会算出不同指纹
        session_store.append_turn(turn.sid, turn.question, answer)
    except Exception as e:
        logger.debug(f"[api] 缓存/会话收尾降级: {e}")

    try:
        from persistence import repo

        repo.submit_turn(
            turn.sid, turn.question, answer,
            user_id=turn.user_id,
            context_fp=turn.cache.fp,
            task_type=(intent or {}).get("task_type"),
            intent=intent,
            citations=citations or [],
            tool_calls=tool_calls or [],
            cache_key=turn.cache.key,
            cached=cached,
            kb_version=turn.cache.kb_version,
            model=model,
            latency_ms=turn.elapsed_ms,
            reflection_count=reflection_count,
            trace_id=trace_id,
            error=error,
        )
    except Exception as e:
        logger.debug(f"[api] 问答落库降级: {e}")


def _final_answer(messages) -> str:
    """取最后一条非工具调用的 AI 回复（澄清等非流式路径也能拿到）。"""
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content
    return ""


# ── API 端点 ──────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, response: Response):
    """查询知识库。支持多轮对话（传入 session_id 保持上下文）。"""
    turn = _prepare_turn(req, allow_wait=True)

    if turn.cache.hit:
        response.headers["X-Cache"] = "HIT"
        _finalize(turn, turn.cache.answer, cached=True,
                  intent=(turn.cache.payload or {}).get("intent"),
                  citations=(turn.cache.payload or {}).get("citations"))
        return QueryResponse(answer=turn.cache.answer, session_id=turn.sid, cached=True)

    response.headers["X-Cache"] = "MISS"
    try:
        # 用 trace_query 包裹真实查询：结束后写入 execution trace，供 /traces 查询
        # 注意：不要把 summary 放进输入——LangGraph 会用它覆盖 checkpoint 里的累积摘要
        with trace_query(req.question) as tr:
            tr.session_id = turn.sid
            result = agent.invoke(
                {"messages": [("user", req.question)], "reflection_count": 0,
                 "intent": {}, "current_question": req.question},
                config={"configurable": {"thread_id": turn.sid}},
            )
        messages = result["messages"]
        answer = _final_answer(messages)

        _finalize(turn, answer, intent=result.get("intent"),
                  citations=_extract_citations(messages),
                  tool_calls=tr.tool_calls,
                  reflection_count=result.get("reflection_count", 0),
                  trace_id=tr.trace_id, model=config.llm.model)
        return QueryResponse(answer=answer or "无响应", session_id=turn.sid)
    except Exception as e:
        _finalize(turn, "", error=str(e), cacheable=False)
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
      done    -> 结束，携带 session_id / answer / cached

    **本函数体（而非生成器内）承担全部会话与缓存的前置工作**：这样 X-Cache 头在
    首字节之前就能确定，也不会在 threadpool 迭代生成器时多占一次线程。
    """
    turn = _prepare_turn(req, allow_wait=False)

    def _event_gen():
        first_answer = True
        streaming = False
        last_status = None
        final_state = None
        trace = None

        def _status(msg: str):
            nonlocal last_status
            if msg != last_status:
                last_status = msg
                return _sse({"type": "status", "message": msg})
            return None

        # ── 命中缓存：不触发 LLM，直接伪流式吐出去 ──
        if turn.cache.hit:
            _finalize(turn, turn.cache.answer, cached=True,
                      intent=(turn.cache.payload or {}).get("intent"),
                      citations=(turn.cache.payload or {}).get("citations"))
            yield _sse({"type": "status", "message": "命中热点缓存，秒回"})
            yield _sse({"type": "token", "token": turn.cache.answer})
            yield _sse({"type": "done", "session_id": turn.sid,
                        "answer": turn.cache.answer, "cached": True})
            return

        yield _sse({"type": "status", "message": "正在理解意图…"})
        error = None
        try:
            with trace_query(req.question) as tr:
                trace = tr
                tr.session_id = turn.sid
                # 同时流式返回 message token 与最终状态（用于非流式路径的完整答案）
                # 不要传 summary：它会覆盖 checkpoint 里累积的摘要
                for mode, data in agent.stream(
                    {"messages": [("user", req.question)], "reflection_count": 0,
                     "intent": {}, "current_question": req.question},
                    config={"configurable": {"thread_id": turn.sid}},
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
            error = str(e)
            yield _sse({"type": "error", "message": error})

        # 从最终状态提取完整答案（澄清等非流式路径兜底）。
        # 必须用 state 里的最终答案而不是累加的 token：反思判 FAIL 时前端已收到 reset，
        # 累加值是那一版被判 FAIL 的文本。
        messages = (final_state or {}).get("messages", [])
        answer = _final_answer(messages)
        if error and not answer:
            answer = ""

        _finalize(turn, answer,
                  intent=(final_state or {}).get("intent"),
                  citations=_extract_citations(messages),
                  tool_calls=(trace.tool_calls if trace else None),
                  reflection_count=(final_state or {}).get("reflection_count", 0),
                  trace_id=(trace.trace_id if trace else None),
                  model=config.llm.model, error=error, cacheable=not error)

        yield _sse({"type": "done", "session_id": turn.sid, "answer": answer,
                    "cached": False})

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Cache": "HIT" if turn.cache.hit else "MISS",
        },
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
    """存活探针。顺带报告存储层状态，但不因存储层挂了而返回非 200。

    存储层是**可选增强**：Redis/MySQL 全挂时服务依然能回答问题，只是没有多轮记忆、
    没有缓存、没有归档。把它做成硬依赖会让这个特性得不偿失。
    """
    from persistence import redis_client, session_store
    from memory.checkpoint import backend as checkpoint_backend

    out = {"status": "ok"}
    try:
        import persistence.db as db

        out["redis"] = redis_client.is_available()
        out["mysql"] = db.is_available()
        out["mysql_dropped_writes"] = _dropped_writes()
        out["checkpoint"] = checkpoint_backend()
        out["active_sessions"] = len(session_store.list_sessions())
    except Exception as e:
        out["storage_error"] = str(e)
    return out


def _dropped_writes() -> int:
    try:
        from persistence.writer import dropped_count

        return dropped_count()
    except Exception:
        return 0


@app.get("/cache/stats")
def cache_stats():
    """热点缓存运行状况：命中数、热门问题排行、kb_version。"""
    from persistence.qa_cache import stats

    return stats()


@app.get("/sessions/{session_id}/qa")
def session_qa(session_id: str, limit: int = 50, offset: int = 0):
    """某个会话的历史问答（来自 MySQL 归档，含被裁剪掉的轮次）。"""
    from persistence import repo

    return {"session_id": session_id, "items": repo.list_session_qa(session_id, limit, offset)}


@app.get("/users/{user_id}/sessions")
def user_sessions(user_id: int, limit: int = 50):
    """某个用户的所有会话。"""
    from persistence import repo, user_store

    return {
        "user_id": user_id,
        "user": user_store.get_info(user_id),
        "items": repo.list_user_sessions(user_id, limit),
    }


@app.get("/documents")
def documents(status: str | None = None, limit: int = 200):
    """文档元数据目录。"""
    from persistence.doc_registry import list_documents

    return {"items": list_documents(status=status, limit=limit)}


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
