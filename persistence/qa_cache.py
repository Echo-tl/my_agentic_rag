"""
热点问答缓存。

正确性优先于命中率：任何一处不确定（Redis 读不到窗口、意图低置信、澄清、联网搜索）
一律不缓存。宁可少缓存，不可答错。

多轮指代问题（「和 ReAct 有什么区别」）直接按问题哈希缓存必然串答案——不同会话里
同一句话的所指对象不同。这里的方案是上下文指纹：
- 问题**自足**（点名了完整对象，如「AutoGen 和 ReAct 有什么区别」）→ fp='self'，跨会话共享
  注意对比句要求**两个**实体才算自足：「和 ReAct 有什么区别」只点名一个，另一方来自
  上下文，仍按不自足处理
- 问题**不自足**（含指代/省略）→ fp='ctx:<hash>'
  hash 取自最近 K 轮的**用户问题**（不是 assistant 回答）
  · 用户输入完全确定性；LLM 输出即使 temperature=0 也可能随版本漂移，作 key 会抖动
  · 窗口本来就要读，零额外成本
- 首轮就问指代问题 → 无上下文可依据 → fp=None → 禁止缓存

连带的好处：Redis 挂掉 → 窗口读不到 → 非自足问题自动 fp=None → 缓存自然全禁用，
不需要额外开关，正确性自动兜底。
"""

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from config import config
from persistence.redis_client import get_redis

logger = logging.getLogger("agentic_rag")

# 归一化算法版本。改算法时必须同步改这个常量，老 key 自然失效。
NORMS_VER = "n1"

_KB_VERSION_KEY = "kbver"

# 礼貌前缀：不影响语义，去掉可提升中文问句命中率
_POLITE_PREFIX = re.compile(r"^(请问|请|帮我|麻烦|能不能|可以|你好[，,]?|hi[，,]?|hello[，,]?)\s*")
_TRAILING_PUNCT = re.compile(r"[。！？!?.,，;；:：、\s]+$")
_CJK = r"一-鿿"

# ── 指代 / 省略线索：命中即判定「不自足」────────────────────
_ANAPHORA_ZH = re.compile(r"这|那|它|他|她|其|该|此|上述|前面|刚才|上面|同样|也是|还有|再|继续|展开")
_ANAPHORA_EN = re.compile(
    r"\b(and|what about|how about|why|then|it|its|they|them|their|this|that|those|these|"
    r"the same|the former|the latter|more|also|too)\b"
)
# 对比线索：命中但**无实体** → 缺主语，不自足（"和ReAct有什么区别"）
_CONTRAST = re.compile(
    r"有什么区别|有何区别|区别是什么|差异|对比|比较|优劣|"
    r"\b(vs|versus|compare|comparison|differ|difference)\b"
)


# ============================================================
# Key 归一化
# ============================================================
def normalize(question: str) -> str:
    """问题归一化：NFKC → casefold → 去礼貌前缀 → 折叠空白 → 去尾部标点。

    中文场景下刻意去掉 CJK 两侧的空格：'ReAct 和 AutoGen' 与 'ReAct和AutoGen'
    必须落到同一个 key。但**不**删除英文词间空格——'what is react' 与 'whatisreact'
    应当是两个不同的问题，全部删空格会引入无谓的碰撞风险。
    """
    s = unicodedata.normalize("NFKC", question or "")
    s = s.casefold()
    # 必须先折叠并去掉首尾空白，否则 '　请问…' 这类带前导空格的串匹配不到礼貌前缀
    s = re.sub(r"\s+", " ", s).strip()
    s = _POLITE_PREFIX.sub("", s)
    # 去掉 CJK 字符两侧的空格
    s = re.sub(rf"(?<=[{_CJK}]) | (?=[{_CJK}])", "", s)
    s = _TRAILING_PUNCT.sub("", s)
    return s


def question_hash(question: str) -> str:
    """归一化问题的 sha1（40 位），与 qa_records.question_hash 同一算法。"""
    return hashlib.sha1(normalize(question).encode("utf-8")).hexdigest()


# ============================================================
# 上下文指纹
# ============================================================
def is_self_contained(question: str, known_papers: Optional[list[str]] = None) -> bool:
    """确定性判定问题是否自足（脱离上下文也能正确回答）。

    判定不确定时一律返回 False——宁可不缓存，也不能答错。
    """
    if known_papers is None:
        from rag.intent import KNOWN_PAPERS
        known_papers = KNOWN_PAPERS

    nq = normalize(question)
    if not nq:
        return False

    entities = [p for p in known_papers if p.casefold() in nq]

    # 1) 对比句但实体不足两个 → 被对比的那一方来自上下文，不自足。
    #    这条必须排在「有实体就自足」前面："和ReAct有什么区别" 只点名了 ReAct，
    #    主语是省略的；而 "AutoGen 和 ReAct 有什么区别" 才是完整自足的。
    if _CONTRAST.search(nq) and len(entities) < 2:
        return False

    # 2) 有明确实体且不是缺对象的对比 → 自足
    if entities:
        return True

    # 3) 指代 / 省略线索 → 不自足
    if _ANAPHORA_ZH.search(nq) or _ANAPHORA_EN.search(nq):
        return False

    # 4) 其余对比句（"对比一下"）→ 缺主语
    if _CONTRAST.search(nq):
        return False

    # 5) 长度门槛：中文信息密度高，门槛低；纯英文门槛高
    floor = 8 if re.search(rf"[{_CJK}]", nq) else 20
    return len(nq) >= floor


def build_fp(question: str, window: Optional[list[dict]] = None,
             ctx_turns: Optional[int] = None) -> Optional[str]:
    """返回上下文指纹：'self' 或 'ctx:<16hex>'；返回 None 表示**禁止缓存**。

    Args:
        question: 用户本轮问题
        window: 会话消息窗口（时间正序），每项 {"role": ..., "content": ...}
    """
    if is_self_contained(question):
        return "self"

    if ctx_turns is None:
        ctx_turns = config.redis.cache_fp_ctx_turns

    nq = normalize(question)
    prior_users = [m.get("content", "") for m in (window or []) if m.get("role") == "user"]

    # 剔除与本次**完全相同**的历史提问。指代句本身不携带「它在指谁」的信息，
    # 所以重复问同一句时的所指必然和上一次相同。不剔除的话：
    #   第 N 轮 「那它的缺点呢」→ 上下文 [q1]      → fp_A
    #   第 N+1 轮 同一句        → 上下文 [q1, 那它的缺点呢] → fp_B ≠ fp_A
    # 用户刷新页面重问一次就必然 MISS，多轮缓存的收益被吃光。
    prior_users = [u for u in prior_users if normalize(u) != nq]
    if not prior_users:
        return None  # 首轮就问指代问题 → 无上下文可依据，不缓存

    raw = "\x1f".join(normalize(u) for u in prior_users[-ctx_turns:])
    return "ctx:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 知识库版本（缓存失效的唯一开关）
# ============================================================
def get_kb_version() -> int:
    """读当前知识库版本。

    **绝不做进程内缓存**：多 worker / 多容器部署时，只有触发摄取的那个进程会 bump，
    一旦本地缓存了版本号，其它进程的缓存就永不失效。
    附带的好处：Redis 重启丢版本号 → 缓存 key 里的版本号一起归零 → 旧缓存也丢了，语义自洽。
    """
    r = get_redis()
    if r is None:
        return 0
    try:
        v = r.get(f"{config.redis.key_prefix}:{_KB_VERSION_KEY}")
        return int(v) if v is not None else 0
    except Exception:
        return 0


def bump_kb_version(added=None, changed=None, removed=None, total: int = 0) -> int:
    """知识库变更后自增版本号，让所有老缓存 key 不可达并自然过期。

    不做 SCAN+DEL：阻塞且低效。版本号内嵌在 key 里，老 key 靠 TTL 自行清理。
    """
    r = get_redis()
    if r is None:
        return 0
    try:
        prefix = config.redis.key_prefix
        pipe = r.pipeline()
        pipe.incr(f"{prefix}:{_KB_VERSION_KEY}")
        meta = {
            "changed_files": json.dumps(
                {"added": list(added or []), "changed": list(changed or []),
                 "removed": list(removed or [])}, ensure_ascii=False),
            "doc_count": str(total),
        }
        pipe.hset(f"{prefix}:{_KB_VERSION_KEY}:meta", mapping=meta)
        ver, _ = pipe.execute()
        logger.info(f"[cache] kb_version → {ver}（缓存已失效）")
        return int(ver)
    except Exception as e:
        logger.warning(f"[cache] kb_version bump 失败（缓存失效降级）: {e}")
        return 0


# ============================================================
# 缓存读写
# ============================================================
def _fp_key(fp: str) -> str:
    """指纹里的 ':' 会破坏 key 分段，替换掉。"""
    return fp.replace(":", "_")


def make_key(fp: str, question: str, kb_version: Optional[int] = None) -> str:
    if kb_version is None:
        kb_version = get_kb_version()
    return (f"{config.redis.key_prefix}:qa:v{kb_version}:{NORMS_VER}"
            f":{_fp_key(fp)}:{question_hash(question)}")


def make_lock_key(fp: str, question: str, kb_version: Optional[int] = None) -> str:
    if kb_version is None:
        kb_version = get_kb_version()
    return (f"{config.redis.key_prefix}:lock:qa:v{kb_version}"
            f":{_fp_key(fp)}:{question_hash(question)}")


@dataclass
class CacheLookup:
    """一次缓存查询的结果。

    key 为 None 表示本次请求**禁止缓存**（要么配置关闭、要么 Redis 不可用、
    要么问题不自足且无上下文依据）——此时既不读也不写。
    """

    key: Optional[str] = None
    answer: Optional[str] = None
    payload: Optional[dict] = None
    locked: bool = False          # 是否抢到 single-flight 锁（只有持有者才写缓存）
    fp: Optional[str] = None
    kb_version: int = 0
    question: str = ""            # 原问题，供释放锁时重建 lock key

    @property
    def hit(self) -> bool:
        return self.answer is not None


def lookup(question: str, window: Optional[list[dict]] = None) -> CacheLookup:
    """查缓存；未命中则登记热度并尝试抢占 single-flight 锁。"""
    if not config.redis.cache_enabled:
        return CacheLookup()

    r = get_redis()
    if r is None:
        return CacheLookup()

    try:
        kb_version = get_kb_version()
        fp = build_fp(question, window)
        if fp is None:
            return CacheLookup(kb_version=kb_version)

        key = make_key(fp, question, kb_version)
        raw = r.get(key)
        if raw:
            payload = json.loads(raw)
            return CacheLookup(key=key, answer=payload.get("answer"),
                               payload=payload, fp=fp, kb_version=kb_version)

        # 未命中：登记热度（用于热点门槛与热门问题排行）
        field = f"{_fp_key(fp)}:{question_hash(question)}"
        try:
            pipe = r.pipeline()
            pipe.hincrby(f"{config.redis.key_prefix}:qafreq:v{kb_version}", field, 1)
            pipe.zincrby(f"{config.redis.key_prefix}:qahot:v{kb_version}", 1, field)
            pipe.execute()
        except Exception:
            pass  # 热度统计失败不影响主流程

        # single-flight：抢到锁的请求负责回填缓存，避免并发同问重复写
        locked = _acquire_lock(r, make_lock_key(fp, question, kb_version))

        return CacheLookup(key=key, fp=fp, locked=locked, kb_version=kb_version,
                           question=question)
    except Exception as e:
        logger.debug(f"[cache] lookup 失败，按未命中处理: {e}")
        return CacheLookup()


def _acquire_lock(r, lock_key: str) -> bool:
    try:
        return bool(r.set(lock_key, "1", nx=True, px=config.redis.cache_lock_ttl_ms))
    except Exception:
        return False


def wait_for_peer(key: str, timeout: float = 2.0, interval: float = 0.05) -> Optional[str]:
    """等并发邻居回填缓存（仅用于非流式端点）。

    SSE 端点**不要**调这个——同步生成器里 sleep 会白占一个 threadpool 线程，
    还让用户看到卡顿；那边直接用「抢不到锁就自己算」的策略。
    """
    import time

    r = get_redis()
    if r is None or not key:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = r.get(key)
            if raw:
                return json.loads(raw).get("answer")
        except Exception:
            return None
        time.sleep(interval)
    return None


def is_cacheable(answer: str, intent: Optional[dict], reflection_count: int = 0) -> tuple:
    """判定答案是否值得缓存。返回 (是否可缓存, 原因)。"""
    task_type = (intent or {}).get("task_type")
    confidence = (intent or {}).get("confidence", 1.0) or 0.0

    if not answer or len(answer) < config.redis.cache_min_answer_chars:
        return False, "answer_too_short"
    if len(answer) > 8000:
        return False, "answer_too_long"       # 超大条目占 Redis 内存不划算
    if task_type == "clarification":
        return False, "clarification"          # 模板化澄清提示，缓存会掩盖意图变化
    if task_type == "web_search":
        return False, "time_sensitive"         # 联网结果强时效
    if confidence < 0.6:
        return False, "low_confidence"         # 意图都没把握，答案可复现性差
    if reflection_count > 0:
        return False, "reflection_retried"     # 首答被自己判 FAIL，质量不可信
    if "No relevant documents found" in answer:
        return False, "retrieval_miss"         # 知识库兜底话术，稍后重试可能命中
    return True, "ok"


def _release_lock(result: CacheLookup):
    """释放 single-flight 锁。

    必须在每次计算结束时释放，不能只靠 TTL 过期：锁 TTL 是 30s，而相邻两次相同提问
    的间隔常常短于 30s，不主动释放的话第二次请求抢不到锁 → locked=False →
    永远不会回填缓存，热点门槛成了死结。
    """
    r = get_redis()
    if r is None:
        return
    try:
        r.delete(make_lock_key(result.fp, result.question, result.kb_version))
    except Exception:
        pass


def observe_and_store(result: CacheLookup, question: str, answer: str,
                      intent: Optional[dict] = None, reflection_count: int = 0,
                      citations: Optional[list] = None) -> bool:
    """回填缓存。只有抢到锁、通过白名单、且达到热度门槛才真正写入。

    调用方必须传**整图结束后的最终答案**，不能用流式 token 累加值——
    反思判 FAIL 时前端会收到 reset 事件清空重答，累加值会是被判 FAIL 的那一版。
    """
    if result.key is None or not result.locked:
        return False

    try:
        ok, reason = is_cacheable(answer, intent, reflection_count)
        if not ok:
            logger.debug(f"[cache] 不缓存（{reason}）")
            return False

        r = get_redis()
        if r is None:
            return False

        # 热点门槛：读实时计数，避免用 lookup 时的旧值
        field = f"{_fp_key(result.fp)}:{question_hash(question)}"
        raw_hits = r.hget(f"{config.redis.key_prefix}:qafreq:v{result.kb_version}", field)
        hits = int(raw_hits) if raw_hits else 1
        if hits < config.redis.cache_min_hits:
            logger.debug(f"[cache] 热度不足（{hits}/{config.redis.cache_min_hits}），暂不缓存")
            return False

        payload = {
            "answer": answer,
            "fp": result.fp,
            "qhash": question_hash(question),
            "kb_version": result.kb_version,
            "intent": intent or {},
            "citations": citations or [],
            "hits": hits,
        }
        r.set(result.key, json.dumps(payload, ensure_ascii=False), ex=config.redis.cache_ttl)
        logger.info(f"[cache] 已缓存问答（热度 {hits}，TTL {config.redis.cache_ttl}s）")
        return True
    except Exception as e:
        logger.warning(f"[cache] 写入失败（降级）: {e}")
        return False
    finally:
        _release_lock(result)


def stats() -> dict:
    """缓存运行状况，供 /cache/stats 暴露。"""
    r = get_redis()
    if r is None:
        return {"enabled": config.redis.cache_enabled, "redis_available": False}

    prefix = config.redis.key_prefix
    kb_version = get_kb_version()
    out = {
        "enabled": config.redis.cache_enabled,
        "redis_available": True,
        "kb_version": kb_version,
        "cache_ttl": config.redis.cache_ttl,
        "min_hits": config.redis.cache_min_hits,
    }
    try:
        _, keys = r.scan(0, match=f"{prefix}:qa:v{kb_version}:*", count=1000)
        out["cached_questions"] = len(keys)
        out["hot_questions"] = [
            {"key": m, "hits": int(s)}
            for s, m in r.zrevrange(f"{prefix}:qahot:v{kb_version}", 0, 9, withscores=True)
        ]
    except Exception as e:
        out["error"] = str(e)
    return out
