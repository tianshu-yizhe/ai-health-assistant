"""
============================================================================
  AI 问答服务 — FastAPI 入口
  ============== ============ ============ ============ ============ ======
  启动: python main.py
  测试: http://127.0.0.1:8000
  文档: http://127.0.0.1:8000/docs
  测试: python tests/test_chat.py

  项目结构:
    .env         → API Key（不提交 Git）
    prompts/     → Prompt 配置（产品/运营都能改）
    src/         → 核心代码
    tests/       → 接口测试
============================================================================
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── 从 src 包导入各模块 ──
from src.llm_client import async_client, record_usage, get_usage_today  # 异步客户端 + 用量统计
from src.prompt_manager import (
    build_system_prompt,                     # Prompt 构建器（YAML → 模板渲染）
    get_model,                               # 模型选择器
)
from src.safety import (
    check_safety, clear_rate_limit, filter_output,  # 安全过滤
    find_stream_block, STREAM_TAIL,                 # 流式输出安全
)
from src.memory import get_history, add_message, clear_history  # 对话记忆
from src.rag_engine import build_knowledge_base, search  # RAG 检索引擎
from src.cache import get_cached, set_cached, cache_stats, r  # Redis 缓存
from src.logger import logger, LOG_DIR  # 日志
from src import metrics  # 运营指标埋点


# ── FastAPI 应用 ──
# 生产环境关闭 /docs、/redoc、/openapi.json，避免 API 结构公网暴露
app = FastAPI(title="AI服务", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════
#  IP 限流中间件 — 防外部刷接口烧钱
#  基于 Redis 固定窗口计数：重启不清零、将来多实例共享
#  ═══════════════════════════════════════════════════════
import time
import redis as redis_lib
from fastapi import Request

IP_LIMIT = 30        # 每个 IP 每分钟最多 30 次请求
IP_WINDOW = 60       # 时间窗口（秒）


@app.middleware("http")
async def rate_limit_by_ip(request: Request, call_next):
    # 管理后台接口不参与业务限流与计数（自身有 token 鉴权 + 登录锁定，避免观察者效应）
    if request.url.path.startswith("/api/admin"):
        return await call_next(request)
    # 只限流 API 接口，静态资源不限
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # nginx 反代后 client.host 恒为 127.0.0.1，取 X-Real-IP 才是真实用户 IP
    ip = request.headers.get("X-Real-IP") or request.client.host
    window = int(time.time() // IP_WINDOW)
    key = f"rl:ip:{ip}:{window}"
    try:
        count = r.incr(key)
        if count == 1:
            r.expire(key, IP_WINDOW + 10)
    except redis_lib.RedisError:
        # Redis 不可用 → 降级放行，不让限流组件拖垮主流程
        return await call_next(request)

    if count > IP_LIMIT:
        logger.warning("IP限流: %s 一分钟内 %d 次请求", ip, count)
        metrics.incr("rate_limited")
        return JSONResponse(status_code=429, content={"answer": "请求过于频繁，请稍后再试。"})

    metrics.incr("requests")
    return await call_next(request)

# ── 健康检查 ──
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ── 用量统计（旧接口，公网已被 nginx 拦截，保留本地查询用）──
@app.get("/api/stats")
async def stats():
    """今日 token 用量 + 缓存命中率"""
    data = get_usage_today()
    data["cache"] = cache_stats()
    return data


# ═══════════════════════════════════════════════════════
#  管理后台：登录鉴权 + 运营数据快照
#  ═══════════════════════════════════════════════════════
import os
import glob
import secrets

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_TOKEN_TTL = 86400        # 登录态 24 小时
LOGIN_MAX_FAIL = 5             # 连续失败 5 次锁定
LOGIN_LOCK_TTL = 900           # 锁 15 分钟


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def _check_admin(request: Request) -> bool:
    """校验 token：query 参数 ?token= 或 Authorization: Bearer"""
    token = request.query_params.get("token") or ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    if not token:
        return False
    try:
        return bool(metrics.r.exists(f"admin:token:{token}"))
    except Exception:
        return False


@app.post("/api/admin/login")
async def admin_login(req: AdminLoginRequest, request: Request):
    ip = request.headers.get("X-Real-IP") or request.client.host
    try:
        if metrics.r.exists(f"admin:lock:{ip}"):
            return JSONResponse(status_code=429, content={"msg": "尝试次数过多，请 15 分钟后再试"})
    except Exception:
        pass
    if ADMIN_PASSWORD and req.username == ADMIN_USER and req.password == ADMIN_PASSWORD:
        token = secrets.token_hex(16)
        try:
            metrics.r.set(f"admin:token:{token}", "1", ex=ADMIN_TOKEN_TTL)
            metrics.r.delete(f"admin:fails:{ip}")
        except Exception:
            pass
        logger.warning("后台登录成功: %s", ip)
        return {"token": token, "expires_in": ADMIN_TOKEN_TTL}
    metrics.incr("login_fails")
    try:
        fails = metrics.r.incr(f"admin:fails:{ip}")
        metrics.r.expire(f"admin:fails:{ip}", LOGIN_LOCK_TTL)
        if int(fails) >= LOGIN_MAX_FAIL:
            metrics.r.set(f"admin:lock:{ip}", "1", ex=LOGIN_LOCK_TTL)
    except Exception:
        pass
    logger.warning("后台登录失败: ip=%s user=%s", ip, req.username)
    return JSONResponse(status_code=401, content={"msg": "用户名或密码错误"})


@app.post("/api/admin/logout")
async def admin_logout(request: Request):
    token = request.query_params.get("token") or ""
    try:
        if token:
            metrics.r.delete(f"admin:token:{token}")
    except Exception:
        pass
    return {"msg": "已退出"}


def _recent_events(limit: int = 200) -> list:
    """从日志文件取最近的操作事件：提问、拦截、限流、错误、失败。
    按文件修改时间排序——午夜轮转后最新内容在 app.log，
    按文件名排会把它排到最后（名字最短），所以必须按 mtime。"""
    events = []
    files = glob.glob(os.path.join(LOG_DIR, "app.log*"))
    files.sort(key=os.path.getmtime, reverse=True)
    lines = []
    for fp in files[:3]:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines() + lines
        except OSError:
            continue
        if len(lines) > 800:
            break
    for line in lines:
        if any(k in line for k in ("提问", "拦截", "限流", "ERROR", "失败")):
            events.append(line.strip())
    # 倒序返回：最新的在最前面
    return list(reversed(events[-limit:]))


@app.get("/api/admin/snapshot")
async def admin_snapshot(request: Request):
    if not _check_admin(request):
        return JSONResponse(status_code=401, content={"msg": "未授权"})
    usage = get_usage_today()
    return {
        "status": "ok",
        "today": {
            "requests": metrics.get_today("requests"),
            "blocked": metrics.get_today("blocked"),
            "llm_errors": metrics.get_today("llm_errors"),
            "rate_limited": metrics.get_today("rate_limited"),
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "sessions": metrics.count_keys("mem:*"),
            "rl_ips": metrics.count_keys("rl:ip:*"),
        },
        "series": {
            "requests": metrics.get_series("requests", 7),
            "usage": metrics.get_usage_series(7),
        },
        "recent_events": _recent_events(200),
    }

# ── 启动时自动建知识库 ──
@app.on_event("startup")
async def startup():
    build_knowledge_base()


# ── 请求体 ──
class ChatRequest(BaseModel):
    question: str


class ChatWithMemoryRequest(BaseModel):
    question: str
    session_id: str = "default"      # 会话ID，区分不同用户/窗口


class ChatDebugRequest(BaseModel):
    question: str
    debug: bool = False              # true 时返回 sources/from_cache 调试字段


class ChatStreamRequest(BaseModel):
    question: str
    session_id: str | None = None    # 多轮对话会话ID；为空则无记忆（兼容旧调用）


# ═══════════════════════════════════════════════════════
#  POST /api/chat          — 无记忆（每问独立）
#  POST /api/chat/memory   — 带记忆（记住上下文）
#  ═══════════════════════════════════════════════════════
@app.post("/api/chat")
async def chat(req: ChatRequest):
    return await _do_chat(req.question)


@app.post("/api/chat/memory")
async def chat_with_memory(req: ChatWithMemoryRequest):
    return await _do_chat_with_memory(req.question, req.session_id)


# ═══════════════════════════════════════════════════════
#  POST /api/chat/rag  — RAG 知识库增强问答
#  ═══════════════════════════════════════════════════════
@app.post("/api/chat/rag")
async def chat_rag(req: ChatDebugRequest):
    return await _do_chat_with_rag(req.question, req.debug)


@app.get("/api/chat/rag")
async def chat_rag_get(
    q: str = Query(..., description="输入问题"),
    debug: bool = Query(False, description="调试模式，返回 sources/from_cache"),
):
    return await _do_chat_with_rag(q, debug)


# ═══════════════════════════════════════════════════════
#  POST /api/chat/rag/stream  — RAG 流式输出
#  ═══════════════════════════════════════════════════════
@app.post("/api/chat/rag/stream")
async def chat_rag_stream(req: ChatStreamRequest):
    question = req.question
    session_id = req.session_id
    has_memory = bool(session_id)
    logger.info("收到流式提问: %s (session=%s)", question[:50], session_id or "-")

    is_safe, reason = check_safety(question, session_id or "default")
    if not is_safe:
        logger.warning("安全拦截(流式): %s → %s", question[:50], reason[:30])
        metrics.incr("blocked")
        # 流式接口拦截也要返回纯文本，不能返回 JSON（前端按文本流显示）
        return StreamingResponse(iter([reason]), media_type="text/plain")

    # 多轮模式下答案依赖上下文，不能按问题缓存；无记忆模式才走缓存
    cached = None
    if not has_memory:
        cached = get_cached(question)
    if cached:
        logger.info("缓存命中(流式): %s", question[:50])
        return StreamingResponse(iter([cached]), media_type="text/plain")

    retrieved_docs = await search(question, top_k=3)
    base_prompt = build_system_prompt()
    model = get_model()

    if retrieved_docs:
        docs_text = "\n---\n".join(d["text"] for d in retrieved_docs)
        system_prompt = f"""{base_prompt}

【参考资料】
以下是从知识库中检索到的相关医学资料，请参考这些资料回答问题。

{docs_text}"""
    else:
        system_prompt = base_prompt

    # 多轮对话：system 提示 + 历史对话 + 当前问题
    messages = [{"role": "system", "content": system_prompt}]
    if has_memory:
        messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": question})

    async def generate():
        full = []
        pending = ""   # 尾部缓冲：安全检测需要跨 chunk 匹配，先压着不发给前端
        blocked = False
        real_usage = None   # 流式最后一个 chunk 携带的真实用量
        t0 = time.time()
        try:
            response = await async_client.chat.completions.create(
                model=model, messages=messages, stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in response:
                if getattr(chunk, "usage", None):
                    real_usage = chunk.usage
                    continue  # usage chunk 没有 choices，跳过
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    full.append(content)
                    pending += content
                    block_msg = find_stream_block(pending)
                    if block_msg:
                        logger.warning("流式输出拦截: %s", question[:50])
                        yield block_msg
                        blocked = True
                        return
                    if len(pending) > STREAM_TAIL:
                        yield pending[:-STREAM_TAIL]
                        pending = pending[-STREAM_TAIL:]
        except Exception:
            metrics.incr("llm_errors")
            yield ERROR_MSG
            return
        if not blocked:
            if pending:
                yield pending
            answer = "".join(full)
            if answer != ERROR_MSG:
                if has_memory:
                    add_message(session_id, "user", question)
                    add_message(session_id, "assistant", answer)
                else:
                    set_cached(question, answer)
                if real_usage:
                    # API 返回的真实用量（含历史对话、检索文档）
                    record_usage(real_usage.prompt_tokens or 0, real_usage.completion_tokens or 0)
                else:
                    # 兜底估算
                    record_usage(len(system_prompt) // 2 + len(question) // 2, len(answer) // 2)
            logger.info("流式完成, 耗时=%.0fms", (time.time() - t0) * 1000)

    return StreamingResponse(generate(), media_type="text/plain")


# ═══════════════════════════════════════════════════════
#  GET /api/chat?q=xxx           — 无记忆
#  GET /api/chat/memory?q=xxx    — 带记忆（默认 session）
#  ═══════════════════════════════════════════════════════
@app.get("/api/chat")
async def chat_get(q: str = Query(..., description="输入你的问题")):
    return await _do_chat(q)


@app.get("/api/chat/memory")
async def chat_get_memory(
    q: str = Query(..., description="输入你的问题"),
    session_id: str = Query("default", description="会话ID"),
):
    return await _do_chat_with_memory(q, session_id)


# ═══════════════════════════════════════════════════════
#  POST /api/chat/clear  — 清空会话记忆
#  ═══════════════════════════════════════════════════════
@app.post("/api/chat/clear")
async def clear_memory(session_id: str = "default"):
    clear_history(session_id)
    clear_rate_limit(session_id)
    return {"message": "会话已重置"}


# ── 核心逻辑（避免重复代码）─────────────────────────────────

ERROR_MSG = "服务暂时不可用，请稍后重试。"


async def _call_llm(model: str, messages: list) -> str:
    """统一 LLM 调用，超时/异常 → 返回错误提示，不让接口 500"""
    try:
        response = await async_client.chat.completions.create(model=model, messages=messages)
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", 0):
            record_usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
        else:
            record_usage(len(messages[-1]["content"]) // 2, len(response.choices[0].message.content or "") // 2)
        logger.info("LLM调用成功, 模型=%s, 输入Token≈%d", model, len(messages[-1]["content"]))
        return response.choices[0].message.content
    except Exception as e:
        logger.error("LLM调用失败: %s", e)
        metrics.incr("llm_errors")
        return ERROR_MSG


async def _do_chat(question: str, session_id: str = "default") -> dict:
    """无记忆模式：每问独立，不拼接历史"""
    is_safe, reason = check_safety(question, session_id)
    if not is_safe:
        metrics.incr("blocked")
        return {"answer": reason}

    system_prompt = build_system_prompt()
    model = get_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = await _call_llm(model, messages)
    return {"answer": filter_output(answer)}


async def _do_chat_with_rag(question: str, debug: bool = False) -> dict:
    """
    RAG 增强模式：先从知识库检索相关文档，再拼进 Prompt 让 LLM 参考回答。
    高频问题走 Redis 缓存，命中直接返回不调 API。
    对外只返回 answer；debug=True 时才带 sources/from_cache 调试字段。
    """
    logger.info("收到RAG提问: %s", question[:50])
    t0 = time.time()

    is_safe, reason = check_safety(question)
    if not is_safe:
        logger.warning("安全拦截: %s → %s", question[:50], reason[:30])
        metrics.incr("blocked")
        return {"answer": reason}

    # 缓存检查：同一个问题，15 分钟内直接返回
    cached = get_cached(question)
    if cached:
        logger.info("缓存命中: %s", question[:50])
        return {"answer": cached} if not debug else {"answer": cached, "from_cache": True}

    # 检索知识库
    retrieved_docs = await search(question, top_k=3)

    # 构建 Prompt：基础系统提示 + 检索到的参考文档 + 用户问题
    base_prompt = build_system_prompt()
    model = get_model()

    if retrieved_docs:
        docs_text = "\n---\n".join(d["text"] for d in retrieved_docs)
        system_prompt = f"""{base_prompt}

【参考资料】
以下是从知识库中检索到的相关医学资料，请参考这些资料回答问题。
如果资料不足以回答用户的问题，请如实说明，不要编造信息。

{docs_text}"""
    else:
        # 知识库为空，退化为普通模式
        system_prompt = base_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = await _call_llm(model, messages)
    answer = filter_output(answer)

    # 存入缓存，下次同样问题直接命中（错误提示不缓存）
    if answer != ERROR_MSG:
        set_cached(question, answer)
        logger.info("写入缓存: %s", question[:50])

    logger.info("RAG完成, 耗时=%.0fms", (time.time() - t0) * 1000)

    if debug:
        return {"answer": answer, "sources": retrieved_docs, "from_cache": False}
    return {"answer": answer}


async def _do_chat_with_memory(question: str, session_id: str) -> dict:
    """有记忆模式：带上历史对话，AI 知道上下文"""
    is_safe, reason = check_safety(question, session_id)
    if not is_safe:
        metrics.incr("blocked")
        return {"answer": reason}

    system_prompt = build_system_prompt()
    model = get_model()

    # 关键：messages = system 提示 + 历史记录 + 当前问题
    messages = [{"role": "system", "content": system_prompt}]  # ① 规矩
    messages.extend(get_history(session_id))                    # ② 历史对话
    messages.append({"role": "user", "content": question})      # ③ 当前问题

    answer = await _call_llm(model, messages)
    answer = filter_output(answer)

    # 把本轮对话写入记忆（供下次使用）
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    return {"answer": answer}


# ═══════════════════════════════════════════════════════
#  GET /  — 中文测试页面
#  ═══════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
    <html lang="zh">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>一折健康助手</title>
    <link rel="icon" href="static/favicon.svg" type="image/svg+xml">
    <style>
        :root { --primary:#0f766e; --primary-h:#0d5f58; --bg:#f5f7f9; --panel:#ffffff; --text:#1f2d3d; --sub:#8a97a5; --line:#e5eaf0; }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'PingFang SC','Microsoft YaHei',sans-serif; background:var(--bg); height:100vh; display:flex; overflow:hidden; color:var(--text); font-size:14px; }
        /* ── 侧边栏 ── */
        #sidebar { width:236px; background:var(--panel); border-right:1px solid var(--line); display:flex; flex-direction:column; flex-shrink:0; z-index:20; transition:margin-left .2s; }
        .brand { display:flex; align-items:center; gap:10px; padding:16px 16px 12px; }
        .brand .logo { width:34px; height:34px; border-radius:9px; background:var(--primary); color:#fff; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:bold; flex-shrink:0; }
        .brand .name { font-size:14px; font-weight:600; }
        .brand .en { font-size:10px; color:var(--sub); margin-top:1px; }
        #new-btn { margin:4px 14px 12px; padding:9px 0; background:var(--primary); color:#fff; border:none; border-radius:8px; font-size:13px; cursor:pointer; }
        #new-btn:hover { background:var(--primary-h); }
        #session-list { flex:1; overflow-y:auto; padding:0 10px; }
        .sess-group { font-size:11px; color:var(--sub); padding:10px 6px 4px; }
        .sess-item { display:flex; align-items:center; padding:9px 10px; border-radius:8px; cursor:pointer; font-size:13px; color:#3d4a58; gap:6px; margin-bottom:2px; }
        .sess-item:hover { background:#f0f4f6; }
        .sess-item.active { background:#e6f2f1; color:var(--primary); font-weight:500; }
        .sess-item .t { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .sess-item .del { visibility:hidden; color:var(--sub); border:none; background:none; cursor:pointer; font-size:14px; padding:0 2px; }
        .sess-item:hover .del { visibility:visible; }
        #side-foot { padding:12px 16px; border-top:1px solid var(--line); font-size:11px; color:var(--sub); }
        #backdrop { display:none; position:fixed; left:0; top:0; right:0; bottom:0; background:rgba(15,23,42,.35); z-index:15; }
        #backdrop.show { display:block; }
        /* ── 主区 ── */
        #main { flex:1; display:flex; flex-direction:column; min-width:0; }
        #topbar { background:var(--panel); border-bottom:1px solid var(--line); height:52px; flex-shrink:0; }
        .topbar-inner { width:min(1600px, calc(100% - 64px)); margin:0 auto; height:52px; display:flex; align-items:center; gap:12px; padding:0 20px; }
        #menu-btn { display:none; background:none; border:none; font-size:18px; cursor:pointer; color:var(--text); }
        #cur-title { font-size:14px; font-weight:600; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        #status { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--sub); }
        #status .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; }
        #chat-box { flex:1; overflow-y:auto; padding:24px 20px; width:min(1600px, calc(100% - 64px)); margin:0 auto; }
        .msg-row { display:flex; gap:10px; margin-bottom:18px; }
        .msg-row.user { justify-content:flex-end; }
        .avatar { width:32px; height:32px; border-radius:8px; background:var(--primary); color:#fff; display:flex; align-items:center; justify-content:center; font-size:17px; font-weight:bold; flex-shrink:0; }
        .logo svg { width:20px; height:20px; }
        .big-logo svg { width:36px; height:36px; }
        .avatar svg { width:18px; height:18px; }
        .msg-inner { max-width:min(72%, 880px); min-width:0; }
        .bubble { padding:10px 14px; border-radius:12px; line-height:1.7; word-break:break-word; white-space:pre-wrap; }
        .msg-row.assistant .bubble { background:#fff; border:1px solid var(--line); border-top-left-radius:4px; }
        .msg-row.user .bubble { background:linear-gradient(135deg,#0f766e,#14b8a6); color:#fff; border-top-right-radius:4px; }
        .meta { font-size:11px; color:var(--sub); margin-top:4px; }
        .msg-row.user .meta { text-align:right; }
        .typing { display:inline-flex; gap:4px; padding:2px 0; }
        .typing i { width:6px; height:6px; background:#9dbcb7; border-radius:50%; animation:blink 1.2s infinite; }
        .typing i:nth-child(2){ animation-delay:.2s; } .typing i:nth-child(3){ animation-delay:.4s; }
        @keyframes blink { 0%,80%,100%{opacity:.25;} 40%{opacity:1;} }
        .welcome { text-align:center; margin-top:9vh; }
        .big-logo { width:60px; height:60px; border-radius:16px; background:linear-gradient(135deg,#0f766e,#14b8a6); color:#fff; font-size:34px; font-weight:bold; display:flex; align-items:center; justify-content:center; margin:0 auto 16px; box-shadow:0 6px 18px rgba(15,118,110,.25); }
        .welcome h2 { font-size:19px; margin-bottom:6px; }
        .welcome p { color:var(--sub); font-size:13px; margin-bottom:24px; }
        .cards { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
        .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 16px; font-size:13px; color:#3d4a58; cursor:pointer; transition:all .15s; }
        .card:hover { border-color:var(--primary); color:var(--primary); }
        #input-wrap { background:var(--panel); border-top:1px solid var(--line); padding:10px 20px 6px; width:min(1600px, calc(100% - 64px)); margin:0 auto; }
        #input-row { display:flex; gap:8px; align-items:center; }
        #q { flex:1; padding:11px 14px; font-size:14px; border:1.5px solid var(--line); border-radius:10px; outline:none; background:#fbfcfd; transition:all .15s; }
        #q:focus { border-color:var(--primary); background:#fff; }
        #send { padding:11px 24px; background:var(--primary); color:#fff; border:none; border-radius:10px; font-size:14px; cursor:pointer; }
        #send:hover { background:var(--primary-h); }
        #send:disabled { background:#b6d6d0; cursor:not-allowed; }
        #disclaimer { font-size:11px; color:var(--sub); text-align:center; padding:6px 0 2px; }
        @media (max-width:720px) {
            #sidebar { position:absolute; left:0; top:0; bottom:0; margin-left:-236px; box-shadow:2px 0 12px rgba(0,0,0,.08); }
            #sidebar.open { margin-left:0; }
            #menu-btn { display:block; }
            #chat-box, #input-wrap, .topbar-inner { width:100%; }
        }
    </style>
    </head>
    <body>
        <aside id="sidebar">
            <div class="brand">
                <div class="logo"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg></div>
                <div>
                    <div class="name">一折健康助手</div>
                    <div class="en">AI Health Assistant</div>
                </div>
            </div>
            <button id="new-btn">＋ 新建对话</button>
            <div id="session-list"></div>
            <div id="side-foot">v1.0 · 智能导诊 / 医学科普 / 健康预防</div>
        </aside>
        <div id="backdrop"></div>
        <div id="main">
            <div id="topbar">
                <div class="topbar-inner">
                    <button id="menu-btn">☰</button>
                    <div id="cur-title">新对话</div>
                    <div id="status"><span class="dot"></span><span id="status-text">连接中…</span></div>
                </div>
            </div>
            <div id="chat-box"></div>
            <div id="input-wrap">
                <div id="input-row">
                    <input id="q" placeholder="描述你的症状或健康疑问…" autofocus />
                    <button id="send">发送</button>
                </div>
                <div id="disclaimer">内容由 AI 生成，仅供参考，不构成医疗建议。如有不适请及时就医。</div>
            </div>
        </div>
        <script>
        var ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>';
        var LS_KEY = 'yz_sessions', CUR_KEY = 'yz_current';
        var currentId = localStorage.getItem(CUR_KEY) || null;
        var box = document.getElementById('chat-box');

        function loadSessions() {
            try { return JSON.parse(localStorage.getItem(LS_KEY)) || []; } catch (e) { return []; }
        }
        function saveSessions(list) { localStorage.setItem(LS_KEY, JSON.stringify(list)); }
        function getSession(id) {
            var list = loadSessions();
            for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
            return null;
        }
        function upsertSession(sess) {
            var list = loadSessions();
            for (var i = 0; i < list.length; i++) {
                if (list[i].id === sess.id) { list[i] = sess; saveSessions(list); return; }
            }
            list.unshift(sess);
            if (list.length > 50) list = list.slice(0, 50);
            saveSessions(list);
        }
        function removeSession(id) {
            var list = loadSessions().filter(function (s) { return s.id !== id; });
            saveSessions(list);
            if (currentId === id) newSession();
        }
        function fmtTime(ts) {
            var d = new Date(ts), p = function (n) { return n < 10 ? '0' + n : '' + n; };
            return p(d.getHours()) + ':' + p(d.getMinutes());
        }
        function fmtDate(ts) {
            var d = new Date(ts), now = new Date();
            if (d.toDateString() === now.toDateString()) return '今天';
            var y = new Date(now.getTime() - 86400000);
            if (d.toDateString() === y.toDateString()) return '昨天';
            return (d.getMonth() + 1) + '月' + d.getDate() + '日';
        }
        function scrollBox() { box.scrollTop = box.scrollHeight; }

        function renderSidebar() {
            var list = loadSessions(), el = document.getElementById('session-list');
            el.innerHTML = '';
            if (list.length === 0) {
                el.innerHTML = '<div class="sess-group">暂无历史会话</div>';
                return;
            }
            var groups = {};
            list.forEach(function (s) {
                var k = fmtDate(s.time);
                if (!groups[k]) groups[k] = [];
                groups[k].push(s);
            });
            Object.keys(groups).forEach(function (g) {
                var gd = document.createElement('div');
                gd.className = 'sess-group';
                gd.textContent = g;
                el.appendChild(gd);
                groups[g].forEach(function (s) {
                    var item = document.createElement('div');
                    item.className = 'sess-item' + (s.id === currentId ? ' active' : '');
                    var t = document.createElement('span');
                    t.className = 't';
                    t.textContent = s.title || '新对话';
                    var d = document.createElement('button');
                    d.className = 'del';
                    d.textContent = '×';
                    d.title = '删除会话';
                    d.addEventListener('click', function (e) {
                        e.stopPropagation();
                        removeSession(s.id);
                        renderSidebar();
                    });
                    item.appendChild(t);
                    item.appendChild(d);
                    item.addEventListener('click', function () {
                        currentId = s.id;
                        localStorage.setItem(CUR_KEY, currentId);
                        renderSession();
                        renderSidebar();
                        closeSidebar();
                    });
                    el.appendChild(item);
                });
            });
        }

        function addMsgRow(role, text, ts) {
            var row = document.createElement('div');
            row.className = 'msg-row ' + role;
            var inner = '';
            if (role === 'assistant') inner = '<div class="avatar">' + ICON + '</div>';
            inner += '<div class="msg-inner"><div class="bubble"></div><div class="meta">' + fmtTime(ts || Date.now()) + '</div></div>';
            row.innerHTML = inner;
            row.querySelector('.bubble').textContent = text;
            box.appendChild(row);
            scrollBox();
            return row;
        }
        function showWelcome() {
            var w = document.createElement('div');
            w.className = 'welcome';
            w.innerHTML = '<div class="big-logo">' + ICON + '</div><h2>你好，我是一折</h2><p>日常养生、饮食调理、小病小痛怎么就诊，都可以问我</p><div class="cards">' +
                '<div class="card" data-q="感冒了应该注意什么">感冒了应该注意什么</div>' +
                '<div class="card" data-q="头痛应该挂哪个科室">头痛挂什么科室</div>' +
                '<div class="card" data-q="经常熬夜怎么调理">经常熬夜怎么调理</div>' +
                '</div>';
            box.appendChild(w);
            var cards = w.querySelectorAll('.card');
            for (var i = 0; i < cards.length; i++) {
                cards[i].addEventListener('click', function () { ask(this.getAttribute('data-q')); });
            }
            scrollBox();
        }

        function renderSession() {
            box.innerHTML = '';
            var s = getSession(currentId);
            document.getElementById('cur-title').textContent = (s && s.title) || '新对话';
            if (!s || !s.msgs || s.msgs.length === 0) { showWelcome(); return; }
            s.msgs.forEach(function (m) { addMsgRow(m.role, m.text, m.time); });
        }

        function newSession() {
            currentId = 's' + Date.now();
            localStorage.setItem(CUR_KEY, currentId);
            renderSession();
            renderSidebar();
            document.getElementById('q').focus();
        }

        function appendMsg(role, text) {
            var s = getSession(currentId);
            if (!s) { s = { id: currentId, title: '', time: Date.now(), msgs: [] }; }
            s.msgs.push({ role: role, text: text, time: Date.now() });
            s.time = Date.now();
            if (!s.title && role === 'user') s.title = text.slice(0, 14);
            upsertSession(s);
        }

        async function ask(q) {
            var text = (typeof q === 'string') ? q.trim() : document.getElementById('q').value.trim();
            if (!text) return;
            if (!getSession(currentId)) upsertSession({ id: currentId, title: text.slice(0, 14), time: Date.now(), msgs: [] });
            document.getElementById('q').value = '';
            addMsgRow('user', text, Date.now());
            appendMsg('user', text);
            renderSidebar();
            var send = document.getElementById('send');
            send.disabled = true;
            var typing = document.createElement('div');
            typing.className = 'msg-row assistant';
            typing.innerHTML = '<div class="avatar">' + ICON + '</div><div class="msg-inner"><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div></div>';
            box.appendChild(typing);
            scrollBox();
            var first = true, bubbleEl = null, fullText = '';
            try {
                var res = await fetch('api/chat/rag/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text, session_id: currentId })
                });
                var reader = res.body.getReader();
                var decoder = new TextDecoder();
                while (true) {
                    var chunk = await reader.read();
                    if (chunk.done) break;
                    var piece = decoder.decode(chunk.value, { stream: true });
                    if (first) {
                        first = false;
                        typing.remove();
                        var row = addMsgRow('assistant', '', Date.now());
                        bubbleEl = row.querySelector('.bubble');
                    }
                    fullText += piece;
                    if (bubbleEl) bubbleEl.textContent = fullText;
                    scrollBox();
                }
                if (first) { typing.remove(); }
            } catch (e) {
                typing.remove();
            }
            send.disabled = false;
            if (fullText) appendMsg('assistant', fullText);
            renderSidebar();
        }

        var backdrop = document.getElementById('backdrop');
        function openSidebar() {
            document.getElementById('sidebar').classList.add('open');
            backdrop.classList.add('show');
        }
        function closeSidebar() {
            document.getElementById('sidebar').classList.remove('open');
            backdrop.classList.remove('show');
        }
        document.getElementById('new-btn').addEventListener('click', function () {
            newSession();
            closeSidebar();
        });
        document.getElementById('send').addEventListener('click', function () { ask(); });
        document.getElementById('q').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') ask();
        });
        document.getElementById('menu-btn').addEventListener('click', function () {
            var sb = document.getElementById('sidebar');
            if (sb.classList.contains('open')) { closeSidebar(); } else { openSidebar(); }
        });
        backdrop.addEventListener('click', closeSidebar);
        window.addEventListener('resize', closeSidebar);
        fetch('healthz').then(function (r) {
            if (r.ok) document.getElementById('status-text').textContent = '服务正常';
            else document.getElementById('status-text').textContent = '服务异常';
        }).catch(function () {
            document.getElementById('status-text').textContent = '服务异常';
        });

        if (currentId && getSession(currentId)) { renderSession(); }
        else { newSession(); }
        renderSidebar();
        </script>
    </body>
    </html>"""


# ═══════════════════════════════════════════════════════
#  GET /admin  — 管理后台页面（隐藏入口，无任何公开链接）
#  ═══════════════════════════════════════════════════════
@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>管理后台 · 一折健康助手</title>
<style>
    :root { --bg:#f0f4f8; --panel:#fff; --text:#1f2d3d; --sub:#7a8a99; --line:#e3eaf1; --primary:#0f766e; --danger:#dc2626; --warn:#d97706; }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'PingFang SC','Microsoft YaHei',sans-serif; background:var(--bg); min-height:100vh; color:var(--text); font-size:14px; }
    .hidden { display:none !important; }
    /* 登录 */
    #login-wrap { min-height:100vh; display:flex; align-items:center; justify-content:center; }
    .login-card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:36px 32px; width:340px; box-shadow:0 8px 30px rgba(15,60,80,.08); }
    .login-card h1 { font-size:18px; margin-bottom:4px; }
    .login-card p { color:var(--sub); font-size:12px; margin-bottom:20px; }
    .login-card input { width:100%; padding:10px 12px; border:1.5px solid var(--line); border-radius:8px; font-size:14px; margin-bottom:12px; outline:none; }
    .login-card input:focus { border-color:var(--primary); }
    .login-card button { width:100%; padding:10px; background:var(--primary); color:#fff; border:none; border-radius:8px; font-size:14px; cursor:pointer; }
    #login-msg { color:var(--danger); font-size:12px; margin-top:10px; min-height:16px; }
    /* 面板 */
    #dash { max-width:1080px; margin:0 auto; padding:20px; }
    .topbar { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
    .topbar h1 { font-size:17px; flex:1; }
    .topbar .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; }
    .topbar button { padding:8px 14px; background:var(--panel); border:1px solid var(--line); border-radius:8px; font-size:13px; cursor:pointer; color:var(--text); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px,1fr)); gap:12px; margin-bottom:18px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
    .card .label { font-size:12px; color:var(--sub); margin-bottom:6px; }
    .card .value { font-size:22px; font-weight:600; }
    .card .value.small { font-size:17px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px; }
    .panel h2 { font-size:14px; margin-bottom:12px; }
    .ev-head { display:flex; align-items:center; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
    .ev-head h2 { flex:1; margin-bottom:0; }
    .ev-head input { padding:7px 10px; border:1.5px solid var(--line); border-radius:8px; font-size:13px; width:200px; outline:none; }
    .ev-head input:focus { border-color:var(--primary); }
    .ev-pager { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--sub); }
    .ev-pager button { padding:5px 10px; background:var(--panel); border:1px solid var(--line); border-radius:6px; font-size:12px; cursor:pointer; color:var(--text); }
    .ev-pager button:disabled { opacity:.4; cursor:not-allowed; }
    .bars { display:flex; align-items:flex-end; gap:8px; height:120px; }
    .bar-col { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; height:100%; justify-content:flex-end; }
    .bar { width:70%; background:linear-gradient(180deg,#14b8a6,#0f766e); border-radius:4px 4px 0 0; min-height:2px; }
    .bar-col .v { font-size:10px; color:var(--sub); }
    .bar-col .d { font-size:10px; color:var(--sub); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
    th { color:var(--sub); font-weight:500; }
    td.mono { font-family:Consolas,monospace; color:#3d4a58; word-break:break-all; }
    .tag { padding:2px 8px; border-radius:10px; font-size:11px; }
    .tag.err { background:#fdeaea; color:var(--danger); }
    .tag.warn { background:#fdf3e2; color:var(--warn); }
    .tag.ok { background:#e4f4f1; color:var(--primary); }
</style>
</head>
<body>
<div id="login-wrap">
    <div class="login-card">
        <h1>一折健康助手</h1>
        <p>管理后台 · 仅限管理员访问</p>
        <input id="u" placeholder="用户名" autocomplete="username" />
        <input id="p" type="password" placeholder="密码" autocomplete="current-password" />
        <button onclick="doLogin()">登 录</button>
        <div id="login-msg"></div>
    </div>
</div>
<div id="dash" class="hidden">
    <div class="topbar">
        <span class="dot"></span>
        <h1>管理后台 · 一折健康助手</h1>
        <button onclick="load()">刷新</button>
        <button onclick="doLogout()">退出登录</button>
    </div>
    <div class="cards" id="cards"></div>
    <div class="panel">
        <h2>近 7 天请求量</h2>
        <div class="bars" id="req-bars"></div>
    </div>
    <div class="panel">
        <h2>近 7 天 Token 用量（绿=输入 蓝=输出）</h2>
        <div class="bars" id="tok-bars"></div>
    </div>
    <div class="panel">
        <div class="ev-head">
            <h2>最近操作事件</h2>
            <input id="ev-search" placeholder="搜索事件内容…" />
            <div class="ev-pager">
                <button id="ev-prev">上一页</button>
                <span id="ev-page">1 / 1</span>
                <button id="ev-next">下一页</button>
            </div>
        </div>
        <table id="events"><thead><tr><th style="width:150px">时间</th><th>事件</th></tr></thead><tbody></tbody></table>
    </div>
</div>
<script>
var TOKEN_KEY = 'yz_admin_token';
var allEvents = [], evPage = 1, evQuery = '';
var EV_PAGE_SIZE = 10;

function renderEvents() {
    var filtered = allEvents;
    var q = evQuery.trim().toLowerCase();
    if (q) {
        filtered = allEvents.filter(function (e) { return e.toLowerCase().indexOf(q) >= 0; });
    }
    var totalPages = Math.max(1, Math.ceil(filtered.length / EV_PAGE_SIZE));
    if (evPage > totalPages) evPage = totalPages;
    if (evPage < 1) evPage = 1;
    var pageItems = filtered.slice((evPage - 1) * EV_PAGE_SIZE, evPage * EV_PAGE_SIZE);
    var tb = document.querySelector('#events tbody');
    tb.innerHTML = '';
    pageItems.forEach(function (e) {
        var tr = document.createElement('tr');
        var kind, label;
        if (e.indexOf('ERROR') >= 0) { kind = 'err'; label = '错误'; }
        else if (e.indexOf('拦截') >= 0 || e.indexOf('限流') >= 0) { kind = 'warn'; label = '拦截'; }
        else { kind = 'ok'; label = '提问'; }
        var time = (e.match(/\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}/) || [''])[0];
        var text = e.replace(time, '').replace(/^[ |]+/, '');
        tr.innerHTML = '<td class="mono">' + time + '</td><td class="mono"><span class="tag ' + kind + '">' + label + '</span> ' + text.replace(/[<>&]/g, function (c) { return { '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]; }) + '</td>';
        tb.appendChild(tr);
    });
    document.getElementById('ev-page').textContent = evPage + ' / ' + totalPages;
    document.getElementById('ev-prev').disabled = evPage <= 1;
    document.getElementById('ev-next').disabled = evPage >= totalPages;
}
function token() { return localStorage.getItem(TOKEN_KEY) || ''; }
function show(which) {
    document.getElementById('login-wrap').classList.toggle('hidden', which !== 'login');
    document.getElementById('dash').classList.toggle('hidden', which !== 'dash');
}
async function doLogin() {
    var u = document.getElementById('u').value.trim();
    var p = document.getElementById('p').value;
    var msg = document.getElementById('login-msg');
    msg.textContent = '';
    try {
        var res = await fetch('api/admin/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        var data = await res.json();
        if (res.ok && data.token) {
            localStorage.setItem(TOKEN_KEY, data.token);
            show('dash');
            load();
        } else {
            msg.textContent = data.msg || '登录失败';
        }
    } catch (e) { msg.textContent = '网络异常，请稍后再试'; }
}
async function doLogout() {
    try { await fetch('api/admin/logout?token=' + encodeURIComponent(token()), { method: 'POST' }); } catch (e) {}
    localStorage.removeItem(TOKEN_KEY);
    show('login');
}
function card(label, value, small) {
    return '<div class="card"><div class="label">' + label + '</div><div class="value' + (small ? ' small' : '') + '">' + value + '</div></div>';
}
function bars(el, data, fmt) {
    var max = 1;
    data.forEach(function (d) { if (d.value > max) max = d.value; });
    el.innerHTML = '';
    data.forEach(function (d) {
        var col = document.createElement('div');
        col.className = 'bar-col';
        var h = max > 0 ? Math.round(d.value / max * 100) : 0;
        col.innerHTML = '<div class="v">' + fmt(d.value) + '</div><div class="bar" style="height:' + Math.max(h, 2) + '%"></div><div class="d">' + d.date.slice(5) + '</div>';
        el.appendChild(col);
    });
}
async function load() {
    if (!token()) { show('login'); return; }
    try {
        var res = await fetch('api/admin/snapshot?token=' + encodeURIComponent(token()));
        if (res.status === 401) { doLogout(); return; }
        var s = await res.json();
        var t = s.today;
        document.getElementById('cards').innerHTML =
            card('今日请求', t.requests) +
            card('安全拦截', t.blocked) +
            card('LLM 错误', t.llm_errors) +
            card('限流触发', t.rate_limited) +
            card('Token 输入', t.tokens_in, true) +
            card('Token 输出', t.tokens_out, true) +
            card('活跃会话', t.sessions) +
            card('受限 IP 数', t.rl_ips);
        bars(document.getElementById('req-bars'), s.series.requests, function (v) { return v; });
        var tok = s.series.usage.map(function (d) { return { date: d.date, value: d.in + d.out }; });
        bars(document.getElementById('tok-bars'), tok, function (v) { return v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v; });
        allEvents = s.recent_events || [];
        renderEvents();
    } catch (e) {}
}
document.getElementById('ev-prev').addEventListener('click', function () { evPage--; renderEvents(); });
document.getElementById('ev-next').addEventListener('click', function () { evPage++; renderEvents(); });
document.getElementById('ev-search').addEventListener('input', function () {
    evQuery = this.value;
    evPage = 1;
    renderEvents();
});
if (token()) { show('dash'); load(); } else { show('login'); }
document.getElementById('p').addEventListener('keydown', function (e) { if (e.key === 'Enter') doLogin(); });
setInterval(function () {
    if (!document.getElementById('dash').classList.contains('hidden')) load();
}, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
