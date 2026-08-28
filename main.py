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

import os

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


class ChatImageRequest(BaseModel):
    image_data: str | list[str] | None = None  # data URL 或数组（多图）；文字问答可为空
    question: str = ""               # 可选：用户附带的问题
    session_id: str = ""             # 会话ID：识别结果进会话记忆，可连续追问


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
def _build_rag_prompt(base_prompt: str, retrieved_docs: list) -> str:
    """把检索到的知识库片段拼进系统提示；无检索结果时退回基础提示。
    流式与非流式接口共用，保证两边的 Prompt 行为一致。"""
    if not retrieved_docs:
        return base_prompt
    docs_text = "\n---\n".join(d["text"] for d in retrieved_docs)
    return f"""{base_prompt}

【参考资料】
以下是从知识库中检索到的相关医学资料，请参考这些资料回答问题。
如果资料不足以回答用户的问题，请如实说明，不要编造信息。

{docs_text}"""


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
    system_prompt = _build_rag_prompt(base_prompt, retrieved_docs)

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


async def _do_chat_with_rag(question: str, debug: bool = False, skip_dosage: bool = False) -> dict:
    """
    RAG 增强模式：先从知识库检索相关文档，再拼进 Prompt 让 LLM 参考回答。
    高频问题走 Redis 缓存，命中直接返回不调 API。
    对外只返回 answer；debug=True 时才带 sources/from_cache 调试字段。
    """
    logger.info("收到RAG提问: %s", question[:50])
    t0 = time.time()

    is_safe, reason = check_safety(question, skip_dosage=skip_dosage)
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
    system_prompt = _build_rag_prompt(base_prompt, retrieved_docs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    answer = await _call_llm(model, messages)
    answer = filter_output(answer, allow_reference=skip_dosage)

    # 存入缓存，下次同样问题直接命中（错误提示不缓存）
    if answer != ERROR_MSG:
        set_cached(question, answer)
        logger.info("写入缓存: %s", question[:50])

    logger.info("RAG完成, 耗时=%.0fms", (time.time() - t0) * 1000)

    if debug:
        return {"answer": answer, "sources": retrieved_docs, "from_cache": False}
    return {"answer": answer}


# ═══════════════════════════════════════════════════════
#  POST /api/chat/image  — 图片识别 + RAG 问答
#  ═══════════════════════════════════════════════════════
async def _recognize_image(image_data: str | list[str]) -> str:
    """调用通义 VL 模型识别图片（支持多图，如药盒正反面），返回图片中的文字/内容。

    用 OpenAI 兼容格式的 image_url 多模态消息；模型由 VISION_MODEL 配置。
    """
    model = os.getenv("VISION_MODEL", "qwen3-vl-30b-a3b-thinking")
    images = image_data if isinstance(image_data, list) else [image_data]
    content: list = [{"type": "image_url", "image_url": {"url": u}} for u in images]
    content.append({"type": "text", "text": (
        "请识别这些图片（可能是一组，如药盒正面和背面），先判断类型再按类型合并提取信息：\n"
        "1) 药盒/药品：逐字准确输出药名、规格、剂量、主要成分、适应症、用法用量、注意事项；\n"
        "2) 化验单/检查报告：逐字准确输出检查项目、结果值、参考范围、异常项（标出偏高/偏低）；\n"
        "3) 症状照片：描述部位、外观特征（颜色/形态/范围）、如有文字一并输出；\n"
        "4) 其他图片：简要描述主体内容，图中文字概括即可。\n"
        "多张图信息合并输出，不要遗漏。只输出识别结果，不要多余解释。"
    )})
    messages = [{"role": "user", "content": content}]
    try:
        resp = await async_client.chat.completions.create(
            model=model, messages=messages, max_tokens=800
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("图片识别失败: %s", e)
        return ""


@app.post("/api/chat/image")
async def chat_image(req: ChatImageRequest):
    """图片识别 + RAG + 会话记忆：VL 识别图片内容（支持多图）；
    健康相关内容拼进问题走知识库问答并写入会话记忆（可连续追问）；
    无关内容（截图/风景等）直接返回识别描述，不做健康兜底拦截。"""
    logger.info("收到图片提问 (data URL 长度=%d)", len(req.image_data))
    image_text = (await _recognize_image(req.image_data) or "").strip()
    logger.info("图片识别结果 repr: %r", image_text[:200])
    if not image_text:
        return {"answer": "抱歉，图片识别失败，请重试或换一张更清晰的图片。"}

    health_kw = ("药", "胶囊", "片", "化验", "检查", "症状", "体温", "血压",
                 "血糖", "医院", "医嘱", "剂量", "成分", "适应症", "服用")
    if not any(k in image_text for k in health_kw):
        return {"answer": f"识别结果：{image_text}"}

    question = req.question.strip() or "请根据这张图片回答"
    q = f"{question}\n【图片内容】{image_text}"
    if req.session_id:
        return await _do_chat_with_rag_memory(q, req.session_id, skip_dosage=True)
    return await _do_chat_with_rag(q, skip_dosage=True)


async def _do_chat_with_rag_memory(question: str, session_id: str, skip_dosage: bool = False) -> dict:
    """RAG + 会话记忆：检索知识库拼进 prompt，同时带历史对话，回答写回记忆。

    图片识别后的追问需要同时有知识库上下文和同一会话的历史。
    """
    is_safe, reason = check_safety(question, session_id, skip_dosage=skip_dosage)
    if not is_safe:
        metrics.incr("blocked")
        return {"answer": reason}

    retrieved_docs = await search(question, top_k=3)
    system_prompt = _build_rag_prompt(build_system_prompt(), retrieved_docs)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": question})

    answer = await _call_llm(get_model(), messages)
    answer = filter_output(answer, allow_reference=skip_dosage)

    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)
    return {"answer": answer}


# ═══════════════════════════════════════════════════════
#  Agent 模式 — function calling 循环（LLM 自主决策调工具）
#  ═══════════════════════════════════════════════════════
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen-max")
_agent_images: list[str] = []  # 本会话注入的图片 data URL（供识图工具读取）

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recognize_image",
            "description": "识别用户上传的图片（药盒/化验单/症状照片，可能多张）。图片已由系统提供，调用即返回识别结果",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在健康知识库中检索与问题相关的医学资料",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "要检索的问题或关键词"}},
                "required": ["query"],
            },
        },
    },
]


async def _agent_run(question: str, session_id: str, images: list[str] | None = None) -> dict:
    """Agent 循环：LLM 自主决定调工具（识图/检索），工具结果回填后给出最终回答。

    最多 4 轮工具调用；支持会话记忆；图片转述场景跳过剂量拦截。
    """
    is_safe, reason = check_safety(question, session_id)
    if not is_safe:
        metrics.incr("blocked")
        return {"answer": reason}

    global _agent_images
    _agent_images = images or []

    system_prompt = build_system_prompt()
    user_content = question
    if _agent_images:
        user_content = f"{question}\n（用户上传了{'%d张' % len(_agent_images)}图片，需要看图时调用 recognize_image 工具）"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(session_id))
    messages.append({"role": "user", "content": user_content})

    answer = ERROR_MSG
    steps: list[dict] = []  # 工具调用轨迹（前端展示决策过程）
    for _ in range(4):
        try:
            resp = await async_client.chat.completions.create(
                model=AGENT_MODEL, messages=messages,
                tools=AGENT_TOOLS, tool_choice="auto", max_tokens=600,
            )
        except Exception as e:
            logger.error("Agent LLM 调用失败: %s", e)
            metrics.incr("llm_errors")
            answer = ERROR_MSG
            break
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ERROR_MSG
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            if name == "recognize_image":
                if _agent_images:
                    result = await _recognize_image(_agent_images)
                    result = f"图片识别结果：{result}"
                    steps.append({"tool": "识图", "detail": f"识别了 {len(_agent_images)} 张图片"})
                else:
                    result = "没有可识别的图片"
            elif name == "search_knowledge":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                docs = await search(args.get("query", question), top_k=3)
                result = json.dumps([d["text"] for d in docs], ensure_ascii=False)[:1500]
                steps.append({"tool": "知识库检索", "detail": f"查询：{args.get('query', question)[:30]}"})
            else:
                result = f"未知工具: {name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        answer = "抱歉，这个问题我思考得太久了，请换个方式再问一次。"

    # agent 场景始终豁免剂量硬过滤：回答转述说明书内容受系统 prompt 约束
    # （禁止推荐用药/替代医嘱），硬过滤放宽不会导致乱建议。
    answer = filter_output(answer, allow_reference=True)
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)
    return {"answer": answer, "steps": steps}


@app.post("/api/chat/agent")
async def chat_agent(req: ChatImageRequest):
    """Agent 问答：LLM 自主决策（识图/检索知识库），支持图片+文字、会话记忆。"""
    logger.info("收到Agent提问: %s", (req.question or "")[:50])
    images = req.image_data if isinstance(req.image_data, list) else ([req.image_data] if req.image_data else None)
    return await _agent_run(req.question.strip() or "请根据图片回答", req.session_id or "default", images)


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
@app.get("/", response_class=HTMLResponse)
async def home():
    """聊天页面（模板见 static/index.html，改样式直接改文件）"""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"), media_type="text/html")


# ═══════════════════════════════════════════════════════
#  GET /admin  — 管理后台页面（隐藏入口，无任何公开链接）
#  ═══════════════════════════════════════════════════════
@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理后台页面（模板见 static/admin.html）"""
    return FileResponse(os.path.join(BASE_DIR, "static", "admin.html"), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
