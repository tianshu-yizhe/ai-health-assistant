"""
大模型客户端
============
同时提供同步和异步客户端：
  - client:       同步，用于启动建库等非请求场景
  - async_client: 异步，用于接口请求，不阻塞事件循环

用量统计：
  - record_usage() 按天累计 token 用量到 Redis（db 1）
  - 用于成本监控，接口 /api/stats 可查
"""

import os
import datetime
import redis as redis_lib
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not API_KEY:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量，请检查 .env 文件")

# 超时配置（秒）：LLM 响应不能无限等待（VL 大图识别耗时，120s）
TIMEOUT = 120

# 同步客户端（启动建库用）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

# 异步客户端（接口请求用，不阻塞事件循环）
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

# 用量统计 Redis（与缓存/记忆同库：db 1）
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
_redis = redis_lib.Redis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)


def record_usage(prompt_tokens: int, completion_tokens: int):
    """按天累计 token 用量（Redis 挂了静默跳过，不影响主流程）"""
    try:
        day = datetime.date.today().strftime("%Y-%m-%d")
        p = _redis.pipeline()
        p.incrby(f"usage:{day}:in", int(prompt_tokens))
        p.incrby(f"usage:{day}:out", int(completion_tokens))
        p.expire(f"usage:{day}:in", 172800)   # 2 天后过期
        p.expire(f"usage:{day}:out", 172800)
        p.execute()
    except redis_lib.RedisError:
        pass


def get_usage_today() -> dict:
    """查询今日用量（Redis 挂了返回 0）"""
    try:
        day = datetime.date.today().strftime("%Y-%m-%d")
        return {
            "date": day,
            "tokens_in": int(_redis.get(f"usage:{day}:in") or 0),
            "tokens_out": int(_redis.get(f"usage:{day}:out") or 0),
        }
    except redis_lib.RedisError:
        return {"date": "", "tokens_in": 0, "tokens_out": 0}
