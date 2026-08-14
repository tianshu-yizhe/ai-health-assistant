"""
Redis 缓存模块
==============
高频问题缓存 LLM 回答，命中直接返回，不调 API（省钱提速）。

原理:
  用户问题 → 归一化（去首尾空格/标点）→ MD5 哈希做 key → 查 Redis
      → 命中: 直接返回缓存答案（0.1ms）
      → 未命中: 调 LLM → 答案存进 Redis → 返回
"""

import hashlib
import os
import re
import redis

# Redis 地址：环境变量 REDIS_HOST 覆盖（服务器上设 127.0.0.1，本地默认连 VM）
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
r = redis.Redis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)

# 缓存有效期（秒）— 60 秒太短浪费，1 小时太长容易过期信息，取 15 分钟
CACHE_TTL = 900

# 命中率统计（演示时可以说"缓存命中率 xx%"）
_cache_hits = 0
_cache_misses = 0

# 归一化：去空白和常见标点，"感冒了怎么办" 和 " 感冒了怎么办？" 命中同一个缓存
_NORMALIZE_PATTERN = re.compile(r"[\s，。！？!?、；;：:\"'“”‘’（）()]+")


def _normalize(question: str) -> str:
    """问题文本归一化：去首尾空格、去标点空白，统一小写"""
    return _NORMALIZE_PATTERN.sub("", question.strip()).lower()


def _key(question: str) -> str:
    """问题 → Redis key（归一化后 MD5 哈希，固定长度）"""
    return "chat:" + hashlib.md5(_normalize(question).encode()).hexdigest()


def get_cached(question: str) -> str | None:
    """查缓存，命中返回答案，未命中返回 None。Redis 挂了就当没缓存（降级）。"""
    global _cache_hits, _cache_misses
    try:
        cached = r.get(_key(question))
        if cached is not None:
            _cache_hits += 1
            return cached
    except redis.RedisError:
        pass  # Redis 不可用 → 降级为未命中，不影响主流程
    _cache_misses += 1
    return None


def set_cached(question: str, answer: str):
    """存缓存（带过期时间）。Redis 挂了静默跳过。"""
    try:
        r.set(_key(question), answer, ex=CACHE_TTL)
    except redis.RedisError:
        pass


def cache_stats() -> dict:
    """缓存命中率统计"""
    total = _cache_hits + _cache_misses
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": f"{_cache_hits / total * 100:.1f}%" if total > 0 else "0%",
    }
