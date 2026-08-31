"""
指标埋点模块
============
按天累计运营指标到 Redis（db 1），供管理后台展示：
  - requests        请求总数（/api/ 前缀）
  - blocked         安全拦截次数
  - llm_errors      LLM 调用失败次数
  - rate_limited    IP 限流触发次数
  - login_fails     后台登录失败次数

Redis 挂了全部静默降级，不影响主流程。
"""

import os
import datetime
import redis as redis_lib

REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
r = redis_lib.Redis(host=REDIS_HOST, port=6379, db=1, decode_responses=True,
                    socket_connect_timeout=2, socket_timeout=2,
                    retry=None)  # Redis 挂时 2s 快速降级（8.x 默认重试会把 2s 放大到 25s），不卡请求

TTL = 15 * 86400  # 指标保留 15 天


def _day() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


def incr(name: str, amount: int = 1):
    try:
        key = f"m:{name}:{_day()}"
        r.incrby(key, amount)
        r.expire(key, TTL)
    except redis_lib.RedisError:
        pass


def get_today(name: str) -> int:
    try:
        return int(r.get(f"m:{name}:{_day()}") or 0)
    except redis_lib.RedisError:
        return 0


def get_series(name: str, days: int = 7) -> list:
    """近 N 天指标序列（升序），画趋势图用"""
    out = []
    for i in range(days):
        d = (datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            v = int(r.get(f"m:{name}:{d}") or 0)
        except redis_lib.RedisError:
            v = 0
        out.append({"date": d, "value": v})
    return list(reversed(out))


def count_keys(pattern: str) -> int:
    """按模式统计 key 数量（如活跃会话 mem:*）"""
    try:
        return sum(1 for _ in r.scan_iter(match=pattern, count=1000))
    except redis_lib.RedisError:
        return 0


def get_usage_series(days: int = 7) -> list:
    """近 N 天 token 用量序列（usage:日期:in/out，升序）"""
    out = []
    for i in range(days):
        d = (datetime.date.today() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            v_in = int(r.get(f"usage:{d}:in") or 0)
            v_out = int(r.get(f"usage:{d}:out") or 0)
        except redis_lib.RedisError:
            v_in = v_out = 0
        out.append({"date": d, "in": v_in, "out": v_out})
    return list(reversed(out))
