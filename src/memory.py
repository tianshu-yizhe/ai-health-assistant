"""
对话记忆模块
===========
模拟 LLM 的"记忆"能力。
原理：每次调 API 时，把之前的对话历史一块塞进 messages 里。
LLM 本身是无状态的，记忆全靠你帮它记住。

存储方式（从简到繁）：
  1. Python dict   → 学习用，重启丢失
  2. Redis         → 生产用，持久化+过期自动清 ← 当前实现
  3. 数据库        → 永久存储，用户翻历史记录

当前实现：Redis 存储（JSON 序列化 + TTL）。
  - 服务重启记忆不丢
  - 会话 24 小时无操作自动过期，防止无人清理的内存/存储泄漏
  - Redis 挂了降级：读返回空历史，写静默跳过，主流程不 500
"""

import json
import os
from typing import List, Dict
import redis as redis_lib

REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
r = redis_lib.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)

# 每个会话最多保留多少轮对话（一轮 = 用户问 + AI 答 = 2 条消息）
MAX_ROUNDS = 10

# 会话过期时间（秒）：24 小时无操作自动清
SESSION_TTL = 86400


def _key(session_id: str) -> str:
    return f"mem:{session_id}"


def get_history(session_id: str) -> List[Dict[str, str]]:
    """获取某个会话的全部历史（Redis 挂了返回空列表，降级为无记忆模式）"""
    try:
        raw = r.get(_key(session_id))
        if raw:
            history = json.loads(raw)
            if isinstance(history, list):
                return history
    except (redis_lib.RedisError, json.JSONDecodeError):
        pass
    return []


def add_message(session_id: str, role: str, content: str):
    """往会话历史里追加一条消息（Redis 挂了静默跳过）"""
    try:
        history = get_history(session_id)
        history.append({"role": role, "content": content})

        # 控制长度：超过限制自动裁剪最旧的对话
        # MAX_ROUNDS * 2 = 用户消息 + AI 回复，10 轮就是 20 条
        max_messages = MAX_ROUNDS * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        r.set(_key(session_id), json.dumps(history, ensure_ascii=False), ex=SESSION_TTL)
    except redis_lib.RedisError:
        pass


def clear_history(session_id: str):
    """清空某个会话（用户说"重新开始"时调用）"""
    try:
        r.delete(_key(session_id))
    except redis_lib.RedisError:
        pass
