"""会话记忆裁剪纯函数单元测试（不依赖 Redis 服务器）"""
from src.memory import _trim_history


def _msgs(n):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


class TestTrim:
    def test_under_limit(self):
        h = _msgs(3)
        assert _trim_history(h, 4) == h

    def test_over_limit_keeps_latest(self):
        out = _trim_history(_msgs(10), 4)
        assert len(out) == 4
        assert out[0]["content"] == "m6"
        assert out[-1]["content"] == "m9"

    def test_default_limit(self):
        # 默认 10 轮 × 2 = 20 条
        out = _trim_history(_msgs(25))
        assert len(out) == 20
