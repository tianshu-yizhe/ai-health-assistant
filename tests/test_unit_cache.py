"""缓存归一化纯函数单元测试（不依赖 Redis 服务器）"""
from src.cache import _normalize, _key


class TestNormalize:
    def test_whitespace(self):
        assert _normalize(" 感冒了怎么办 ") == "感冒了怎么办"

    def test_punctuation(self):
        assert _normalize("感冒了怎么办？") == "感冒了怎么办"

    def test_case(self):
        assert _normalize("Hello") == "hello"


class TestKey:
    def test_same_question_same_key(self):
        assert _key("感冒了怎么办") == _key("感冒了怎么办？！！")

    def test_different_question_different_key(self):
        assert _key("感冒了怎么办") != _key("头痛怎么办")

    def test_key_prefix(self):
        assert _key("x").startswith("chat:")
