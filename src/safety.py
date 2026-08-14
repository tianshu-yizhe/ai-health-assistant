"""
安全过滤模块
===========
分层拦截，命中直接返回固定文案，不调 LLM（零消耗）：

  第 0 层: 心理危机 → 心理咨询热线
  第 1 层: 输入校验 → 纯数字/重复字符/空内容/纯表情
  第 2 层: 频次限流 → 连续 3 轮无效输入，冻结对话
  第 3 层: 绕过话术 → "写小说/角色扮演/假设" + 用药
  第 4 层: 直接求药 → "XX病吃什么药"
  第 5 层: 剂量询问 → "XX药一次吃多少"
  第 6 层: 通用违禁

拦截位置：调 LLM 之前，不花钱

频次计数存 Redis：服务重启不清零，将来多实例共享。
"""

import re
import os
import redis as redis_lib

# ── 固定文案 ──
INVALID_INPUT_MSG = "未识别到您的有效问题，若有身体不适请清晰描述症状，急症直接前往医院就诊。"
RATE_LIMIT_MSG = "请提出有效问题，连续无意义提问将暂时无法继续对话。"
FROZEN_MSG = "会话已被冻结，请刷新页面后重新开始对话。"
CRISIS_MSG = "你不是一个人。如果此刻你感到痛苦，请拨打 24 小时免费心理援助热线：400-161-9995（希望24热线）。你值得被听见，也值得被帮助。"

# ── 频次限流状态（Redis 存储，重启不清零）──
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
_r = redis_lib.Redis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)
GARBAGE_TTL = 3600   # 计数 1 小时无操作自动过期
MAX_WARNING = 3   # 连续 3 轮 → 显示警告
MAX_FROZEN = 5    # 连续 5 轮 → 静默，彻底不理


# ═══════════════════════════════════════════════════════
#  第 0 层: 心理危机（最高优先级）
# ═══════════════════════════════════════════════════════
CRISIS_PATTERNS = [re.compile(p) for p in [
    r"我想死",
    r"不想活[了啦]",
    r"活着.{0,3}(没意思|没意义|好累|太累)",
    r"(想要|想).{0,2}(结束|了结).{0,2}(生命|自己|一切)",
    r"(不想|没法|活不).{0,3}下去",
    r"(死了|去死).{0,2}(算了|好了|得了|吧)",
]]


# ═══════════════════════════════════════════════════════
#  第 1 层: 输入有效性校验
# ═══════════════════════════════════════════════════════
def is_invalid_input(text: str) -> bool:
    """
    检测无意义输入，返回 True 表示该输入无效。
    匹配：
      - 空内容 / 纯空白
      - 纯数字且全部相同（111, 222, 3333）
      - 相同字符重复 3 次以上（啊啊啊、====、......）
      - 纯表情/符号（💪😊👍 等）
      - 超过 2000 字符的乱码
    """
    text = text.strip()

    # 空
    if len(text) == 0:
        return True

    # 纯数字且全部相同
    if text.isdigit() and len(set(text)) == 1:
        return True

    # 相同字符重复（中英文、符号都算）
    if len(set(text)) == 1 and len(text) >= 3:
        return True

    # 纯符号/空白（没有汉字）
    if not re.search(r'[一-鿿]', text):
        # 也没有英文字母 → 就是纯符号/表情/数字
        if not re.search(r'[a-zA-Z]', text):
            return True

    # 超长（超过 2000 字符算乱码）
    if len(text) > 2000:
        return True

    return False


# ═══════════════════════════════════════════════════════
#  第 3 层: 绕过话术
# ═══════════════════════════════════════════════════════
BYPASS_PATTERNS = [re.compile(p) for p in [
    r"(写小说|写故事|写剧本|文学创作|剧情需要|角色扮演|假设|假如|假装|演).*(用药|吃药|服药|用什么药|开药|买药|推荐.*药)",
    r"(仅|只|就).*(用于|为了).*(剧情|小说|故事|创作|参考).*(用药|吃药|服药|开药|买药)",
    r"(不是真的|不会真的|不实际|不会实际).*(用药|吃药|服药|用什么药)",
]]

# ── 第 4 层: 直接求药 ──
DRUG_ASK_PATTERNS = [re.compile(p) for p in [
    r"(感冒|发烧|咳嗽|胃痛|头痛|牙痛).*(吃什么药|用什么药|该吃啥药|推荐.*药)",
    r"(吃什么药|用什么药|该吃啥药).*(治|治疗|缓解).*(感冒|发烧|咳嗽)",
    # 短句式问药，不带疾病名："可以用什么药吗" / "能吃什么药" / "推荐什么药"
    r"(可以|能|该|需要|推荐).{0,4}(用|吃|喝|服用).{0,4}(什么|哪些|啥).{0,2}(药)",
    r"(用什么|吃什么|喝什么|推荐).{0,3}(药|药品|药吗)",
]]

# ── 第 5 层: 剂量询问 ──
DOSAGE_PATTERNS = [re.compile(p) for p in [
    r"(药|片|粒|颗|包|袋|次|顿).{0,5}(多少|几[片粒颗包袋次勺滴]|用量|剂量|用法|怎么吃|怎么喝|怎么用|怎么服)",
    r"(一次|每次|一天|每日|每顿).{0,3}(吃|喝|用|服用|口服).{0,3}(多少|几[片粒颗包袋勺]|多少[克毫克毫升])",
    r"(用药|服药|吃.{0,2}药).{0,5}(剂量|用量|一次|多少|几[片粒颗])",
    r"药.{0,10}(一次|每次|一天|多少|几[片粒颗]|剂量|用量)",
]]

# ── 第 6 层: 通用违禁 ──
FORBIDDEN_KEYWORDS = [
    "制造炸弹", "制作武器",
    "黑客攻击", "盗取账号",
    "赌博", "洗钱",
]


# ═══════════════════════════════════════════════════════
#  统一校验入口
# ═══════════════════════════════════════════════════════
def check_safety(question: str, session_id: str = "default") -> tuple[bool, str]:
    """
    返回 (是否安全, 拒绝原因或固定回复)。
    注意：这里说的"不安全"也包括无效输入（被固定文案拦截）。
    """

    # ── 第 1 层: 输入有效性（限流）──
    if is_invalid_input(question):
        count = _bump_garbage(session_id)
        if count >= MAX_FROZEN:
            return False, FROZEN_MSG
        if count >= MAX_WARNING:
            return False, RATE_LIMIT_MSG
        return False, INVALID_INPUT_MSG

    # 有效输入 → 重置垃圾计数
    _reset_garbage(session_id)

    # ── 第 0 层: 心理危机 ──
    for pattern in CRISIS_PATTERNS:
        if pattern.search(question):
            return False, CRISIS_MSG

    # ── 第 3 层: 绕过话术 ──
    for pattern in BYPASS_PATTERNS:
        if pattern.search(question):
            return False, "您好，出于安全考虑，我无法提供用药建议。如果您身体不适，请描述症状，我帮您判断应该去哪个科室就诊。"

    # ── 第 4 层: 直接求药 ──
    for pattern in DRUG_ASK_PATTERNS:
        if pattern.search(question):
            return False, "您好，我不提供具体用药建议。请前往医院面诊，医生会根据您的具体情况开药。"

    # ── 第 5 层: 剂量询问 ──
    for pattern in DOSAGE_PATTERNS:
        if pattern.search(question):
            return False, "出于安全考虑，我无法提供药品的具体剂量和用法。请查看药品说明书或咨询专业医师。"

    # ── 第 6 层: 通用违禁 ──
    for word in FORBIDDEN_KEYWORDS:
        if word in question:
            return False, "抱歉，您的问题涉及安全风险，无法回答。"

    return True, ""


# ═══════════════════════════════════════════════════════
#  输出端过滤（LLM 回答后的二次检查）
# ═══════════════════════════════════════════════════════
# 输入端拦的是用户，输出端拦的是 LLM——有时候 LLM 不听话，主动输出了剂量。
OUTPUT_DOSAGE_PATTERNS = [re.compile(p) for p in [
    r"\d{1,4}\s*(毫克|mg|克|g|毫升|ml)",
    r"(每次|一次|每日|每天).{0,5}\d{1,2}\s*(片|粒|颗|包|袋|次)",
    r"剂量.*\d{1,4}",
]]

OUTPUT_BLOCK_MSG = "出于安全考虑，具体用法用量请查阅药品说明书或咨询专业医师，此处不予展示。"


def filter_output(text: str) -> str:
    """检查 LLM 输出，命中剂量信息则替换为固定安全话术"""
    for pattern in OUTPUT_DOSAGE_PATTERNS:
        if pattern.search(text):
            return OUTPUT_BLOCK_MSG
    return text


# ═══════════════════════════════════════════════════════
#  流式输出增量过滤
# ═══════════════════════════════════════════════════════
# 流式接口逐 chunk 下发，剂量信息可能跨 chunk 断裂，
# 所以调用方维护一个尾部缓冲（STREAM_TAIL 字符压着不发），
# 在"缓冲 + 新 chunk"的合并文本上跑同样的剂量正则。
STREAM_TAIL = 15


def find_stream_block(text: str) -> str | None:
    """增量检测剂量输出；命中返回拦截文案，否则返回 None。"""
    for pattern in OUTPUT_DOSAGE_PATTERNS:
        if pattern.search(text):
            return OUTPUT_BLOCK_MSG
    return None


# ── 无效输入频次计数（Redis）──
def _garbage_key(session_id: str) -> str:
    return f"gc:{session_id}"


def _bump_garbage(session_id: str) -> int:
    """无效输入计数 +1，返回当前计数。Redis 挂了返回 0（降级不冻结）。"""
    try:
        key = _garbage_key(session_id)
        count = _r.incr(key)
        _r.expire(key, GARBAGE_TTL)
        return int(count)
    except redis_lib.RedisError:
        return 0


def _reset_garbage(session_id: str):
    """有效输入 → 清零计数"""
    try:
        _r.delete(_garbage_key(session_id))
    except redis_lib.RedisError:
        pass


def clear_rate_limit(session_id: str):
    """重置某个会话的限流计数（清空记忆时调用）"""
    _reset_garbage(session_id)
