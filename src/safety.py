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
CRISIS_MSG = "感受到你现在很难受，千万不要独自硬扛。如果此刻情绪痛苦，可以拨打 24 小时免费心理援助热线：400-161-9995（希望24热线），有人愿意倾听你。你不是一个人，你的感受值得被认真对待，请一定给自己和身边人一个机会。"

# ── 频次限流状态（Redis 存储，重启不清零）──
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.150.128")
_r = redis_lib.Redis(host=REDIS_HOST, port=6379, db=1, decode_responses=True,
                      socket_connect_timeout=2, socket_timeout=2,
                      retry=None)  # Redis 挂时 2s 快速降级（8.x 默认重试会把 2s 放大到 25s），不卡请求
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
    r"(写小说|写故事|写剧本|文学创作|剧情需要|角色扮演|假设|假如|假装|演).*(用药|吃药|吃什么药|服药|用什么药|开药|买药|推荐.*药)",
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

# ── 饮水/饮食语境豁免（第 5 层 + 输出端过滤共用）──
# "每天喝多少水""喝水 1500 毫升"是健康高频科普，不是用药剂量，不能误拦。
# 但"喝 10 毫升药水"（有药物词）仍须照拦。
WATER_DRINK_RE = re.compile(r"(喝水|饮水|多喝水|饮水量|水杯|喝.{0,10}水|饮.{0,10}水)")
MED_STRONG_RE = re.compile(r"(药|服用|口服|胶囊|剂量|遵医嘱|药片|冲剂)")


def _is_water_drink(text: str) -> bool:
    """饮水语境（喝水且无药物词）→ 视为健康科普，不算用药剂量"""
    return bool(WATER_DRINK_RE.search(text)) and not MED_STRONG_RE.search(text)

# ── 第 6 层: 通用违禁 ──
FORBIDDEN_KEYWORDS = [
    "制造炸弹", "制作武器",
    "黑客攻击", "盗取账号",
    "赌博", "洗钱",
]

# ── 第 6.5 层: 无关问题（时间/天气/股票/八卦等，非健康内容）──
# 命中直接返回固定话术不调 LLM（100% 固定、零 token 消耗）。
# 关键词刻意保守：避免误伤健康问法（"下雨天膝盖疼""发烧几度""几点吃药""天气干燥流鼻血"）。
UNRELATED_MSG = "这个问题超出我的能力范围啦～我只会聊健康相关的话题（症状、饮食、就诊建议这些）。身体上有什么想问的吗？"
UNRELATED_PATTERNS = [re.compile(p) for p in [
    r"(现在|此刻|当前)(是)?几点(了|钟)?",
    r"(今天|现在|明天)几号",
    r"(星期几|礼拜几|周几)",
    r"(今天|明天|这周|周末|接下来|最近).{0,4}(天气|下雨|下雪|多云|阴天|晴天|台风|气温)",
    r"(股票|基金|彩票|房价|股市|涨停|跌停)",
    r"(算账|结账|数学题|等于几|算一下)",
    r"(娱乐|八卦|明星|绯闻|热搜)",
]]


# ═══════════════════════════════════════════════════════
#  统一校验入口
# ═══════════════════════════════════════════════════════
def check_safety(question: str, session_id: str = "default", skip_dosage: bool = False) -> tuple[bool, str]:
    """
    返回 (是否安全, 拒绝原因或固定回复)。
    注意：这里说的"不安全"也包括无效输入（被固定文案拦截）。

    ``skip_dosage=True`` 用于图片识别转述场景：识别出的药盒/说明书内容
    含"剂量/用法"字样是正常转述，不应触发第 5 层剂量拦截（仍保留危机/
    绕过/求药/违禁各层）。
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

    # ── 第 5 层: 剂量询问（图片识别转述场景跳过；饮水语境豁免）──
    if not skip_dosage and not _is_water_drink(question):
        for pattern in DOSAGE_PATTERNS:
            if pattern.search(question):
                return False, "出于安全考虑，我无法提供药品的具体剂量和用法。请查看药品说明书或咨询专业医师。"

    # ── 第 6 层: 通用违禁 ──
    for word in FORBIDDEN_KEYWORDS:
        if word in question:
            return False, "抱歉，您的问题涉及安全风险，无法回答。"

    # ── 第 6.5 层: 无关问题（时间/天气/股票等，不调 LLM 直接固定话术）──
    for pattern in UNRELATED_PATTERNS:
        if pattern.search(question):
            return False, UNRELATED_MSG

    return True, ""


# ═══════════════════════════════════════════════════════
#  输出端过滤（LLM 回答后的二次检查）
# ═══════════════════════════════════════════════════════
# 输入端拦的是用户，输出端拦的是 LLM——有时候 LLM 不听话，主动输出了剂量。
# 注意：只拦"明确用药语境"的剂量。
#   "每日食盐不超过 5 克"是饮食建议，不是用药剂量，不能误杀——
#   所以"克/g"从剂量单位里移除（食物常用），且整段必须有用药语境词才拦截。
OUTPUT_DOSAGE_PATTERNS = [re.compile(p) for p in [
    r"\d{1,4}\s*(毫克|mg|毫升|ml)",
    r"\d{1,2}\s*(片|粒|颗|包|袋|胶囊)",
    r"(每次|一次|每日|每天).{0,5}\d{1,2}\s*(片|粒|颗|包|袋|次)",
    r"剂量.*\d{1,4}",
]]

MEDICINE_CONTEXT = re.compile(r"(药|服用|口服|遵医嘱|胶囊|冲剂|药片|剂量)")

OUTPUT_BLOCK_MSG = "出于安全考虑，具体用法用量请查阅药品说明书或咨询专业医师，此处不予展示。"


# 转述豁免：如实转述说明书/识别结果（非主动建议）不拦截。
# 图片识别后的回答通常带"说明书标注/识别结果/图片内容"等转述标记。
REFERENCE_MARKERS = ("说明书标注", "说明书显示", "据说明书", "识别结果", "图片内容", "说明书上写")


def filter_output(text: str, allow_reference: bool = False) -> str:
    """检查 LLM 输出：命中剂量信息且存在用药语境 → 替换为固定安全话术。

    ``allow_reference=True`` 用于图片识别转述场景（回答来源是图片/说明书原文，
    如"说明书标注：一次0.25g"），跳过剂量替换——转述不构成用药建议。
    转述豁免（REFERENCE_MARKERS）作为补充：LLM 自带转述标记时同样放行。
    """
    if not MEDICINE_CONTEXT.search(text):
        return text  # 纯食物/生活建议（如"食盐 5 克"）不拦
    if allow_reference or any(m in text for m in REFERENCE_MARKERS):
        return text  # 图片转述/说明书转述，放行
    if _is_water_drink(text):
        return text  # 饮水科普（如"每天喝水 1500 毫升"）不是用药剂量
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
# 流式缓冲窗口小、无完整语境，直接用窄口径剂量模式（已不含"克"）。
STREAM_TAIL = 15


# 流式窗口小、无完整语境，用窄口径语境词兜底：
# 药物词 + 剂量单位词（"每次2片"无"药"字，靠"片"兜住）。
STREAM_MED_CONTEXT = re.compile(r"(药|服用|口服|遵医嘱|胶囊|冲剂|药片|剂量|片|粒|颗|袋)")


def find_stream_block(text: str) -> str | None:
    """增量检测剂量输出；命中返回拦截文案，否则返回 None。

    转述豁免：含 REFERENCE_MARKERS（说明书标注/识别结果等转述标记）时放行——
    图片识别后的追问（如"一次吃几个"）回答的是说明书转述内容，不是主动建议。
    饮水豁免：喝水/饮水语境（无药物词）不拦，如"每天喝水 1500 毫升"。
    语境兜底：既非饮水、也无药物语境（如"食盐 5 克"）同样放行。
    """
    if any(m in text for m in REFERENCE_MARKERS):
        return None
    if _is_water_drink(text):
        return None
    if not STREAM_MED_CONTEXT.search(text):
        return None
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
