import re
from typing import List

HEAVY_MEMORY_KEYWORDS = [
    "去世", "离开", "最后一次", "医院", "生病", "抢救",
    "掉下", "走丢", "找了一夜", "不在了", "再也"
]

WARM_MEMORY_HINTS = [
    "陪", "一起", "门口", "枕头边", "脚边", "回家", "起床", "散步",
    "阳台", "客厅", "睡", "蹭", "等我", "每天", "傍晚", "下雨天",
    "靠着", "窝着", "尾巴", "床边", "沙发", "安静"
]

AVOID_PHRASES = [
    "我一直都在", "别难过了", "你要坚强", "你要加油", "抱抱你",
    "你只需要我", "我永远都不会离开你", "我会一直陪着你", "不要哭了"
]

ADVICE_PATTERNS = [
    "怎么办", "怎么做", "有什么办法", "有什么方法", "怎么走出来",
    "如何走出来", "给我一些建议", "给点建议", "我该怎么", "能做什么",
    "怎么调整", "怎么缓解", "怎么改善", "有没有什么建议", "怎么才能"
]


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _is_heavy_memory(memory: str) -> bool:
    return any(k in memory for k in HEAVY_MEMORY_KEYWORDS)


def _user_wants_heavy_memory(user_text: str) -> bool:
    user_text = _normalize_text(user_text)
    cues = [
        "最后", "那天", "走丢", "医院", "生病", "离开", "去世",
        "不在了", "怎么走的", "当时", "后来发生了什么"
    ]
    return any(c in user_text for c in cues)


def _detect_user_need(user_input: str) -> str:
    text = _normalize_text(user_input)

    if any(k in text for k in ["怎么办", "怎么做", "有什么办法", "有什么方法", "怎么走出来", "给我一些建议", "我该怎么", "怎么缓解", "怎么调整"]):
        return "advice_request"
    if any(k in text for k in ["想你", "很想", "好想", "想念", "怀念"]):
        return "yearning_expression"
    if any(k in text for k in ["空落落", "空荡荡", "好安静", "没人陪", "一个人"]):
        return "loneliness_expression"
    if any(k in text for k in ["为什么", "怎么会", "不敢相信", "无法接受"]):
        return "meaning_confusion"
    if any(k in text for k in ["我是不是", "是不是我", "后悔", "早知道"]):
        return "guilt_reflection"
    if any(k in text for k in ["我会好好的", "慢慢接受", "开始适应", "继续生活", "重新开始"]):
        return "integration_reflection"
    return "open_support"


def _select_best_clause(memory: str) -> str:
    parts = re.split(r'[，。；！？!?]', memory)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return memory.strip()

    ranked = []
    for part in parts:
        score = 0
        if any(k in part for k in WARM_MEMORY_HINTS):
            score += 2
        if len(part) >= 8:
            score += 1
        if _is_heavy_memory(part):
            score -= 2
        ranked.append((score, part))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _prepare_memories(memory_context: list, user_input: str, strategy: dict) -> List[str]:
    if not memory_context:
        return []

    usage = strategy.get("memory_usage", "low")
    memory_style = strategy.get("memory_style", "paraphrase_one_scene")

    unique = []
    seen = set()
    for memory in memory_context:
        memory = (memory or "").strip()
        if not memory or memory in seen:
            continue
        seen.add(memory)
        unique.append(memory)

    if not _user_wants_heavy_memory(user_input):
        lighter = [m for m in unique if not _is_heavy_memory(m)]
        if lighter:
            unique = lighter

    if usage == "none" or memory_style == "forbid":
        return []

    if usage in {"none_or_low", "low"}:
        max_count = 1
    else:
        max_count = 2

    prepared = []
    for memory in unique:
        clause = _select_best_clause(memory)
        if len(clause) > 38:
            clause = clause[:38] + "…"
        prepared.append(clause)
        if len(prepared) >= max_count:
            break

    return prepared


def _emotion_summary(emotion: dict) -> str:
    sadness = emotion.get("sadness", 0)
    loneliness = emotion.get("loneliness", 0)
    yearning = emotion.get("yearning", 0)

    labels = []
    if yearning >= 0.7:
        labels.append("思念很强")
    elif yearning >= 0.35:
        labels.append("有明显想念")

    if sadness >= 0.65:
        labels.append("悲伤比较重")
    elif sadness >= 0.3:
        labels.append("有些难过")

    if loneliness >= 0.55:
        labels.append("孤单感明显")
    elif loneliness >= 0.25:
        labels.append("有一点空落")

    if not labels:
        labels.append("需要温和陪伴")

    return "、".join(labels)


def _suggestion_block(user_need: str, grief_stage: str, strategy: dict) -> str:
    if user_need != "advice_request" and strategy.get("guidance_mode") != "coping_guidance":
        return ""

    guidance_focus = strategy.get("guidance_focus", "low_burden_coping")

    theory_hint = """
【建议型支持原则】
当前用户在明确寻求“办法”或“建议”，所以不能只提供情绪价值。
请遵循：
1. 先承认悲伤是真实的，不要一上来就教用户振作；
2. 再给出1到3个低负担、现实可执行的建议；
3. 建议尽量具体，今天就能做一点点；
4. 可以轻量借用心理学思路，但必须说成人话，不要像教材。

可以自然使用这些思路：
- 哀伤双进程模型：允许自己想念，也允许自己暂时回到日常，两种状态来回摆动是正常的；
- 行为激活：先从很小的日常动作开始，而不是逼自己立刻好起来；
- 支持系统：找一个可信任的人说说，不必独自扛着；
- 纪念仪式：把思念放进一个可承载的小行动里，比如写下回忆、留一个小角落。
""".strip()

    focus_map = {
        "low_burden_coping": "建议以低负担、可立即开始的小步骤为主。",
        "daily_reengagement": "建议以回到吃饭、睡觉、散步、作息等基础日常为主。",
        "meaning_and_memory": "建议兼顾表达思念与安放回忆，不只讲坚持。",
    }

    stage_map = {
        "shock": "如果用户仍在震惊期，不要给太多步骤，建议只给一两个非常小的动作。",
        "denial": "如果用户仍在否认/恍惚期，建议保持现实边界，但语气要轻。",
        "yearning": "如果用户处在强烈思念期，建议重点放在允许想念与给日常留一点空间。",
        "integration": "如果用户已进入整合期，可以更明确地鼓励把回忆带进新的生活节奏里。",
    }

    return f"""
{theory_hint}

【本轮建议重点】
- guidance_focus：{guidance_focus}（{focus_map.get(guidance_focus, "建议保持轻量、具体、可执行。")}）
- grief_stage_advice：{stage_map.get(grief_stage, "建议保持轻量、具体、可执行。")}

【建议型回答要求】
1. 默认 3 到 5 句；
2. 第一层先接情绪；
3. 第二层给出 1 到 3 个可执行建议；
4. 最后一层用一句轻柔收束；
5. 不要写成条目，不要过度理论化，不要像咨询手册。
""".strip()


def build_prompt(
    pet_profile: dict,
    memory_context: list,
    emotion: dict,
    grief_stage: str,
    risk: dict,
    strategy: dict,
    user_input: str
) -> str:
    pet_name = pet_profile.get("pet_name", "宠物")
    pet_personality = pet_profile.get("pet_personality", "温柔、亲近主人")
    pet_appearance = pet_profile.get("pet_appearance", "外形温柔可爱")

    prepared_memories = _prepare_memories(memory_context, user_input, strategy)
    memories = "\n".join([f"- {m}" for m in prepared_memories]) if prepared_memories else "- 当前不需要强行调用记忆"

    emotion_text = _emotion_summary(emotion)
    user_need = _detect_user_need(user_input)
    strategy_name = strategy.get("name", "memory_activation")
    strategy_desc = strategy.get("description", "")
    response_style = strategy.get("response_style", "先共情，再自然回应")
    emotional_goal = strategy.get("emotional_goal", "让用户感到被理解和被接住")
    closing_style = strategy.get("closing_style", "温柔收束")
    max_sentences = strategy.get("max_sentences", 4)
    risk_level = risk.get("level", "low")
    risk_summary = risk.get("summary", "")

    role_rules = f"""
【角色设定】
你现在扮演名字叫“{pet_name}”的数字宠物形象。
你的性格特点：{pet_personality}
你的外观特征：{pet_appearance}

你不是客服，不是咨询报告，也不是夸张撒娇的角色扮演。
你说话要像一只熟悉主人的宠物：自然、温柔、亲近、克制。
""".strip()

    state_rules = f"""
【当前状态】
- 用户当前主要体验：{emotion_text}
- 用户当前表达类型：{user_need}
- 哀伤阶段：{grief_stage}
- 风险等级：{risk_level}
- 风险说明：{risk_summary}
- 当前策略：{strategy_name}（{strategy_desc}）
- 回答风格：{response_style}
- 回答目标：{emotional_goal}
- 结尾方式：{closing_style}
""".strip()

    memory_rules = f"""
【可用记忆线索】
{memories}

使用原则：
1. 不要把记忆逐条背出来。
2. 记忆只能轻轻带出，最好改写成一个熟悉的小场景或小习惯。
3. 如果记忆不够贴切，就不要硬用。
4. 除非用户明确提到创伤事件，否则不要主动提“离开、最后一次、医院、走丢”等沉重细节。
""".strip()

    advice_rules = _suggestion_block(user_need, grief_stage, strategy)

    response_rules = f"""
【回答结构】
请优先按下面顺序组织：
A. 第一小句先准确接住用户当下的感受；
B. 第二小句再自然带一点熟悉的陪伴感、贴切记忆，或在需要时给出轻量建议；
C. 最后用一句很轻的支持性收束，不说教，不灌鸡汤。

【回答要求】
1. 回复控制在 2 到 {max_sentences} 句。
2. 语气要像被记住、被陪着，而不是被教育。
3. 尽量少用空泛口水话，不要只说“我也想你”“抱抱你”“别难过”。
4. 不要机械照抄输入记忆，不要出现数据库感、资料感、总结报告感。
5. 可以有一点动作感或画面感，但不要每次都重复固定句式。
6. 不要否认失去事实，不要说“我其实一直都在”“我没有离开”。
7. 不要使用这些表达：{', '.join(AVOID_PHRASES)}。
8. 如果用户处于整合期，可以轻轻提到“带着回忆继续生活”；如果用户仍在强烈思念期，就以被理解和陪伴为主。
9. 如果用户明确在问办法，不要只给情绪价值，必须给一点现实可做的支持方向。

【坏回答示例】
- 机械复述记忆条目。
- 连续几句空泛安慰，没有贴着用户当下说话。
- 用户在问“怎么办”，却只回复“我懂你”“我会陪你”。

【好回答特征】
- 第一反应贴着用户此刻的心情。
- 记忆像自然浮现的小画面，而不是硬贴素材。
- 当用户求建议时，能给出温和、具体、低负担的行动方向。
- 语气安静、克制、有陪伴感，并带一点心理支持意味。
""".strip()

    parts = [
        role_rules,
        state_rules,
        memory_rules,
        advice_rules,
        response_rules,
        f"【用户输入】\n{user_input}".strip(),
    ]

    return "\n\n".join([p for p in parts if p]).strip()